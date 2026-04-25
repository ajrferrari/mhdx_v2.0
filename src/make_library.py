"""
make_library.py
===============
Aggregate N protein-identification CSVs (from multiple undeuterated runs)
into a single consensus library master list.

Processing order
----------------
1. Read and tag all input CSVs with a run index.
2. Concatenate into one DataFrame.
3. Filter: im_mono > im_min, idotp >= idotp_min, abs_ppm <= abs_ppm_max.
4. Sort by [name, charge, RT, abs_ppm].
5. Cluster by RT proximity per (name, charge) — vectorized sequential scan.
6. Aggregate per (name, charge, cluster_id):
     weighted_average_rt  (weight = ab_cluster_total)
     weighted_average_im  (weight = ab_cluster_total)
     n_runs               (distinct run_idx values seen)
     representative row   (lowest abs_ppm entry)
7. Within each (name, cluster_id): keep one charge state (lowest abs_ppm).
8. Build name_rt-group label: f"{name}_{weighted_average_rt:.2f}".
9. Write output CSV.

Usage
-----
    python make_library.py run1_ids.csv run2_ids.csv run3_ids.csv \\
        --output results/library/library_master.csv \\
        --rt_group_cutoff 0.2 \\
        --idotp_min 0.8 \\
        --abs_ppm_max 10.0 \\
        --im_min 10

Output columns
--------------
name_rt-group, weighted_average_rt, weighted_average_im, n_runs,
name, sequence, MW, charge, RT, im_mono, ab_cluster_total,
expect_mz, obs_mz, ppm, abs_ppm, idotp
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

_OUTPUT_COLUMNS = [
    "name_rt-group",
    "weighted_average_rt",
    "weighted_average_im",
    "n_runs",
    "name",
    "sequence",
    "MW",
    "charge",
    "RT",
    "im_mono",
    "ab_cluster_total",
    "expect_mz",
    "obs_mz",
    "ppm",
    "abs_ppm",
    "idotp",
]

_REQUIRED_COLUMNS = {
    "name", "sequence", "RT", "im_mono", "ab_cluster_total",
    "MW", "charge", "expect_mz", "obs_mz", "ppm", "abs_ppm", "idotp",
}


def load_and_tag(input_paths: list[str]) -> pd.DataFrame:
    """Read N identification CSVs and tag each row with run_idx (0-based)."""
    frames = []
    for idx, path in enumerate(input_paths):
        df = pd.read_csv(path)
        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{path}: missing required columns: {missing}")
        df["run_idx"] = idx
        frames.append(df)
        print(f"  [{idx}] {path}: {len(df)} rows")
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(combined)} rows from {len(input_paths)} file(s)")
    return combined


def apply_filters(
    df: pd.DataFrame,
    *,
    im_min: float = 10.0,
    idotp_min: float = 0.0,
    abs_ppm_max: float = float("inf"),
) -> pd.DataFrame:
    """Apply hard-cut quality filters. Prints row counts at each step."""
    n0 = len(df)
    df = df[df["im_mono"] > im_min].copy()
    print(f"  After im_mono > {im_min}: {len(df)} rows ({n0 - len(df)} dropped)")

    if idotp_min > 0.0:
        n1 = len(df)
        df = df[df["idotp"].notna() & (df["idotp"] >= idotp_min)].copy()
        print(f"  After idotp >= {idotp_min}: {len(df)} rows ({n1 - len(df)} dropped)")

    if abs_ppm_max < float("inf"):
        n2 = len(df)
        df = df[df["abs_ppm"] <= abs_ppm_max].copy()
        print(f"  After abs_ppm <= {abs_ppm_max}: {len(df)} rows ({n2 - len(df)} dropped)")

    return df.reset_index(drop=True)


def assign_rt_clusters(df: pd.DataFrame, rt_group_cutoff: float = 0.2) -> pd.DataFrame:
    """Assign a globally unique cluster_id per protein name and RT-proximity group.

    Within each (name) group sorted by RT, a new cluster starts whenever the
    gap to the previous row exceeds rt_group_cutoff.  Different charge states
    at similar RTs share the same cluster_id, which allows
    keep_best_charge_per_cluster to select between them.
    """
    df = df.sort_values(["name", "RT", "abs_ppm"]).reset_index(drop=True)
    prev_rt = df.groupby("name")["RT"].shift(1)
    gap = (df["RT"] - prev_rt).fillna(0.0)
    is_new_group = df["name"].ne(df["name"].shift())
    new_cluster = (gap > rt_group_cutoff) | is_new_group
    df["cluster_id"] = (new_cluster.cumsum() - 1).astype(int)
    return df


def aggregate_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse each (name, charge, cluster_id) into one representative row.

    Cluster-level stats (weighted_average_rt, weighted_average_im, n_runs)
    are computed across ALL charge states in the cluster, matching the legacy
    behavior of 4_make_library_master_list.py.  The representative row for
    each (name, charge, cluster_id) is the entry with the lowest abs_ppm.
    """
    def _cluster_stats(g):
        w = g["ab_cluster_total"].clip(lower=0).values.astype(np.float64)
        w_sum = w.sum() or 1.0
        return pd.Series({
            "weighted_average_rt": float(np.dot(w, g["RT"].values) / w_sum),
            "weighted_average_im": float(np.dot(w, g["im_mono"].values) / w_sum),
            "n_runs": int(g["run_idx"].nunique()),
        })

    cluster_stats = (
        df.groupby(["name", "cluster_id"], sort=False)
          .apply(_cluster_stats)
          .reset_index()
    )

    rep_idx = df.groupby(["name", "charge", "cluster_id"])["abs_ppm"].idxmin()
    reps = df.loc[rep_idx].reset_index(drop=True)

    return reps.merge(cluster_stats, on=["name", "cluster_id"], how="left")


def keep_best_charge_per_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """Within each (name, cluster_id), keep the charge with the lowest abs_ppm."""
    df = df.sort_values(["name", "cluster_id", "abs_ppm", "charge"])
    best_idx = df.groupby(["name", "cluster_id"])["abs_ppm"].idxmin()
    return df.loc[best_idx].reset_index(drop=True)


def build_name_rt_group(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 'name_rt-group' label: f'{name}_{weighted_average_rt:.2f}'."""
    df = df.copy()
    df["name_rt-group"] = [
        f"{row['name']}_{row['weighted_average_rt']:.2f}"
        for _, row in df.iterrows()
    ]
    return df


def make_library(
    input_paths: list[str],
    output_path: str,
    *,
    rt_group_cutoff: float = 0.2,
    idotp_min: float = 0.0,
    abs_ppm_max: float = float("inf"),
    im_min: float = 10.0,
) -> pd.DataFrame:
    """Aggregate N identification CSVs into a consensus library master list."""
    print(f"[make_library] Reading {len(input_paths)} input file(s)...")
    df = load_and_tag(input_paths)
    n_input = len(df)

    print("[make_library] Applying filters...")
    df = apply_filters(df, im_min=im_min, idotp_min=idotp_min, abs_ppm_max=abs_ppm_max)
    n_filtered = len(df)

    if df.empty:
        print("[make_library] WARNING: No rows remain after filtering. Writing empty output.")
        out = pd.DataFrame(columns=_OUTPUT_COLUMNS)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if 'csv' in output_path:
            out.to_csv(output_path, index=False)
        elif 'json' in output_path:
            out.to_json(output_path, index=False)
        return out

    print(f"[make_library] Assigning RT clusters (cutoff={rt_group_cutoff} min)...")
    df = assign_rt_clusters(df, rt_group_cutoff=rt_group_cutoff)

    print("[make_library] Aggregating clusters...")
    df = aggregate_clusters(df)

    print("[make_library] Selecting best charge per (name, cluster)...")
    df = keep_best_charge_per_cluster(df)

    print("[make_library] Building name_rt-group labels...")
    df = build_name_rt_group(df)

    for col in _OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    df = df[_OUTPUT_COLUMNS]

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if 'csv' in output_path:
        df.to_csv(output_path, index=False)
    elif 'json' in output_path:
        df.to_json(output_path)

    print(
        f"\n[make_library] Summary:\n"
        f"  Input rows (all files):  {n_input}\n"
        f"  After filters:           {n_filtered}\n"
        f"  Library entries:         {len(df)}\n"
        f"  Output -> {output_path}"
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate N protein-identification CSVs into a consensus "
            "library master list."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input identification CSV files (one per undeuterated run). "
             "Shell globbing is supported: make_library.py results/id/*/*_identifications.csv ...",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the library master list CSV.",
    )
    parser.add_argument(
        "--rt_group_cutoff",
        type=float,
        default=0.2,
        help="Maximum RT gap [min] within an RT cluster (default: 0.2).",
    )
    parser.add_argument(
        "--idotp_min",
        type=float,
        default=0.0,
        help="Minimum idotp to retain a row (default: 0.0, disabled). "
             "Rows with NaN idotp are kept when this is 0.0.",
    )
    parser.add_argument(
        "--abs_ppm_max",
        type=float,
        default=float("inf"),
        help="Maximum abs_ppm to retain a row (default: no limit).",
    )
    parser.add_argument(
        "--im_min",
        type=float,
        default=10.0,
        help="Minimum im_mono (drift time bins) to retain a row (default: 10).",
    )

    args = parser.parse_args()
    make_library(
        input_paths=args.inputs,
        output_path=args.output,
        rt_group_cutoff=args.rt_group_cutoff,
        idotp_min=args.idotp_min,
        abs_ppm_max=args.abs_ppm_max,
        im_min=args.im_min,
    )

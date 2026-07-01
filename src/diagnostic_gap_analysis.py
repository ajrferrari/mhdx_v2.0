"""Compare NTF pipeline output against DriftScope reference to identify missing signals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def match_signals(
    ref: pd.DataFrame,
    pipe: pd.DataFrame,
    mz_ppm: float = 10.0,
    rt_tol: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (matched_ref_rows, unmatched_ref_rows).

    A ref row is matched when a pipeline row exists with the same charge,
    obs_mz within mz_ppm, and RT within rt_tol.
    """
    matched_idx: list[int] = []
    for i, row in ref.iterrows():
        z = int(row["charge"])
        cands = pipe[pipe["charge"] == z]
        if cands.empty:
            continue
        ppm = np.abs(cands["monoisotopic_mz"] - row["obs_mz"]) / row["obs_mz"] * 1e6
        rt_diff = np.abs(cands["rt_center"] - row["RT"])
        if not cands[(ppm <= mz_ppm) & (rt_diff <= rt_tol)].empty:
            matched_idx.append(i)
    mask = ref.index.isin(matched_idx)
    return ref[mask].copy(), ref[~mask].copy()


def charge_family_gaps(
    ref: pd.DataFrame,
    matched_names: set[str],
    recovered_charges: dict[str, set[int]],
) -> dict[str, dict]:
    """For each detected protein, report which reference charge states are missing.

    Returns:
        {protein_name: {"found": set, "missing": set, "all_ref": set}}
    """
    result: dict[str, dict] = {}
    for name in matched_names:
        all_ref_z = set(ref[ref["name"] == name]["charge"].astype(int))
        found_z = recovered_charges.get(name, set())
        missing_z = all_ref_z - found_z
        if missing_z:
            result[name] = {"found": found_z, "missing": missing_z, "all_ref": all_ref_z}
    return result


def run_gap_analysis(
    ref_csv: str | Path,
    pipe_csv: str | Path,
    mz_ppm: float = 10.0,
    rt_tol: float = 0.3,
    output_csv: str | Path | None = None,
) -> None:
    """Print recovery summary and optionally save unmatched signals."""
    ref = pd.read_csv(ref_csv)
    pipe = pd.read_csv(pipe_csv)

    if "is_best" in pipe.columns:
        pipe = pipe[pipe["is_best"] == True].copy()

    print(f"Reference signals:          {len(ref)}")
    print(f"Pipeline signals (is_best): {len(pipe)}")

    matched, unmatched = match_signals(ref, pipe, mz_ppm=mz_ppm, rt_tol=rt_tol)
    recovery = len(matched) / len(ref) * 100 if len(ref) else 0.0
    print(f"\nOverall recovery: {len(matched)}/{len(ref)} = {recovery:.1f}%")

    print("\nRecovery by charge state:")
    for z in sorted(ref["charge"].unique()):
        ref_z = ref[ref["charge"] == z]
        match_z = matched[matched["charge"] == z]
        pct = len(match_z) / len(ref_z) * 100 if len(ref_z) else 0.0
        print(f"  z={int(z):2d}: {len(match_z):4d}/{len(ref_z):4d} = {pct:5.1f}%")

    recovered_charges: dict[str, set[int]] = {}
    for _, row in matched.iterrows():
        recovered_charges.setdefault(row["name"], set()).add(int(row["charge"]))

    gaps = charge_family_gaps(ref, set(matched["name"]), recovered_charges)
    print(f"\nProteins with partial charge-state recovery: {len(gaps)}")
    for name, info in sorted(gaps.items(), key=lambda x: -len(x[1]["missing"]))[:20]:
        print(f"  {name}: found z={sorted(info['found'])}, missing z={sorted(info['missing'])}")

    print(f"\nUnmatched signals: {len(unmatched)}")
    if output_csv:
        unmatched.to_csv(output_csv, index=False)
        print(f"Saved unmatched to {output_csv}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Gap analysis: NTF pipeline vs reference")
    p.add_argument("ref_csv", help="Reference CSV (260107_AF2501_0.mzML_intermediate.csv)")
    p.add_argument("pipe_csv", help="Pipeline output (*_isotopes_filtered.csv)")
    p.add_argument("--mz_ppm", type=float, default=10.0)
    p.add_argument("--rt_tol", type=float, default=0.3)
    p.add_argument("--output_csv", default=None, help="Save unmatched signals here")
    args = p.parse_args()
    run_gap_analysis(args.ref_csv, args.pipe_csv, args.mz_ppm, args.rt_tol, args.output_csv)

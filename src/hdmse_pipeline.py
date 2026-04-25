# src/hdmse_pipeline.py
"""
hdmse_pipeline.py
=================
SDK-facing orchestrator for the HDMS^E anchor-and-project library pipeline.

This module imports the Waters SDK transitively via ``waters_reader`` and
therefore only runs on x86-64 Linux/Windows inside the Singularity
container. The pure-numpy library logic lives in ``hdmse_library`` and is
unit-tested separately.

Public functions
----------------
* ``process_lce_window`` — process one LCE m/z window for one open reader,
  returns a list of library rows.
* ``process_raw_file``   — iterate all LCE m/z windows for one .raw file
  and write a single Parquet library.

Both functions mutate no global state and write only the file specified in
``output_path``.
"""
from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from hdmse_library import (
    LIBRARY_SCHEMA,
    build_library_row,
    extract_anchor,
    extract_fragments,
    project_anchor_onto_hce,
    write_library_parquet,
)


def _full_dt_range(reader, function: int) -> Tuple[int, int]:
    """Return (dt_lo, dt_hi) covering every drift bin of *function*."""
    meta = reader.metadata()
    n_drift = meta.n_drift_bins[function]
    return 0, int(n_drift - 1)


def _full_rt_range(reader, function: int) -> Tuple[float, float]:
    rt_lo, rt_hi = reader.metadata().rt_range[function]
    return float(rt_lo), float(rt_hi)


def process_lce_window(
    reader,
    *,
    sample: str,
    lce_function: int,
    hce_function: int,
    lce_mz_lo: float,
    lce_mz_hi: float,
    hce_mz_lo: float,
    hce_mz_hi: float,
    hce_mz_slab: float,
    rho_threshold: float,
    min_fragment_intensity: float,
    peak_distance_da: float,
    rank_init: int,
    rank_max: int,
    n_restarts: int,
    rt_r2_min: float,
    dt_r2_min: float,
    charge_range: Tuple[int, int],
    min_cosine: float,
    min_peaks_per_cluster: int,
    mz_bin: float,
    gauss_sigma_rt: float,
    gauss_sigma_dt: float,
    intensity_floor: float,
    verbose: bool = False,
) -> List[Dict[str, object]]:
    """Run the anchor-and-project pipeline on one LCE m/z window.

    Steps
    -----
    1. ``tensor_analysis.analyze_chunk`` on (full RT × full DT × LCE window).
    2. ``isotope_analysis.process_all_factors`` → per-factor precursor rows.
    3. For each kept factor: ``extract_anchor`` from A[:, k], B[:, k].
    4. Iterate HCE m/z slabs: ``tensor_analysis.build_tensor`` →
       ``project_anchor_onto_hce`` → ``extract_fragments``. Concatenate.
    5. ``build_library_row`` per (factor, cluster) pair.

    Returns
    -------
    list of dicts in LIBRARY_SCHEMA shape (may be empty).
    """
    from tensor_analysis import analyze_chunk, build_tensor
    from isotope_analysis import process_all_factors

    rt_lo, rt_hi = _full_rt_range(reader, lce_function)
    dt_lo, dt_hi = _full_dt_range(reader, lce_function)

    if verbose:
        print(f"[hdmse] LCE NTF  m/z={lce_mz_lo:.1f}-{lce_mz_hi:.1f}  RT={rt_lo:.2f}-{rt_hi:.2f}")
    lce = analyze_chunk(
        reader, lce_function,
        rt_lo, rt_hi, dt_lo, dt_hi, lce_mz_lo, lce_mz_hi,
        mz_bin=mz_bin,
        gauss_sigma_rt=gauss_sigma_rt, gauss_sigma_dt=gauss_sigma_dt,
        intensity_floor=intensity_floor,
        rank_init=rank_init, rank_max=rank_max, n_restarts=n_restarts,
        rt_r2_min=rt_r2_min, dt_r2_min=dt_r2_min,
        apply_quality_filter=True,
        plot=False,
        verbose=verbose,
    )
    A, B, C = lce.get("A"), lce.get("B"), lce.get("C")
    if A is None or A.shape[1] == 0:
        if verbose:
            print("[hdmse]   no LCE factors survived quality filter")
        return []

    precursors_df = process_all_factors(
        A, B, C,
        mz_axis=lce["mz_axis_ntf"],
        rt_axis=lce["rt_axis_ntf"],
        dt_axis=lce["dt_axis_ntf"],
        charge_range=charge_range,
        min_cosine=min_cosine,
        min_peaks_per_cluster=min_peaks_per_cluster,
        output_dir=None,
        verbose=verbose,
        mz_axis_full=lce.get("mz_axis"),
        mask_mz=lce.get("mask_mz"),
    )
    precursors_df = precursors_df[precursors_df["k"] == 0].copy()
    if len(precursors_df) == 0:
        return []

    rt_axis_ntf = lce["rt_axis_ntf"]
    dt_axis_ntf = lce["dt_axis_ntf"]
    anchors_by_factor: Dict[int, Dict[str, object]] = {}
    for factor_idx in precursors_df["factor_idx"].astype(int).unique():
        anchors_by_factor[int(factor_idx)] = extract_anchor(
            a_vec=A[:, int(factor_idx)],
            b_vec=B[:, int(factor_idx)],
            rt_axis=rt_axis_ntf,
            dt_axis=dt_axis_ntf,
        )

    slab_starts = np.arange(hce_mz_lo, hce_mz_hi, hce_mz_slab, dtype=np.float64)
    slabs = [(float(s), float(min(s + hce_mz_slab, hce_mz_hi))) for s in slab_starts]
    if verbose:
        print(f"[hdmse]   {len(precursors_df)} precursors × {len(slabs)} HCE slabs")

    fragments_by_factor: Dict[int, List[Dict[str, float]]] = {
        f: [] for f in anchors_by_factor
    }

    for slab_lo, slab_hi in slabs:
        hce_tensor, hce_rt_axis, hce_dt_axis, hce_mz_axis = build_tensor(
            reader, hce_function,
            rt_lo, rt_hi,
            dt_lo, dt_hi,
            slab_lo, slab_hi,
            mz_bin=mz_bin,
            gauss_sigma_rt=gauss_sigma_rt, gauss_sigma_dt=gauss_sigma_dt,
            intensity_floor=intensity_floor,
        )
        if hce_tensor.size == 0 or float(hce_tensor.max()) == 0.0:
            continue

        mask_rt = lce.get("mask_rt")
        mask_dt = lce.get("mask_dt")
        if mask_rt is not None:
            hce_tensor = hce_tensor[mask_rt, :, :]
        if mask_dt is not None:
            hce_tensor = hce_tensor[:, mask_dt, :]

        for factor_idx, anchor in anchors_by_factor.items():
            rho, intensity = project_anchor_onto_hce(hce_tensor, anchor)
            slab_fragments = extract_fragments(
                mz_axis=hce_mz_axis,
                rho=rho,
                intensity=intensity,
                rho_threshold=rho_threshold,
                min_intensity=min_fragment_intensity,
                peak_distance_da=peak_distance_da,
            )
            fragments_by_factor[factor_idx].extend(slab_fragments)

    for factor_idx, fragments in fragments_by_factor.items():
        best_by_fragment = {}
        for fragment in fragments:
            key = round(float(fragment["mz"]), 3)
            current = best_by_fragment.get(key)
            if current is None or float(fragment["intensity"]) > float(current["intensity"]):
                best_by_fragment[key] = fragment
        fragments_by_factor[factor_idx] = list(best_by_fragment.values())

    rows: List[Dict[str, object]] = []
    for _, prec in precursors_df.iterrows():
        factor_idx = int(prec["factor_idx"])
        anchor = anchors_by_factor.get(factor_idx)
        if anchor is None:
            continue
        rows.append(build_library_row(
            sample=sample,
            anchor=anchor,
            precursor=dict(
                factor_idx=factor_idx,
                cluster_idx=int(prec["cluster_idx"]),
                charge=int(prec["charge"]),
                monoisotopic_mz=float(prec["monoisotopic_mz"]),
                monoisotopic_mass_da=float(prec["monoisotopic_mass_da"]),
                cluster_intensity=float(prec["cluster_intensity"]),
                cosine_similarity=float(prec["cosine_similarity"]),
            ),
            fragments=fragments_by_factor.get(factor_idx, []),
        ))
    return rows


def process_raw_file(
    raw_path: str,
    license_path: str,
    output_path: str,
    *,
    lce_function: int = 0,
    hce_function: int = 1,
    lce_mz_window: float = 50.0,
    lce_mz_step: float = 50.0,
    hce_mz_lo: float = 100.0,
    hce_mz_hi: float = 2000.0,
    hce_mz_slab: float = 100.0,
    rho_threshold: float = 0.85,
    min_fragment_intensity: float = 50.0,
    peak_distance_da: float = 0.5,
    rank_init: int = 5,
    rank_max: int = 15,
    n_restarts: int = 3,
    rt_r2_min: float = 0.85,
    dt_r2_min: float = 0.85,
    charge_range: Tuple[int, int] = (3, 15),
    min_cosine: float = 0.5,
    min_peaks_per_cluster: int = 3,
    mz_bin: float = 0.001,
    gauss_sigma_rt: float = 1.0,
    gauss_sigma_dt: float = 1.0,
    intensity_floor: float = 10.0,
    verbose: bool = False,
) -> None:
    """Iterate LCE m/z windows for one raw file and write a Parquet library."""
    from waters_reader import WatersRawReader, read_license

    sample = Path(raw_path).stem
    license_key = read_license(license_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    with WatersRawReader(raw_path, license=license_key) as reader:
        meta = reader.metadata()
        n_fn = meta.n_functions
        for name, idx in (("lce_function", lce_function), ("hce_function", hce_function)):
            if not 0 <= idx < n_fn:
                raise ValueError(
                    f"{name}={idx} not present (file has {n_fn} functions)"
                )
        if meta.n_scans[lce_function] != meta.n_scans[hce_function]:
            raise ValueError(
                f"LCE/HCE scan counts differ "
                f"({meta.n_scans[lce_function]} vs {meta.n_scans[hce_function]}) "
                "— interleaved acquisition assumption violated."
            )

        lce_mz_min, lce_mz_max = meta.mass_range[lce_function]
        starts = np.arange(lce_mz_min, lce_mz_max, lce_mz_step, dtype=np.float64)
        failed_windows: List[str] = []
        for s in starts:
            lo = float(s)
            hi = float(min(s + lce_mz_window, lce_mz_max))
            try:
                rows = process_lce_window(
                    reader,
                    sample=sample,
                    lce_function=lce_function, hce_function=hce_function,
                    lce_mz_lo=lo, lce_mz_hi=hi,
                    hce_mz_lo=hce_mz_lo, hce_mz_hi=hce_mz_hi,
                    hce_mz_slab=hce_mz_slab,
                    rho_threshold=rho_threshold,
                    min_fragment_intensity=min_fragment_intensity,
                    peak_distance_da=peak_distance_da,
                    rank_init=rank_init, rank_max=rank_max,
                    n_restarts=n_restarts,
                    rt_r2_min=rt_r2_min, dt_r2_min=dt_r2_min,
                    charge_range=charge_range,
                    min_cosine=min_cosine,
                    min_peaks_per_cluster=min_peaks_per_cluster,
                    mz_bin=mz_bin,
                    gauss_sigma_rt=gauss_sigma_rt,
                    gauss_sigma_dt=gauss_sigma_dt,
                    intensity_floor=intensity_floor,
                    verbose=verbose,
                )
                all_rows.extend(rows)
            except Exception as exc:
                print(f"[hdmse] WARNING: window m/z={lo:.1f}-{hi:.1f} failed: {exc}")
                failed_windows.append(f"{lo:.1f}-{hi:.1f}: {exc}")
                traceback.print_exc()

    write_library_parquet(all_rows, output_path)
    print(f"[hdmse] wrote {len(all_rows)} library rows → {output_path}")
    if failed_windows:
        failures = "; ".join(failed_windows)
        raise RuntimeError(f"One or more LCE windows failed: {failures}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HDMS^E anchor-and-project pseudo-MS2 library builder"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("process_raw", help="Build library from one .raw file")
    p.add_argument("raw_path")
    p.add_argument("license_path")
    p.add_argument("--output", required=True, help="Output Parquet path")
    p.add_argument("--lce_function",  type=int,   default=0)
    p.add_argument("--hce_function",  type=int,   default=1)
    p.add_argument("--lce_mz_window", type=float, default=50.0)
    p.add_argument("--lce_mz_step",   type=float, default=50.0)
    p.add_argument("--hce_mz_lo",     type=float, default=100.0)
    p.add_argument("--hce_mz_hi",     type=float, default=2000.0)
    p.add_argument("--hce_mz_slab",   type=float, default=100.0)
    p.add_argument("--rho_threshold", type=float, default=0.85)
    p.add_argument("--min_fragment_intensity", type=float, default=50.0)
    p.add_argument("--peak_distance_da", type=float, default=0.5)
    p.add_argument("--rank_init",  type=int, default=5)
    p.add_argument("--rank_max",   type=int, default=15)
    p.add_argument("--n_restarts", type=int, default=3)
    p.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.cmd == "process_raw":
        process_raw_file(
            raw_path=args.raw_path,
            license_path=args.license_path,
            output_path=args.output,
            lce_function=args.lce_function,
            hce_function=args.hce_function,
            lce_mz_window=args.lce_mz_window,
            lce_mz_step=args.lce_mz_step,
            hce_mz_lo=args.hce_mz_lo,
            hce_mz_hi=args.hce_mz_hi,
            hce_mz_slab=args.hce_mz_slab,
            rho_threshold=args.rho_threshold,
            min_fragment_intensity=args.min_fragment_intensity,
            peak_distance_da=args.peak_distance_da,
            rank_init=args.rank_init,
            rank_max=args.rank_max,
            n_restarts=args.n_restarts,
            verbose=args.verbose,
        )

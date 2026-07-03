"""
troubleshoot.py
===============
Interactive diagnostic tool for inspecting individual RT×DT×m/z slices at
each stage of the mhdx pipeline: tensor build → NTF factorization →
isotopic-cluster assignment.

Every function writes a self-contained output directory whose layout is
designed so you can share it with Claude and have an immediate, structured
conversation about what the algorithm found, what it missed, and how it
could be improved.

Public API
----------
save_tensor(raw_path, license_path, ...)
    Stage 1: build the pre-NTF tensor and save it as 01_tensor.npz plus a
    human-readable summary.json.  Use this to verify that the raw data looks
    as expected before running factorization.

save_factors(raw_path, license_path, ...)
    Stages 1-2: build tensor + run NTF.  Saves 01_tensor.npz, 02_factors.npz,
    and 02_factors_meta.json with Gaussian R² purity scores per factor.

save_clusters(raw_path, license_path, ...)
    Stages 1-3 (focused): runs the full pipeline but saves only the compact
    cluster-level diagnostics (03_clusters.json).  Each cluster entry contains
    the m/z window profile, RT/DT vectors, theoretical envelope, detected
    peaks, and all quality metrics — the minimum you need to discuss any
    cluster in detail.

inspect_slice(raw_path, license_path, ...)
    One call → saves everything: (a) tensor, (b) factor matrices, (c) cluster
    diagnostics, plus optional PNG plots.  The recommended entry point for
    interactive debugging sessions.

Output layout
-------------
<out_dir>/<slice_tag>/
    summary.json            slice coordinates, shapes, all parameters used
    01_tensor.npz           raw tensor + masked tensor + three pairs of axes
    02_factors.npz          A (RT), B (DT), C (m/z) matrices + masked axes
    02_factors_meta.json    per-factor Gaussian R², n_peaks, rank, correlation
    03_clusters.json        per-cluster diagnostics with embedded profile arrays
    plots/                  optional PNG images (factor-level + cluster-level)

Quick-start example
-------------------
>>> from troubleshoot import inspect_slice
>>> inspect_slice(
...     raw_path="data/sample.raw",
...     license_path="license.key",
...     function=0,
...     rt_lo=4.5, rt_hi=5.5,
...     dt_lo=25,  dt_hi=75,
...     mz_lo=700, mz_hi=800,
...     out_dir="debug_output",
... )
# Writes debug_output/RT4.5-5.5_DT25-75_mz700-800/
# Open summary.json for the overview, then 03_clusters.json for per-cluster detail.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slice_tag(rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi) -> str:
    return f"RT{rt_lo:.2f}-{rt_hi:.2f}_DT{dt_lo}-{dt_hi}_mz{mz_lo:.1f}-{mz_hi:.1f}"


def _make_out_dir(out_dir: str, tag: str) -> str:
    path = os.path.join(out_dir, tag)
    os.makedirs(path, exist_ok=True)
    return path


def _to_json_safe(obj):
    """Recursively convert numpy scalars / arrays to JSON-serialisable types."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (v != v) else v   # NaN → null
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and obj != obj:
        return None                       # NaN → null
    return obj


def _write_json(path: str, data) -> None:
    with open(path, "w") as fh:
        json.dump(_to_json_safe(data), fh, indent=2)


def _zip_profile(axis: np.ndarray, values: np.ndarray) -> List[List]:
    """Return [[axis_val, intensity], ...] for non-zero entries only."""
    nz = values > 0
    if not nz.any():
        return [[float(axis[0]), 0.0], [float(axis[-1]), 0.0]]
    return [[float(a), float(v)] for a, v in zip(axis[nz], values[nz])]


def _cluster_mz_window(
    mz_axis: np.ndarray,
    mz_profile: np.ndarray,
    mono_mz: float,
    charge: int,
    averagine_offsets: np.ndarray,
    margin_da: float = 0.5,
) -> Dict:
    """Extract the m/z profile in the cluster's envelope window.

    Returns a dict with mz_lo, mz_hi, and paired [mz, intensity] lists for
    both the raw masked-axis profile and the non-zero bins only.
    """
    env_mz = mono_mz + averagine_offsets / charge
    win_lo = float(env_mz[0])  - margin_da
    win_hi = float(env_mz[-1]) + margin_da

    mask = (mz_axis >= win_lo) & (mz_axis <= win_hi)
    mz_win   = mz_axis[mask]
    prof_win = mz_profile[mask]

    return {
        "mz_lo": round(win_lo, 4),
        "mz_hi": round(win_hi, 4),
        "n_bins": int(mask.sum()),
        # All non-zero bins in the window — the profile Claude will discuss
        "profile": _zip_profile(mz_win, prof_win),
    }


def _theoretical_envelope_entry(
    mono_mz: float,
    charge: int,
    mass_da: float,
) -> Dict:
    """Compute and return the averagine envelope as a list of peak dicts."""
    from isotope_analysis import averagine_envelope
    offsets, rel = averagine_envelope(mass_da, min_relative=0.01)
    peaks = [
        {"mz": round(float(mono_mz + off / charge), 5), "rel_intensity": round(float(r), 4)}
        for off, r in zip(offsets, rel)
    ]
    return {
        "mono_mz": round(mono_mz, 5),
        "charge": int(charge),
        "neutral_mass_da": round(mass_da, 3),
        "peaks": peaks,
    }


# ---------------------------------------------------------------------------
# Core pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(
    raw_path: str,
    license_path: str,
    function: int,
    rt_lo: float, rt_hi: float,
    dt_lo: int,   dt_hi: int,
    mz_lo: float, mz_hi: float,
    *,
    mz_bin: float = 0.001,
    gauss_sigma_rt: float = 1.0,
    gauss_sigma_dt: float = 1.0,
    intensity_floor: float = 10.0,
    rank_init: int = 5,
    corr_threshold: float = 0.17,
    n_iter_max: int = 10_000,
    rank_max: int = 15,
    n_restarts: int = 3,
    rt_r2_min: float = 0.85,
    dt_r2_min: float = 0.85,
    charge_range: Tuple[int, int] = (3, 15),
    min_cosine: float = 0.5,
    min_peaks_per_cluster: int = 3,
    stage: int = 3,
    verbose: bool = True,
) -> Dict:
    """Internal runner.  stage=1 → tensor only; 2 → + NTF; 3 → + clusters."""
    from waters_reader import WatersRawReader, read_license
    from tensor_analysis import analyze_chunk
    from isotope_analysis import find_isotopic_clusters

    license_key = read_license(license_path)
    out: Dict = {}

    with WatersRawReader(raw_path, license=license_key) as reader:
        # Stage 1 — tensor
        if verbose:
            print(f"[troubleshoot] Building tensor for {_slice_tag(rt_lo,rt_hi,dt_lo,dt_hi,mz_lo,mz_hi)} …")

        result = analyze_chunk(
            reader, function,
            rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi,
            mz_bin=mz_bin,
            gauss_sigma_rt=gauss_sigma_rt,
            gauss_sigma_dt=gauss_sigma_dt,
            intensity_floor=intensity_floor,
            rank_init=rank_init,
            corr_threshold=corr_threshold,
            n_iter_max=n_iter_max,
            rank_max=rank_max,
            n_restarts=n_restarts,
            rt_r2_min=rt_r2_min,
            dt_r2_min=dt_r2_min,
            apply_quality_filter=(stage >= 2),
            plot=False,
            verbose=verbose,
        )

        out["tensor"]       = result["tensor"]
        out["rt_axis"]      = result["rt_axis"]
        out["dt_axis"]      = result["dt_axis"]
        out["mz_axis"]      = result["mz_axis"]
        out["tensor_ntf"]   = result.get("tensor_ntf", result["tensor"])
        out["rt_axis_ntf"]  = result.get("rt_axis_ntf", result["rt_axis"])
        out["dt_axis_ntf"]  = result.get("dt_axis_ntf", result["dt_axis"])
        out["mz_axis_ntf"]  = result.get("mz_axis_ntf", result["mz_axis"])
        out["mask_rt"]      = result.get("mask_rt")
        out["mask_dt"]      = result.get("mask_dt")
        out["mask_mz"]      = result.get("mask_mz")

        if stage < 2:
            return out

        # Stage 2 — NTF factors
        A = result.get("A")
        B = result.get("B")
        C = result.get("C")
        out["A"] = A
        out["B"] = B
        out["C"] = C
        out["final_rank"]    = result.get("final_rank", 0)
        out["max_corr"]      = result.get("max_corr", float("nan"))
        out["kept_indices"]  = result.get("kept_indices", [])
        out["quality_info"]  = result.get("quality_info", {})
        out["spectra"]       = result.get("spectra")

        if stage < 3 or A is None or A.shape[1] == 0:
            return out

        # Stage 3 — isotopic clusters (with profile arrays for debugging)
        if verbose:
            print(f"[troubleshoot] Running isotope analysis on {A.shape[1]} factors …")

        all_clusters = []
        rt_ax  = out["rt_axis_ntf"]
        dt_ax  = out["dt_axis_ntf"]
        mz_ax  = out["mz_axis_ntf"]

        for r in range(A.shape[1]):
            A_r = A[:, r]
            B_r = B[:, r]
            C_r = C[:, r]

            clusters = find_isotopic_clusters(
                A_r, B_r, C_r,
                mz_axis=mz_ax,
                rt_axis=rt_ax,
                dt_axis=dt_ax,
                charge_range=charge_range,
                min_cosine=min_cosine,
                min_peaks_per_cluster=min_peaks_per_cluster,
                factor_idx=r,
                output_dir=None,
                verbose=verbose,
                mz_axis_full=result.get("mz_axis"),
                mask_mz=result.get("mask_mz"),
            )

            # Reconstruct the mz profile (same formula used inside find_isotopic_clusters)
            mz_profile = C_r.astype(np.float64) * float(A_r.sum()) * float(B_r.sum())

            # Enrich each cluster dict with the profile arrays needed for discussion
            for cl in clusters:
                entry = dict(cl)

                # --- RT and DT profiles (paired axis, intensity lists) ---
                entry["rt_profile"] = _zip_profile(rt_ax, A_r.astype(np.float64))
                entry["dt_profile"] = _zip_profile(dt_ax, B_r.astype(np.float64))

                # --- m/z profile in the cluster window ---
                mono  = cl.get("monoisotopic_mz") or cl.get("k0_monoisotopic_mz", 0.0)
                mass  = cl.get("monoisotopic_mass_da") or cl.get("k0_monoisotopic_mass_da", 0.0)
                z     = int(cl["charge"])

                # Compute averagine offsets for the window calculation
                from isotope_analysis import averagine_envelope
                avg_offsets, avg_rel = averagine_envelope(mass, min_relative=0.01)

                entry["mz_window"] = _cluster_mz_window(
                    mz_ax, mz_profile, mono, z, avg_offsets,
                )

                # --- Theoretical envelope ---
                entry["theoretical_envelope"] = _theoretical_envelope_entry(mono, z, mass)

                all_clusters.append(entry)

        out["clusters"] = all_clusters
    return out


# ---------------------------------------------------------------------------
# Stage 1 — save_tensor
# ---------------------------------------------------------------------------

def save_tensor(
    raw_path: str,
    license_path: str,
    function: int,
    rt_lo: float, rt_hi: float,
    dt_lo: int,   dt_hi: int,
    mz_lo: float, mz_hi: float,
    out_dir: str = "troubleshoot_out",
    *,
    mz_bin: float = 0.001,
    gauss_sigma_rt: float = 1.0,
    gauss_sigma_dt: float = 1.0,
    intensity_floor: float = 10.0,
    verbose: bool = True,
) -> str:
    """Build and save the pre-NTF tensor for one slice.

    Useful for verifying that the raw data is sensible before running NTF.
    Check tensor shape, intensity ranges, and whether the masking step
    removed the expected empty bins.

    Returns
    -------
    The output directory path (contains summary.json and 01_tensor.npz).
    """
    tag     = _slice_tag(rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi)
    work    = _make_out_dir(out_dir, tag)

    data = _run_pipeline(
        raw_path, license_path, function,
        rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi,
        mz_bin=mz_bin, gauss_sigma_rt=gauss_sigma_rt,
        gauss_sigma_dt=gauss_sigma_dt, intensity_floor=intensity_floor,
        stage=1, verbose=verbose,
    )

    tensor     = data["tensor"]
    tensor_ntf = data["tensor_ntf"]

    # --- 01_tensor.npz ---
    save_kw = dict(
        tensor=tensor,
        rt_axis=data["rt_axis"],
        dt_axis=data["dt_axis"],
        mz_axis=data["mz_axis"],
        tensor_ntf=tensor_ntf,
        rt_axis_ntf=data["rt_axis_ntf"],
        dt_axis_ntf=data["dt_axis_ntf"],
        mz_axis_ntf=data["mz_axis_ntf"],
    )
    for name in ("mask_rt", "mask_dt", "mask_mz"):
        if data.get(name) is not None:
            save_kw[name] = data[name]
    np.savez_compressed(os.path.join(work, "01_tensor.npz"), **save_kw)

    # --- summary.json ---
    summary = {
        "stage": "tensor",
        "raw_path": str(raw_path),
        "function": function,
        "slice": {"rt_lo": rt_lo, "rt_hi": rt_hi,
                  "dt_lo": int(dt_lo), "dt_hi": int(dt_hi),
                  "mz_lo": mz_lo, "mz_hi": mz_hi},
        "params": {"mz_bin": mz_bin, "gauss_sigma_rt": gauss_sigma_rt,
                   "gauss_sigma_dt": gauss_sigma_dt,
                   "intensity_floor": intensity_floor},
        "tensor_shape": list(tensor.shape),
        "tensor_ntf_shape": list(tensor_ntf.shape),
        "tensor_stats": {
            "min": float(tensor.min()),
            "max": float(tensor.max()),
            "nnz": int((tensor > 0).sum()),
            "nnz_fraction": round(float((tensor > 0).mean()), 4),
        },
        "tensor_ntf_stats": {
            "min": float(tensor_ntf.min()),
            "max": float(tensor_ntf.max()),
            "nnz": int((tensor_ntf > 0).sum()),
            "nnz_fraction": round(float((tensor_ntf > 0).mean()), 4),
        },
        "axes": {
            "rt_range": [float(data["rt_axis"].min()), float(data["rt_axis"].max())],
            "dt_range": [int(data["dt_axis"].min()), int(data["dt_axis"].max())],
            "mz_range": [float(data["mz_axis"].min()), float(data["mz_axis"].max())],
            "rt_ntf_range": [float(data["rt_axis_ntf"].min()), float(data["rt_axis_ntf"].max())],
            "dt_ntf_range": [int(data["dt_axis_ntf"].min()), int(data["dt_axis_ntf"].max())],
            "mz_ntf_range": [float(data["mz_axis_ntf"].min()), float(data["mz_axis_ntf"].max())],
        },
        "files": ["summary.json", "01_tensor.npz"],
    }
    _write_json(os.path.join(work, "summary.json"), summary)
    if verbose:
        print(f"[troubleshoot] Saved tensor stage → {work}")
    return work


# ---------------------------------------------------------------------------
# Stage 2 — save_factors
# ---------------------------------------------------------------------------

def save_factors(
    raw_path: str,
    license_path: str,
    function: int,
    rt_lo: float, rt_hi: float,
    dt_lo: int,   dt_hi: int,
    mz_lo: float, mz_hi: float,
    out_dir: str = "troubleshoot_out",
    *,
    mz_bin: float = 0.001,
    gauss_sigma_rt: float = 1.0,
    gauss_sigma_dt: float = 1.0,
    intensity_floor: float = 10.0,
    rank_init: int = 5,
    corr_threshold: float = 0.17,
    n_iter_max: int = 10_000,
    rank_max: int = 15,
    n_restarts: int = 3,
    rt_r2_min: float = 0.85,
    dt_r2_min: float = 0.85,
    verbose: bool = True,
) -> str:
    """Build tensor + run NTF and save both stages.

    The factor matrices A (RT), B (DT), C (m/z) are stored in 02_factors.npz.
    The JSON file 02_factors_meta.json contains per-factor Gaussian R², peak
    counts, and the rank/correlation summary — useful for discussing whether
    factors are clean unimodal peaks or multimodal artefacts.

    Returns
    -------
    The output directory path.
    """
    tag  = _slice_tag(rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi)
    work = _make_out_dir(out_dir, tag)

    data = _run_pipeline(
        raw_path, license_path, function,
        rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi,
        mz_bin=mz_bin, gauss_sigma_rt=gauss_sigma_rt,
        gauss_sigma_dt=gauss_sigma_dt, intensity_floor=intensity_floor,
        rank_init=rank_init, corr_threshold=corr_threshold,
        n_iter_max=n_iter_max, rank_max=rank_max, n_restarts=n_restarts,
        rt_r2_min=rt_r2_min, dt_r2_min=dt_r2_min,
        stage=2, verbose=verbose,
    )

    tensor     = data["tensor"]
    tensor_ntf = data["tensor_ntf"]

    # --- 01_tensor.npz (same as save_tensor) ---
    t_kw = dict(
        tensor=tensor,
        rt_axis=data["rt_axis"],
        dt_axis=data["dt_axis"],
        mz_axis=data["mz_axis"],
        tensor_ntf=tensor_ntf,
        rt_axis_ntf=data["rt_axis_ntf"],
        dt_axis_ntf=data["dt_axis_ntf"],
        mz_axis_ntf=data["mz_axis_ntf"],
    )
    for name in ("mask_rt", "mask_dt", "mask_mz"):
        if data.get(name) is not None:
            t_kw[name] = data[name]
    np.savez_compressed(os.path.join(work, "01_tensor.npz"), **t_kw)

    # --- 02_factors.npz ---
    A, B, C = data["A"], data["B"], data["C"]
    f_kw = dict(
        rt_axis_ntf=data["rt_axis_ntf"],
        dt_axis_ntf=data["dt_axis_ntf"],
        mz_axis_ntf=data["mz_axis_ntf"],
    )
    if A is not None:
        f_kw.update(A=A, B=B, C=C)
    if data.get("spectra") is not None:
        f_kw["spectra"] = data["spectra"]
    np.savez_compressed(os.path.join(work, "02_factors.npz"), **f_kw)

    # --- 02_factors_meta.json ---
    qi = data.get("quality_info") or {}
    factors_meta = {
        "final_rank": int(data.get("final_rank", 0)),
        "max_corr": data.get("max_corr"),
        "corr_threshold": corr_threshold,
        "kept_indices": [int(i) for i in (data.get("kept_indices") or [])],
        "n_factors": int(A.shape[1]) if A is not None else 0,
        # Per-factor quality (each key is a factor index string)
        "factors": {},
    }
    if A is not None:
        mz_ax = data["mz_axis_ntf"]
        rt_ax = data["rt_axis_ntf"]
        dt_ax = data["dt_axis_ntf"]
        for r in range(A.shape[1]):
            mz_profile = C[:, r].astype(np.float64) * float(A[:, r].sum()) * float(B[:, r].sum())
            fi = qi.get(r, {})
            factors_meta["factors"][str(r)] = {
                "rt_gaussian_r2": fi.get("rt_r2"),
                "dt_gaussian_r2": fi.get("dt_r2"),
                "rt_n_peaks":     fi.get("rt_n_peaks"),
                "dt_n_peaks":     fi.get("dt_n_peaks"),
                "rt_multimodal":  fi.get("rt_multimodal"),
                "dt_multimodal":  fi.get("dt_multimodal"),
                "factor_bpi":     round(float(mz_profile.max()), 2),
                "factor_tic":     round(float(mz_profile.sum()), 2),
                "rt_center":      round(float(np.average(rt_ax, weights=A[:, r])), 4),
                "dt_center":      round(float(np.average(dt_ax, weights=B[:, r])), 4),
                "mz_range": [
                    round(float(mz_ax[C[:, r] > 0].min()), 4) if (C[:, r] > 0).any() else None,
                    round(float(mz_ax[C[:, r] > 0].max()), 4) if (C[:, r] > 0).any() else None,
                ],
            }
    _write_json(os.path.join(work, "02_factors_meta.json"), factors_meta)

    # --- summary.json ---
    summary = {
        "stage": "factors",
        "raw_path": str(raw_path),
        "function": function,
        "slice": {"rt_lo": rt_lo, "rt_hi": rt_hi,
                  "dt_lo": int(dt_lo), "dt_hi": int(dt_hi),
                  "mz_lo": mz_lo, "mz_hi": mz_hi},
        "params": {
            "mz_bin": mz_bin, "gauss_sigma_rt": gauss_sigma_rt,
            "gauss_sigma_dt": gauss_sigma_dt, "intensity_floor": intensity_floor,
            "rank_init": rank_init, "corr_threshold": corr_threshold,
            "n_iter_max": n_iter_max, "rank_max": rank_max,
            "n_restarts": n_restarts, "rt_r2_min": rt_r2_min, "dt_r2_min": dt_r2_min,
        },
        "tensor_shape":     list(tensor.shape),
        "tensor_ntf_shape": list(tensor_ntf.shape),
        "final_rank":       factors_meta["final_rank"],
        "max_corr":         factors_meta["max_corr"],
        "files": ["summary.json", "01_tensor.npz", "02_factors.npz", "02_factors_meta.json"],
    }
    _write_json(os.path.join(work, "summary.json"), summary)
    if verbose:
        print(f"[troubleshoot] Saved factors stage → {work}")
    return work


# ---------------------------------------------------------------------------
# Stage 3 — save_clusters
# ---------------------------------------------------------------------------

def save_clusters(
    raw_path: str,
    license_path: str,
    function: int,
    rt_lo: float, rt_hi: float,
    dt_lo: int,   dt_hi: int,
    mz_lo: float, mz_hi: float,
    out_dir: str = "troubleshoot_out",
    *,
    mz_bin: float = 0.001,
    gauss_sigma_rt: float = 1.0,
    gauss_sigma_dt: float = 1.0,
    intensity_floor: float = 10.0,
    rank_init: int = 5,
    corr_threshold: float = 0.17,
    n_iter_max: int = 10_000,
    rank_max: int = 15,
    n_restarts: int = 3,
    rt_r2_min: float = 0.85,
    dt_r2_min: float = 0.85,
    charge_range: Tuple[int, int] = (3, 15),
    min_cosine: float = 0.5,
    min_peaks_per_cluster: int = 3,
    verbose: bool = True,
) -> str:
    """Run the full pipeline and save compact cluster-level diagnostics.

    This is the lightest-weight entry point for cluster debugging: it does
    not save the large tensor or factor matrices, only 03_clusters.json.
    Each cluster entry contains:

    - All quality metrics (cosine, adjusted_score, gap_fraction, peak_rmse, …)
    - RT profile:  [[rt_min, intensity], ...] for the factor's A vector
    - DT profile:  [[dt_bin, intensity], ...] for the factor's B vector
    - mz_window:   the m/z profile in the cluster's envelope window (±0.5 Da)
    - theoretical_envelope: the averagine stick pattern at the assigned charge

    Returns
    -------
    The output directory path.
    """
    tag  = _slice_tag(rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi)
    work = _make_out_dir(out_dir, tag)

    data = _run_pipeline(
        raw_path, license_path, function,
        rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi,
        mz_bin=mz_bin, gauss_sigma_rt=gauss_sigma_rt,
        gauss_sigma_dt=gauss_sigma_dt, intensity_floor=intensity_floor,
        rank_init=rank_init, corr_threshold=corr_threshold,
        n_iter_max=n_iter_max, rank_max=rank_max, n_restarts=n_restarts,
        rt_r2_min=rt_r2_min, dt_r2_min=dt_r2_min,
        charge_range=charge_range, min_cosine=min_cosine,
        min_peaks_per_cluster=min_peaks_per_cluster,
        stage=3, verbose=verbose,
    )

    clusters = data.get("clusters", [])
    _write_json(os.path.join(work, "03_clusters.json"), clusters)

    summary = {
        "stage": "clusters",
        "raw_path": str(raw_path),
        "function": function,
        "slice": {"rt_lo": rt_lo, "rt_hi": rt_hi,
                  "dt_lo": int(dt_lo), "dt_hi": int(dt_hi),
                  "mz_lo": mz_lo, "mz_hi": mz_hi},
        "params": {
            "mz_bin": mz_bin, "gauss_sigma_rt": gauss_sigma_rt,
            "gauss_sigma_dt": gauss_sigma_dt, "intensity_floor": intensity_floor,
            "rank_init": rank_init, "corr_threshold": corr_threshold,
            "n_iter_max": n_iter_max, "rank_max": rank_max,
            "n_restarts": n_restarts, "rt_r2_min": rt_r2_min, "dt_r2_min": dt_r2_min,
            "charge_range": list(charge_range), "min_cosine": min_cosine,
            "min_peaks_per_cluster": min_peaks_per_cluster,
        },
        "final_rank": int(data.get("final_rank", 0)),
        "max_corr": data.get("max_corr"),
        "n_clusters": len([c for c in clusters if c.get("k", 0) == 0]),
        "n_cluster_rows": len(clusters),
        "files": ["summary.json", "03_clusters.json"],
    }
    _write_json(os.path.join(work, "summary.json"), summary)
    if verbose:
        print(f"[troubleshoot] Saved clusters stage → {work}  "
              f"({summary['n_clusters']} clusters)")
    return work


# ---------------------------------------------------------------------------
# All-in-one — inspect_slice
# ---------------------------------------------------------------------------

def inspect_slice(
    raw_path: str,
    license_path: str,
    function: int,
    rt_lo: float, rt_hi: float,
    dt_lo: int,   dt_hi: int,
    mz_lo: float, mz_hi: float,
    out_dir: str = "troubleshoot_out",
    *,
    mz_bin: float = 0.001,
    gauss_sigma_rt: float = 1.0,
    gauss_sigma_dt: float = 1.0,
    intensity_floor: float = 10.0,
    rank_init: int = 5,
    corr_threshold: float = 0.17,
    n_iter_max: int = 10_000,
    rank_max: int = 15,
    n_restarts: int = 3,
    rt_r2_min: float = 0.85,
    dt_r2_min: float = 0.85,
    charge_range: Tuple[int, int] = (3, 15),
    min_cosine: float = 0.5,
    min_peaks_per_cluster: int = 3,
    save_plots: bool = True,
    plot_format: str = "png",
    png_dpi: int = 100,
    verbose: bool = True,
) -> str:
    """Run the full pipeline and save all three diagnostic stages.

    Writes:
        a) 01_tensor.npz       — pre-NTF tensor and masked tensor with axes
        b) 02_factors.npz      — A, B, C factor matrices
           02_factors_meta.json — per-factor quality summary
        c) 03_clusters.json    — per-cluster diagnostics with profile arrays
        d) plots/              — factor PNG + cluster PNGs (if save_plots=True)
        e) summary.json        — overall summary and file index

    This is the recommended entry point for interactive debugging sessions.
    Start by reading summary.json, then 03_clusters.json for the cluster
    details you want to discuss.

    Returns
    -------
    The output directory path.
    """
    tag  = _slice_tag(rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi)
    work = _make_out_dir(out_dir, tag)

    data = _run_pipeline(
        raw_path, license_path, function,
        rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi,
        mz_bin=mz_bin, gauss_sigma_rt=gauss_sigma_rt,
        gauss_sigma_dt=gauss_sigma_dt, intensity_floor=intensity_floor,
        rank_init=rank_init, corr_threshold=corr_threshold,
        n_iter_max=n_iter_max, rank_max=rank_max, n_restarts=n_restarts,
        rt_r2_min=rt_r2_min, dt_r2_min=dt_r2_min,
        charge_range=charge_range, min_cosine=min_cosine,
        min_peaks_per_cluster=min_peaks_per_cluster,
        stage=3, verbose=verbose,
    )

    tensor     = data["tensor"]
    tensor_ntf = data["tensor_ntf"]
    A, B, C    = data.get("A"), data.get("B"), data.get("C")
    clusters   = data.get("clusters", [])

    # (a) tensor
    t_kw = dict(
        tensor=tensor,
        rt_axis=data["rt_axis"], dt_axis=data["dt_axis"], mz_axis=data["mz_axis"],
        tensor_ntf=tensor_ntf,
        rt_axis_ntf=data["rt_axis_ntf"],
        dt_axis_ntf=data["dt_axis_ntf"],
        mz_axis_ntf=data["mz_axis_ntf"],
    )
    for name in ("mask_rt", "mask_dt", "mask_mz"):
        if data.get(name) is not None:
            t_kw[name] = data[name]
    np.savez_compressed(os.path.join(work, "01_tensor.npz"), **t_kw)

    # (b) factors
    f_kw = dict(
        rt_axis_ntf=data["rt_axis_ntf"],
        dt_axis_ntf=data["dt_axis_ntf"],
        mz_axis_ntf=data["mz_axis_ntf"],
    )
    if A is not None:
        f_kw.update(A=A, B=B, C=C)
    if data.get("spectra") is not None:
        f_kw["spectra"] = data["spectra"]
    np.savez_compressed(os.path.join(work, "02_factors.npz"), **f_kw)

    qi = data.get("quality_info") or {}
    factors_meta: Dict = {
        "final_rank": int(data.get("final_rank", 0)),
        "max_corr": data.get("max_corr"),
        "corr_threshold": corr_threshold,
        "kept_indices": [int(i) for i in (data.get("kept_indices") or [])],
        "n_factors": int(A.shape[1]) if A is not None else 0,
        "factors": {},
    }
    if A is not None:
        mz_ax = data["mz_axis_ntf"]
        rt_ax = data["rt_axis_ntf"]
        dt_ax = data["dt_axis_ntf"]
        for r in range(A.shape[1]):
            mz_profile = C[:, r].astype(np.float64) * float(A[:, r].sum()) * float(B[:, r].sum())
            fi = qi.get(r, {})
            factors_meta["factors"][str(r)] = {
                "rt_gaussian_r2": fi.get("rt_r2"),
                "dt_gaussian_r2": fi.get("dt_r2"),
                "rt_n_peaks":     fi.get("rt_n_peaks"),
                "dt_n_peaks":     fi.get("dt_n_peaks"),
                "rt_multimodal":  fi.get("rt_multimodal"),
                "dt_multimodal":  fi.get("dt_multimodal"),
                "factor_bpi":     round(float(mz_profile.max()), 2),
                "factor_tic":     round(float(mz_profile.sum()), 2),
                "rt_center":      round(float(np.average(rt_ax, weights=A[:, r])), 4),
                "dt_center":      round(float(np.average(dt_ax, weights=B[:, r])), 4),
                "mz_range": [
                    round(float(mz_ax[C[:, r] > 0].min()), 4) if (C[:, r] > 0).any() else None,
                    round(float(mz_ax[C[:, r] > 0].max()), 4) if (C[:, r] > 0).any() else None,
                ],
            }
    _write_json(os.path.join(work, "02_factors_meta.json"), factors_meta)

    # (c) clusters
    _write_json(os.path.join(work, "03_clusters.json"), clusters)

    # (d) plots
    files = ["summary.json", "01_tensor.npz",
             "02_factors.npz", "02_factors_meta.json",
             "03_clusters.json"]

    if save_plots and A is not None and A.shape[1] > 0:
        import sys; sys.path.insert(0, os.path.dirname(__file__))
        import matplotlib
        matplotlib.use("Agg")

        plots_dir = os.path.join(work, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        from tensor_analysis import (
            plot_raw as _plot_raw,
            plot_factors as _plot_factors,
            plot_correlations as _plot_corr,
        )
        from isotope_analysis import find_isotopic_clusters as _find_iso, plot_isotopic_cluster

        import matplotlib.pyplot as _plt

        slice_label = tag

        # Factor-level plots (raw tensor, NTF factors, correlations)
        try:
            fig_r = _plot_raw(tensor_ntf, data["rt_axis_ntf"],
                              data["dt_axis_ntf"], data["mz_axis_ntf"])
            fig_r.savefig(os.path.join(plots_dir, f"factor_raw.{plot_format}"),
                          dpi=png_dpi, bbox_inches="tight")
            _plt.close(fig_r)
        except Exception as exc:
            warnings.warn(f"plot_raw failed: {exc}", RuntimeWarning)

        try:
            fig_f = _plot_factors(A, B, C, data["rt_axis_ntf"],
                                  data["dt_axis_ntf"], data["mz_axis_ntf"])
            fig_f.savefig(os.path.join(plots_dir, f"factor_ntf.{plot_format}"),
                          dpi=png_dpi, bbox_inches="tight")
            _plt.close(fig_f)
        except Exception as exc:
            warnings.warn(f"plot_factors failed: {exc}", RuntimeWarning)

        try:
            fig_c = _plot_corr(A, B, C)
            fig_c.savefig(os.path.join(plots_dir, f"factor_corr.{plot_format}"),
                          dpi=png_dpi, bbox_inches="tight")
            _plt.close(fig_c)
        except Exception as exc:
            warnings.warn(f"plot_correlations failed: {exc}", RuntimeWarning)

        # Cluster-level plots (one per assigned cluster)
        for r in range(A.shape[1]):
            cl_list = _find_iso(
                A[:, r], B[:, r], C[:, r],
                mz_axis=data["mz_axis_ntf"],
                rt_axis=data["rt_axis_ntf"],
                dt_axis=data["dt_axis_ntf"],
                charge_range=charge_range,
                min_cosine=min_cosine,
                min_peaks_per_cluster=min_peaks_per_cluster,
                factor_idx=r,
                output_dir=plots_dir,
                mz_axis_full=data.get("mz_axis"),
                mask_mz=data.get("mask_mz"),
                png_dpi=png_dpi,
                plot_format=plot_format,
                verbose=False,
            )

        files.append("plots/")

    # (e) summary
    summary = {
        "stage": "inspect_slice (all)",
        "raw_path": str(raw_path),
        "function": function,
        "slice": {"rt_lo": rt_lo, "rt_hi": rt_hi,
                  "dt_lo": int(dt_lo), "dt_hi": int(dt_hi),
                  "mz_lo": mz_lo, "mz_hi": mz_hi},
        "params": {
            "mz_bin": mz_bin, "gauss_sigma_rt": gauss_sigma_rt,
            "gauss_sigma_dt": gauss_sigma_dt, "intensity_floor": intensity_floor,
            "rank_init": rank_init, "corr_threshold": corr_threshold,
            "n_iter_max": n_iter_max, "rank_max": rank_max,
            "n_restarts": n_restarts, "rt_r2_min": rt_r2_min, "dt_r2_min": dt_r2_min,
            "charge_range": list(charge_range), "min_cosine": min_cosine,
            "min_peaks_per_cluster": min_peaks_per_cluster,
        },
        "tensor_shape":     list(tensor.shape),
        "tensor_ntf_shape": list(tensor_ntf.shape),
        "final_rank":   factors_meta["final_rank"],
        "max_corr":     factors_meta["max_corr"],
        "n_clusters":   len([c for c in clusters if c.get("k", 0) == 0]),
        "n_cluster_rows": len(clusters),
        "cluster_summary": [
            {
                "factor_idx":          c.get("factor_idx"),
                "cluster_idx":         c.get("cluster_idx"),
                "k":                   c.get("k"),
                "charge":              c.get("charge"),
                "monoisotopic_mz":     c.get("monoisotopic_mz"),
                "monoisotopic_mass_da":c.get("monoisotopic_mass_da"),
                "cosine_similarity":   c.get("cosine_similarity"),
                "adjusted_score":      c.get("adjusted_score"),
                "fit_quality":         c.get("fit_quality"),
                "rt_center":           c.get("rt_center"),
                "dt_center":           c.get("dt_center"),
            }
            for c in clusters
        ],
        "files": files,
        "how_to_read": (
            "Start with this file for the overview. "
            "Open 03_clusters.json for full per-cluster detail (m/z profiles, "
            "RT/DT vectors, theoretical envelopes). "
            "Open 02_factors_meta.json for factor-level Gaussian purity metrics. "
            "Large array data lives in the .npz files."
        ),
    }
    _write_json(os.path.join(work, "summary.json"), summary)

    if verbose:
        n_cl = summary["n_clusters"]
        print(f"[troubleshoot] inspect_slice complete → {work}")
        print(f"  tensor shape:     {tensor.shape}  →  {tensor_ntf.shape} (after masking)")
        print(f"  NTF rank:         {factors_meta['final_rank']}")
        print(f"  Assigned clusters: {n_cl}")

    return work


# ---------------------------------------------------------------------------
# trace_missing_signal: per-signal diagnostic tracer
# ---------------------------------------------------------------------------

def _best_centered_slice(
    candidates: list,
    obs_mz: float,
    rt: float,
    im_mono: float,
) -> dict:
    """Return the slice where the signal is most interior.

    Scores each candidate by the minimum fractional distance from any of its
    six faces.  A signal exactly at the centre scores 0.5; one touching an
    edge scores 0.  Returns the highest-scoring candidate (first on ties).
    """
    best_s, best_score = candidates[0], -1.0
    for s in candidates:
        rt_span = s["rt_hi"] - s["rt_lo"]
        dt_span = s["dt_hi"] - s["dt_lo"]
        mz_span = s["mz_hi"] - s["mz_lo"]
        rt_m = min(rt      - s["rt_lo"], s["rt_hi"] - rt)      / rt_span if rt_span > 0 else 0.0
        dt_m = min(im_mono - s["dt_lo"], s["dt_hi"] - im_mono) / dt_span if dt_span > 0 else 0.0
        mz_m = min(obs_mz  - s["mz_lo"], s["mz_hi"] - obs_mz) / mz_span if mz_span > 0 else 0.0
        score = min(rt_m, dt_m, mz_m)
        if score > best_score:
            best_s, best_score = s, score
    return best_s

def trace_missing_signal(
    raw_path: str,
    license_path: str,
    obs_mz: float,
    charge: int,
    rt: float,
    im_mono: float,
    config: Dict,
    function: int = 0,
    mz_ppm: float = 10.0,
    rt_tol: float = 0.3,
    dt_tol: float = 10.0,
    verbose: bool = False,
) -> Dict:
    """Trace why a reference signal is absent from the pipeline output.

    Finds all slices that contain the target (obs_mz, RT, im_mono) position,
    re-runs each pipeline stage on those slices, and reports the furthest
    stage the signal reached before being dropped.

    Returns a dict with keys:
        lost_at       : one of 'not_in_any_slice' | 'bpi_tic' | 'ntf_no_factors' |
                        'ntf_gaussian' | 'isotope_min_peaks' | 'isotope_cosine' |
                        'post_filter' | 'recovered'
        n_slices      : number of overlapping slices examined
        best_cosine   : best cosine found across all slices (None if never reached)
        ntf_factors   : (final_rank, n_passing_gauss) tuple for the best slice
        slice_coords  : (rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi) of the best slice
    """
    from pipeline import generate_slice_grid, _compute_slice_bpi_tic
    from waters_reader import WatersRawReader, read_license

    ntf_cfg  = config.get("ntf", {})
    iso_cfg  = config.get("isotope", {})
    filt_cfg = config.get("filters", {})
    tsr_cfg  = config.get("tensor", {})
    slc_cfg  = config.get("slice", {})

    license_key = read_license(license_path)

    stage_order = [
        "not_in_any_slice", "bpi_tic", "ntf_no_factors", "ntf_gaussian",
        "isotope_min_peaks", "isotope_cosine", "post_filter", "recovered",
    ]
    best: Dict = {"lost_at": "not_in_any_slice", "n_slices": 0,
                  "best_cosine": None, "ntf_factors": None, "slice_coords": None}

    with WatersRawReader(raw_path, license=license_key) as reader:
        # Read file axis ranges to build the slice grid (same pattern as write_slice_list)
        meta     = reader.metadata()
        rt_range = meta.rt_range[function]
        mz_range = meta.mass_range[function]
        n_drift  = meta.n_drift_bins[function]

        grid = generate_slice_grid(
            rt_min=float(rt_range[0]),  rt_max=float(rt_range[1]),
            dt_min=0,                    dt_max=int(n_drift - 1),
            mz_min=float(mz_range[0]),  mz_max=float(mz_range[1]),
            dt_width=slc_cfg.get("dt_width", 50),
            dt_step=slc_cfg.get("dt_step",   25),
            rt_width=slc_cfg.get("rt_width",  1.0),
            rt_step=slc_cfg.get("rt_step",    0.5),
            mz_width=slc_cfg.get("mz_width", 100.0),
            mz_step=slc_cfg.get("mz_step",    50.0),
        )

        candidates = [
            s for s in grid
            if s["mz_lo"] <= obs_mz  <= s["mz_hi"]
            and s["rt_lo"] <= rt      <= s["rt_hi"]
            and s["dt_lo"] <= im_mono <= s["dt_hi"]
        ]

        if not candidates:
            return best

        best["n_slices"] = len(candidates)
        best["lost_at"]  = "bpi_tic"

        # Use only the most interior slice — the one where the signal sits
        # farthest from any edge.  Running all overlapping slices (up to 8
        # due to 50% overlap in RT × DT × m/z) multiplies NTF cost by ~8×,
        # making each batch take 12–16 h instead of the allocated 4 h.
        s = _best_centered_slice(candidates, obs_mz, rt, im_mono)
        result = _trace_one_slice(
            reader, s, obs_mz, charge, rt, im_mono,
            function, ntf_cfg, iso_cfg, filt_cfg, tsr_cfg,
            mz_ppm, rt_tol, dt_tol, verbose,
        )
        if stage_order.index(result["lost_at"]) > stage_order.index(best["lost_at"]):
            best = result

    return best


def _trace_one_slice(
    reader,
    s: Dict,
    obs_mz: float,
    charge: int,
    rt: float,
    im_mono: float,
    function: int,
    ntf_cfg: Dict,
    iso_cfg: Dict,
    filt_cfg: Dict,
    tsr_cfg: Dict,
    mz_ppm: float,
    rt_tol: float,
    dt_tol: float,
    verbose: bool,
) -> Dict:
    """Run all pipeline stages on one slice; return at which stage signal is lost."""
    from tensor_analysis import analyze_chunk
    from isotope_analysis import find_isotopic_clusters
    from pipeline import _compute_slice_bpi_tic
    import numpy as np

    coords = (s["rt_lo"], s["rt_hi"], s["dt_lo"], s["dt_hi"], s["mz_lo"], s["mz_hi"])
    base   = {"slice_coords": coords, "best_cosine": None, "ntf_factors": None}

    # Stage 1 — BPI/TIC gate
    bpi, tic = _compute_slice_bpi_tic(
        reader, function,
        s["rt_lo"], s["rt_hi"], s["dt_lo"], s["dt_hi"], s["mz_lo"], s["mz_hi"],
    )
    bpi_min = filt_cfg.get("bpi_min", 0.0)
    tic_min = filt_cfg.get("tic_min", 0.0)
    if bpi < bpi_min or tic < tic_min:
        return {**base, "lost_at": "bpi_tic"}

    # Stage 2 — NTF
    chunk = analyze_chunk(
        reader, function,
        s["rt_lo"], s["rt_hi"], s["dt_lo"], s["dt_hi"], s["mz_lo"], s["mz_hi"],
        mz_bin=tsr_cfg.get("mz_bin", 0.001),
        gauss_sigma_rt=tsr_cfg.get("gauss_sigma_rt", 1.0),
        gauss_sigma_dt=tsr_cfg.get("gauss_sigma_dt", 1.0),
        intensity_floor=tsr_cfg.get("intensity_floor", 10.0),
        rank_init=ntf_cfg.get("rank_init", 5),
        corr_threshold=ntf_cfg.get("corr_threshold", 0.17),
        n_iter_max=ntf_cfg.get("n_iter_max", 10000),
        rank_max=ntf_cfg.get("rank_max", 15),
        n_restarts=ntf_cfg.get("n_restarts", 3),
        rt_r2_min=ntf_cfg.get("rt_r2_min", 0.75),
        dt_r2_min=ntf_cfg.get("dt_r2_min", 0.75),
        apply_quality_filter=True,
        plot=False,
        verbose=verbose,
    )

    A = chunk.get("A")
    final_rank = chunk.get("final_rank", 0)

    if A is None or final_rank == 0:
        return {**base, "lost_at": "ntf_no_factors"}

    n_passing = A.shape[1]
    base["ntf_factors"] = (final_rank, n_passing)

    if n_passing == 0:
        return {**base, "lost_at": "ntf_gaussian"}

    # Stage 3 — isotope detection (one call per NTF factor, same pattern as _run_pipeline)
    B     = chunk["B"]
    C     = chunk["C"]
    mz_ax = chunk.get("mz_axis_ntf", chunk.get("mz_axis"))
    rt_ax = chunk.get("rt_axis_ntf", chunk.get("rt_axis"))
    dt_ax = chunk.get("dt_axis_ntf", chunk.get("dt_axis"))

    if mz_ax is None:
        return {**base, "lost_at": "ntf_no_factors"}

    charge_range = tuple(iso_cfg.get("charge_range", [2, 15]))
    min_cosine   = iso_cfg.get("min_cosine", 0.5)
    min_peaks    = iso_cfg.get("min_peaks_per_cluster", 2)

    clusters_all: list = []
    for r in range(A.shape[1]):
        clusters_all.extend(find_isotopic_clusters(
            A[:, r], B[:, r], C[:, r],
            mz_axis=mz_ax,
            rt_axis=rt_ax,
            dt_axis=dt_ax,
            charge_range=charge_range,
            min_cosine=min_cosine,
            min_peaks_per_cluster=min_peaks,
            factor_idx=r,
            output_dir=None,
            verbose=False,
            mz_axis_full=chunk.get("mz_axis"),
            mask_mz=chunk.get("mask_mz"),
        ))

    if not clusters_all:
        # Distinguish isotope_min_peaks from isotope_cosine by trying looser params
        relaxed: list = []
        for r in range(A.shape[1]):
            relaxed.extend(find_isotopic_clusters(
                A[:, r], B[:, r], C[:, r],
                mz_axis=mz_ax, rt_axis=rt_ax, dt_axis=dt_ax,
                charge_range=charge_range,
                min_cosine=0.1,
                min_peaks_per_cluster=2,
                factor_idx=r, output_dir=None, verbose=False,
                mz_axis_full=chunk.get("mz_axis"), mask_mz=chunk.get("mask_mz"),
            ))
        return {**base, "lost_at": "isotope_min_peaks" if relaxed else "isotope_cosine"}

    # Check if the target signal is among detected clusters
    best_cosine = None
    for cluster in clusters_all:
        if int(cluster.get("charge", 0)) != charge:
            continue
        cmz = float(cluster.get("monoisotopic_mz", 0.0))
        ppm = abs(cmz - obs_mz) / (obs_mz + 1e-12) * 1e6
        if ppm <= mz_ppm:
            cos = float(cluster.get("cosine_similarity", 0.0))
            if best_cosine is None or cos > best_cosine:
                best_cosine = cos

    base["best_cosine"] = best_cosine

    if best_cosine is None:
        return {**base, "lost_at": "isotope_cosine"}

    # Stage 4 — post-filter: check NTF Gaussian R² for any passing factor
    rt_r2_min    = filt_cfg.get("rt_gaussian_r2_min", 0.80)
    dt_r2_min    = filt_cfg.get("dt_gaussian_r2_min", 0.80)
    quality_info = chunk.get("quality_info", {})
    if quality_info:
        any_good = any(
            q.get("rt_r2", 0.0) >= rt_r2_min and q.get("dt_r2", 0.0) >= dt_r2_min
            for q in quality_info.values()
        )
        if not any_good:
            return {**base, "lost_at": "post_filter"}

    return {**base, "lost_at": "recovered"}


def batch_trace_missing(
    raw_path: str,
    license_path: str,
    unmatched_csv: str,
    config: Dict,
    n_signals: int = -1,
    output_csv: str = "trace_report.csv",
    function: int = 0,
) -> None:
    """Run trace_missing_signal on rows of unmatched_csv.

    n_signals <= 0 means process all rows (default); positive values cap the run.
    """
    import pandas as pd

    df = pd.read_csv(unmatched_csv)
    unmatched = df if n_signals <= 0 else df.head(n_signals)
    print(f"Tracing {len(unmatched)} missing signals from {unmatched_csv}")

    records = []
    for i, row in unmatched.iterrows():
        if i % 10 == 0:
            print(f"  {i}/{len(unmatched)} …")
        result = trace_missing_signal(
            raw_path, license_path,
            obs_mz=float(row["obs_mz"]),
            charge=int(row["charge"]),
            rt=float(row["RT"]),
            im_mono=float(row["im_mono"]),
            config=config,
            function=function,
        )
        records.append({
            "name":             row.get("name", ""),
            "obs_mz":           row["obs_mz"],
            "charge":           row["charge"],
            "RT":               row["RT"],
            "im_mono":          row["im_mono"],
            "lost_at":          result["lost_at"],
            "best_cosine":      result["best_cosine"],
            "ntf_n_factors":    result["ntf_factors"][0] if result["ntf_factors"] else None,
            "ntf_passing_gauss": result["ntf_factors"][1] if result["ntf_factors"] else None,
            "n_slices":         result["n_slices"],
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\nTrace report saved to {output_csv}")
    print("\nlost_at breakdown:")
    print(df["lost_at"].value_counts().to_string())


# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    # Dispatch to trace_missing subcommand if requested
    if len(sys.argv) > 1 and sys.argv[1] == "trace_missing":
        import yaml
        tp = argparse.ArgumentParser(
            description="Batch-trace missing signals against the pipeline.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        tp.add_argument("raw_path",       help="Path to Waters .raw directory")
        tp.add_argument("license_path",   help="Path to MassLynx license key file")
        tp.add_argument("unmatched_csv",  help="CSV of unmatched reference signals")
        tp.add_argument("--config",       default="src/config.yaml",
                        help="Pipeline config YAML")
        tp.add_argument("--n_signals",    type=int, default=-1,
                        help="Number of signals to trace (-1 = all rows)")
        tp.add_argument("--output",       default="trace_report.csv",
                        help="Output CSV path")
        tp.add_argument("--function",     type=int, default=0)
        targs = tp.parse_args(sys.argv[2:])

        with open(targs.config) as f:
            cfg = yaml.safe_load(f)

        batch_trace_missing(
            raw_path=targs.raw_path,
            license_path=targs.license_path,
            unmatched_csv=targs.unmatched_csv,
            config=cfg,
            n_signals=targs.n_signals,
            output_csv=targs.output,
            function=targs.function,
        )
        sys.exit(0)

    p = argparse.ArgumentParser(
        description="Troubleshoot a single RT×DT×m/z slice from a Waters .raw file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("raw_path",      help="Path to Waters .raw directory")
    p.add_argument("license_path",  help="Path to MassLynx license key file")
    p.add_argument("--function",    type=int,   default=0)
    p.add_argument("--rt-lo",       type=float, required=True)
    p.add_argument("--rt-hi",       type=float, required=True)
    p.add_argument("--dt-lo",       type=int,   required=True)
    p.add_argument("--dt-hi",       type=int,   required=True)
    p.add_argument("--mz-lo",       type=float, required=True)
    p.add_argument("--mz-hi",       type=float, required=True)
    p.add_argument("--out-dir",     default="troubleshoot_out")
    p.add_argument("--stage",
                   choices=["tensor", "factors", "clusters", "all"],
                   default="all",
                   help="How far to run the pipeline")
    p.add_argument("--no-plots",    action="store_true")
    p.add_argument("--min-cosine",  type=float, default=0.5)
    p.add_argument("--rank-init",   type=int,   default=5)
    p.add_argument("--rank-max",    type=int,   default=15)
    p.add_argument("--charge-min",  type=int,   default=3)
    p.add_argument("--charge-max",  type=int,   default=15)
    p.add_argument("--quiet",       action="store_true")
    args = p.parse_args()

    common = dict(
        raw_path=args.raw_path,
        license_path=args.license_path,
        function=args.function,
        rt_lo=args.rt_lo, rt_hi=args.rt_hi,
        dt_lo=args.dt_lo, dt_hi=args.dt_hi,
        mz_lo=args.mz_lo, mz_hi=args.mz_hi,
        out_dir=args.out_dir,
        verbose=not args.quiet,
    )

    if args.stage == "tensor":
        save_tensor(**common)
    elif args.stage == "factors":
        save_factors(**common, rank_init=args.rank_init, rank_max=args.rank_max)
    elif args.stage == "clusters":
        save_clusters(**common,
                      rank_init=args.rank_init, rank_max=args.rank_max,
                      min_cosine=args.min_cosine,
                      charge_range=(args.charge_min, args.charge_max))
    else:
        inspect_slice(**common,
                      rank_init=args.rank_init, rank_max=args.rank_max,
                      min_cosine=args.min_cosine,
                      charge_range=(args.charge_min, args.charge_max),
                      save_plots=not args.no_plots)

"""
pipeline.py
===========
High-level pipeline functions for distributed HDX-MS isotope analysis.

Two main entry points
---------------------
``process_slice``
    Takes a raw file path and a (RT, DT, m/z) slice specification, runs
    the full tensor-build → NTF → isotope-analysis pipeline, writes a CSV
    and per-cluster PNGs, and returns the resulting DataFrame.  Returns an
    empty DataFrame (with correct schema) for slices below BPI/TIC
    thresholds or with no detectable isotopic clusters.

``scan_slice_metrics``
    Rapid diagnostic: iterates over the full slice grid of a raw file and
    reports BPI and TIC per slice without running NTF.  Use this to decide
    which slices are worth processing before launching the full pipeline.

Slice grid (defaults match config.yaml)
----------------------------------------
DT  :  width=50 bins,   step=25 bins   → 50 % overlap
RT  :  width=1.0 min,   step=0.5 min   → 50 % overlap
m/z :  width=100.0 Th,  step=50.0 Th   → 50 % overlap

All slice boundaries are clipped to the file's actual metadata ranges so
no slice ever requests data outside the file.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema for an empty / failed slice result
# ---------------------------------------------------------------------------

_RESULT_COLUMNS = [
    "sample",
    "rt_lo", "rt_hi", "dt_lo", "dt_hi", "mz_lo", "mz_hi",
    "factor_idx", "cluster_idx", "k", "charge",
    "monoisotopic_mz", "monoisotopic_mass_da", "cosine_similarity",
    "left_ratio", "left_shifted",
    "rt_center", "dt_center",
    # Intensity hierarchy: slice tensor → NTF factor → isotopic cluster
    "bpi", "tic",                      # raw .raw slice (initial tensor)
    "factor_bpi", "factor_tic",        # NTF factor m/z profile (factor_tic = sum of mz_profile)
    "cluster_bpi", "cluster_tic",      # isotopic cluster envelope window
    "cluster_intensity",               # cosine-weighted sum of observed isotope peak intensities
    "intensity_fraction",              # cluster_intensity / factor_tic
    "adjusted_score", "gap_fraction", "unexplained_peak_fraction",
    "fit_quality", "n_isotope_peaks_observed",
    # Per-peak intensities integrated at ±10 ppm around each expected isotope position,
    # normalized by max. Pipe-separated string. Independent for k=0 and k=1.
    "peak_intensities",
    # Peak noise metric: average RMSE of per-peak Gaussian fits to the smoothed m/z profile.
    # Low = clean Gaussian peaks; high = noisy, asymmetric, or blended peaks.
    "peak_rmse",
    # RT / DT Gaussian purity metrics (factor-level, same value for all clusters of a factor)
    # gaussian_r2: R² of single-Gaussian fit; 1.0 = clean unimodal, lower = shoulder/multimodal
    # n_peaks: number of significant local maxima (prominence ≥ 10 % of profile max)
    # multimodal: True when n_peaks > 1
    "rt_gaussian_r2", "dt_gaussian_r2",
    "rt_n_peaks", "dt_n_peaks",
    "rt_multimodal", "dt_multimodal",
    # Deduplication flag: True for the best representative of each unique species
    # (seed-based 5 ppm m/z grouping, ranked by cluster_bpi; a second entry is also
    # marked True if its RT_center differs by > 30 s from the primary, preserving
    # genuine same-m/z signals at different retention times).
    "is_best",
]


def _empty_result(
    sample: str,
    rt_lo: float, rt_hi: float,
    dt_lo: int,   dt_hi: int,
    mz_lo: float, mz_hi: float,
) -> pd.DataFrame:
    """Return a zero-row DataFrame with the canonical result schema."""
    return pd.DataFrame(columns=_RESULT_COLUMNS).astype({
        "rt_lo": float, "rt_hi": float,
        "dt_lo": int,   "dt_hi": int,
        "mz_lo": float, "mz_hi": float,
        "factor_idx": int, "cluster_idx": int, "k": int,
        "charge": int,
        "bpi": float, "tic": float,
        "factor_bpi": float, "factor_tic": float,
        "cluster_bpi": float, "cluster_tic": float,
        "is_best": bool,
    })


# ---------------------------------------------------------------------------
# Deduplication helper
# ---------------------------------------------------------------------------

def _annotate_is_best(
    df: pd.DataFrame,
    ppm: float = 5.0,
    rt_sep_min: float = 0.5,
) -> pd.DataFrame:
    """Add an ``is_best`` boolean column marking the best representative of
    each unique molecular species.

    Strategy
    --------
    1. Work on k=0 rows only (one unique row per cluster).
    2. Sort by monoisotopic_mz, then group using a seed-based ppm threshold
       (compare each signal to the *first* m/z seen in the group — prevents
       single-linkage chaining across wide m/z ranges).
    3. Within each m/z group, rank by ``cluster_bpi`` descending.
    4. Mark the highest-BPI entry as ``is_best=True``.
    5. Also mark a second entry as ``is_best=True`` if its RT_center differs
       by more than *rt_sep_min* minutes from the primary selection — this
       preserves genuine same-m/z signals at different retention times.
    6. Propagate ``is_best`` from k=0 to k=1 rows of the same cluster.
    7. All k=0 singletons (no duplicates in their group) are marked True.

    Implementation uses plain numpy loops (no pandas groupby or merge) to
    keep memory usage flat even on large aggregate datasets.

    Parameters
    ----------
    df : DataFrame with at least ``k``, ``monoisotopic_mz``, ``cluster_bpi``,
         ``rt_center``, ``sample``, ``factor_idx``, ``cluster_idx`` columns.
    ppm : seed-based ppm threshold for grouping [default 5.0].
    rt_sep_min : minimum RT separation to mark a second entry [minutes;
                 default 0.5 = 30 s].
    """
    if df.empty:
        df = df.copy()
        df["is_best"] = False
        return df

    df = df.copy()
    df["is_best"] = False

    k0_mask = (df["k"] == 0).values
    if not k0_mask.any():
        return df

    # --- Work on k=0 rows, sorted by monoisotopic_mz ---
    k0 = df.loc[k0_mask].sort_values("monoisotopic_mz")
    orig_idx  = k0.index.values                              # df index positions
    mz_vals   = k0["monoisotopic_mz"].values.astype(np.float64)
    bpi_vals  = k0["cluster_bpi"].values.astype(np.float64)
    rt_vals   = k0["rt_center"].values.astype(np.float64)
    n = len(mz_vals)

    # --- Seed-based ppm grouping ---
    labels = np.zeros(n, dtype=np.int64)
    if n > 0:
        gid  = 0
        seed = mz_vals[0]
        for i in range(1, n):
            if seed == 0.0 or (mz_vals[i] - seed) / seed * 1e6 > ppm:
                gid  += 1
                seed  = mz_vals[i]
            labels[i] = gid
    n_groups = int(labels[-1]) + 1 if n > 0 else 0

    # --- For each group pick primary (max BPI) + optional secondary (RT gap) ---
    best_flags = np.zeros(n, dtype=bool)
    for g in range(n_groups):
        pos = np.where(labels == g)[0]          # positions within k0 arrays
        if len(pos) == 1:
            best_flags[pos[0]] = True
            continue
        order    = pos[np.argsort(bpi_vals[pos])[::-1]]
        best_flags[order[0]] = True
        best_rt  = rt_vals[order[0]]
        for idx in order[1:]:
            if abs(rt_vals[idx] - best_rt) > rt_sep_min:
                best_flags[idx] = True
                break

    df.loc[orig_idx[best_flags], "is_best"] = True

    # --- Propagate is_best to k=1 rows via dict lookup (avoids merge) ---
    k1_mask = (df["k"] == 1).values
    if k1_mask.any():
        k0_sub = df.loc[k0_mask, ["sample", "factor_idx", "cluster_idx", "is_best"]]
        lookup: Dict = dict(zip(
            zip(k0_sub["sample"],
                k0_sub["factor_idx"].astype(int),
                k0_sub["cluster_idx"].astype(int)),
            k0_sub["is_best"],
        ))
        k1_sub = df.loc[k1_mask, ["sample", "factor_idx", "cluster_idx"]]
        df.loc[k1_mask, "is_best"] = np.array([
            lookup.get((s, int(f), int(c)), False)
            for s, f, c in zip(
                k1_sub["sample"],
                k1_sub["factor_idx"],
                k1_sub["cluster_idx"],
            )
        ], dtype=bool)

    return df


# ---------------------------------------------------------------------------
# Slice grid generator
# ---------------------------------------------------------------------------

def generate_slice_grid(
    rt_min: float,  rt_max: float,
    dt_min: int,    dt_max: int,
    mz_min: float,  mz_max: float,
    *,
    dt_width: int   = 50,
    dt_step:  int   = 25,
    rt_width: float = 1.0,
    rt_step:  float = 0.5,
    mz_width: float = 100.0,
    mz_step:  float = 50.0,
) -> List[Dict]:
    """Generate overlapping slice grid clipped to file metadata ranges.

    Parameters
    ----------
    rt_min, rt_max : float — RT range of the raw file [min]
    dt_min, dt_max : int  — DT range of the raw file [bins, 0-based]
    mz_min, mz_max : float — m/z range of the raw file [Da]

    Returns
    -------
    List of dicts with keys rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi.
    """
    slices = []

    # RT windows
    rt_starts = np.arange(rt_min, rt_max, rt_step)
    rt_windows = [
        (float(s), float(min(s + rt_width, rt_max)))
        for s in rt_starts
        if s < rt_max
    ]

    # DT windows
    dt_starts = np.arange(dt_min, dt_max, dt_step)
    dt_windows = [
        (int(s), int(min(s + dt_width, dt_max)))
        for s in dt_starts
        if s < dt_max
    ]

    # m/z windows
    mz_starts = np.arange(mz_min, mz_max, mz_step)
    mz_windows = [
        (float(s), float(min(s + mz_width, mz_max)))
        for s in mz_starts
        if s < mz_max
    ]

    for (rt_lo, rt_hi) in rt_windows:
        for (dt_lo, dt_hi) in dt_windows:
            for (mz_lo, mz_hi) in mz_windows:
                slices.append(dict(
                    rt_lo=rt_lo, rt_hi=rt_hi,
                    dt_lo=dt_lo, dt_hi=dt_hi,
                    mz_lo=mz_lo, mz_hi=mz_hi,
                ))

    return slices


# ---------------------------------------------------------------------------
# BPI / TIC helpers
# ---------------------------------------------------------------------------

def _compute_slice_bpi_tic(
    reader,
    function: int,
    rt_lo: float, rt_hi: float,
    dt_lo: int,   dt_hi: int,
    mz_lo: float, mz_hi: float,
    mz_bin_da: float = 0.01,
) -> Tuple[float, float]:
    """Compute base-peak intensity and total-ion current for one slice.

    Iterates over all (scan, drift) combinations in the slice window and
    accumulates centroid intensities into a binned m/z histogram.  Returns
    (bpi, tic) where:

    ``tic`` — total intensity of the whole 3-D tensor within the window
              (Σ over all RT scans, DT bins, and m/z centroids).  This is
              the same quantity as the TIC you would read off the data after
              restricting to [rt_lo, rt_hi] × [dt_lo, dt_hi] × [mz_lo, mz_hi].

    ``bpi`` — base-peak intensity of the *integrated* m/z profile: each m/z
              bin accumulates signal from all (RT scan, DT bin) pairs, then
              ``bpi`` = max of that summed profile.  This matches what you
              would see in MassLynx as the peak of a composite spectrum
              co-added across the RT and DT window, and is directly
              comparable to a BPC (base-peak chromatogram) value at a
              specific m/z.

    ``mz_bin_da`` controls the bin width for the m/z accumulation (default
    0.01 Da — sufficient to separate adjacent isotopes without fine
    centroid-level resolution).
    """
    meta = reader.metadata()
    rt_axis_fn = meta.rt_axis[function]  # float32 array of RT values

    # Convert RT range to scan indices (0-based)
    scan_mask = (rt_axis_fn >= rt_lo) & (rt_axis_fn <= rt_hi)
    scan_indices = np.where(scan_mask)[0]

    if len(scan_indices) == 0:
        return 0.0, 0.0

    n_drift = meta.n_drift_bins[function]
    drift_lo = max(0, int(dt_lo))
    drift_hi = min(n_drift - 1, int(dt_hi))
    drift_range = range(drift_lo, drift_hi + 1)

    # Pre-allocate a numpy array for m/z bin accumulation (bins relative to
    # mz_lo to keep the array small).  np.add.at accumulates all peaks from
    # one drift scan in a single vectorised call, replacing the per-peak
    # Python loop.
    n_bins = max(1, int(round((mz_hi - mz_lo) / mz_bin_da)) + 2)
    mz_accum = np.zeros(n_bins, dtype=np.float64)
    has_data = False

    for scan in scan_indices:
        for drift in drift_range:
            try:
                mz_arr, int_arr = reader.read_drift_scan(function, int(scan), int(drift))
            except Exception:
                continue
            if len(mz_arr) == 0:
                continue
            mz_arr  = np.asarray(mz_arr,  dtype=np.float64)
            int_arr = np.asarray(int_arr, dtype=np.float64)
            mask_mz = (mz_arr >= mz_lo) & (mz_arr <= mz_hi)
            if not np.any(mask_mz):
                continue
            mz_in  = mz_arr[mask_mz]
            int_in = int_arr[mask_mz]
            bin_keys = np.clip(
                np.round((mz_in - mz_lo) / mz_bin_da).astype(np.int64),
                0, n_bins - 1,
            )
            np.add.at(mz_accum, bin_keys, int_in)
            has_data = True

    if not has_data:
        return 0.0, 0.0

    tic = float(mz_accum.sum())
    bpi = float(mz_accum.max())
    return bpi, tic


# ---------------------------------------------------------------------------
# Public function 1: process_slice
# ---------------------------------------------------------------------------

def process_slice(
    raw_path: str,
    license_path: str,
    function: int,
    rt_lo: float,
    rt_hi: float,
    dt_lo: int,
    dt_hi: int,
    mz_lo: float,
    mz_hi: float,
    output_dir: str,
    output_csv: Optional[str] = None,
    *,
    bpi_threshold: float = 0.0,
    tic_threshold: float = 0.0,
    # tensor / NTF parameters
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
    # isotope analysis parameters
    charge_range: Tuple[int, int] = (3, 15),
    min_cosine: float = 0.5,
    min_peaks_per_cluster: int = 3,
    png_dpi: int = 100,
    plot_format: str = "png",
    save_clusters: bool = True,
    save_factors: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run the full pipeline on one (RT, DT, m/z) slice of a raw file.

    Steps
    -----
    1. Open the raw file and compute BPI / TIC for this slice.
    2. If BPI < bpi_threshold or TIC < tic_threshold, write an empty CSV
       and return an empty DataFrame.
    3. Build tensor → NTF → quality filter (via ``analyze_chunk``).
    4. Optionally save factor-level diagnostic plots (raw, NTF, correlations)
       to *output_dir*/factors/ when *save_factors* is True.
    5. Run isotope analysis on all surviving factors
       (via ``process_all_factors``), saving cluster plots to
       *output_dir*/clusters/ when *save_clusters* is True.
    6. Annotate every row with slice coordinates, BPI, TIC, and sample name.
    7. Write the DataFrame to *output_csv* and return it.

    Parameters
    ----------
    raw_path : str
        Path to the Waters .raw directory.
    license_path : str
        Path to the MassLynx SDK license key file.
    function : int
        0-based function index (typically 0 for the IMS function).
    rt_lo, rt_hi : float
        Retention-time window [min].
    dt_lo, dt_hi : int
        Drift-bin window [0-based].
    mz_lo, mz_hi : float
        m/z window [Da].
    output_dir : str
        Directory in which per-cluster PNG files are saved.
    output_csv : str
        Full path for the output CSV file.
    bpi_threshold, tic_threshold : float
        Skip slices with BPI or TIC below these values (default 0 = keep all).
    verbose : bool
        Print factorization progress.

    Returns
    -------
    pd.DataFrame — the result rows (empty if the slice was unproductive).
    """
    from waters_reader import WatersRawReader, read_license
    from tensor_analysis import analyze_chunk
    from isotope_analysis import process_all_factors

    sample = Path(raw_path).stem
    os.makedirs(output_dir, exist_ok=True)
    if output_csv is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    try:
        return _process_slice_inner(
            raw_path, license_path, function,
            rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi,
            output_dir, output_csv,
            bpi_threshold=bpi_threshold, tic_threshold=tic_threshold,
            mz_bin=mz_bin, gauss_sigma_rt=gauss_sigma_rt,
            gauss_sigma_dt=gauss_sigma_dt, intensity_floor=intensity_floor,
            rank_init=rank_init, corr_threshold=corr_threshold,
            n_iter_max=n_iter_max, rank_max=rank_max, n_restarts=n_restarts,
            rt_r2_min=rt_r2_min, dt_r2_min=dt_r2_min,
            charge_range=charge_range, min_cosine=min_cosine,
            min_peaks_per_cluster=min_peaks_per_cluster,
            png_dpi=png_dpi, plot_format=plot_format,
            save_clusters=save_clusters, save_factors=save_factors,
            verbose=verbose, sample=sample,
        )
    except Exception as exc:
        import traceback
        print(f"[process_slice] ERROR on slice RT[{rt_lo},{rt_hi}] "
              f"DT[{dt_lo},{dt_hi}] mz[{mz_lo},{mz_hi}]: {exc}", flush=True)
        traceback.print_exc()
        df = _empty_result(sample, rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi)
        if output_csv is not None:
            df.to_csv(output_csv, index=False)
        return df


def _process_slice_inner(
    raw_path, license_path, function,
    rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi,
    output_dir, output_csv, *, sample,
    bpi_threshold, tic_threshold, mz_bin, gauss_sigma_rt, gauss_sigma_dt,
    intensity_floor, rank_init, corr_threshold, n_iter_max, rank_max,
    n_restarts, rt_r2_min, dt_r2_min, charge_range, min_cosine,
    min_peaks_per_cluster, png_dpi, plot_format, save_clusters, save_factors,
    verbose,
):
    from waters_reader import WatersRawReader, read_license
    from tensor_analysis import analyze_chunk
    from isotope_analysis import process_all_factors

    license_key = read_license(license_path)

    with WatersRawReader(raw_path, license=license_key) as reader:
        # --- Step 1: BPI / TIC check ---
        bpi, tic = _compute_slice_bpi_tic(
            reader, function,
            rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi,
        )

        if bpi < bpi_threshold or tic < tic_threshold:
            if verbose:
                print(
                    f"[process_slice] Skipping slice "
                    f"RT[{rt_lo:.2f},{rt_hi:.2f}] "
                    f"DT[{dt_lo},{dt_hi}] "
                    f"mz[{mz_lo:.1f},{mz_hi:.1f}]: "
                    f"BPI={bpi:.2e} TIC={tic:.2e} "
                    f"(thresholds BPI={bpi_threshold:.2e}, TIC={tic_threshold:.2e})"
                )
            df = _empty_result(sample, rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi)
            if output_csv is not None:
                df.to_csv(output_csv, index=False)
            return df

        # --- Step 2: Tensor build + NTF ---
        # Generate factor plots in the same call when save_factors is True
        # (avoids re-running NTF; figures are returned in result dict).
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
            apply_quality_filter=True,
            plot=save_factors,          # generate figures only when needed
            verbose=verbose,
        )

        A = result.get("A")
        if A is None or A.shape[1] == 0:
            # NTF produced no valid factors (too weak or all filtered)
            df = _empty_result(sample, rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi)
            if output_csv is not None:
                df.to_csv(output_csv, index=False)
            return df

        B = result["B"]
        C = result["C"]

        # --- Step 2b: Save factor-level diagnostic plots ---
        if save_factors:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            factors_dir = os.path.join(output_dir, "factors")
            os.makedirs(factors_dir, exist_ok=True)
            _slice_tag = (
                f"RT{rt_lo:.1f}-{rt_hi:.1f}"
                f"_DT{dt_lo}-{dt_hi}"
                f"_mz{mz_lo:.1f}-{mz_hi:.1f}"
            )
            for _fig_key, _fig_suffix in [
                ("fig_raw",     "raw"),
                ("fig_factors", "factors"),
                ("fig_corr",    "corr"),
            ]:
                _fig = result.get(_fig_key)
                if _fig is not None:
                    _fpath = os.path.join(
                        factors_dir,
                        f"{_slice_tag}_{_fig_suffix}.{plot_format}",
                    )
                    try:
                        _fig.savefig(_fpath, dpi=png_dpi, bbox_inches="tight")
                    except Exception as _exc:
                        warnings.warn(
                            f"Factor plot save failed ({_fig_key}): {_exc}",
                            RuntimeWarning,
                        )
                    finally:
                        _plt.close(_fig)

        # --- Step 3: Isotope analysis ---
        clusters_dir = os.path.join(output_dir, "clusters") if save_clusters else None
        if clusters_dir is not None:
            os.makedirs(clusters_dir, exist_ok=True)

        df = process_all_factors(
            A, B, C,
            mz_axis=result["mz_axis_ntf"],
            rt_axis=result["rt_axis_ntf"],
            dt_axis=result["dt_axis_ntf"],
            charge_range=charge_range,
            min_cosine=min_cosine,
            min_peaks_per_cluster=min_peaks_per_cluster,
            output_dir=clusters_dir,
            verbose=verbose,
            mz_axis_full=result.get("mz_axis"),
            mask_mz=result.get("mask_mz"),
            png_dpi=png_dpi,
            plot_format=plot_format,
        )

    # --- Step 4: Annotate ---
    df["sample"]  = sample
    df["rt_lo"]   = rt_lo
    df["rt_hi"]   = rt_hi
    df["dt_lo"]   = dt_lo
    df["dt_hi"]   = dt_hi
    df["mz_lo"]   = mz_lo
    df["mz_hi"]   = mz_hi
    df["bpi"]     = bpi
    df["tic"]     = tic

    # Ensure all schema columns are present (fill missing with NaN)
    for col in _RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")

    df = df[_RESULT_COLUMNS]
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
    return df


# ---------------------------------------------------------------------------
# Public function 2: process_batch
# ---------------------------------------------------------------------------

def process_batch(
    raw_path: str,
    license_path: str,
    function: int,
    slice_list_json: str,
    batch_idx: int,
    batch_size: int,
    output_dir: str,
    output_csv: str,
    *,
    bpi_threshold: float = 0.0,
    tic_threshold: float = 0.0,
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
    png_dpi: int = 100,
    plot_format: str = "png",
    save_clusters: bool = True,
    save_factors: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Process a contiguous batch of slices from slice_list.json.

    Reads ``batch_size`` slices starting at index ``batch_idx * batch_size``
    from *slice_list_json*, runs the full pipeline on each in sequence, and
    writes a single concatenated CSV to *output_csv*.  Individual slice errors
    are caught by ``process_slice``'s own handler; the batch always produces
    its output CSV so Snakemake is never blocked by a single-slice crash.

    Parameters
    ----------
    slice_list_json : str
        Path to the JSON file written by ``write_slice_list``.
    batch_idx : int
        Zero-based batch index (Snakemake wildcard ``batch_id`` cast to int).
    batch_size : int
        Number of slices per batch (must match value used to build the DAG).
    output_csv : str
        Path for the combined batch output CSV.
    """
    with open(slice_list_json) as fh:
        all_slices = json.load(fh)

    start = batch_idx * batch_size
    end   = min(start + batch_size, len(all_slices))
    batch = all_slices[start:end]

    sample = Path(raw_path).stem
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    frames: List[pd.DataFrame] = []
    for i, s in enumerate(batch):
        print(
            f"[process_batch] {batch_idx:04d}  "
            f"slice {start + i + 1}/{len(all_slices)}  "
            f"RT[{s['rt_lo']:.2f},{s['rt_hi']:.2f}] "
            f"DT[{s['dt_lo']},{s['dt_hi']}] "
            f"mz[{s['mz_lo']:.1f},{s['mz_hi']:.1f}]",
            flush=True,
        )
        df = process_slice(
            raw_path, license_path, function,
            s["rt_lo"], s["rt_hi"],
            s["dt_lo"], s["dt_hi"],
            s["mz_lo"], s["mz_hi"],
            output_dir=output_dir,
            output_csv=None,           # batch manages its own combined CSV
            bpi_threshold=bpi_threshold,
            tic_threshold=tic_threshold,
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
            charge_range=charge_range,
            min_cosine=min_cosine,
            min_peaks_per_cluster=min_peaks_per_cluster,
            png_dpi=png_dpi,
            plot_format=plot_format,
            save_clusters=save_clusters,
            save_factors=save_factors,
            verbose=verbose,
        )
        frames.append(df)

    result = (
        pd.concat(frames, ignore_index=True)
        if frames
        else _empty_result(sample, 0.0, 0.0, 0, 0, 0.0, 0.0)
    )
    result.to_csv(output_csv, index=False)
    print(f"[process_batch] {batch_idx:04d}  written → {output_csv}", flush=True)
    return result


# ---------------------------------------------------------------------------
# Public function 3: scan_slice_metrics
# ---------------------------------------------------------------------------

def scan_slice_metrics(
    raw_path: str,
    license_path: str,
    function: int = 0,
    *,
    dt_width: int   = 50,
    dt_step:  int   = 25,
    rt_width: float = 1.0,
    rt_step:  float = 0.5,
    mz_width: float = 100.0,
    mz_step:  float = 50.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute BPI and TIC for every slice in the grid without running NTF.

    Iterates over all (RT × DT × m/z) slice combinations defined by the
    grid parameters and calls ``_compute_slice_bpi_tic`` for each.  Returns
    a DataFrame with columns:
        sample, rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi, bpi, tic

    Parameters
    ----------
    raw_path : str
        Path to the Waters .raw directory.
    license_path : str
        Path to the MassLynx SDK license key file.
    function : int
        0-based function index.
    dt_width, dt_step, rt_width, rt_step, mz_width, mz_step :
        Slice grid parameters (see ``generate_slice_grid``).
    verbose : bool
        Print per-slice progress.

    Returns
    -------
    pd.DataFrame with one row per slice.
    """
    from waters_reader import WatersRawReader, read_license

    sample = Path(raw_path).stem
    license_key = read_license(license_path)

    rows = []
    with WatersRawReader(raw_path, license=license_key) as reader:
        meta = reader.metadata()
        rt_range = meta.rt_range[function]      # (rt_min, rt_max)
        mz_range = meta.mass_range[function]    # (mz_lo, mz_hi)
        n_drift  = meta.n_drift_bins[function]

        slices = generate_slice_grid(
            rt_min=float(rt_range[0]),  rt_max=float(rt_range[1]),
            dt_min=0,                    dt_max=int(n_drift - 1),
            mz_min=float(mz_range[0]),  mz_max=float(mz_range[1]),
            dt_width=dt_width,  dt_step=dt_step,
            rt_width=rt_width,  rt_step=rt_step,
            mz_width=mz_width,  mz_step=mz_step,
        )

        n_total = len(slices)
        for i, s in enumerate(slices):
            if verbose:
                print(
                    f"[scan_slice_metrics] {i+1}/{n_total}  "
                    f"RT[{s['rt_lo']:.2f},{s['rt_hi']:.2f}] "
                    f"DT[{s['dt_lo']},{s['dt_hi']}] "
                    f"mz[{s['mz_lo']:.1f},{s['mz_hi']:.1f}]"
                )
            bpi, tic = _compute_slice_bpi_tic(
                reader, function,
                s["rt_lo"], s["rt_hi"],
                s["dt_lo"], s["dt_hi"],
                s["mz_lo"], s["mz_hi"],
            )
            rows.append(dict(
                sample=sample,
                rt_lo=s["rt_lo"],  rt_hi=s["rt_hi"],
                dt_lo=s["dt_lo"],  dt_hi=s["dt_hi"],
                mz_lo=s["mz_lo"],  mz_hi=s["mz_hi"],
                bpi=bpi, tic=tic,
            ))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Slice list JSON helper (used by Snakemake checkpoint)
# ---------------------------------------------------------------------------

def write_slice_list(
    raw_path: str,
    license_path: str,
    function: int,
    output_json: str,
    *,
    dt_width: int   = 50,
    dt_step:  int   = 25,
    rt_width: float = 1.0,
    rt_step:  float = 0.5,
    mz_width: float = 100.0,
    mz_step:  float = 50.0,
    tic_threshold: float = 0.0,
    bpi_threshold: float = 0.0,
    metrics_csv: Optional[str] = None,
) -> None:
    """Write the filtered slice grid for one raw file as a JSON array.

    **Chromatographic pre-filter** — uses the TIC chromatogram (one fast SDK
    call) to compute a per-slice RT-window TIC and BPI proxy.  Only slices
    that pass both thresholds are written to *output_json* and will become
    Snakemake batch jobs.  All slices are written to *metrics_csv* for
    diagnostics.

    IMPORTANT — unit mismatch with the 3-D batch filter
    ----------------------------------------------------
    The quantities measured here are:
      ``rt_tic_sum``  — sum of per-scan TIC values within the RT window,
                        integrated over ALL drift bins and ALL m/z.
      ``rt_tic_max``  — maximum per-scan TIC within the RT window
                        (similarly all-mass, all-drift).

    These are orders of magnitude larger than the 3-D window TIC/BPI
    computed inside ``_process_slice_inner`` (which reads only the specific
    DT × m/z sub-volume).  The thresholds passed here should therefore come
    from the ``pre_filter`` config block, **not** from the ``thresholds``
    block that controls the 3-D filter in the batch jobs.

    Typical scale guidance (IMS-MS protein data):
      chromatographic TIC sum  → 1e7–1e9
      chromatographic TIC max  → 1e5–1e7
      3-D window TIC (batch)   → 1e3–1e6 (depends on DT/mz window width)

    Parameters
    ----------
    output_json : str
        Path for the filtered slice list (Snakemake checkpoint output).
    tic_threshold : float
        Minimum chromatographic RT-window TIC sum.  Default 0 (keep all).
    bpi_threshold : float
        Minimum chromatographic RT-window TIC max.  Default 0 (keep all).
    metrics_csv : str, optional
        If given, write a CSV with one row per slice including rt_tic_sum,
        rt_tic_max, and a passes_filter boolean.
    """
    from waters_reader import WatersRawReader, read_license

    sample      = Path(raw_path).stem
    license_key = read_license(license_path)

    with WatersRawReader(raw_path, license=license_key) as reader:
        meta     = reader.metadata()
        rt_range = meta.rt_range[function]
        mz_range = meta.mass_range[function]
        n_drift  = meta.n_drift_bins[function]

        # One SDK call for the full TIC chromatogram
        rt_tic_axis, tic_values = reader.read_tic(function)

    rt_tic_axis = np.asarray(rt_tic_axis, dtype=np.float64)
    tic_values  = np.asarray(tic_values,  dtype=np.float64)

    all_slices = generate_slice_grid(
        rt_min=float(rt_range[0]),  rt_max=float(rt_range[1]),
        dt_min=0,                    dt_max=int(n_drift - 1),
        mz_min=float(mz_range[0]),  mz_max=float(mz_range[1]),
        dt_width=dt_width,  dt_step=dt_step,
        rt_width=rt_width,  rt_step=rt_step,
        mz_width=mz_width,  mz_step=mz_step,
    )

    metrics_rows: List[Dict] = []
    productive:  List[Dict] = []

    for s in all_slices:
        rt_mask     = (rt_tic_axis >= s["rt_lo"]) & (rt_tic_axis <= s["rt_hi"])
        rt_tic_sum  = float(tic_values[rt_mask].sum())  if rt_mask.any() else 0.0
        rt_tic_max  = float(tic_values[rt_mask].max())  if rt_mask.any() else 0.0
        passes      = (rt_tic_sum >= tic_threshold) and (rt_tic_max >= bpi_threshold)

        slice_id = (
            f"RT{s['rt_lo']:.2f}-{s['rt_hi']:.2f}"
            f"_DT{s['dt_lo']}-{s['dt_hi']}"
            f"_mz{s['mz_lo']:.1f}-{s['mz_hi']:.1f}"
        )

        metrics_rows.append(dict(
            sample=sample,
            rt_lo=s["rt_lo"],  rt_hi=s["rt_hi"],
            dt_lo=s["dt_lo"],  dt_hi=s["dt_hi"],
            mz_lo=s["mz_lo"],  mz_hi=s["mz_hi"],
            slice_id=slice_id,
            rt_tic_sum=rt_tic_sum,
            rt_tic_max=rt_tic_max,
            passes_filter=passes,
        ))

        if passes:
            s["slice_id"]    = slice_id
            s["rt_tic_sum"]  = rt_tic_sum
            s["rt_tic_max"]  = rt_tic_max
            productive.append(s)

    n_total = len(all_slices)
    n_pass  = len(productive)
    print(
        f"[write_slice_list] {sample}: {n_total} slices total, "
        f"{n_pass} pass filter "
        f"(tic>={tic_threshold:.2e}, bpi>={bpi_threshold:.2e}), "
        f"{n_total - n_pass} skipped.",
        flush=True,
    )

    if metrics_csv:
        os.makedirs(os.path.dirname(os.path.abspath(metrics_csv)), exist_ok=True)
        pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False)
        print(f"[write_slice_list] metrics → {metrics_csv}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as fh:
        json.dump(productive, fh, indent=2)


# ---------------------------------------------------------------------------
# apply_filters: post-aggregate quality filter
# ---------------------------------------------------------------------------

def apply_filters(
    input_csv: str,
    output_csv: str,
    # Signal intensity cuts
    cluster_bpi_min: float = 5e3,
    cluster_tic_bpi_ratio_min: float = 10.0,
    # Gaussian shape quality
    rt_gaussian_r2_min: float = 0.9,
    dt_gaussian_r2_min: float = 0.9,
    # Edge guard: cluster centre must be at least this far from the slice boundary
    rt_edge_margin: float = 0.1,   # minutes
    dt_edge_margin: float = 5.0,   # drift bins
) -> None:
    """Apply post-aggregate quality filters and write a filtered CSV.

    Filters applied (all must pass):

    1. ``cluster_bpi >= cluster_bpi_min``              — minimum peak intensity
    2. ``cluster_tic / cluster_bpi >= cluster_tic_bpi_ratio_min``  — peak width proxy
    3. ``rt_gaussian_r2 >= rt_gaussian_r2_min``        — RT Gaussian fit quality
    4. ``dt_gaussian_r2 >= dt_gaussian_r2_min``        — DT Gaussian fit quality
    5. ``rt_lo + rt_edge_margin < rt_center < rt_hi - rt_edge_margin``
       — RT centre not at slice edge
    6. ``dt_lo + dt_edge_margin < dt_center < dt_hi - dt_edge_margin``
       — DT centre not at slice edge

    Filters are applied to *all* rows (k=0 and k=1).  Because k=0 and k=1
    rows of the same cluster share identical cluster-level metadata, a cluster
    either survives as both rows or is dropped entirely.

    Parameters
    ----------
    input_csv :
        Path to the aggregate isotopes CSV (e.g. ``*_isotopes_cal.csv``).
    output_csv :
        Output path for the filtered CSV.
    cluster_bpi_min :
        Minimum ``cluster_bpi``.  Default 5 000.
    cluster_tic_bpi_ratio_min :
        Minimum ``cluster_tic / cluster_bpi`` ratio.  Default 10.
    rt_gaussian_r2_min :
        Minimum RT Gaussian R².  Default 0.9.
    dt_gaussian_r2_min :
        Minimum DT Gaussian R².  Default 0.9.
    rt_edge_margin :
        RT slice edge guard in minutes.  Default 0.1 min.
    dt_edge_margin :
        DT slice edge guard in drift bins.  Default 5 bins.
    """
    df = pd.read_csv(input_csv)
    n_in = len(df)
    print(f"[apply_filters] Loaded {n_in} rows from {input_csv}", flush=True)

    mask = pd.Series(True, index=df.index)

    # 1. Minimum cluster BPI
    if "cluster_bpi" in df.columns:
        mask &= df["cluster_bpi"] >= cluster_bpi_min

    # 2. TIC / BPI ratio (isotope peak width proxy)
    if "cluster_tic" in df.columns and "cluster_bpi" in df.columns:
        ratio = df["cluster_tic"] / (df["cluster_bpi"] + 1e-12)
        mask &= ratio >= cluster_tic_bpi_ratio_min

    # 3. RT Gaussian shape
    if "rt_gaussian_r2" in df.columns:
        mask &= df["rt_gaussian_r2"] >= rt_gaussian_r2_min

    # 4. DT Gaussian shape
    if "dt_gaussian_r2" in df.columns:
        mask &= df["dt_gaussian_r2"] >= dt_gaussian_r2_min

    # 5. RT centre not at slice edge
    if all(c in df.columns for c in ["rt_lo", "rt_hi", "rt_center"]):
        mask &= df["rt_center"] > df["rt_lo"] + rt_edge_margin
        mask &= df["rt_center"] < df["rt_hi"] - rt_edge_margin

    # 6. DT centre not at slice edge
    if all(c in df.columns for c in ["dt_lo", "dt_hi", "dt_center"]):
        mask &= df["dt_center"] > df["dt_lo"] + dt_edge_margin
        mask &= df["dt_center"] < df["dt_hi"] - dt_edge_margin

    filtered = df[mask].copy()
    n_out = len(filtered)
    n_dropped = n_in - n_out
    print(
        f"[apply_filters] {n_out} / {n_in} rows passed "
        f"({n_dropped} dropped, {100.*n_out/max(n_in,1):.1f}% kept).",
        flush=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    filtered.to_csv(output_csv, index=False)
    print(f"[apply_filters] Filtered CSV → {output_csv}", flush=True)


# ---------------------------------------------------------------------------
# CLI entry points (called by Snakemake shell directives)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HDX-MS slice pipeline — process one slice or scan metrics"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- process_slice ----
    p_proc = sub.add_parser("process_slice", help="Run full pipeline on one slice")
    p_proc.add_argument("raw_path")
    p_proc.add_argument("license_path")
    p_proc.add_argument("--function",    type=int,   default=0)
    p_proc.add_argument("--rt_lo",       type=float, required=True)
    p_proc.add_argument("--rt_hi",       type=float, required=True)
    p_proc.add_argument("--dt_lo",       type=int,   required=True)
    p_proc.add_argument("--dt_hi",       type=int,   required=True)
    p_proc.add_argument("--mz_lo",       type=float, required=True)
    p_proc.add_argument("--mz_hi",       type=float, required=True)
    p_proc.add_argument("--output_dir",  required=True)
    p_proc.add_argument("--output_csv",  required=True)
    p_proc.add_argument("--bpi_threshold", type=float, default=0.0)
    p_proc.add_argument("--tic_threshold", type=float, default=0.0)
    p_proc.add_argument("--rank_init",   type=int,   default=5)
    p_proc.add_argument("--rank_max",    type=int,   default=15)
    p_proc.add_argument("--n_restarts",  type=int,   default=3)
    p_proc.add_argument("--png_dpi",     type=int,   default=100,
                        help="DPI for per-cluster PNG output (default: 100)")
    p_proc.add_argument("--verbose",     action="store_true")

    # ---- write_slice_list ----
    p_list = sub.add_parser("write_slice_list", help="Write filtered slice grid JSON + metrics CSV")
    p_list.add_argument("raw_path")
    p_list.add_argument("license_path")
    p_list.add_argument("--function",       type=int,   default=0)
    p_list.add_argument("--output_json",    required=True)
    p_list.add_argument("--metrics_csv",    default=None,
                        help="Path for full per-slice metrics CSV (all slices, pass+fail)")
    p_list.add_argument("--dt_width",       type=int,   default=50)
    p_list.add_argument("--dt_step",        type=int,   default=25)
    p_list.add_argument("--rt_width",       type=float, default=1.0)
    p_list.add_argument("--rt_step",        type=float, default=0.5)
    p_list.add_argument("--mz_width",       type=float, default=100.0)
    p_list.add_argument("--mz_step",        type=float, default=50.0)
    p_list.add_argument("--tic_threshold",  type=float, default=0.0)
    p_list.add_argument("--bpi_threshold",  type=float, default=0.0)

    # ---- scan_metrics ----
    p_scan = sub.add_parser("scan_metrics", help="Report BPI/TIC per slice")
    p_scan.add_argument("raw_path")
    p_scan.add_argument("license_path")
    p_scan.add_argument("--function",  type=int,   default=0)
    p_scan.add_argument("--output_csv", required=True)
    p_scan.add_argument("--dt_width",  type=int,   default=50)
    p_scan.add_argument("--dt_step",   type=int,   default=25)
    p_scan.add_argument("--rt_width",  type=float, default=1.0)
    p_scan.add_argument("--rt_step",   type=float, default=0.5)
    p_scan.add_argument("--mz_width",  type=float, default=100.0)
    p_scan.add_argument("--mz_step",   type=float, default=50.0)
    p_scan.add_argument("--verbose",   action="store_true")

    # ---- scan_slice ----
    p_ss = sub.add_parser("scan_slice", help="Report BPI/TIC for a single slice")
    p_ss.add_argument("raw_path")
    p_ss.add_argument("license_path")
    p_ss.add_argument("--function",   type=int,   default=0)
    p_ss.add_argument("--rt_lo",      type=float, required=True)
    p_ss.add_argument("--rt_hi",      type=float, required=True)
    p_ss.add_argument("--dt_lo",      type=int,   required=True)
    p_ss.add_argument("--dt_hi",      type=int,   required=True)
    p_ss.add_argument("--mz_lo",      type=float, required=True)
    p_ss.add_argument("--mz_hi",      type=float, required=True)
    p_ss.add_argument("--output_csv", required=True)

    # ---- process_batch ----
    p_batch = sub.add_parser("process_batch", help="Run full pipeline on a batch of slices")
    p_batch.add_argument("raw_path")
    p_batch.add_argument("license_path")
    p_batch.add_argument("--function",      type=int,   default=0)
    p_batch.add_argument("--slice_list",    required=True,
                         help="Path to slice_list.json written by write_slice_list")
    p_batch.add_argument("--batch_idx",     type=int,   required=True,
                         help="Zero-based batch index")
    p_batch.add_argument("--batch_size",    type=int,   default=50,
                         help="Number of slices per batch")
    p_batch.add_argument("--output_dir",    required=True)
    p_batch.add_argument("--output_csv",    required=True)
    p_batch.add_argument("--bpi_threshold", type=float, default=0.0)
    p_batch.add_argument("--tic_threshold", type=float, default=0.0)
    p_batch.add_argument("--rank_init",     type=int,   default=5)
    p_batch.add_argument("--rank_max",      type=int,   default=15)
    p_batch.add_argument("--n_restarts",    type=int,   default=3)
    p_batch.add_argument("--png_dpi",       type=int,   default=100,
                         help="DPI for saved images (default: 100)")
    p_batch.add_argument("--plot_format",   default="png", choices=["png", "pdf"],
                         help="Image format: png (default) or pdf")
    p_batch.add_argument("--save_clusters", action="store_true", default=True,
                         help="Save per-cluster isotope envelope plots (default on)")
    p_batch.add_argument("--no_save_clusters", dest="save_clusters",
                         action="store_false",
                         help="Disable cluster plot saving")
    p_batch.add_argument("--save_factors",  action="store_true", default=False,
                         help="Save factor-level diagnostic plots (raw+NTF+corr)")
    p_batch.add_argument("--verbose",       action="store_true")

    # ---- aggregate ----
    p_agg = sub.add_parser("aggregate", help="Concatenate per-slice CSVs into one file")
    p_agg.add_argument("--input_pattern", required=True,
                       help="Glob pattern matching per-slice CSV files (quote it!)")
    p_agg.add_argument("--output_csv", required=True)

    # ---- apply_filters ----
    p_filt = sub.add_parser("apply_filters",
                            help="Apply quality filters to aggregate CSV")
    p_filt.add_argument("--input_csv",  required=True)
    p_filt.add_argument("--output_csv", required=True)
    p_filt.add_argument("--cluster_bpi_min",          type=float, default=5e3,
                        help="Minimum cluster_bpi (default: 5000)")
    p_filt.add_argument("--cluster_tic_bpi_ratio_min", type=float, default=10.0,
                        help="Minimum cluster_tic / cluster_bpi ratio (default: 10)")
    p_filt.add_argument("--rt_gaussian_r2_min",       type=float, default=0.9,
                        help="Minimum rt_gaussian_r2 (default: 0.9)")
    p_filt.add_argument("--dt_gaussian_r2_min",       type=float, default=0.9,
                        help="Minimum dt_gaussian_r2 (default: 0.9)")
    p_filt.add_argument("--rt_edge_margin",           type=float, default=0.1,
                        help="RT edge guard in minutes (default: 0.1)")
    p_filt.add_argument("--dt_edge_margin",           type=float, default=5.0,
                        help="DT edge guard in drift bins (default: 5)")

    args = parser.parse_args()

    if args.cmd == "process_slice":
        process_slice(
            raw_path=args.raw_path,
            license_path=args.license_path,
            function=args.function,
            rt_lo=args.rt_lo,   rt_hi=args.rt_hi,
            dt_lo=args.dt_lo,   dt_hi=args.dt_hi,
            mz_lo=args.mz_lo,   mz_hi=args.mz_hi,
            output_dir=args.output_dir,
            output_csv=args.output_csv,
            bpi_threshold=args.bpi_threshold,
            tic_threshold=args.tic_threshold,
            rank_init=args.rank_init,
            rank_max=args.rank_max,
            n_restarts=args.n_restarts,
            png_dpi=args.png_dpi,
            verbose=args.verbose,
        )

    elif args.cmd == "write_slice_list":
        write_slice_list(
            raw_path=args.raw_path,
            license_path=args.license_path,
            function=args.function,
            output_json=args.output_json,
            metrics_csv=args.metrics_csv,
            dt_width=args.dt_width,        dt_step=args.dt_step,
            rt_width=args.rt_width,        rt_step=args.rt_step,
            mz_width=args.mz_width,        mz_step=args.mz_step,
            tic_threshold=args.tic_threshold,
            bpi_threshold=args.bpi_threshold,
        )

    elif args.cmd == "process_batch":
        process_batch(
            raw_path=args.raw_path,
            license_path=args.license_path,
            function=args.function,
            slice_list_json=args.slice_list,
            batch_idx=args.batch_idx,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            output_csv=args.output_csv,
            bpi_threshold=args.bpi_threshold,
            tic_threshold=args.tic_threshold,
            rank_init=args.rank_init,
            rank_max=args.rank_max,
            n_restarts=args.n_restarts,
            png_dpi=args.png_dpi,
            plot_format=args.plot_format,
            save_clusters=args.save_clusters,
            save_factors=args.save_factors,
            verbose=args.verbose,
        )

    elif args.cmd == "scan_metrics":
        df = scan_slice_metrics(
            raw_path=args.raw_path,
            license_path=args.license_path,
            function=args.function,
            dt_width=args.dt_width,   dt_step=args.dt_step,
            rt_width=args.rt_width,   rt_step=args.rt_step,
            mz_width=args.mz_width,   mz_step=args.mz_step,
            verbose=args.verbose,
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Wrote {len(df)} rows → {args.output_csv}")

    elif args.cmd == "scan_slice":
        from waters_reader import WatersRawReader, read_license
        sample = Path(args.raw_path).stem
        license_key = read_license(args.license_path)
        with WatersRawReader(args.raw_path, license=license_key) as reader:
            bpi, tic = _compute_slice_bpi_tic(
                reader, args.function,
                args.rt_lo, args.rt_hi,
                args.dt_lo, args.dt_hi,
                args.mz_lo, args.mz_hi,
            )
        row = pd.DataFrame([dict(
            sample=sample,
            rt_lo=args.rt_lo, rt_hi=args.rt_hi,
            dt_lo=args.dt_lo, dt_hi=args.dt_hi,
            mz_lo=args.mz_lo, mz_hi=args.mz_hi,
            bpi=bpi, tic=tic,
        )])
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        row.to_csv(args.output_csv, index=False)

    elif args.cmd == "aggregate":
        import glob
        csv_files = sorted(glob.glob(args.input_pattern))
        frames = []
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                if len(df) > 0:
                    frames.append(df)
            except Exception as exc:
                print(f"WARNING: could not read {csv_path}: {exc}")
        if frames:
            result = pd.concat(frames, ignore_index=True)
        else:
            result = pd.DataFrame(columns=_RESULT_COLUMNS)

        # Compute is_best across the full sample (requires all batches present)
        print("[aggregate] Annotating is_best …", flush=True)
        result = _annotate_is_best(result, ppm=5.0, rt_sep_min=0.5)
        n_best = int(result["is_best"].sum()) if "is_best" in result.columns else 0
        print(f"[aggregate] is_best=True: {n_best} / {len(result)} rows", flush=True)

        # Ensure all schema columns present and in canonical order
        for col in _RESULT_COLUMNS:
            if col not in result.columns:
                result[col] = float("nan")
        result = result[_RESULT_COLUMNS]

        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        result.to_csv(args.output_csv, index=False)
        print(f"Aggregated {len(frames)} non-empty slices → {len(result)} rows → {args.output_csv}")

    elif args.cmd == "apply_filters":
        apply_filters(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            cluster_bpi_min=args.cluster_bpi_min,
            cluster_tic_bpi_ratio_min=args.cluster_tic_bpi_ratio_min,
            rt_gaussian_r2_min=args.rt_gaussian_r2_min,
            dt_gaussian_r2_min=args.dt_gaussian_r2_min,
            rt_edge_margin=args.rt_edge_margin,
            dt_edge_margin=args.dt_edge_margin,
        )

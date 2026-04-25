"""
calibration.py
==============
Mass calibration from the lock-mass function of a Waters .raw file.

The lock-mass function is always the **last** function in the file
(0-based index: ``meta.n_functions - 1``).  Sodium Formate is the
default reference compound.

Three main entry points
-----------------------
``extract_calibration``
    Reads the lock-mass function, matches peaks to theoretical masses,
    fits Gaussian models, builds one polynomial calibration curve per
    RT chunk, and writes a JSON calibration file plus a diagnostic PDF.

``apply_calibration(mz, rt, cal_chunks)``
    Given a measured m/z and a retention time, returns the corrected m/z
    using the nearest RT chunk's polynomial.

``calibrate_csv``
    Applies per-RT calibration to the ``monoisotopic_mz`` column of an
    aggregate CSV written by the pipeline, writing a new corrected CSV.

CLI
---
    python calibration.py extract  raw_path license_path --output_json cal.json ...
    python calibration.py apply_csv --input_csv agg.csv --cal_json cal.json --output_csv agg_cal.csv
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Theoretical reference masses
# ---------------------------------------------------------------------------

_SODIUM_FORMATE_MZ = np.array([
     90.977190,  158.964613,  226.952035,  294.939457,  362.926880,
    430.914302,  498.901724,  566.889146,  634.876569,  702.863991,
    770.851413,  838.838836,  906.826258,  974.813680, 1042.801103,
   1110.788525, 1178.775947, 1246.763369, 1314.750792, 1382.738214,
   1450.725636, 1518.713059, 1586.700481, 1654.687903, 1722.675326,
   1790.662748, 1858.650170, 1926.637592, 1994.625015, 2062.612437,
   2130.599859, 2198.587282, 2266.574704, 2334.562126, 2402.549549,
   2470.536971, 2538.524393, 2606.511815, 2674.499238, 2742.486660,
   2810.474082, 2878.461505, 2946.448927,
])

_GLUFIB_FRAGMENTS_MZ = np.array([
     72.081300,   120.081300,  175.119500,  187.071900,  246.156600,
    333.188600,   382.172600,  497.199600,  627.325400,  684.346900,
    813.389500,   942.432100, 1056.475000, 1171.502000, 1285.544800,
])

THEORETICAL_MZ: Dict[str, np.ndarray] = {
    "SodiumFormate":    _SODIUM_FORMATE_MZ,
    "GluFibFragments":  _GLUFIB_FRAGMENTS_MZ,
}


def get_theoretical_mz(
    compound: str,
    mz_min: float = 50.0,
    mz_max: float = 2200.0,
) -> np.ndarray:
    """Return theoretical m/z values for *compound* within [mz_min, mz_max]."""
    if compound not in THEORETICAL_MZ:
        raise ValueError(
            f"Unknown compound {compound!r}. "
            f"Available: {list(THEORETICAL_MZ)}"
        )
    arr = THEORETICAL_MZ[compound]
    return arr[(arr >= mz_min) & (arr <= mz_max)]


# ---------------------------------------------------------------------------
# Gaussian helpers (same convention as the old mhdx_tools code)
# ---------------------------------------------------------------------------

def _gaussian(x: np.ndarray, H: float, A: float, x0: float, sigma: float) -> np.ndarray:
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def _gauss_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """Fit a Gaussian to (x, y) and return (H, A, x0, sigma)."""
    mean  = float(np.sum(x * y) / np.sum(y))
    sigma = float(np.sqrt(np.sum(y * (x - mean) ** 2) / np.sum(y)))
    popt, _ = curve_fit(_gaussian, x, y, p0=[0.0, float(y.max()), mean, sigma])
    return tuple(popt)  # type: ignore[return-value]


def _match_peak(
    mzs: np.ndarray,
    spectrum: np.ndarray,
    mz_thr: float,
    ppm_radius: float,
    min_intensity: float,
) -> Tuple[Optional[float], float]:
    """Fit a Gaussian to the region near *mz_thr*.

    Returns
    -------
    obs_mz : float or None
        Gaussian centre if the peak passes quality cuts, else None.
    intensity : float
        Fitted peak amplitude A (0 if fit failed).
    """
    mz_lo = mz_thr * (1.0 - ppm_radius * 1e-6)
    mz_hi = mz_thr * (1.0 + ppm_radius * 1e-6)
    mask  = (mzs >= mz_lo) & (mzs <= mz_hi)

    if mask.sum() < 3:
        return None, 0.0

    try:
        H, A, x0, sigma = _gauss_fit(mzs[mask], spectrum[mask])
    except Exception:
        return None, 0.0

    ppm_err = abs((x0 - mz_thr) * 1e6 / mz_thr)
    if ppm_err <= ppm_radius and A >= min_intensity:
        return float(x0), float(A)
    return None, float(A)


# ---------------------------------------------------------------------------
# Lock-mass tensor extraction
# ---------------------------------------------------------------------------

def extract_lockmass_tensor(
    raw_path: str,
    license_path: str,
    *,
    compound: str = "SodiumFormate",
    ppm_radius: float = 100.0,
    mz_min: float = 50.0,
    mz_max: float = 2200.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Read the lock-mass function and build a (n_scans × n_mz) tensor.

    The lock-mass function is the last function in the raw file
    (0-based index ``meta.n_functions - 1``).  Only m/z bins within
    *ppm_radius* of any theoretical mass are kept, keeping memory low.

    Parameters
    ----------
    raw_path : str
        Path to the Waters .raw directory.
    license_path : str
        Path to the MassLynx SDK license key file.
    compound : str
        Reference compound name (key in THEORETICAL_MZ).
    ppm_radius : float
        m/z window [ppm] around each theoretical peak used for filtering.
    mz_min, mz_max : float
        Global m/z range to consider.

    Returns
    -------
    rt_axis : float32 (n_scans,)
        Retention time of each scan [min].
    mzs : float32 (n_mz,)
        Filtered m/z axis (near-theoretical only).
    tensor : float32 (n_scans, n_mz)
        Intensity per (scan, mz_bin).
    lockmass_fn : int
        0-based function index of the lock-mass function.
    """
    from waters_reader import WatersRawReader, read_license

    theoretical_mz = get_theoretical_mz(compound, mz_min=mz_min, mz_max=mz_max)
    license_key    = read_license(license_path)

    with WatersRawReader(raw_path, license=license_key) as reader:
        meta         = reader.metadata()
        lockmass_fn  = meta.n_functions - 1
        rt_axis      = meta.rt_axis[lockmass_fn].copy()
        n_scans      = len(rt_axis)

        print(
            f"[calibration] lock-mass function index {lockmass_fn}  "
            f"({n_scans} scans, "
            f"m/z {meta.mass_range[lockmass_fn][0]:.1f}–"
            f"{meta.mass_range[lockmass_fn][1]:.1f})",
            flush=True,
        )

        # First pass: collect all near-theoretical m/z values across all scans
        all_mz_near: List[np.ndarray] = []
        scan_data:   List[Tuple[np.ndarray, np.ndarray]] = []

        for scan_1based in range(0, n_scans):
            mz, ints = reader.read_scan(lockmass_fn, scan_1based)
            if mz.size == 0:
                scan_data.append((np.empty(0, dtype=np.float32),
                                  np.empty(0, dtype=np.float32)))
                continue
            # Keep only peaks near theoretical masses
            near = np.any(
                np.abs(mz[:, None] - theoretical_mz[None, :])
                / theoretical_mz[None, :] * 1e6
                <= ppm_radius,
                axis=1,
            )
            mz_f  = mz[near].astype(np.float32)
            int_f = ints[near].astype(np.float32)
            scan_data.append((mz_f, int_f))
            if mz_f.size:
                all_mz_near.append(mz_f)

        if not all_mz_near:
            raise ValueError(
                f"No lock-mass peaks found within {ppm_radius} ppm of "
                f"{compound} theoretical masses."
            )

        # Build a common, sorted m/z axis
        mzs = np.unique(np.concatenate(all_mz_near)).astype(np.float32)

        # Second pass: fill the dense tensor
        tensor = np.zeros((n_scans, len(mzs)), dtype=np.float32)
        for scan_idx, (mz_f, int_f) in enumerate(scan_data):
            if mz_f.size:
                idx = np.searchsorted(mzs, mz_f)
                # Guard against off-by-one from float rounding
                idx = np.clip(idx, 0, len(mzs) - 1)
                tensor[scan_idx, idx] = int_f

    print(
        f"[calibration] tensor shape: {tensor.shape}  "
        f"({len(theoretical_mz)} theoretical peaks in range)",
        flush=True,
    )
    return rt_axis, mzs, tensor, lockmass_fn


# ---------------------------------------------------------------------------
# Peak matching per RT chunk
# ---------------------------------------------------------------------------

def match_peaks_per_chunk(
    rt_axis:        np.ndarray,
    mzs:            np.ndarray,
    tensor:         np.ndarray,
    theoretical_mz: np.ndarray,
    *,
    n_chunks:      int   = 6,
    ppm_radius:    float = 100.0,
    min_intensity: float = 1e3,
) -> List[Dict]:
    """For each RT chunk, sum spectra and match peaks to theoretical masses.

    Parameters
    ----------
    rt_axis : float32 (n_scans,)
    mzs     : float32 (n_mz,)
    tensor  : float32 (n_scans, n_mz)
    theoretical_mz : float array of reference masses to match.
    n_chunks : int
        Number of equal-width RT windows to divide the run into.
    ppm_radius : float
        Search radius around each theoretical peak [ppm].
    min_intensity : float
        Minimum Gaussian amplitude to accept a match.

    Returns
    -------
    list of dicts, one per chunk:
        chunk_idx  : int
        rt_lo, rt_hi : float  [min]
        thr_mz     : list[float]   — theoretical m/z of matched peaks
        obs_mz     : list[float]   — Gaussian-fitted observed m/z
        intensities: list[float]   — fitted amplitude per matched peak
    """
    rt_lo_global = float(rt_axis[0])
    rt_hi_global = float(rt_axis[-1])
    chunk_width  = (rt_hi_global - rt_lo_global) / n_chunks

    chunks = []
    for i in range(n_chunks):
        rt_lo = rt_lo_global + i * chunk_width
        rt_hi = rt_lo + chunk_width if i < n_chunks - 1 else rt_hi_global + 1e-9

        mask    = (rt_axis >= rt_lo) & (rt_axis < rt_hi)
        sumspec = tensor[mask].sum(axis=0)

        thr_list, obs_list, int_list = [], [], []
        for mz_thr in theoretical_mz:
            obs_mz, amp = _match_peak(mzs, sumspec, mz_thr, ppm_radius, min_intensity)
            if obs_mz is not None:
                thr_list.append(float(mz_thr))
                obs_list.append(float(obs_mz))
                int_list.append(float(amp))

        chunks.append({
            "chunk_idx":   i,
            "rt_lo":       float(rt_lo),
            "rt_hi":       float(min(rt_hi, rt_hi_global)),
            "thr_mz":      thr_list,
            "obs_mz":      obs_list,
            "intensities": int_list,
        })

        print(
            f"[calibration] chunk {i:2d}  "
            f"RT {rt_lo:.2f}–{rt_hi:.2f} min  "
            f"{len(thr_list)}/{len(theoretical_mz)} peaks matched",
            flush=True,
        )

    return chunks


# ---------------------------------------------------------------------------
# Calibration curve fitting
# ---------------------------------------------------------------------------

def build_calibration(
    matched_chunks: List[Dict],
    *,
    polyfit_deg: int = 1,
) -> List[Dict]:
    """Fit a polynomial calibration curve for each RT chunk.

    The polynomial maps *observed* m/z → *theoretical* m/z:
        thr_mz ≈ polyval(coeffs, obs_mz)

    Parameters
    ----------
    matched_chunks : output of :func:`match_peaks_per_chunk`.
    polyfit_deg : int
        Polynomial degree (1 = linear, default).

    Returns
    -------
    List of calibration dicts, one per chunk:
        chunk_idx, rt_lo, rt_hi,
        n_peaks, polyfit_deg, polyfit_coeffs (list, highest power first),
        thr_mz, obs_mz,
        ppm_error_before, ppm_error_after   (lists, [ppm])
    """
    cal_chunks = []

    for chunk in matched_chunks:
        thr = np.array(chunk["thr_mz"])
        obs = np.array(chunk["obs_mz"])
        n   = len(thr)

        if n < polyfit_deg + 1:
            warnings.warn(
                f"Chunk {chunk['chunk_idx']}: only {n} matched peaks, "
                f"cannot fit degree-{polyfit_deg} polynomial. "
                f"Using identity (no correction)."
            )
            # Identity: polyval([1, 0], x) = x
            coeffs    = [1.0, 0.0]
            obs_corr  = obs.copy()
        else:
            coeffs   = np.polyfit(obs, thr, deg=polyfit_deg).tolist()
            obs_corr = np.polyval(coeffs, obs)

        ppm_before = ((obs      - thr) / thr * 1e6).tolist() if n else []
        ppm_after  = ((obs_corr - thr) / thr * 1e6).tolist() if n else []

        cal_chunks.append({
            "chunk_idx":        chunk["chunk_idx"],
            "rt_lo":            chunk["rt_lo"],
            "rt_hi":            chunk["rt_hi"],
            "n_peaks":          n,
            "polyfit_deg":      polyfit_deg,
            "polyfit_coeffs":   coeffs,
            "thr_mz":           thr.tolist(),
            "obs_mz":           obs.tolist(),
            "ppm_error_before": ppm_before,
            "ppm_error_after":  ppm_after,
        })

        if n:
            print(
                f"[calibration] chunk {chunk['chunk_idx']:2d}  "
                f"median ppm before {np.median(np.abs(ppm_before)):.2f}  "
                f"after  {np.median(np.abs(ppm_after)):.2f}",
                flush=True,
            )

    return cal_chunks


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def save_calibration(
    cal_chunks: List[Dict],
    output_json: str,
    *,
    metadata: Optional[Dict] = None,
) -> None:
    """Write calibration data to *output_json*."""
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "chunks":   cal_chunks,
    }
    with open(output_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[calibration] calibration written → {output_json}", flush=True)


def load_calibration(json_path: str) -> List[Dict]:
    """Load calibration chunks from *json_path*. Returns list of chunk dicts."""
    with open(json_path) as fh:
        payload = json.load(fh)
    return payload["chunks"]


# ---------------------------------------------------------------------------
# Apply calibration
# ---------------------------------------------------------------------------

def apply_calibration(
    mz: float,
    rt: float,
    cal_chunks: List[Dict],
) -> float:
    """Return calibration-corrected m/z for a given (mz, rt) pair.

    Finds the RT chunk whose window contains *rt* (nearest-chunk fallback
    for RT values outside the calibration range) and evaluates its
    polynomial at *mz*.

    Parameters
    ----------
    mz : float
        Measured monoisotopic m/z.
    rt : float
        Retention time [min] of the feature.
    cal_chunks : list
        Output of :func:`load_calibration` or :func:`build_calibration`.

    Returns
    -------
    float — corrected m/z.
    """
    if not cal_chunks:
        return mz

    # Find the chunk whose RT window contains rt
    best_chunk = cal_chunks[0]
    for chunk in cal_chunks:
        if chunk["rt_lo"] <= rt < chunk["rt_hi"]:
            best_chunk = chunk
            break
    else:
        # rt is past the last chunk's upper bound — use the last chunk
        best_chunk = cal_chunks[-1]

    return float(np.polyval(best_chunk["polyfit_coeffs"], mz))


def calibrate_csv(
    input_csv:  str,
    cal_json:   str,
    output_csv: str,
) -> pd.DataFrame:
    """Apply mass calibration to the ``monoisotopic_mz`` column of an aggregate CSV.

    The ``rt_center`` column (midpoint of the slice RT window) is used to
    select the appropriate RT-chunk calibration.  A new column
    ``monoisotopic_mz_raw`` preserves the original uncalibrated values.

    Parameters
    ----------
    input_csv : str
        Path to aggregate CSV written by ``pipeline.py aggregate``.
    cal_json : str
        Path to calibration JSON written by :func:`extract_calibration`.
    output_csv : str
        Path for the corrected output CSV.

    Returns
    -------
    pd.DataFrame — the corrected DataFrame.
    """
    df         = pd.read_csv(input_csv)
    cal_chunks = load_calibration(cal_json)

    if "monoisotopic_mz" not in df.columns:
        raise ValueError("Input CSV has no 'monoisotopic_mz' column.")

    # Preserve original values
    df["monoisotopic_mz_raw"] = df["monoisotopic_mz"].copy()

    # Use rt_center if available, otherwise midpoint of rt_lo/rt_hi
    if "rt_center" in df.columns:
        rt_col = df["rt_center"]
    elif {"rt_lo", "rt_hi"}.issubset(df.columns):
        rt_col = (df["rt_lo"] + df["rt_hi"]) / 2.0
    else:
        warnings.warn("No RT column found; applying calibration with rt=0 for all rows.")
        rt_col = pd.Series(0.0, index=df.index)

    df["monoisotopic_mz"] = [
        apply_calibration(mz, rt, cal_chunks)
        for mz, rt in zip(df["monoisotopic_mz_raw"], rt_col)
    ]

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(
        f"[calibration] calibrated CSV written → {output_csv}  "
        f"({len(df)} rows)",
        flush=True,
    )
    return df


# ---------------------------------------------------------------------------
# Diagnostic PDF
# ---------------------------------------------------------------------------

def plot_calibration(
    rt_axis:        np.ndarray,
    mzs:            np.ndarray,
    tensor:         np.ndarray,
    theoretical_mz: np.ndarray,
    cal_chunks:     List[Dict],
    output_pdf:     str,
    *,
    ppm_radius: float = 100.0,
) -> None:
    """Generate a two-section diagnostic PDF.

    Section 1 — Peak extraction grid
        Rows = theoretical peaks, columns = RT chunks.
        Each panel shows the summed spectrum (orange), the Gaussian fit
        (blue), the theoretical m/z (red dashed), and the fitted centre
        (blue dashed).  Panels with no match are shown in grey with a
        red label.

    Section 2 — Calibration error plots
        One panel per RT chunk showing ppm error before (filled circles)
        and after (×) correction vs m/z, plus the fitted polynomial curve.
    """
    n_chunks    = len(cal_chunks)
    n_peaks_thr = len(theoretical_mz)

    rt_lo_global = float(rt_axis[0])
    rt_hi_global = float(rt_axis[-1])
    chunk_width  = (rt_hi_global - rt_lo_global) / n_chunks

    # ------------------------------------------------------------------ #
    # Section 1: Peak extraction grid                                     #
    # ------------------------------------------------------------------ #
    fig1, axes1 = plt.subplots(
        n_peaks_thr, n_chunks,
        figsize=(max(3 * n_chunks, 10), max(2 * n_peaks_thr, 6)),
        dpi=150,
    )
    # Ensure axes1 is always 2-D
    if n_peaks_thr == 1:
        axes1 = axes1[np.newaxis, :]
    if n_chunks == 1:
        axes1 = axes1[:, np.newaxis]

    for ci, chunk in enumerate(cal_chunks):
        rt_lo = rt_lo_global + ci * chunk_width
        rt_hi = rt_lo + chunk_width if ci < n_chunks - 1 else rt_hi_global + 1e-9
        mask  = (rt_axis >= rt_lo) & (rt_axis < rt_hi)
        sumspec = tensor[mask].sum(axis=0)

        # Build quick lookup: thr_mz → obs_mz for this chunk
        matched = dict(zip(chunk["thr_mz"], chunk["obs_mz"]))

        for ri, mz_thr in enumerate(theoretical_mz):
            ax = axes1[ri, ci]

            mz_lo = mz_thr * (1.0 - ppm_radius * 1e-6)
            mz_hi = mz_thr * (1.0 + ppm_radius * 1e-6)
            win   = (mzs >= mz_lo) & (mzs <= mz_hi)

            if win.any():
                ax.plot(mzs[win], sumspec[win], c="orange", lw=1.0)

                try:
                    H, A, x0, sigma = _gauss_fit(mzs[win], sumspec[win])
                    xs_fit = np.linspace(mzs[win][0], mzs[win][-1], 80)
                    ax.plot(xs_fit, _gaussian(xs_fit, H, A, x0, sigma),
                            c="steelblue", lw=1.2)
                    ppm_err = (x0 - mz_thr) * 1e6 / mz_thr
                    ax.text(0.97, 0.88,
                            f"{ppm_err:+.1f} ppm",
                            ha="right", va="top",
                            transform=ax.transAxes,
                            fontsize=6,
                            color="steelblue" if abs(ppm_err) < ppm_radius else "red")
                except Exception:
                    x0 = mz_thr

                obs_mz = matched.get(mz_thr)
                ax.axvline(mz_thr, ls="--", lw=0.8, c="red",  alpha=0.7)
                if obs_mz is not None:
                    ax.axvline(obs_mz, ls="--", lw=0.8, c="steelblue", alpha=0.7)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=6, color="grey")

            ax.set_yticks([])
            ax.set_xticks([mz_thr])
            ax.tick_params(axis="x", labelsize=5)

            # Column header (RT range) on top row
            if ri == 0:
                ax.set_title(
                    f"{chunk['rt_lo']:.1f}–{chunk['rt_hi']:.1f} min",
                    fontsize=7,
                )
            # Row label (theoretical m/z) on leftmost column
            if ci == 0:
                ax.set_ylabel(f"{mz_thr:.2f}", fontsize=5, rotation=0,
                              labelpad=28, va="center")

    fig1.suptitle("Lock-mass peak extraction", fontsize=10, y=1.002)
    fig1.tight_layout()

    # ------------------------------------------------------------------ #
    # Section 2: ppm error before / after correction                     #
    # ------------------------------------------------------------------ #
    fig2, axes2 = plt.subplots(
        1, n_chunks,
        figsize=(max(4 * n_chunks, 8), 4),
        dpi=150,
        sharey=True,
    )
    if n_chunks == 1:
        axes2 = [axes2]

    mz_curve = np.linspace(
        float(theoretical_mz.min()) * 0.9,
        float(theoretical_mz.max()) * 1.1,
        500,
    )

    for ci, chunk in enumerate(cal_chunks):
        ax  = axes2[ci]
        thr = np.array(chunk["thr_mz"])
        ppm_b = np.array(chunk["ppm_error_before"])
        ppm_a = np.array(chunk["ppm_error_after"])

        if thr.size:
            ax.scatter(thr, ppm_b, s=18, label="before", zorder=3,
                       color="tomato", alpha=0.85)
            ax.scatter(thr, ppm_a, s=18, marker="x", label="after", zorder=3,
                       color="steelblue", alpha=0.85)

            # Draw the polynomial correction as a ppm-error curve
            coeffs    = chunk["polyfit_coeffs"]
            corr_curve = np.polyval(coeffs, mz_curve)
            ppm_curve  = (corr_curve - mz_curve) / mz_curve * 1e6
            ax.plot(mz_curve, ppm_curve, "--", lw=1.0, color="steelblue", alpha=0.6)

            med_b = float(np.median(np.abs(ppm_b)))
            med_a = float(np.median(np.abs(ppm_a)))
            ax.text(0.03, 0.96,
                    f"before: {med_b:.2f} ppm\nafter:  {med_a:.2f} ppm",
                    transform=ax.transAxes, fontsize=7, va="top",
                    color="black")

        ax.axhline(0, ls=":", lw=0.8, color="grey")
        ax.set_ylim(-ppm_radius * 1.1, ppm_radius * 1.1)
        ax.set_xlabel("m/z", fontsize=8)
        ax.set_title(
            f"{chunk['rt_lo']:.1f}–{chunk['rt_hi']:.1f} min\n"
            f"{chunk['n_peaks']} peaks",
            fontsize=8,
        )
        if ci == 0:
            ax.set_ylabel("ppm error", fontsize=8)
        ax.legend(fontsize=6)

    fig2.suptitle("Calibration ppm error before / after correction", fontsize=10)
    fig2.tight_layout()

    # ------------------------------------------------------------------ #
    # Save both sections to one PDF                                       #
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf)), exist_ok=True)
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig1, bbox_inches="tight")
        pdf.savefig(fig2, bbox_inches="tight")
    plt.close("all")
    print(f"[calibration] diagnostic PDF written → {output_pdf}", flush=True)


# ---------------------------------------------------------------------------
# Main orchestrating function
# ---------------------------------------------------------------------------

def extract_calibration(
    raw_path:      str,
    license_path:  str,
    *,
    compound:      str   = "SodiumFormate",
    n_chunks:      int   = 6,
    polyfit_deg:   int   = 1,
    ppm_radius:    float = 100.0,
    min_intensity: float = 1e3,
    mz_min:        float = 50.0,
    mz_max:        float = 2200.0,
    output_json:   Optional[str] = None,
    output_pdf:    Optional[str] = None,
) -> List[Dict]:
    """Full calibration pipeline for one raw file.

    Steps
    -----
    1. Read lock-mass function → sparse (rt, mz, tensor).
    2. For each of *n_chunks* equal RT windows, sum spectra and match
       Gaussian-fitted peaks to theoretical masses.
    3. Fit a degree-*polyfit_deg* polynomial (obs → thr) per chunk.
    4. Optionally write JSON and diagnostic PDF.

    Returns
    -------
    list of calibration chunk dicts (same as :func:`build_calibration`).
    """
    theoretical_mz = get_theoretical_mz(compound, mz_min=mz_min, mz_max=mz_max)
    print(
        f"[calibration] {Path(raw_path).stem}  "
        f"compound={compound}  "
        f"n_chunks={n_chunks}  "
        f"polyfit_deg={polyfit_deg}  "
        f"theoretical peaks in range: {len(theoretical_mz)}",
        flush=True,
    )

    rt_axis, mzs, tensor, lockmass_fn = extract_lockmass_tensor(
        raw_path, license_path,
        compound=compound,
        ppm_radius=ppm_radius,
        mz_min=mz_min,
        mz_max=mz_max,
    )

    matched_chunks = match_peaks_per_chunk(
        rt_axis, mzs, tensor, theoretical_mz,
        n_chunks=n_chunks,
        ppm_radius=ppm_radius,
        min_intensity=min_intensity,
    )

    cal_chunks = build_calibration(matched_chunks, polyfit_deg=polyfit_deg)

    if output_json is not None:
        save_calibration(
            cal_chunks,
            output_json,
            metadata={
                "sample":       Path(raw_path).stem,
                "compound":     compound,
                "lockmass_fn":  lockmass_fn,
                "n_chunks":     n_chunks,
                "polyfit_deg":  polyfit_deg,
                "ppm_radius":   ppm_radius,
                "min_intensity": min_intensity,
                "mz_min":       mz_min,
                "mz_max":       mz_max,
            },
        )

    if output_pdf is not None:
        plot_calibration(
            rt_axis, mzs, tensor, theoretical_mz, cal_chunks,
            output_pdf,
            ppm_radius=ppm_radius,
        )

    return cal_chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HDX-MS lock-mass calibration — extract or apply"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- extract ----
    p_ext = sub.add_parser(
        "extract",
        help="Extract calibration from the lock-mass function of a .raw file",
    )
    p_ext.add_argument("raw_path",     help="Path to Waters .raw directory")
    p_ext.add_argument("license_path", help="Path to MassLynx SDK license key")
    p_ext.add_argument("--compound",      default="SodiumFormate",
                       help="Reference compound (default: SodiumFormate)")
    p_ext.add_argument("--n_chunks",      type=int,   default=6,
                       help="Number of equal RT windows (default: 6)")
    p_ext.add_argument("--polyfit_deg",   type=int,   default=1,
                       help="Polynomial degree for calibration curve (default: 1)")
    p_ext.add_argument("--ppm_radius",    type=float, default=100.0,
                       help="Peak search radius [ppm] (default: 100)")
    p_ext.add_argument("--min_intensity", type=float, default=1e3,
                       help="Minimum peak amplitude to accept match (default: 1e3)")
    p_ext.add_argument("--mz_min",        type=float, default=50.0,
                       help="Lower m/z bound for theoretical masses (default: 50)")
    p_ext.add_argument("--mz_max",        type=float, default=2200.0,
                       help="Upper m/z bound for theoretical masses (default: 2200)")
    p_ext.add_argument("--output_json",   required=True,
                       help="Output JSON calibration file")
    p_ext.add_argument("--output_pdf",    default=None,
                       help="Optional diagnostic PDF path")

    # ---- apply_csv ----
    p_apl = sub.add_parser(
        "apply_csv",
        help="Apply calibration to monoisotopic_mz in an aggregate CSV",
    )
    p_apl.add_argument("--input_csv",  required=True,
                       help="Aggregate CSV from pipeline.py aggregate")
    p_apl.add_argument("--cal_json",   required=True,
                       help="Calibration JSON from 'extract'")
    p_apl.add_argument("--output_csv", required=True,
                       help="Output corrected CSV path")

    args = parser.parse_args()

    if args.cmd == "extract":
        extract_calibration(
            raw_path=args.raw_path,
            license_path=args.license_path,
            compound=args.compound,
            n_chunks=args.n_chunks,
            polyfit_deg=args.polyfit_deg,
            ppm_radius=args.ppm_radius,
            min_intensity=args.min_intensity,
            mz_min=args.mz_min,
            mz_max=args.mz_max,
            output_json=args.output_json,
            output_pdf=args.output_pdf,
        )

    elif args.cmd == "apply_csv":
        calibrate_csv(
            input_csv=args.input_csv,
            cal_json=args.cal_json,
            output_csv=args.output_csv,
        )

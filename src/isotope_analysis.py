"""
isotope_analysis.py
===================
Isotopic cluster detection, charge-state assignment, and monoisotopic mass
inference for intact-protein HDX-MS data.

The module operates on the output of ``tensor_analysis.factorize`` /
``tensor_analysis.filter_factors``: for each NTF factor the m/z profile
(C[:, r]) and the RT / DT profiles (A[:, r], B[:, r]) are used to detect
all isotopic envelopes present in that factor, assign charge states, and
infer the monoisotopic mass via the averagine model.

Averagine model notes (Senko 1995)
----------------------------------
The averagine composition per 111.1028 Da unit is
    C₄.₉₃₈₄ H₇.₇₅₈₃ N₁.₃₅₇₇ O₁.₄₇₇₃ S₀.₀₄₁₇

The isotope envelope is computed by convolving the exact binomial
distributions for each element.  Carbon dominates (λ_C = 0.0107 × n_C).
At 4 kDa the monoisotopic peak has ~14 % relative intensity; at 8 kDa ~2 %;
at 15 kDa ~0.06 %.  The model shape is accurate across this range — the
monoisotopic peak simply becomes unobservable and must be inferred from the
best-fit alignment of the visible envelope.

Typical peak counts above 1 % relative intensity:
    4  kDa  ≈  8–12 peaks
    8  kDa  ≈ 14–18 peaks
    15 kDa  ≈ 20–25 peaks

Workflow
--------
1. ``averagine_envelope``   — isotope distribution for a given neutral mass
2. ``find_isotopic_clusters`` — for ONE factor (A[:,r], B[:,r], C[:,r]):
      * detect peak groups in the m/z profile
      * for each group try every charge state in ``charge_range``
      * score alignments by cosine similarity
      * infer the monoisotopic m/z
3. ``process_all_factors``  — loop over all factors → pandas DataFrame
4. ``plot_isotopic_cluster`` — 3×1 PNG: RT profile | DT profile | m/z slice
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # safe for headless / HPC use
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as _scipy_find_peaks, savgol_filter as _savgol_filter
from scipy.stats import binom as _binom
from scipy.interpolate import interp1d as _interp1d
from scipy.optimize import curve_fit as _curve_fit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Averagine composition (Senko 1995) per 111.1028 Da residue unit
_AVERAGINE_MASS: float = 111.1028
_AVERAGINE_COMP: Dict[str, float] = {
    "C": 4.9384,
    "H": 7.7583,
    "N": 1.3577,
    "O": 1.4773,
    "S": 0.0417,
}

# Natural isotope abundances used for the envelope calculation
# Each entry is (light_isotope_fraction, heavy_isotope_fraction, ...)
# Only the *heavy* fraction(s) drive peak spacing in the envelope.
_ISO_PROBS: Dict[str, Tuple[float, ...]] = {
    "C": (0.9893, 0.0107),                       # 12C, 13C
    "H": (0.999885, 0.000115),                   # 1H, 2H  (tiny — kept for accuracy)
    "N": (0.99632, 0.00368),                     # 14N, 15N
    "O": (0.99757, 0.00038, 0.00205),            # 16O, 17O, 18O
    "S": (0.9499, 0.0075, 0.0425, 0.0, 0.0001), # 32S, 33S, 34S, 35S, 36S
}

NEUTRON: float = 1.003355    # 13C – 12C mass difference [Da]
PROTON:  float = 1.007276    # proton mass [Da]


# ===========================================================================
# Section 1 — Averagine isotope envelope
# ===========================================================================

def _element_dist(n_atoms: float, probs: Tuple[float, ...], max_peaks: int) -> np.ndarray:
    """Exact isotope probability distribution for *n_atoms* of one element.

    Uses the exact binomial (for 2-isotope elements) or sequential
    multinomial convolution (for 3+ isotopes), truncated to *max_peaks*.

    Parameters
    ----------
    n_atoms : float
        (Fractional) number of atoms; rounded to nearest integer.
    probs : tuple of float
        Isotope abundances in order of increasing mass.
    max_peaks : int
        Truncation length of the output vector.

    Returns
    -------
    dist : float64 (max_peaks,)
        Probability of each nominal isotope offset (0, 1, 2, …).
    """
    n = max(1, round(n_atoms))
    dist = np.zeros(max_peaks, dtype=np.float64)

    if len(probs) == 2:
        # Exact binomial
        k = np.arange(min(n + 1, max_peaks))
        vals = _binom.pmf(k, n, probs[1])
        dist[:len(vals)] = vals
    else:
        # Build single-atom distribution then convolve n times (use log trick
        # for large n: convolve the distribution with itself via FFT power).
        single = np.array(probs[:max_peaks], dtype=np.float64)
        single = single[:max_peaks]
        single /= single.sum()

        # Fast exponentiation via repeated squaring in probability space
        result = np.zeros(max_peaks); result[0] = 1.0
        base = np.zeros(max_peaks); base[:len(single)] = single
        exp = n
        while exp > 0:
            if exp % 2 == 1:
                result = np.convolve(result, base)[:max_peaks]
            base = np.convolve(base, base)[:max_peaks]
            exp //= 2
        dist = result

    s = dist.sum()
    return dist / s if s > 0 else dist


def averagine_envelope(
    mass_da: float,
    min_relative: float = 0.01,
    max_peaks: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the isotope envelope for a neutral monoisotopic mass.

    Parameters
    ----------
    mass_da :
        Neutral monoisotopic mass [Da].
    min_relative :
        Only return peaks with relative intensity ≥ this fraction (default 0.01 = 1 %).
    max_peaks :
        Maximum number of isotope peaks to compute.

    Returns
    -------
    offsets : float64 (n_peaks,)
        Mass offset from monoisotopic peak [Da]:  0, NEUTRON, 2*NEUTRON, …
        for each retained peak.
    rel_intensities : float64 (n_peaks,)
        Relative intensities (most abundant peak = 1.0).

    Notes
    -----
    For proteins ≥ 8 kDa the monoisotopic offset=0 peak will have
    relative intensity well below *min_relative* and will be absent from
    the returned arrays.  The monoisotopic mass is still inferred during
    charge assignment by fitting the visible envelope and extrapolating.
    """
    n_units = mass_da / _AVERAGINE_MASS

    # Combined isotope distribution (convolve all elements)
    dist = np.zeros(max_peaks, dtype=np.float64)
    dist[0] = 1.0
    for element, per_unit in _AVERAGINE_COMP.items():
        n_atoms = per_unit * n_units
        if n_atoms < 0.5:
            continue
        edist = _element_dist(n_atoms, _ISO_PROBS[element], max_peaks)
        dist = np.convolve(dist, edist)[:max_peaks]

    dist = np.maximum(dist, 0.0)
    if dist.sum() > 0:
        dist /= dist.sum()

    # Trim to min_relative threshold (relative to maximum)
    peak = dist.max()
    if peak == 0:
        return np.array([0.0]), np.array([1.0])
    rel = dist / peak
    keep = rel >= min_relative
    indices = np.where(keep)[0]

    offsets = indices.astype(np.float64) * NEUTRON
    return offsets, rel[indices]


# ===========================================================================
# Section 2 — Peak detection and grouping
# ===========================================================================

def _find_profile_peaks(
    profile: np.ndarray,
    mz_axis: np.ndarray,
    min_height_frac: float = 0.02,
    min_distance_da: float = 0.04,
) -> np.ndarray:
    """Return indices of local maxima in *profile* above a noise floor.

    Uses physical-space non-maximum suppression so that the minimum
    separation is expressed in Daltons rather than array bins.  This is
    essential for masked m/z axes where bin spacing is highly non-uniform:
    the sub-peaks within each isotope peak (typically ~0.012 Da apart) are
    suppressed, while genuine isotope peaks (≥NEUTRON/z_max ≈ 0.067 Da for
    z_max=15) are preserved.

    Parameters
    ----------
    profile :
        1-D intensity array (m/z dimension of a factor's C vector).
    mz_axis :
        Corresponding m/z positions [Da].  Must have the same length as
        *profile*.
    min_height_frac :
        Minimum peak height as a fraction of the profile maximum.
    min_distance_da :
        Minimum physical separation between accepted peaks [Da].
        Default 0.04 Da — just below NEUTRON/15 ≈ 0.067 Da, so it
        suppresses intra-peak noise without merging adjacent isotope peaks.

    Returns
    -------
    peak_indices : int array, sorted ascending
    """
    if profile.max() <= 0:
        return np.array([], dtype=int)
    threshold = profile.max() * min_height_frac
    # Find all local maxima (no distance constraint yet)
    all_peaks, _ = _scipy_find_peaks(profile, height=threshold)
    if len(all_peaks) == 0:
        return all_peaks
    # Non-maximum suppression in physical m/z space:
    # Process peaks from highest to lowest intensity; accept a peak only if
    # it is at least min_distance_da away from every already-accepted peak.
    intensities = profile[all_peaks]
    order = np.argsort(intensities)[::-1]   # descending intensity
    kept: List[int] = []
    kept_mz: List[float] = []
    for idx in order:
        p = int(all_peaks[idx])
        mz_p = float(mz_axis[p])
        if all(abs(mz_p - m) >= min_distance_da for m in kept_mz):
            kept.append(p)
            kept_mz.append(mz_p)
    return np.sort(np.array(kept, dtype=int))


def _group_peaks(
    peak_indices: np.ndarray,
    mz_axis: np.ndarray,
    max_gap_da: float = 3.0,
) -> List[np.ndarray]:
    """Split *peak_indices* into groups separated by gaps > *max_gap_da*.

    Returns
    -------
    groups : list of int arrays
        Each array contains the peak indices for one candidate cluster.
    """
    if len(peak_indices) == 0:
        return []
    groups: List[np.ndarray] = []
    current = [peak_indices[0]]
    for idx in peak_indices[1:]:
        gap = mz_axis[idx] - mz_axis[current[-1]]
        if gap > max_gap_da:
            if len(current) >= 2:
                groups.append(np.array(current))
            current = [idx]
        else:
            current.append(idx)
    if len(current) >= 2:
        groups.append(np.array(current))
    return groups


# ===========================================================================
# Section 3 — Charge assignment and cosine fit
# ===========================================================================

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two non-negative vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _extract_intensities_at_mz(
    mz_targets: np.ndarray,
    mz_axis: np.ndarray,
    profile: np.ndarray,
    tol_da: float = 0.01,
) -> np.ndarray:
    """Extract profile intensities at a list of target m/z values.

    For each target, sums profile bins within ±*tol_da*.

    Parameters
    ----------
    mz_targets : (n,) — target m/z positions [Da]
    mz_axis    : (K,) — m/z axis of *profile*
    profile    : (K,) — intensity array
    tol_da     : search window radius [Da]

    Returns
    -------
    intensities : (n,) float64
    """
    # Vectorised: broadcast (n_targets, K) boolean mask, sum in one call.
    within = np.abs(mz_axis[None, :] - mz_targets[:, None]) <= tol_da  # (n, K)
    return (profile[None, :] * within).sum(axis=1).astype(np.float64)


def _smooth_mz_profile(
    profile: np.ndarray,
    mz_axis: np.ndarray,
    window_bins: int = 3,
    polyorder: int = 2,
    step_da: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Savitzky-Golay smoothing on the original masked axis, then interpolate.

    Savitzky-Golay is used instead of Gaussian because it preserves peak
    positions and heights much better: the polynomial fit within each window
    suppresses high-frequency noise (intra-peak sub-structure at ~0.012 Da)
    while leaving the true isotope peak centroids essentially unchanged.

    Parameters
    ----------
    profile     : (K,) — raw intensity profile on the masked m/z axis
    mz_axis     : (K,) — corresponding m/z values [Da]
    window_bins : half-width of the SG window in **original-axis bins**;
                  full window = 2*window_bins + 1 = 7 for the default of 3.
                  With typical masked-axis spacing ~0.012–0.025 Da this covers
                  ~0.05–0.18 Da, suppressing sub-peak noise without broadening
                  the ~0.14–0.33 Da isotope peak spacing.
    polyorder   : SG polynomial order (must be < window_length); default 2
    step_da     : fine-grid step size for interpolation [Da]; default 0.001 Da

    Returns
    -------
    mz_fine             : (N,) uniform m/z grid spanning [mz_axis[0], mz_axis[-1]]
    profile_fine_raw    : (N,) linearly interpolated raw profile
    profile_fine_smooth : (N,) SG-smoothed profile interpolated to fine grid
    """
    window_length = 2 * window_bins + 1          # 7 by default
    # SG requires window_length > polyorder and window_length <= len(data)
    if window_length <= polyorder:
        window_length = polyorder + (1 if polyorder % 2 == 0 else 2)
    # Clamp to data length; must remain odd and > polyorder
    n_pts = len(profile)
    if window_length > n_pts:
        window_length = n_pts if n_pts % 2 == 1 else max(1, n_pts - 1)
        if window_length % 2 == 0:
            window_length -= 1

    # Apply SG on the original (sparse) axis — this preserves peak shapes well.
    # If data is too short for SG (fewer points than polyorder+1), fall back to raw.
    if window_length >= polyorder + 1 and n_pts >= polyorder + 1:
        profile_sg = _savgol_filter(
            profile.astype(np.float64), window_length=window_length, polyorder=polyorder
        )
        profile_sg = np.maximum(profile_sg, 0.0)
    else:
        profile_sg = profile.astype(np.float64).copy()

    # Interpolate both raw and smoothed to a uniform fine grid for the scan
    mz_lo   = float(mz_axis[0])
    mz_hi   = float(mz_axis[-1])
    mz_fine = np.arange(mz_lo, mz_hi + step_da * 0.5, step_da)

    fn_raw = _interp1d(mz_axis, profile, kind="linear", bounds_error=False, fill_value=0.0)
    # Cubic spline for the SG-smoothed profile: this gives a continuously
    # differentiable fine-grid representation so that the intensity-weighted
    # centroid computed in _refine_mono_mz lands at the true peak centre
    # rather than being biased by the piecewise-linear kinks of a linear
    # interpolant.  Falls back to linear when there are fewer than 4 points.
    kind_sg = "cubic" if len(mz_axis) >= 4 else "linear"
    fn_sg  = _interp1d(mz_axis, profile_sg, kind=kind_sg, bounds_error=False, fill_value=0.0)

    profile_fine_raw    = np.maximum(fn_raw(mz_fine).astype(np.float64), 0.0)
    profile_fine_smooth = np.maximum(fn_sg(mz_fine).astype(np.float64),  0.0)

    return mz_fine, profile_fine_raw, profile_fine_smooth


def _vectorised_stick_scan(
    mz_fine: np.ndarray,
    profile_fine_smooth: np.ndarray,
    env_offsets: np.ndarray,
    env_rel: np.ndarray,
    z: int,
    mono_mz_lo: float,
    mono_mz_hi: float,
    scan_step_da: float = 0.001,
    tol_da: float = 0.01,
) -> Tuple[float, float, np.ndarray]:
    """Vectorised sliding-window cosine scan for charge-state alignment.

    Slides the theoretical averagine stick pattern (env_offsets/z relative to
    mono_mz) across [mono_mz_lo, mono_mz_hi] and returns the alignment that
    maximises cosine similarity with the Gaussian-smoothed profile.

    The cumulative-sum trick makes extraction of windowed integrals O(1) per
    position, so the full scan over thousands of positions runs in < 1 ms.

    Parameters
    ----------
    mz_fine : (N,) uniform fine m/z grid [Da]
    profile_fine_smooth : (N,) Gaussian-smoothed profile on fine grid
    env_offsets : (n_env,) averagine offsets from monoisotopic peak [Da]
    env_rel     : (n_env,) averagine relative intensities (max = 1.0)
    z           : charge state
    mono_mz_lo, mono_mz_hi : scan range for monoisotopic m/z [Da]
    scan_step_da : scan step size [Da]; default 0.005 Da
    tol_da       : half-width of integration window per stick [Da]

    Returns
    -------
    best_mono_mz : float — monoisotopic m/z at best cosine
    best_cosine  : float — cosine similarity at best alignment
    obs_at_best  : (n_env,) — observed windowed intensities at best alignment
    """
    step = float(mz_fine[1] - mz_fine[0])          # fine-grid step (Da)
    tol_bins = max(1, round(tol_da / step))
    N = len(mz_fine)
    mz_lo_fine = float(mz_fine[0])

    # Cumulative sum for O(1) windowed integrals
    cs = np.empty(N + 1, dtype=np.float64)
    cs[0] = 0.0
    cs[1:] = np.cumsum(profile_fine_smooth)

    # Scan positions
    mono_scan = np.arange(mono_mz_lo, mono_mz_hi + scan_step_da * 0.5, scan_step_da)
    if len(mono_scan) == 0:
        return float(mono_mz_lo), 0.0, np.zeros(len(env_offsets))

    n_scan = len(mono_scan)
    n_env  = len(env_offsets)

    # Stick m/z positions for every scan point: (n_scan, n_env)
    stick_mz = mono_scan[:, None] + (env_offsets / z)[None, :]

    # Convert to fine-grid indices (round to nearest)
    stick_idx = np.round((stick_mz - mz_lo_fine) / step).astype(np.int64)

    # Windowed integral via cumsum
    lo_idx = np.clip(stick_idx - tol_bins,     0, N).astype(np.int64)
    hi_idx = np.clip(stick_idx + tol_bins + 1, 0, N).astype(np.int64)
    obs_matrix = cs[hi_idx] - cs[lo_idx]          # (n_scan, n_env)

    # Cosine similarity: normalise each row by its own max, compare to env_rel
    row_max = obs_matrix.max(axis=1, keepdims=True)
    row_max = np.maximum(row_max, 1e-12)
    obs_norm = obs_matrix / row_max                # (n_scan, n_env) in [0,1]

    pred = env_rel.astype(np.float64)              # already max-normalised
    dots  = obs_norm @ pred                        # (n_scan,)
    obs_norms = np.linalg.norm(obs_norm, axis=1)   # (n_scan,)
    pred_norm_val = float(np.linalg.norm(pred))

    denom = obs_norms * pred_norm_val
    with np.errstate(invalid="ignore", divide="ignore"):
        cosines = np.where(denom > 1e-12, dots / denom, 0.0)

    best_i       = int(np.argmax(cosines))
    best_cosine  = float(cosines[best_i])
    best_mono_mz = float(mono_scan[best_i])
    obs_at_best  = obs_matrix[best_i]              # unnormalised windowed ints

    return best_mono_mz, best_cosine, obs_at_best


def _refine_mono_mz(
    best_mono_mz: float,
    env_offsets: np.ndarray,
    env_rel: np.ndarray,
    z: int,
    mz_fine: np.ndarray,
    profile_fine_smooth: np.ndarray,
    tol_da: float = 0.020,
    min_peak_fraction: float = 0.05,
) -> float:
    """Refine monoisotopic m/z by intensity-weighted centroid alignment.

    The coarse sliding scan places the monoisotopic position on a discrete
    grid (scan_step_da = 0.001 Da), so it can be off by up to ±0.0005 Da
    from the true peak centroid.  This function corrects that by, for each
    theoretical stick position, computing the intensity-weighted centroid of
    the SG-smoothed profile inside the integration window and measuring its
    offset from the theoretical position.  The weighted-mean offset across
    all sufficiently-bright sticks is then applied as a rigid shift to
    ``best_mono_mz``.

    Parameters
    ----------
    best_mono_mz :
        Initial monoisotopic m/z from the sliding scan.
    env_offsets :
        Averagine mass offsets [Da] from the monoisotopic peak.
    env_rel :
        Averagine relative intensities (max-normalised).
    z :
        Charge state.
    mz_fine :
        Uniform fine m/z grid (0.001 Da spacing).
    profile_fine_smooth :
        SG-smoothed intensity profile on *mz_fine*.
    tol_da :
        Half-width of the integration window [Da].  Should match the
        value used in the scan so the same region is sampled.
    min_peak_fraction :
        Stick is ignored for centroid computation if the windowed integral
        is below this fraction of the profile maximum.  Prevents noisy or
        absent peaks from biasing the centroid.

    Returns
    -------
    refined_mono_mz : float
        Monoisotopic m/z after centroid correction.
    """
    step   = float(mz_fine[1] - mz_fine[0])
    mz_lo  = float(mz_fine[0])
    N      = len(mz_fine)
    prof_max = float(profile_fine_smooth.max()) + 1e-12
    threshold = min_peak_fraction * prof_max

    deltas  = []
    weights = []

    for off, rel in zip(env_offsets, env_rel):
        stick_mz = best_mono_mz + float(off) / z
        lo_idx = max(0, int(round((stick_mz - tol_da - mz_lo) / step)))
        hi_idx = min(N - 1, int(round((stick_mz + tol_da - mz_lo) / step)))
        if hi_idx <= lo_idx:
            continue
        window_int = profile_fine_smooth[lo_idx : hi_idx + 1]
        total = float(window_int.sum())
        if total < threshold:
            continue
        window_mz = mz_fine[lo_idx : hi_idx + 1]
        centroid  = float(np.dot(window_mz, window_int) / total)
        deltas.append(centroid - stick_mz)
        # Weight by the observed integral (bright peaks anchor the shift more)
        weights.append(total)

    if not deltas:
        return best_mono_mz

    deltas  = np.array(deltas,  dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    shift   = float(np.dot(deltas, weights) / weights.sum())
    return best_mono_mz + shift


def _integrate_isotope_peaks(
    cs_smooth, fine_step, fine_lo, fine_N,
    mono_mz, charge, n_peaks, ppm=10.0,
):
    """Integrate smoothed m/z profile at each expected isotope position (±ppm window).

    Returns a normalized array of length n_peaks. For absent peaks (e.g. M+0 in
    k=1 assignments) the integration yields ~0 from background noise.
    """
    spacing = NEUTRON / charge
    result = np.zeros(n_peaks, dtype=np.float64)
    for i in range(n_peaks):
        mz_i = mono_mz + i * spacing
        half_da = mz_i * ppm * 1e-6
        half_bins = max(1, round(half_da / fine_step))
        idx = int(round((mz_i - fine_lo) / fine_step))
        lo = max(0, idx - half_bins)
        hi = min(fine_N, idx + half_bins + 1)
        result[i] = cs_smooth[hi] - cs_smooth[lo]
    mx = result.max()
    if mx > 0:
        result /= mx
    return result


def _assign_charge(
    group_indices: np.ndarray,
    mz_axis: np.ndarray,
    profile: np.ndarray,
    mz_fine: np.ndarray,
    profile_fine_raw: np.ndarray,
    profile_fine_smooth: np.ndarray,
    charge_range: Tuple[int, int] = (3, 15),
    min_cosine: float = 0.5,
    max_isotope_offset: int = 25,
    scan_step_da: float = 0.001,
    factor_idx: int = 0,
    group_idx: int = 0,
    ambiguity_threshold: float = 0.15,
    verbose: bool = False,
) -> Optional[Dict]:
    """Assign charge state to one peak group via sliding stick scan.

    The cosine scan runs on the **SG-smoothed** fine-grid profile.  SG ±3 bins
    preserves individual isotope peak positions while suppressing sub-peak noise.

    Integration window per stick is charge-adaptive:
        tol_da_z = 0.25 × NEUTRON / z
    This is 25 % of the isotope peak spacing, tight enough that adjacent isotope
    peaks never bleed into the integration window even at high charge states.

    A **gap penalty** is then applied to each candidate charge state:
        adjusted_score = cosine × exp(−2 × gap_fraction)
        gap_fraction   = Σ(smoothed intensity at inter-stick midpoints)
                         / (Σ(smoothed intensity at stick positions) + ε)
    For the correct charge, inter-stick gaps are genuinely empty.  For a z/2
    alias (e.g. z=5 when truth is z=10), every other z=10 peak falls exactly at
    the midpoint of a z=5 stick pair, so gap_fraction ≈ 0.5–1 and the alias is
    heavily penalised.  The gap window is 0.20 × spacing (narrower than the
    stick window) so it samples the midpoint without contaminating sticks.

    Parameters
    ----------
    group_indices :
        Indices into *mz_axis* / *profile* for the candidate cluster peaks.
    mz_axis :
        Masked m/z axis [Da].
    profile :
        m/z intensity profile (C[:,r] × A_sum × B_sum).
    mz_fine :
        Uniform fine m/z grid.
    profile_fine_raw :
        Linearly interpolated (unsmoothed) profile on *mz_fine*.
        **Used for the cosine scan.**
    profile_fine_smooth :
        SG-smoothed profile on *mz_fine*.  Stored in the result for plotting.
    charge_range :
        (min_z, max_z) inclusive.
    min_cosine :
        Minimum cosine similarity to accept a match.
    max_isotope_offset :
        Maximum isotope index offset the first visible peak can correspond to.
    scan_step_da :
        Step size for the mono_mz scan [Da].
    factor_idx, group_idx :
        Used in printed warnings.
    ambiguity_threshold :
        Fraction below best cosine within which a half/double alias triggers
        a printed warning.

    Returns
    -------
    result : dict or None
        Keys: charge, monoisotopic_mz, monoisotopic_mass_da,
              cosine_similarity, n_isotope_peaks_observed, rough_mass_da,
              averagine_offsets, averagine_rel,
              observed_mz, observed_int, cluster_intensity,
              mz_fine, profile_fine_smooth.
    """
    if len(group_indices) < 2:
        return None

    observed_mz  = mz_axis[group_indices]
    observed_int = profile[group_indices].astype(np.float64)
    obs_int_norm = observed_int / (observed_int.max() + 1e-12)

    group_mz_min = float(observed_mz[0])
    group_mz_max = float(observed_mz[-1])
    apex_mz      = float(mz_axis[group_indices[np.argmax(observed_int)]])

    # Pre-compute cumulative sum of the smoothed profile for fast gap extraction.
    _fine_step = float(mz_fine[1] - mz_fine[0])
    _fine_N    = len(mz_fine)
    _fine_lo   = float(mz_fine[0])
    _cs_smooth = np.empty(_fine_N + 1, dtype=np.float64)
    _cs_smooth[0] = 0.0
    _cs_smooth[1:] = np.cumsum(profile_fine_smooth)

    def _windowed_sum(centers_mz: np.ndarray, half_da: float) -> np.ndarray:
        """Windowed integral from *profile_fine_smooth* at each center m/z."""
        half_bins = max(1, round(half_da / _fine_step))
        idx = np.round((centers_mz - _fine_lo) / _fine_step).astype(np.int64)
        lo  = np.clip(idx - half_bins,     0, _fine_N)
        hi  = np.clip(idx + half_bins + 1, 0, _fine_N)
        return _cs_smooth[hi] - _cs_smooth[lo]

    candidates: List[Tuple[float, int, Dict]] = []

    for z in range(charge_range[0], charge_range[1] + 1):
        spacing    = NEUTRON / z
        rough_mass = apex_mz * z - z * PROTON
        if rough_mass <= 0:
            continue

        env_offsets, env_rel = averagine_envelope(
            rough_mass, min_relative=0.005, max_peaks=60
        )

        # Fixed integration half-window: 0.020 Da for all charge states.
        # This is wide enough to integrate the full instrument peak (typical
        # FWHM ~0.04–0.08 Da) while remaining well inside the valley between
        # any two adjacent isotope peaks across the entire charge range:
        #   z= 3: nearest neighbour at 0.167 Da  (0.020 << 0.167) ✓
        #   z= 5: nearest neighbour at 0.100 Da  (0.020 << 0.100) ✓
        #   z=10: nearest neighbour at 0.050 Da  (0.020 <  0.050) ✓
        # Using a fixed window also removes any charge-state dependence from
        # the intensity integration, keeping comparisons across z fair.
        tol_da_z = 0.020

        mono_mz_lo = group_mz_min - max_isotope_offset * spacing
        mono_mz_hi = group_mz_max

        # Scan on the SG-smoothed fine-grid profile.
        # SG ±3 bins preserves individual isotope peak positions while
        # suppressing sub-peak noise that could confuse the cosine score.
        best_mono_mz, best_cos, obs_at_best = _vectorised_stick_scan(
            mz_fine, profile_fine_smooth,
            env_offsets, env_rel, z,
            mono_mz_lo, mono_mz_hi,
            scan_step_da=scan_step_da,
            tol_da=tol_da_z,
        )

        if best_cos < min_cosine:
            continue

        # Centroid refinement: shift mono_mz so each stick is centred on
        # the true peak rather than the nearest scan-grid position.
        # Threshold relative to the cluster's own brightest stick window
        # (not the global profile max) so weak clusters aren't silently
        # excluded.  Iterate until the residual shift is < 0.0001 Da.
        _cluster_max = float(obs_at_best.max()) + 1e-12
        _cluster_threshold_frac = 0.05 * _cluster_max / (
            float(profile_fine_smooth.max()) + 1e-12
        )
        # Broad-window pass: converge the rigid shift using the same ±20 mDa
        # window as the scan (captures full peak + slight shoulders).
        for _ in range(10):
            _prev = best_mono_mz
            best_mono_mz = _refine_mono_mz(
                best_mono_mz, env_offsets, env_rel, z,
                mz_fine, profile_fine_smooth, tol_da=tol_da_z,
                min_peak_fraction=_cluster_threshold_frac,
            )
            if abs(best_mono_mz - _prev) < 5e-5:    # 0.05 mDa
                break
        # Tight-window pass: ≈ ±10 mDa — close to one native centroid
        # spacing.  This removes residual positive bias that can arise when
        # the broad window asymmetrically samples peak tails (right tail of
        # a rising isotope envelope looks heavier than the left tail).
        _tol_tight = 0.010
        for _ in range(5):
            _prev = best_mono_mz
            best_mono_mz = _refine_mono_mz(
                best_mono_mz, env_offsets, env_rel, z,
                mz_fine, profile_fine_smooth, tol_da=_tol_tight,
                min_peak_fraction=_cluster_threshold_frac,
            )
            if abs(best_mono_mz - _prev) < 5e-5:
                break

        mono_mass_da = best_mono_mz * z - z * PROTON
        n_obs = int((obs_at_best > 0.05 * (obs_at_best.max() + 1e-12)).sum())

        # ------------------------------------------------------------------
        # Gap penalty: penalise z/2 aliases by checking inter-stick signal.
        # For z=5 aliasing z=10: every z=10 peak sits at the midpoint between
        # consecutive z=5 sticks, so gap signal is ~50 % of stick signal.
        # For the true z=10: midpoints fall in empty valleys → gap ≈ 0.
        # ------------------------------------------------------------------
        exp_mz = best_mono_mz + env_offsets / z
        mask_in = (exp_mz >= mz_fine[0]) & (exp_mz <= mz_fine[-1])
        exp_in  = exp_mz[mask_in]
        if len(exp_in) >= 2:
            gap_mz    = 0.5 * (exp_in[:-1] + exp_in[1:])   # inter-stick midpoints
            gap_half  = 0.020                                # same fixed window as sticks
            obs_stick = _windowed_sum(exp_in, tol_da_z)
            obs_gap   = _windowed_sum(gap_mz, gap_half)
            gap_frac  = float(obs_gap.sum()) / (float(obs_stick.sum()) + 1e-12)
        else:
            gap_frac = 0.0

        # ------------------------------------------------------------------
        # Peak-coverage penalty: fraction of observed NMS peaks NOT explained
        # by any theoretical stick.
        #
        # Only peaks within the theoretical envelope range are counted —
        # peaks beyond the last stick are real low-intensity isotope peaks
        # below the averagine min_relative cutoff, not evidence of a bad fit.
        #
        # A slightly wider match tolerance (0.030 Da) is used here rather
        # than tol_da_z (0.020 Da): NMS peaks are on the sparse original axis
        # (~0.007 Da spacing), so they can be up to ~0.010 Da away from the
        # centroid-corrected fine-grid stick position purely due to
        # quantisation.  0.030 Da comfortably absorbs that offset while
        # remaining well inside the inter-peak spacing for all z ≥ 3.
        #
        # For wrong-charge aliases (e.g. z=3 on z=9 data), sticks land on
        # only 1 in 3 observed peaks → unexplained_frac ≈ 0.67.
        # For the correct charge, nearly all in-range peaks are covered → ≈ 0.
        # ------------------------------------------------------------------
        sticks_mz    = best_mono_mz + env_offsets / z
        first_stick  = float(sticks_mz[0])
        last_stick   = float(sticks_mz[-1])
        match_tol    = 0.030    # slightly wider than integration tol for sparse-axis peaks
        in_range     = (observed_mz >= first_stick - match_tol) & \
                       (observed_mz <= last_stick  + match_tol)
        obs_in_range = observed_mz[in_range]
        if len(obs_in_range) > 0:
            dists_to_sticks = np.abs(
                obs_in_range[:, None] - sticks_mz[None, :]
            )                                                # (n_in_range, n_sticks)
            n_explained      = int((dists_to_sticks.min(axis=1) < match_tol).sum())
            unexplained_frac = (len(obs_in_range) - n_explained) / len(obs_in_range)
        else:
            n_explained      = 0
            unexplained_frac = 0.0

        # Combined score: cosine × gap-penalty × peak-coverage
        adjusted = (
            best_cos
            * float(np.exp(-2.0 * gap_frac))
            * (1.0 - unexplained_frac)
        )

        candidates.append((adjusted, z, dict(
            charge=z,
            monoisotopic_mz=float(best_mono_mz),
            monoisotopic_mass_da=float(mono_mass_da),
            cosine_similarity=float(best_cos),
            adjusted_score=float(adjusted),
            gap_fraction=float(gap_frac),
            unexplained_peak_fraction=float(unexplained_frac),
            n_isotope_peaks_observed=n_obs,
            rough_mass_da=float(rough_mass),
            averagine_offsets=env_offsets,
            averagine_rel=env_rel,
            observed_mz=observed_mz,
            observed_int=obs_int_norm,
            cluster_intensity=float(profile[group_indices].sum()),
            mz_fine=mz_fine,
            profile_fine_raw=profile_fine_raw,
            profile_fine_smooth=profile_fine_smooth,
        )))

    if not candidates:
        return None

    # Best combined score wins (cosine × gap-penalty × peak-coverage)
    best_score, best_z, best_dict = max(candidates, key=lambda x: x[0])

    # ------------------------------------------------------------------
    # Fit quality flag
    # 'good'       : unexplained < 0.20  — nearly all observed peaks covered
    # 'suspicious' : unexplained 0.20–0.40 — noticeable residual signal
    #                (e.g. mild coelution, or weak second species)
    # 'poor'       : unexplained > 0.40  — major fraction of observed peaks
    #                not accounted for; likely coeluting species or wrong z
    # ------------------------------------------------------------------
    upf = best_dict['unexplained_peak_fraction']
    if upf < 0.20:
        fit_quality = 'good'
    elif upf < 0.40:
        fit_quality = 'suspicious'
    else:
        fit_quality = 'poor'
    best_dict['fit_quality'] = fit_quality

    if verbose:
        # Print winner diagnostics + full ranked candidate table
        print(
            f"  [F{factor_idx}-G{group_idx}] WINNER z={best_z}  "
            f"cos={best_dict['cosine_similarity']:.4f}  "
            f"gap={best_dict['gap_fraction']:.3f}  "
            f"unexp={upf:.2f}  "
            f"adj={best_dict['adjusted_score']:.4f}  "
            f"quality={fit_quality}"
        )
        sorted_cands = sorted(candidates, key=lambda x: x[0], reverse=True)
        for rank, (adj, z, d) in enumerate(sorted_cands[:5], 1):
            marker = " <-- WINNER" if z == best_z else ""
            print(
                f"    #{rank} z={z:2d}  cos={d['cosine_similarity']:.4f}  "
                f"gap={d['gap_fraction']:.3f}  unexp={d['unexplained_peak_fraction']:.2f}  "
                f"adj={adj:.4f}{marker}"
            )

        # Warn when a half/double-z alias is within ambiguity_threshold of the winner
        for alt_adj, alt_z, alt_d in candidates:
            if alt_z == best_z:
                continue
            is_alias = (alt_z == best_z * 2) or (best_z % 2 == 0 and alt_z == best_z // 2)
            if is_alias and alt_adj >= (1.0 - ambiguity_threshold) * best_score:
                print(
                    f"  [AMBIGUITY] Factor {factor_idx} group {group_idx}: "
                    f"z={best_z} (adj={best_score:.4f}) vs "
                    f"z={alt_z} (adj={alt_adj:.4f})  —  keeping z={best_z}"
                )

    # ------------------------------------------------------------------
    # Left-boundary ratio check  (Task 2)
    # ------------------------------------------------------------------
    # Cosine similarity only evaluates sticks *inside* the claimed
    # envelope window, so it is blind to signal at M-1 (one isotope
    # step to the left of the current monoisotopic position).
    # A high left_ratio = signal(M-1) / signal(M+0) means the true
    # monoisotopic peak is one step to the left — the assignment is
    # off by +1 in isotope index.
    # Threshold 0.25: averagine predicts < 1% relative intensity at M-1
    # for any mass above ~500 Da, so any ratio > 0.25 is real signal.
    _LEFT_RATIO_THRESHOLD = 0.25

    _z_lb    = best_dict['charge']
    _mono_lb = best_dict['monoisotopic_mz']
    _tol_lb  = 0.020

    # Reuse the cumulative sum and grid variables already computed above.
    _hb_lb = max(1, round(_tol_lb / _fine_step))

    def _lb_signal(mz_center: float) -> float:
        idx = int(round((mz_center - _fine_lo) / _fine_step))
        lo  = max(0, idx - _hb_lb)
        hi  = min(_fine_N, idx + _hb_lb + 1)
        return float(_cs_smooth[hi] - _cs_smooth[lo])

    _sig_left = _lb_signal(_mono_lb - NEUTRON / _z_lb)
    _sig_mono = _lb_signal(_mono_lb)
    _left_ratio = _sig_left / (_sig_mono + 1e-12)

    # ---- Always store k=0 (current / no-shift) metrics ----
    best_dict['left_ratio'] = float(_left_ratio)
    best_dict['k0_monoisotopic_mz']      = float(_mono_lb)
    best_dict['k0_monoisotopic_mass_da'] = float(_mono_lb * _z_lb - _z_lb * PROTON)
    best_dict['k0_cosine_similarity']    = float(best_dict['cosine_similarity'])
    best_dict['k0_averagine_offsets']    = best_dict['averagine_offsets'].copy()
    best_dict['k0_averagine_rel']        = best_dict['averagine_rel'].copy()

    # ---- k=0 integrated peak intensities (±10 ppm per position) ----
    _k0_ints = _integrate_isotope_peaks(
        _cs_smooth, _fine_step, _fine_lo, _fine_N,
        _mono_lb, _z_lb, best_dict['n_isotope_peaks_observed'],
    )
    best_dict['k0_peak_intensities'] = "|".join(f"{v:.4f}" for v in _k0_ints)
    best_dict['k1_peak_intensities'] = ""   # default; filled below if k=1 computed

    # ---- Default k=1 (shifted) to NaN — filled in below if computed ----
    _nan = float('nan')
    best_dict['k1_monoisotopic_mz']      = _nan
    best_dict['k1_monoisotopic_mass_da'] = _nan
    best_dict['k1_cosine_similarity']    = _nan

    if verbose:
        print(
            f"  [F{factor_idx}-G{group_idx}] left_ratio = {_left_ratio:.3f}  "
            f"(M-1 = {_sig_left:.3e}, M+0 = {_sig_mono:.3e})"
        )

    if _left_ratio > _LEFT_RATIO_THRESHOLD:
        # -- Compute the k=1 assignment: mono_mz shifted one step left --
        _shifted_mz   = _mono_lb - NEUTRON / _z_lb
        _shifted_mass = _shifted_mz * _z_lb - _z_lb * PROTON

        if _shifted_mass > 100.0:
            _env_off_s, _env_rel_s = averagine_envelope(
                _shifted_mass, min_relative=0.005, max_peaks=60
            )

            # Narrow re-scan ± 5 mDa around the shifted position.
            _s_mz, _s_cos, _s_obs = _vectorised_stick_scan(
                mz_fine, profile_fine_smooth,
                _env_off_s, _env_rel_s, _z_lb,
                _shifted_mz - 0.005, _shifted_mz + 0.005,
                scan_step_da=0.001, tol_da=0.020,
            )

            # Centroid refinement on the shifted candidate
            _s_cluster_max = float(_s_obs.max()) + 1e-12
            _s_thr = 0.05 * _s_cluster_max / (
                float(profile_fine_smooth.max()) + 1e-12
            )
            for _ in range(10):
                _prev_s = _s_mz
                _s_mz = _refine_mono_mz(
                    _s_mz, _env_off_s, _env_rel_s, _z_lb,
                    mz_fine, profile_fine_smooth, tol_da=0.020,
                    min_peak_fraction=_s_thr,
                )
                if abs(_s_mz - _prev_s) < 5e-5:
                    break
            for _ in range(5):
                _prev_s = _s_mz
                _s_mz = _refine_mono_mz(
                    _s_mz, _env_off_s, _env_rel_s, _z_lb,
                    mz_fine, profile_fine_smooth, tol_da=0.010,
                    min_peak_fraction=_s_thr,
                )
                if abs(_s_mz - _prev_s) < 5e-5:
                    break

            # Always record k=1 metrics regardless of acceptance.
            best_dict['k1_monoisotopic_mz']      = float(_s_mz)
            best_dict['k1_monoisotopic_mass_da'] = float(
                _s_mz * _z_lb - _z_lb * PROTON
            )
            best_dict['k1_cosine_similarity']    = float(_s_cos)
            best_dict['k1_averagine_offsets']    = _env_off_s
            best_dict['k1_averagine_rel']        = _env_rel_s

            # k=1 integrated peak intensities — starts at the shifted mono_mz
            # (one spacing to the left), so position 0 is the absent M+0 peak.
            _k1_ints = _integrate_isotope_peaks(
                _cs_smooth, _fine_step, _fine_lo, _fine_N,
                float(_s_mz), _z_lb, best_dict['n_isotope_peaks_observed'],
            )
            best_dict['k1_peak_intensities'] = "|".join(f"{v:.4f}" for v in _k1_ints)

            # Accept shifted assignment only if the cosine degradation is
            # small (≤ 0.020, i.e. the two positions score comparably).
            # A 0.020 tolerance was chosen empirically: it passes genuine
            # off-by-one cases (typical Δcos ≤ 0.002) while rejecting
            # neighbouring-state interference (typical Δcos ≈ 0.04+).
            _accept_cos = best_dict['k0_cosine_similarity'] - 0.020
            if _s_cos >= _accept_cos:
                if verbose:
                    print(
                        f"  [LEFT-SHIFT ACCEPTED] F{factor_idx}-G{group_idx}: "
                        f"k=0 mono={_mono_lb:.4f} → k=1 mono={_s_mz:.4f} Da  "
                        f"(left_ratio={_left_ratio:.3f}, "
                        f"cos {best_dict['k0_cosine_similarity']:.4f} → {_s_cos:.4f})"
                    )
                # Update the primary assignment to the shifted (k=1) position
                best_dict['monoisotopic_mz']      = float(_s_mz)
                best_dict['monoisotopic_mass_da'] = float(
                    _s_mz * _z_lb - _z_lb * PROTON
                )
                best_dict['cosine_similarity']    = float(_s_cos)
                best_dict['averagine_offsets']    = _env_off_s
                best_dict['averagine_rel']        = _env_rel_s
                best_dict['left_shifted']         = True
            else:
                if verbose:
                    print(
                        f"  [LEFT-SHIFT REJECTED] F{factor_idx}-G{group_idx}: "
                        f"k=1 cos={_s_cos:.4f} < {_accept_cos:.4f} threshold  "
                        f"(left_ratio={_left_ratio:.3f}, k=1 stored for reference)"
                    )
                best_dict['left_shifted'] = False
        else:
            best_dict['left_shifted'] = False
    else:
        best_dict['left_shifted'] = False

    return best_dict


# ===========================================================================
# Section 4 — Per-cluster PNG
# ===========================================================================

def plot_isotopic_cluster(
    result: Dict,
    A_r: np.ndarray,
    B_r: np.ndarray,
    C_r: np.ndarray,
    rt_axis: np.ndarray,
    dt_axis: np.ndarray,
    mz_axis: np.ndarray,
    output_path: str,
    factor_idx: int = 0,
    cluster_idx: int = 0,
    mz_axis_full: Optional[np.ndarray] = None,
    mask_mz: Optional[np.ndarray] = None,
    png_dpi: int = 100,
) -> None:
    """Save a 3×1 PNG for one isotopic cluster.

    Panels
    ------
    Top    : RT distribution — integral of factor over DT and m/z
    Middle : DT distribution — integral of factor over RT and m/z
    Bottom : m/z slice.  Three overlaid traces:
               • Blue solid  — raw m/z profile (C_r * A_sum * B_sum)
               • Red dashed  — Gaussian-smoothed profile used for the scan
                               (from result['profile_fine_smooth'], if present)
               • Green stems — theoretical averagine sticks at the best-fit
                               monoisotopic m/z, height = relative intensity

    Parameters
    ----------
    result :
        Dict returned by ``_assign_charge``.
    A_r, B_r, C_r :
        Factor vectors for one component (1-D arrays, in the masked coordinate
        space matching *rt_axis* / *dt_axis* / *mz_axis*).
    rt_axis, dt_axis, mz_axis :
        Axis arrays matching the factor vectors.
    output_path :
        Full path (including filename) for the PNG.
    factor_idx, cluster_idx :
        Used for figure suptitle labelling.
    """
    z           = result["charge"]
    mono_mz     = result["monoisotopic_mz"]
    cos         = result["cosine_similarity"]
    env_offsets = result["averagine_offsets"]
    env_rel     = result["averagine_rel"]

    # --- marginal projections (Σ over the other two modes) ---
    b_sum = float(B_r.sum())
    c_sum = float(C_r.sum())
    a_sum = float(A_r.sum())

    rt_profile = A_r.astype(np.float64) * b_sum * c_sum
    dt_profile = B_r.astype(np.float64) * a_sum * c_sum
    mz_profile = C_r.astype(np.float64) * a_sum * b_sum

    rt_norm = rt_profile / (rt_profile.max() + 1e-12)
    dt_norm = dt_profile / (dt_profile.max() + 1e-12)

    # m/z plot window: tight around the theoretical envelope (0.3 Da margin).
    # This keeps the snapshot focused on the cluster itself and avoids
    # contaminating the intensity scale with signal from other clusters.
    avg_mz       = mono_mz + env_offsets / z
    mz_margin    = 0.30
    mz_lo_plot   = float(avg_mz[0])  - mz_margin
    mz_hi_plot   = float(avg_mz[-1]) + mz_margin

    # Raw trace: take the non-zero NTF bins in the plot window, then merge any
    # adjacent-bin pairs that are ≤ 2 mDa apart.  Such pairs are artefacts of
    # the old tent reprofiling (each centroid split across two ~0.001 Da bins);
    # with nearest-bin reprofiling they do not arise.  Merging is harmless for
    # nearest-bin data (consecutive bins are ≥ 12 mDa apart → no merges).
    raw_mask      = (mz_axis >= mz_lo_plot) & (mz_axis <= mz_hi_plot)
    mz_raw_all    = mz_axis[raw_mask].astype(np.float64)
    int_raw_all   = mz_profile[raw_mask].astype(np.float64)

    # Merge adjacent tent-split pairs: walk sorted bins, collapse any pair
    # whose m/z separation is ≤ 0.002 Da into a single intensity-weighted point.
    if len(mz_raw_all) > 1:
        mz_merged, int_merged = [], []
        i = 0
        while i < len(mz_raw_all):
            if (i + 1 < len(mz_raw_all)
                    and (mz_raw_all[i + 1] - mz_raw_all[i]) <= 0.002):
                total = int_raw_all[i] + int_raw_all[i + 1]
                wmz   = (mz_raw_all[i]   * int_raw_all[i]
                         + mz_raw_all[i+1] * int_raw_all[i+1]) / (total + 1e-12)
                mz_merged.append(wmz)
                int_merged.append(total)
                i += 2
            else:
                mz_merged.append(mz_raw_all[i])
                int_merged.append(int_raw_all[i])
                i += 1
        mz_raw_slice  = np.array(mz_merged)
        int_raw_slice = np.array(int_merged)
    else:
        mz_raw_slice  = mz_raw_all
        int_raw_slice = int_raw_all

    cluster_bpi   = float(int_raw_slice.max()) if len(int_raw_slice) else 0.0
    cluster_total = float(int_raw_slice.sum()) if len(int_raw_slice) else 0.0
    scale         = cluster_bpi + 1e-12
    int_raw_norm  = int_raw_slice / scale

    # Smoothed profile slice (fine uniform grid from result, if available)
    has_smooth = "mz_fine" in result and "profile_fine_smooth" in result
    if has_smooth:
        mz_fine    = result["mz_fine"]
        pf_smooth  = result["profile_fine_smooth"]
        fine_mask  = (mz_fine >= mz_lo_plot) & (mz_fine <= mz_hi_plot)
        mz_sm      = mz_fine[fine_mask]
        int_sm     = pf_smooth[fine_mask]
        int_sm_norm = int_sm / scale
    else:
        mz_sm = int_sm_norm = None

    fig, axes = plt.subplots(3, 1, figsize=(6, 8))
    fig.suptitle(
        f"Factor {factor_idx} · cluster {cluster_idx} · z={z}\n"
        f"Monoisotopic m/z = {mono_mz:.4f} Da · cosine = {cos:.3f}",
        fontsize=9,
    )

    # --- RT ---
    rt_bpi   = float(rt_profile.max())
    rt_total = float(rt_profile.sum())
    ax = axes[0]
    ax.plot(rt_axis, rt_norm, color="steelblue", linewidth=1.2)
    ax.text(
        0.98, 0.97,
        f"BPI = {rt_bpi:.3e}\nTotal = {rt_total:.3e}",
        transform=ax.transAxes,
        fontsize=7, va="top", ha="right",
        color="dimgray",
        linespacing=1.5,
    )
    ax.set_xlabel("Retention time (min)", fontsize=8)
    ax.set_ylabel("Intensity (norm.)", fontsize=8)
    ax.set_title("RT distribution  (Σ DT, m/z)", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)

    # --- DT ---
    dt_bpi   = float(dt_profile.max())
    dt_total = float(dt_profile.sum())
    ax = axes[1]
    ax.plot(dt_axis, dt_norm, color="darkorange", linewidth=1.2)
    ax.text(
        0.98, 0.97,
        f"BPI = {dt_bpi:.3e}\nTotal = {dt_total:.3e}",
        transform=ax.transAxes,
        fontsize=7, va="top", ha="right",
        color="dimgray",
        linespacing=1.5,
    )
    ax.set_xlabel("Drift bin", fontsize=8)
    ax.set_ylabel("Intensity (norm.)", fontsize=8)
    ax.set_title("DT distribution  (Σ RT, m/z)", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)

    # --- m/z slice ---
    ax = axes[2]

    # 1. Raw profile — blue solid, thin
    ax.plot(mz_raw_slice, int_raw_norm,
            color="dodgerblue", linewidth=0.8, label="Raw", zorder=3)

    # 2. SG-smoothed profile — red dashed, thin (if available)
    if has_smooth and mz_sm is not None and len(mz_sm) > 0:
        ax.plot(mz_sm, int_sm_norm,
                color="firebrick", linewidth=0.8, linestyle="--",
                label="Smoothed", zorder=4, alpha=0.85)

    # 3. Theoretical averagine sticks at best-fit mono_mz — green stems + tips
    in_plot = (avg_mz >= mz_lo_plot) & (avg_mz <= mz_hi_plot)
    stick_mzs  = avg_mz[in_plot]
    stick_rels = env_rel[in_plot]
    ax.vlines(stick_mzs, 0, stick_rels, colors="seagreen", linewidth=0.7, zorder=2)
    ax.plot(stick_mzs, stick_rels, "^", color="seagreen", markersize=3, zorder=2)

    # Monoisotopic position marker (dotted navy line)
    ax.axvline(mono_mz, color="navy", linewidth=0.6, linestyle=":",
               label=f"Mono {mono_mz:.4f} Da", zorder=1)

    # Intensity annotations — BPI and total in scientific notation
    ax.text(
        0.98, 0.97,
        f"BPI = {cluster_bpi:.3e}\nTotal = {cluster_total:.3e}",
        transform=ax.transAxes,
        fontsize=7, va="top", ha="right",
        color="dimgray",
        linespacing=1.5,
    )

    ax.set_xlabel("m/z (Da)", fontsize=8)
    ax.set_ylabel("Intensity (norm.)", fontsize=8)
    ax.set_title(
        f"m/z spectrum  (Σ RT, DT)   cosine={cos:.3f}",
        fontsize=8,
    )
    ax.set_xlim(mz_lo_plot, mz_hi_plot)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Section 5 — Per-factor search
# ===========================================================================

def _profile_purity(
    profile: np.ndarray,
    axis: np.ndarray,
    peak_prominence_frac: float = 0.10,
) -> Dict[str, object]:
    """Compute Gaussian shape purity metrics for a 1-D factor profile (RT or DT).

    Parameters
    ----------
    profile : (n,) — A_r or B_r factor vector (non-negative).
    axis    : (n,) — Corresponding axis values (RT [min] or DT [bins]).
    peak_prominence_frac :
        Local maxima whose prominence is below this fraction of the profile
        maximum are ignored when counting peaks.  0.10 means a secondary
        peak must reach at least 10 % of the dominant peak's height to be
        counted as a separate mode.

    Returns
    -------
    dict with keys:
        gaussian_r2  : float — Coefficient of determination of the best-fit
                       single Gaussian to the normalised profile.  1.0 = perfect
                       Gaussian, lower values indicate asymmetry, shoulders, or
                       multimodal distributions.  Can be negative for very
                       poor fits (bounded to [-1, 1]).
        n_peaks      : int  — Number of significant local maxima in the profile.
                       1 = unimodal, >1 = multimodal.
        multimodal   : bool — True when n_peaks > 1.
    """
    profile = np.asarray(profile, dtype=np.float64)
    axis    = np.asarray(axis,    dtype=np.float64)

    if profile.size < 5 or float(profile.max()) <= 0:
        return {"gaussian_r2": float("nan"), "n_peaks": 0, "multimodal": False}

    # ---- peak counting -------------------------------------------------------
    prominence_thresh = peak_prominence_frac * float(profile.max())
    peaks, _ = _scipy_find_peaks(profile, prominence=prominence_thresh)
    n_peaks   = int(len(peaks))
    multimodal = n_peaks > 1

    # ---- Gaussian fit --------------------------------------------------------
    # Normalise to [0, 1] so the amplitude bound is stable regardless of
    # absolute intensity scale.
    p_norm = profile / float(profile.max())
    x      = axis

    # Weighted mean and standard deviation as initial guess.
    w   = profile / (float(profile.sum()) + 1e-12)
    mu0 = float(np.dot(x, w))
    sig0 = float(np.sqrt(np.dot(w, (x - mu0) ** 2)))
    if sig0 < 1e-9:
        sig0 = float((x[-1] - x[0]) / 4.0) + 1e-9

    def _gauss(x, amp, mu, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    try:
        popt, _ = _curve_fit(
            _gauss, x, p_norm,
            p0=[1.0, mu0, sig0],
            bounds=(
                [0.0,    float(x[0]),  1e-9],
                [2.0,    float(x[-1]), 4.0 * float(x[-1] - x[0]) + 1e-9],
            ),
            maxfev=600,
        )
        fitted  = _gauss(x, *popt)
        ss_res  = float(np.sum((p_norm - fitted) ** 2))
        ss_tot  = float(np.sum((p_norm - float(p_norm.mean())) ** 2))
        r2      = float(np.clip(1.0 - ss_res / (ss_tot + 1e-12), -1.0, 1.0))
    except Exception:
        r2 = float("nan")

    return {"gaussian_r2": r2, "n_peaks": n_peaks, "multimodal": multimodal}


def _per_peak_gaussian_rmse(
    observed_mz: np.ndarray,
    mz_fine: np.ndarray,
    profile_fine_smooth: np.ndarray,
    half_window_da: float = 0.04,
) -> float:
    """Average per-peak Gaussian RMSE across all isotope peaks in a cluster.

    For each observed isotope peak, extracts the Gaussian-smoothed profile in
    a ±half_window_da window, fits a single Gaussian, and computes the RMSE
    between observed and fitted profiles.  Returns the mean RMSE across all
    peaks that could be fitted (NaN if none fitted successfully).

    A low peak_rmse indicates each peak is well-described by a single Gaussian
    (clean, symmetric peak shape).  A high value signals noisy, asymmetric, or
    blended peaks.

    Parameters
    ----------
    observed_mz : (n_peaks,) observed isotope peak m/z positions [Da]
    mz_fine : uniform fine m/z grid [Da]
    profile_fine_smooth : Gaussian-smoothed intensity profile on fine grid
    half_window_da : half-width of fitting window around each peak [Da]
    """
    def _gauss(x, amp, cen, sig):
        return amp * np.exp(-0.5 * ((x - cen) / (sig + 1e-12)) ** 2)

    step = float(mz_fine[1] - mz_fine[0])
    half_bins = max(5, int(round(half_window_da / step)))
    mz0 = float(mz_fine[0])

    rmse_vals: List[float] = []
    for mz_peak in observed_mz:
        center_idx = int(round((mz_peak - mz0) / step))
        lo = max(0, center_idx - half_bins)
        hi = min(len(mz_fine), center_idx + half_bins + 1)
        if hi - lo < 5:
            continue
        x = mz_fine[lo:hi]
        y = profile_fine_smooth[lo:hi]
        amp0 = float(y.max()) if y.max() > 0 else 1.0
        try:
            popt, _ = _curve_fit(
                _gauss, x, y,
                p0=[amp0, mz_peak, half_window_da / 3.0],
                bounds=(
                    [0.0,    float(x[0]),  1e-9],
                    [np.inf, float(x[-1]), half_window_da],
                ),
                maxfev=500,
            )
            y_fit = _gauss(x, *popt)
            rmse_vals.append(float(np.sqrt(np.mean((y - y_fit) ** 2))))
        except Exception:
            pass

    return float(np.mean(rmse_vals)) if rmse_vals else float("nan")


def find_isotopic_clusters(
    A_r: np.ndarray,
    B_r: np.ndarray,
    C_r: np.ndarray,
    mz_axis: np.ndarray,
    rt_axis: np.ndarray,
    dt_axis: np.ndarray,
    charge_range: Tuple[int, int] = (3, 15),
    min_cosine: float = 0.5,
    min_peaks_per_cluster: int = 3,
    min_height_frac: float = 0.02,
    min_distance_da: float = 0.04,
    max_gap_da: float = 3.0,
    tol_da: float = 0.01,
    factor_idx: int = 0,
    output_dir: Optional[str] = None,
    verbose: bool = False,
    mz_axis_full: Optional[np.ndarray] = None,
    mask_mz: Optional[np.ndarray] = None,
    png_dpi: int = 100,
    plot_format: str = "png",
) -> List[Dict]:
    """Find all isotopic clusters in a single NTF factor.

    Parameters
    ----------
    A_r : (n_rt,) — RT factor vector
    B_r : (n_dt,) — DT factor vector
    C_r : (n_mz,) — m/z factor vector
    mz_axis : (n_mz,) — m/z values [Da]
    rt_axis : (n_rt,) — RT values [min]
    dt_axis : (n_dt,) — DT bin indices
    charge_range :
        (min_z, max_z) inclusive.  Default (3, 15).
    min_cosine :
        Minimum cosine similarity to accept a cluster assignment.
    min_peaks_per_cluster :
        Discard groups with fewer detected peaks.
    min_height_frac :
        Peaks below this fraction of the maximum m/z profile intensity
        are ignored during peak detection.
    min_distance_da :
        Minimum physical separation between accepted peaks [Da].
        Should be slightly less than NEUTRON/z_max (e.g. 0.04 Da for
        z_max=15) so that intra-peak sub-structure is suppressed while
        consecutive isotope peaks are preserved.
    max_gap_da :
        Peaks separated by more than this [Da] are treated as distinct
        clusters.
    tol_da :
        Tolerance for mapping averagine positions onto the m/z grid.
    factor_idx :
        Factor number (used for labelling and image filenames).
    output_dir :
        If given, save one image per accepted cluster here.
    plot_format :
        Image format: "png" (default) or "pdf".

    Returns
    -------
    clusters : list of dict
        One dict per accepted cluster with keys:
            factor_idx, cluster_idx, charge,
            monoisotopic_mz, monoisotopic_mass_da,
            rt_center, dt_center,
            cosine_similarity, n_isotope_peaks_observed,
            cluster_intensity, intensity_fraction, peak_rmse.
    """
    # --- shape sanity check ---
    # C_r must match mz_axis.  A common mistake is passing result['mz_axis']
    # (the full axis, e.g. 40001 elements) instead of result['mz_axis_ntf']
    # (the masked axis that matches C, e.g. 5604 elements).
    if len(C_r) != len(mz_axis):
        raise ValueError(
            f"Shape mismatch: C_r has {len(C_r)} elements but mz_axis has "
            f"{len(mz_axis)} elements.  Pass result['mz_axis_ntf'] (the masked "
            f"m/z axis whose length equals C.shape[0]), not result['mz_axis'] "
            f"(the full axis)."
        )

    # m/z profile = marginal over RT and DT
    mz_profile = C_r.astype(np.float64) * float(A_r.sum()) * float(B_r.sum())

    # Pre-compute SG-smoothed fine-grid profile (shared across all groups)
    # window_bins=2  →  5-bin SG window ≈ 5 × 12.5 mDa = 62.5 mDa.
    # This is ~1.25× the typical isotope-peak FWHM (≈50 mDa at z=9).
    # The previous window_bins=3 (87.5 mDa) was broad enough to let
    # signal from adjacent isotope peaks bleed into the centroid window
    # and pull centroid estimates rightward by ~1–2 mDa.
    mz_fine, profile_fine_raw, profile_fine_smooth = _smooth_mz_profile(
        mz_profile, mz_axis, window_bins=2, polyorder=2, step_da=0.001
    )

    # RT / DT centers (intensity-weighted means of the factor profiles)
    rt_weights = A_r.astype(np.float64)
    dt_weights = B_r.astype(np.float64)
    rt_center = float(
        np.average(rt_axis, weights=rt_weights)
        if rt_weights.sum() > 0 else rt_axis.mean()
    )
    dt_center = float(
        np.average(dt_axis, weights=dt_weights)
        if dt_weights.sum() > 0 else dt_axis.mean()
    )
    total_factor_intensity = float(mz_profile.sum())
    factor_bpi             = float(mz_profile.max()) if mz_profile.size > 0 else 0.0
    factor_tic             = total_factor_intensity

    # --- RT / DT Gaussian purity metrics -------------------------------------
    # Computed once per factor and shared across all clusters from this factor.
    # gaussian_r2 close to 1.0 means a clean, unimodal Gaussian distribution.
    # n_peaks > 1 flags a multimodal factor (two species co-eluting or
    # co-drifting in this slice).
    _rt_purity = _profile_purity(A_r, rt_axis)
    _dt_purity = _profile_purity(B_r, dt_axis)

    # Detect peaks and group into candidate clusters
    peak_idx = _find_profile_peaks(
        mz_profile, mz_axis,
        min_height_frac=min_height_frac,
        min_distance_da=min_distance_da,
    )
    groups = _group_peaks(peak_idx, mz_axis, max_gap_da=max_gap_da)

    if verbose and len(peak_idx) > 0:
        peak_mzs = mz_axis[peak_idx]
        print(
            f"  [Factor {factor_idx}] NMS peaks ({len(peak_idx)}): "
            + ", ".join(f"{m:.3f}" for m in peak_mzs)
        )
        print(
            f"  [Factor {factor_idx}] Groups ({len(groups)}): "
            + "; ".join(
                f"[{mz_axis[g[0]]:.3f}–{mz_axis[g[-1]]:.3f} Da, {len(g)} peaks]"
                for g in groups
            )
        )

    clusters: List[Dict] = []
    for ci, group in enumerate(groups):
        if len(group) < min_peaks_per_cluster:
            continue

        if verbose:
            center_mz = float(np.mean(mz_axis[group]))
            print(
                f"    Group {ci}: center ≈ {center_mz:.3f} Da  "
                f"({len(group)} peaks, "
                f"range {mz_axis[group[0]]:.3f}–{mz_axis[group[-1]]:.3f} Da)"
            )

        match = _assign_charge(
            group, mz_axis, mz_profile,
            mz_fine=mz_fine,
            profile_fine_raw=profile_fine_raw,
            profile_fine_smooth=profile_fine_smooth,
            charge_range=charge_range,
            min_cosine=min_cosine,
            factor_idx=factor_idx,
            group_idx=ci,
            verbose=verbose,
        )
        if match is None:
            if verbose:
                print(f"      → no charge assignment (cosine < {min_cosine})")
            continue

        if verbose:
            print(
                f"      → z={match['charge']}  "
                f"mono={match['monoisotopic_mz']:.4f} Da  "
                f"cos={match['cosine_similarity']:.3f}  "
                f"n_peaks={match['n_isotope_peaks_observed']}"
            )

        intensity_fraction = (
            match["cluster_intensity"] / total_factor_intensity
            if total_factor_intensity > 0 else float("nan")
        )

        # BPI / TIC of the isotopic cluster's m/z window (same window as the PNG).
        # This is the full envelope ± 0.30 Da margin, matching plot_isotopic_cluster.
        _env_mz    = match["monoisotopic_mz"] + match["averagine_offsets"] / match["charge"]
        _cwin      = (
            (mz_axis >= float(_env_mz[0])  - 0.30) &
            (mz_axis <= float(_env_mz[-1]) + 0.30)
        )
        cluster_bpi = float(mz_profile[_cwin].max()) if _cwin.any() else 0.0
        cluster_tic = float(mz_profile[_cwin].sum()) if _cwin.any() else 0.0

        # Per-peak Gaussian RMSE: average goodness-of-fit of a single Gaussian
        # to each isotope peak's local profile.  Low = clean Gaussian peaks;
        # high = noisy, asymmetric, or blended peaks.
        peak_rmse = _per_peak_gaussian_rmse(
            match["observed_mz"],
            match["mz_fine"],
            match["profile_fine_smooth"],
        )

        # Shared fields that are the same for every k of this cluster.
        _shared = dict(
            factor_idx=factor_idx,
            cluster_idx=ci,
            charge=match["charge"],
            left_ratio=match.get("left_ratio", float("nan")),
            left_shifted=match.get("left_shifted", False),
            rt_center=rt_center,
            dt_center=dt_center,
            cluster_intensity=match["cluster_intensity"],
            intensity_fraction=intensity_fraction,
            factor_bpi=factor_bpi,
            factor_tic=factor_tic,
            cluster_bpi=cluster_bpi,
            cluster_tic=cluster_tic,
            peak_rmse=peak_rmse,
            # ---- RT / DT Gaussian purity (factor-level, shared per cluster) ----
            # rt_gaussian_r2 / dt_gaussian_r2: R² of single-Gaussian fit to the
            # factor's RT (A_r) and DT (B_r) profile.  1.0 = perfect Gaussian,
            # lower = asymmetric, shoulder, or multimodal distribution.
            # rt_n_peaks / dt_n_peaks: number of significant local maxima
            # (prominence ≥ 10 % of profile max).  >1 flags multimodal factors.
            rt_gaussian_r2=_rt_purity["gaussian_r2"],
            dt_gaussian_r2=_dt_purity["gaussian_r2"],
            rt_n_peaks=_rt_purity["n_peaks"],
            dt_n_peaks=_dt_purity["n_peaks"],
            rt_multimodal=_rt_purity["multimodal"],
            dt_multimodal=_dt_purity["multimodal"],
        )

        # k=0 row — always present.
        # Full scan / fit metrics are only computed for the primary scan
        # position (k=0), so they live here.
        clusters.append({
            **_shared,
            "k": 0,
            "monoisotopic_mz":      match.get("k0_monoisotopic_mz",      float("nan")),
            "monoisotopic_mass_da": match.get("k0_monoisotopic_mass_da", float("nan")),
            "cosine_similarity":    match.get("k0_cosine_similarity",    float("nan")),
            "adjusted_score":               match["adjusted_score"],
            "gap_fraction":                 match["gap_fraction"],
            "unexplained_peak_fraction":    match["unexplained_peak_fraction"],
            "fit_quality":                  match["fit_quality"],
            "n_isotope_peaks_observed":     match["n_isotope_peaks_observed"],
            "peak_intensities":             match.get("k0_peak_intensities", ""),
        })

        # k=1 row — present only when the left-shifted candidate was evaluated
        # (i.e. left_ratio > threshold).  The monoisotopic_mz and cosine differ
        # from k=0 (different stick position), but cluster quality metrics
        # (gap_fraction, unexplained_peak_fraction, fit_quality, adjusted_score,
        # n_isotope_peaks_observed) describe the same underlying cluster and are
        # identical to k=0.
        _k1_mz = match.get("k1_monoisotopic_mz", float("nan"))
        if not (isinstance(_k1_mz, float) and np.isnan(_k1_mz)):
            clusters.append({
                **_shared,
                "k": 1,
                "monoisotopic_mz":           match.get("k1_monoisotopic_mz",      float("nan")),
                "monoisotopic_mass_da":      match.get("k1_monoisotopic_mass_da", float("nan")),
                "cosine_similarity":         match.get("k1_cosine_similarity",    float("nan")),
                "adjusted_score":            match["adjusted_score"],
                "gap_fraction":              match["gap_fraction"],
                "unexplained_peak_fraction": match["unexplained_peak_fraction"],
                "fit_quality":               match["fit_quality"],
                "n_isotope_peaks_observed":  match["n_isotope_peaks_observed"],
                "peak_intensities":          match.get("k1_peak_intensities", ""),
            })

        # Save cluster plot
        if output_dir is not None:
            # Filename uses RT/DT centers and monoisotopic_mz (k=0 position).
            # Format: RT{center:.1f}_DT{center:.1f}_mz{mono:.3f}_Factor{N:02d}_cluster{N:02d}_charge{z}.{ext}
            # Both k=0 and k=1 rows of the same cluster share a single image
            # (the plot encodes the left-shifted flag in its title/label).
            _mono_mz_k0 = match.get("k0_monoisotopic_mz", match.get("monoisotopic_mz", 0.0))
            _fname = (
                f"RT{rt_center:.1f}"
                f"_DT{dt_center:.1f}"
                f"_mz{_mono_mz_k0:.3f}"
                f"_Factor{factor_idx:02d}"
                f"_cluster{ci:02d}"
                f"_charge{match['charge']}"
                f".{plot_format}"
            )
            plot_path = os.path.join(output_dir, _fname)
            try:
                plot_isotopic_cluster(
                    match, A_r, B_r, C_r,
                    rt_axis, dt_axis, mz_axis,
                    output_path=plot_path,
                    factor_idx=factor_idx,
                    cluster_idx=ci,
                    mz_axis_full=mz_axis_full,
                    mask_mz=mask_mz,
                    png_dpi=png_dpi,
                )
            except Exception as exc:
                warnings.warn(
                    f"plot_isotopic_cluster failed for factor {factor_idx} "
                    f"cluster {ci}: {exc}",
                    RuntimeWarning,
                )

    return clusters


# ===========================================================================
# Section 6 — Process all factors → DataFrame
# ===========================================================================

def process_all_factors(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    mz_axis: np.ndarray,
    rt_axis: np.ndarray,
    dt_axis: np.ndarray,
    charge_range: Tuple[int, int] = (3, 15),
    min_cosine: float = 0.5,
    min_peaks_per_cluster: int = 3,
    min_height_frac: float = 0.02,
    min_distance_da: float = 0.04,
    max_gap_da: float = 3.0,
    tol_da: float = 0.01,
    output_dir: Optional[str] = None,
    verbose: bool = False,
    mz_axis_full: Optional[np.ndarray] = None,
    mask_mz: Optional[np.ndarray] = None,
    png_dpi: int = 100,
    plot_format: str = "png",
) -> pd.DataFrame:
    """Run isotope analysis on all NTF factors.

    Parameters
    ----------
    A : (n_rt, R) — RT factor matrix
    B : (n_dt, R) — DT factor matrix
    C : (n_mz, R) — m/z factor matrix
    mz_axis : (n_mz,)
        **Must be the masked m/z axis that matches C.shape[0].**
        Use ``result['mz_axis_ntf']`` from ``analyze_chunk``, NOT
        ``result['mz_axis']`` (the full axis).  Passing the wrong axis causes
        all charge-consistency checks to fail silently → 0 clusters found.
    rt_axis, dt_axis :
        Axis arrays matching A.shape[0] and B.shape[0] respectively.
        Use ``result['rt_axis_ntf']`` and ``result['dt_axis_ntf']``.
    charge_range :
        (min_z, max_z) inclusive.  Default (3, 15).
    min_cosine :
        Minimum cosine similarity threshold.
    min_peaks_per_cluster :
        Minimum observed peaks per cluster.
    min_height_frac :
        Peak-detection noise floor (fraction of profile max).
    max_gap_da :
        Maximum m/z gap within one cluster.
    tol_da :
        Bin-search tolerance [Da] for intensity extraction.
    output_dir :
        Directory for per-cluster plots.  None → no plots saved.
    verbose :
        Print progress.
    plot_format :
        Image format: "png" (default) or "pdf".

    Returns
    -------
    df : pandas DataFrame
        Columns: factor_idx, cluster_idx, charge, monoisotopic_mz,
                 monoisotopic_mass_da, rt_center, dt_center,
                 cosine_similarity, n_isotope_peaks_observed,
                 cluster_intensity, intensity_fraction, peak_rmse.
        Sorted by factor_idx, then monoisotopic_mz.
    """
    R = A.shape[1]
    all_records: List[Dict] = []

    for r in range(R):
        if verbose:
            print(f"[process_all_factors] Factor {r}/{R-1} …")
        clusters = find_isotopic_clusters(
            A[:, r], B[:, r], C[:, r],
            mz_axis=mz_axis,
            rt_axis=rt_axis,
            dt_axis=dt_axis,
            charge_range=charge_range,
            min_cosine=min_cosine,
            min_peaks_per_cluster=min_peaks_per_cluster,
            min_height_frac=min_height_frac,
            min_distance_da=min_distance_da,
            max_gap_da=max_gap_da,
            tol_da=tol_da,
            factor_idx=r,
            output_dir=output_dir,
            mz_axis_full=mz_axis_full,
            mask_mz=mask_mz,
            png_dpi=png_dpi,
            plot_format=plot_format,
        )
        all_records.extend(clusters)
        if verbose:
            print(f"  → {len(clusters)} cluster(s) found")

    if not all_records:
        return pd.DataFrame(columns=[
            "factor_idx", "cluster_idx", "charge",
            "monoisotopic_mz", "monoisotopic_mass_da",
            "rt_center", "dt_center",
            "cosine_similarity", "n_isotope_peaks_observed",
            "cluster_intensity", "intensity_fraction", "peak_rmse",
        ])

    df = pd.DataFrame(all_records)
    # Sort so that within each (factor, cluster) group k=0 always comes
    # before k=1, and groups are ordered by factor then by cluster.
    df = df.sort_values(["factor_idx", "cluster_idx", "k"]).reset_index(drop=True)
    return df

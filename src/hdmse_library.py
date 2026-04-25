# src/hdmse_library.py
"""
hdmse_library.py
================
Pure-NumPy anchor-and-project routines that build a pseudo-MS2 fragment
library from HDMS^E data.

This module never touches the Waters SDK — all functions take in-memory
NumPy arrays produced by ``tensor_analysis`` / ``isotope_analysis`` and
``waters_reader``. That separation lets the algorithmic core be unit-tested
on ARM laptops while the SDK-facing orchestration in ``hdmse_pipeline``
runs only inside the x86-64 Singularity container.

Phases (per the design spec)
----------------------------
* Phase 2 — ``extract_anchor``: RT/DT centroid, sigma, and L2-normalized
            RT/DT vectors (a_norm, b_norm) from one LCE factor.
* Phase 3 — ``project_anchor_onto_hce``: weighted projection of the
            factor's outer-product RT×DT signature onto an HCE m/z slab,
            yielding a per-m/z Pearson rho and projected intensity profile.
* Phase 3 — ``extract_fragments``: peak detection on the projected
            intensity profile, gated by the Pearson rho threshold.
* Phase 4 — ``build_library_row`` / ``LIBRARY_SCHEMA`` /
            ``write_library_parquet``: assemble the per-precursor records
            into a target-decoy-ready Parquet library.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.signal import find_peaks as _find_peaks


# ---------------------------------------------------------------------------
# Phase 2 — anchor extraction
# ---------------------------------------------------------------------------

def extract_anchor(
    a_vec: np.ndarray,
    b_vec: np.ndarray,
    rt_axis: np.ndarray,
    dt_axis: np.ndarray,
) -> Dict[str, object]:
    """Compute RT/DT centroid and sigma of a single LCE factor.

    Uses intensity-weighted first and second moments rather than refitting
    a Gaussian. For non-negative profiles this is exact for an ideal
    Gaussian and robust to mild peak asymmetry; it also avoids a SciPy
    curve_fit call per factor.

    Parameters
    ----------
    a_vec, b_vec :
        Non-negative RT and DT profiles for one factor (typically a column
        of the NTF ``A`` and ``B`` matrices).
    rt_axis, dt_axis :
        Coordinates corresponding to ``a_vec`` and ``b_vec``.

    Returns
    -------
    dict with keys
        ``rt_center``, ``rt_sigma`` : float — RT centroid and stddev (min)
        ``dt_center``, ``dt_sigma`` : float — DT centroid and stddev (bins)
        ``a_norm`` : ndarray of float32, shape (n_rt,) — L2-normalized a_vec
        ``b_norm`` : ndarray of float32, shape (n_dt,) — L2-normalized b_vec
    """
    a = np.asarray(a_vec, dtype=np.float64)
    b = np.asarray(b_vec, dtype=np.float64)
    rt = np.asarray(rt_axis, dtype=np.float64)
    dt = np.asarray(dt_axis, dtype=np.float64)

    a_sum = a.sum()
    b_sum = b.sum()
    if a_sum <= 0 or b_sum <= 0:
        raise ValueError("extract_anchor: a_vec or b_vec has non-positive sum")

    rt_center = np.dot(a, rt) / a_sum
    dt_center = np.dot(b, dt) / b_sum
    rt_var = np.dot(a, (rt - rt_center) ** 2) / a_sum
    dt_var = np.dot(b, (dt - dt_center) ** 2) / b_sum
    rt_sigma = np.sqrt(max(rt_var, 0.0))
    dt_sigma = np.sqrt(max(dt_var, 0.0))

    a_norm = (a / np.linalg.norm(a)).astype(np.float32)
    b_norm = (b / np.linalg.norm(b)).astype(np.float32)

    return dict(
        rt_center=rt_center, rt_sigma=rt_sigma,
        dt_center=dt_center, dt_sigma=dt_sigma,
        a_norm=a_norm, b_norm=b_norm,
    )


# ---------------------------------------------------------------------------
# Phase 3 — projection and correlation
# ---------------------------------------------------------------------------

def project_anchor_onto_hce(
    hce_tensor: np.ndarray,
    anchor: Dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Project an LCE anchor onto an HCE tensor — vectorised over m/z.

    For an anchor with L2-normalized RT vector ``a`` (shape n_rt) and DT
    vector ``b`` (shape n_dt) and an HCE tensor ``H`` of shape
    ``(n_rt, n_dt, n_mz)``, returns two length-``n_mz`` arrays:

        rho[k]       = Pearson(vec(H[:,:,k]), vec(a ⊗ b))
        intensity[k] = sum_{i,j} H[i,j,k] * a[i] * b[j]

    Implementation uses a single ``(n_rt*n_dt, n_mz)`` reshape and one
    matrix multiplication for intensity, plus three reductions for the
    Pearson numerator/denominator. Pearson is defined as 0 (not NaN) for
    bins whose RT×DT slice is constant (e.g. all-zero).

    Parameters
    ----------
    hce_tensor : ndarray of float32, shape (n_rt, n_dt, n_mz)
    anchor : dict with ``a_norm`` (n_rt,) and ``b_norm`` (n_dt,) —
             L2-normalized; ``extract_anchor`` already normalizes.

    Returns
    -------
    rho : ndarray of float32, shape (n_mz,) — Pearson correlation per m/z bin
    intensity : ndarray of float32, shape (n_mz,) — projected intensity per m/z bin
    """
    H = np.asarray(hce_tensor, dtype=np.float32)
    a = np.asarray(anchor["a_norm"], dtype=np.float32)
    b = np.asarray(anchor["b_norm"], dtype=np.float32)

    n_rt, n_dt, n_mz = H.shape
    if a.shape[0] != n_rt or b.shape[0] != n_dt:
        raise ValueError(
            f"Anchor shape ({a.shape}, {b.shape}) does not match "
            f"HCE tensor RT/DT dims ({n_rt}, {n_dt})."
        )

    anchor_flat = np.outer(a, b).reshape(-1).astype(np.float32)  # (N,)
    H_flat = H.reshape(n_rt * n_dt, n_mz)                        # (N, n_mz)
    n = float(anchor_flat.size)

    sum_ah = anchor_flat @ H_flat                                  # (n_mz,)
    intensity = sum_ah.astype(np.float32)                         # (n_mz,)

    # Pearson: rho = (n·Σxy − Σx·Σy) / sqrt((n·Σx² − (Σx)²)·(n·Σy² − (Σy)²))
    sum_a = float(anchor_flat.sum())
    sum_a2 = float((anchor_flat * anchor_flat).sum())
    sum_h = H_flat.sum(axis=0)              # (n_mz,)
    sum_h2 = (H_flat * H_flat).sum(axis=0)  # (n_mz,)

    num = n * sum_ah - sum_a * sum_h
    den_a = n * sum_a2 - sum_a * sum_a
    den_h = n * sum_h2 - sum_h * sum_h
    den = np.sqrt(np.maximum(den_a * den_h, 0.0))

    rho = np.zeros(n_mz, dtype=np.float32)
    nonzero = den > 0
    rho[nonzero] = (num[nonzero] / den[nonzero]).astype(np.float32)

    return rho, intensity


# ---------------------------------------------------------------------------
# Phase 3 — fragment extraction
# ---------------------------------------------------------------------------

def extract_fragments(
    mz_axis: np.ndarray,
    rho: np.ndarray,
    intensity: np.ndarray,
    rho_threshold: float = 0.85,
    min_intensity: float = 0.0,
    peak_distance_da: float = 0.5,
) -> List[Dict[str, float]]:
    """Detect fragment peaks on the anchor-projected HCE intensity profile.

    A peak is accepted only if BOTH:
      * its projected intensity is a local maximum at least *min_intensity*
        tall, separated from the nearest kept peak by *peak_distance_da* Da
      * the maximum Pearson rho within ±*peak_distance_da*/2 around the peak
        is at least *rho_threshold* (the neighborhood check tolerates 1-2 bin
        offsets between the intensity and rho maxima)

    Parameters
    ----------
    mz_axis : ndarray of float32, shape (n_mz,)
    rho : ndarray of float32, shape (n_mz,) — from ``project_anchor_onto_hce``
    intensity : ndarray of float32, shape (n_mz,) — from ``project_anchor_onto_hce``
    rho_threshold : minimum Pearson rho to accept a peak (default 0.85)
    min_intensity : minimum projected intensity (default 0)
    peak_distance_da : minimum m/z separation between kept peaks (Da)

    Returns
    -------
    list of dicts with keys ``mz``, ``intensity``, ``rho``, sorted by
    descending intensity.
    """
    mz = np.asarray(mz_axis, dtype=np.float64)
    rho_a = np.asarray(rho, dtype=np.float64)
    inten = np.asarray(intensity, dtype=np.float64)
    if mz.shape != rho_a.shape or mz.shape != inten.shape:
        raise ValueError("mz_axis, rho, and intensity must have identical shape")

    if len(mz) == 0 or inten.size == 0:
        return []
    if inten.max() <= 0:
        return []

    bin_da = float(np.median(np.diff(mz))) if len(mz) > 1 else peak_distance_da
    distance_bins = max(1, int(round(peak_distance_da / max(bin_da, 1e-12))))

    peak_idx, _ = _find_peaks(inten, height=min_intensity, distance=distance_bins)
    if len(peak_idx) == 0:
        return []

    halfwidth_bins = max(1, distance_bins // 2)
    fragments: List[Dict[str, float]] = []
    for k in peak_idx:
        lo = max(0, int(k) - halfwidth_bins)
        hi = min(len(mz), int(k) + halfwidth_bins + 1)
        local_rho = float(rho_a[lo:hi].max())
        if local_rho < rho_threshold:
            continue
        fragments.append(dict(
            mz=float(mz[k]),
            intensity=float(inten[k]),
            rho=local_rho,
        ))

    fragments.sort(key=lambda f: f["intensity"], reverse=True)
    return fragments


# ---------------------------------------------------------------------------
# Phase 4 — pseudo-MS2 library assembly
# ---------------------------------------------------------------------------

LIBRARY_SCHEMA: tuple = (
    "sample",
    "obs_mz",
    "charge",
    "MW",
    "RT",
    "im_mono",
    "ab_cluster_total",
    "lce_factor_idx",
    "lce_cluster_idx",
    "lce_cosine_similarity",
    "rt_sigma_min",
    "dt_sigma_bins",
    "n_fragments",
    "fragments",
)


def build_library_row(
    sample: str,
    anchor: Dict[str, object],
    precursor: Dict[str, object],
    fragments: List[Dict[str, float]],
) -> Dict[str, object]:
    """Assemble one library row from anchor + precursor + fragment list.

    Parameters
    ----------
    sample : source raw-file stem
    anchor : output of ``extract_anchor``
    precursor : per-factor record from ``isotope_analysis.process_all_factors``
                (keys: ``factor_idx``, ``cluster_idx``, ``charge``,
                ``monoisotopic_mz``, ``monoisotopic_mass_da``,
                ``cluster_intensity``, ``cosine_similarity``)
    fragments : list of ``{mz, intensity, rho}`` dicts from ``extract_fragments``

    Returns
    -------
    dict whose keys are exactly LIBRARY_SCHEMA
    """
    row = dict(
        sample=str(sample),
        obs_mz=float(precursor["monoisotopic_mz"]),
        charge=int(precursor["charge"]),
        MW=float(precursor["monoisotopic_mass_da"]),
        RT=float(anchor["rt_center"]),
        im_mono=float(anchor["dt_center"]),
        ab_cluster_total=float(precursor["cluster_intensity"]),
        lce_factor_idx=int(precursor["factor_idx"]),
        lce_cluster_idx=int(precursor["cluster_idx"]),
        lce_cosine_similarity=float(precursor["cosine_similarity"]),
        rt_sigma_min=float(anchor["rt_sigma"]),
        dt_sigma_bins=float(anchor["dt_sigma"]),
        n_fragments=int(len(fragments)),
        fragments=list(fragments),
    )
    if set(row.keys()) != set(LIBRARY_SCHEMA):
        raise RuntimeError(
            f"build_library_row schema mismatch: {set(row.keys())} vs {set(LIBRARY_SCHEMA)}"
        )
    return row


def write_library_parquet(rows: List[Dict[str, object]], output_path: str) -> None:
    """Write library rows to a Parquet file via PyArrow.

    The ``fragments`` column is stored as a list-of-struct so per-fragment
    ``mz``, ``intensity``, ``rho`` are preserved without row explosion.
    An empty ``rows`` list produces an empty Parquet with the correct schema.
    """
    df = pd.DataFrame(rows, columns=list(LIBRARY_SCHEMA))
    df.to_parquet(output_path, engine="pyarrow", index=False)

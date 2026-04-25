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

from typing import Dict

import numpy as np


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

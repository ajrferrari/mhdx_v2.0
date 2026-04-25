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
* Phase 2 — ``extract_anchor``: RT/DT centroid + sigma from one LCE factor.
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
        ``a_norm`` : float32 (n_rt,) — L2-normalized a_vec
        ``b_norm`` : float32 (n_dt,) — L2-normalized b_vec
    """
    a = np.asarray(a_vec, dtype=np.float64)
    b = np.asarray(b_vec, dtype=np.float64)
    rt = np.asarray(rt_axis, dtype=np.float64)
    dt = np.asarray(dt_axis, dtype=np.float64)

    a_sum = float(a.sum())
    b_sum = float(b.sum())
    if a_sum <= 0 or b_sum <= 0:
        raise ValueError("extract_anchor: a_vec or b_vec has non-positive sum")

    rt_center = float(np.dot(a, rt) / a_sum)
    dt_center = float(np.dot(b, dt) / b_sum)
    rt_var = float(np.dot(a, (rt - rt_center) ** 2) / a_sum)
    dt_var = float(np.dot(b, (dt - dt_center) ** 2) / b_sum)
    rt_sigma = float(np.sqrt(max(rt_var, 0.0)))
    dt_sigma = float(np.sqrt(max(dt_var, 0.0)))

    a_norm = (a / np.linalg.norm(a)).astype(np.float32)
    b_norm = (b / np.linalg.norm(b)).astype(np.float32)

    return dict(
        rt_center=rt_center, rt_sigma=rt_sigma,
        dt_center=dt_center, dt_sigma=dt_sigma,
        a_norm=a_norm, b_norm=b_norm,
    )

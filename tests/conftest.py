# tests/conftest.py
"""Shared fixtures for hdmse_library tests.

All tensors are built in pure NumPy with planted Gaussian anchors so we can
verify the anchor-and-project pipeline against known ground truth without the
Waters SDK (which is x86-64 only).
"""
from __future__ import annotations

import numpy as np
import pytest


def _gauss_1d(x: np.ndarray, center: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


@pytest.fixture
def axes():
    """Common RT, DT, m/z axes used by both LCE and HCE synthetic tensors."""
    rt_axis = np.linspace(5.0, 6.0, 60, dtype=np.float32)        # min, 60 scans
    dt_axis = np.arange(50, 100, dtype=np.float32)                # 50 drift bins
    mz_lce = np.linspace(800.0, 850.0, 5001, dtype=np.float32)   # 0.01 Da bin
    mz_hce = np.linspace(100.0, 1500.0, 14001, dtype=np.float32) # 0.1 Da bin
    return dict(rt=rt_axis, dt=dt_axis, mz_lce=mz_lce, mz_hce=mz_hce)


@pytest.fixture
def planted_precursor_anchor(axes):
    """Single rank-1 LCE factor planted with a known RT and DT Gaussian."""
    rt_center, rt_sigma = 5.45, 0.08
    dt_center, dt_sigma = 72.0, 4.0
    a_vec = _gauss_1d(axes["rt"], rt_center, rt_sigma).astype(np.float32)
    b_vec = _gauss_1d(axes["dt"], dt_center, dt_sigma).astype(np.float32)
    return dict(
        a_vec=a_vec, b_vec=b_vec,
        rt_center=rt_center, rt_sigma=rt_sigma,
        dt_center=dt_center, dt_sigma=dt_sigma,
    )


@pytest.fixture
def hce_tensor_with_three_fragments(axes, planted_precursor_anchor):
    """HCE tensor (n_rt, n_dt, n_mz_hce) with three planted fragments:
        * fragment_a at m/z 250.000 — co-elutes with precursor (rho ≈ 1.0)
        * fragment_b at m/z 600.000 — co-elutes with precursor (rho ≈ 1.0)
        * fragment_c at m/z 900.000 — RT-shifted by 0.20 min (chimera, rho ≪ 0.85)
    Each fragment's RT×DT slice is a unit-normalized Gaussian outer product
    multiplied by the fragment's intrinsic intensity.
    """
    rt = axes["rt"]; dt = axes["dt"]; mz = axes["mz_hce"]
    a = planted_precursor_anchor["a_vec"]
    b = planted_precursor_anchor["b_vec"]
    n_rt, n_dt, n_mz = len(rt), len(dt), len(mz)
    tensor = np.zeros((n_rt, n_dt, n_mz), dtype=np.float32)

    def _deposit(mz_target: float, intensity: float, a_use: np.ndarray, b_use: np.ndarray) -> None:
        # Centroid m/z deposited as ±0.05 Da Gaussian (≈ instrument peak shape)
        c_vec = _gauss_1d(mz, mz_target, 0.05).astype(np.float32)
        tensor[:, :, :] += intensity * np.einsum("i,j,k->ijk", a_use, b_use, c_vec)

    _deposit(250.000, 1.0e4, a, b)
    _deposit(600.000, 5.0e3, a, b)
    # Chimera: shifted RT center by 0.20 min
    a_shifted = _gauss_1d(rt, planted_precursor_anchor["rt_center"] + 0.20,
                          planted_precursor_anchor["rt_sigma"]).astype(np.float32)
    _deposit(900.000, 7.0e3, a_shifted, b)

    # Add small flat noise so correlation isn't pathologically perfect everywhere
    rng = np.random.default_rng(0)
    tensor += rng.normal(0.0, 1.0, tensor.shape).astype(np.float32) * 0.5
    np.maximum(tensor, 0.0, out=tensor)
    return tensor


@pytest.fixture
def planted_precursor_record():
    """Mock precursor record matching what isotope_analysis.process_all_factors
    emits for one (factor, charge, monoisotopic_mz) assignment."""
    return dict(
        factor_idx=0,
        cluster_idx=0,
        charge=8,
        monoisotopic_mz=812.345,
        monoisotopic_mass_da=6490.708,
        cluster_intensity=1.5e5,
        cosine_similarity=0.94,
        rt_center=5.45,
        dt_center=72.0,
    )

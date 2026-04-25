# tests/test_hdmse_library.py
"""Unit tests for src/hdmse_library.py — pure-numpy library functions.

The tests run on ARM (no Waters SDK) using fixtures from tests/conftest.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from hdmse_library import extract_anchor, project_anchor_onto_hce


def test_extract_anchor_raises_on_zero_vec(axes):
    zero = np.zeros(len(axes["rt"]), dtype=np.float32)
    valid_b = np.ones(len(axes["dt"]), dtype=np.float32)
    with pytest.raises(ValueError, match="extract_anchor"):
        extract_anchor(
            a_vec=zero,
            b_vec=valid_b,
            rt_axis=axes["rt"],
            dt_axis=axes["dt"],
        )


def test_extract_anchor_truncated_gaussian(axes):
    """Truncated profile (peak near axis edge) should not raise and should
    return normalized vectors with centroid within the axis bounds."""
    # Gaussian centered at axis start → heavily left-truncated
    rt = axes["rt"]
    dt = axes["dt"]
    a_trunc = np.exp(-0.5 * ((rt - rt[0]) / 0.05) ** 2).astype(np.float32)
    b_trunc = np.exp(-0.5 * ((dt - dt[10]) / 3.0) ** 2).astype(np.float32)
    anchor = extract_anchor(a_vec=a_trunc, b_vec=b_trunc, rt_axis=rt, dt_axis=dt)
    assert rt[0] <= anchor["rt_center"] <= rt[-1]
    assert dt[0] <= anchor["dt_center"] <= dt[-1]
    assert np.linalg.norm(anchor["a_norm"]) == pytest.approx(1.0, abs=1e-5)
    assert np.linalg.norm(anchor["b_norm"]) == pytest.approx(1.0, abs=1e-5)


def test_extract_anchor_recovers_planted_centroids(axes, planted_precursor_anchor):
    anchor = extract_anchor(
        a_vec=planted_precursor_anchor["a_vec"],
        b_vec=planted_precursor_anchor["b_vec"],
        rt_axis=axes["rt"],
        dt_axis=axes["dt"],
    )
    # Intensity-weighted centroid of a Gaussian recovers its center to ~1e-3
    assert anchor["rt_center"] == pytest.approx(planted_precursor_anchor["rt_center"], abs=1e-3)
    assert anchor["dt_center"] == pytest.approx(planted_precursor_anchor["dt_center"], abs=1e-2)
    # Sigma recovered to ~5% (weighted-moment estimator is consistent)
    assert anchor["rt_sigma"] == pytest.approx(planted_precursor_anchor["rt_sigma"], rel=0.05)
    assert anchor["dt_sigma"] == pytest.approx(planted_precursor_anchor["dt_sigma"], rel=0.05)
    # Anchor vectors are L2-normalized so projection magnitudes are comparable
    assert np.linalg.norm(anchor["a_norm"]) == pytest.approx(1.0, abs=1e-5)
    assert np.linalg.norm(anchor["b_norm"]) == pytest.approx(1.0, abs=1e-5)


def test_project_anchor_onto_hce_high_rho_at_planted_fragments(
    axes, planted_precursor_anchor, hce_tensor_with_three_fragments,
):
    anchor = dict(
        a_norm=planted_precursor_anchor["a_vec"] /
               np.linalg.norm(planted_precursor_anchor["a_vec"]),
        b_norm=planted_precursor_anchor["b_vec"] /
               np.linalg.norm(planted_precursor_anchor["b_vec"]),
    )
    rho, intensity = project_anchor_onto_hce(
        hce_tensor=hce_tensor_with_three_fragments,
        anchor=anchor,
    )
    mz_hce = axes["mz_hce"]
    assert rho.shape == (len(mz_hce),)
    assert intensity.shape == (len(mz_hce),)

    def _rho_near(mz_target: float, halfwidth_da: float = 0.2) -> float:
        mask = np.abs(mz_hce - mz_target) <= halfwidth_da
        return float(rho[mask].max())

    # Co-eluting fragments → rho close to 1
    assert _rho_near(250.000) > 0.95
    assert _rho_near(600.000) > 0.95
    # RT-shifted chimera → rho falls well below the 0.85 threshold
    assert _rho_near(900.000) < 0.80


def test_project_anchor_onto_hce_zero_intensity_bins_have_zero_rho(
    axes, planted_precursor_anchor,
):
    """For all-zero HCE bins rho must be 0 (not NaN), so downstream peak
    finding is well-behaved."""
    anchor = dict(
        a_norm=planted_precursor_anchor["a_vec"] /
               np.linalg.norm(planted_precursor_anchor["a_vec"]),
        b_norm=planted_precursor_anchor["b_vec"] /
               np.linalg.norm(planted_precursor_anchor["b_vec"]),
    )
    n_rt, n_dt, n_mz = len(axes["rt"]), len(axes["dt"]), 100
    zero_hce = np.zeros((n_rt, n_dt, n_mz), dtype=np.float32)
    rho, intensity = project_anchor_onto_hce(zero_hce, anchor)
    assert np.all(rho == 0.0)
    assert np.all(intensity == 0.0)

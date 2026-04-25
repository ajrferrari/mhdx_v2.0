# tests/test_hdmse_library.py
"""Unit tests for src/hdmse_library.py — pure-numpy library functions.

The tests run on ARM (no Waters SDK) using fixtures from tests/conftest.py.
"""
from __future__ import annotations

import numpy as np
import pytest

import pandas as pd

from hdmse_library import (
    LIBRARY_SCHEMA,
    build_library_row,
    extract_anchor,
    extract_fragments,
    project_anchor_onto_hce,
    write_library_parquet,
)


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


def test_extract_fragments_keeps_correlated_rejects_chimera(
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
    fragments = extract_fragments(
        mz_axis=axes["mz_hce"],
        rho=rho,
        intensity=intensity,
        rho_threshold=0.85,
        min_intensity=10.0,
        peak_distance_da=0.5,
    )
    mz_kept = sorted(f["mz"] for f in fragments)
    assert len(fragments) == 2
    assert mz_kept[0] == pytest.approx(250.000, abs=0.05)
    assert mz_kept[1] == pytest.approx(600.000, abs=0.05)
    for f in fragments:
        assert f["rho"] >= 0.85
        assert f["intensity"] > 0


def test_extract_fragments_returns_empty_when_threshold_above_all(
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
    fragments = extract_fragments(
        mz_axis=axes["mz_hce"],
        rho=rho,
        intensity=intensity,
        rho_threshold=1.1,   # above the maximum possible Pearson rho
        min_intensity=10.0,
        peak_distance_da=0.5,
    )
    assert fragments == []


def test_library_schema_includes_identification_compatible_columns():
    cols = set(LIBRARY_SCHEMA)
    assert {"obs_mz", "charge", "MW", "RT", "im_mono",
            "ab_cluster_total"}.issubset(cols)
    assert {"sample", "lce_factor_idx", "rt_sigma_min", "dt_sigma_bins",
            "n_fragments", "fragments"}.issubset(cols)


def test_build_library_row_assembles_precursor_and_fragments(
    planted_precursor_record, planted_precursor_anchor,
):
    fragments = [
        dict(mz=250.0, intensity=1.0e4, rho=0.99),
        dict(mz=600.0, intensity=5.0e3, rho=0.97),
    ]
    row = build_library_row(
        sample="260424_AF2501_04_0s",
        anchor=planted_precursor_anchor,
        precursor=planted_precursor_record,
        fragments=fragments,
    )
    assert row["obs_mz"] == pytest.approx(812.345)
    assert row["charge"] == 8
    assert row["MW"] == pytest.approx(6490.708)
    assert row["RT"] == pytest.approx(planted_precursor_anchor["rt_center"])
    assert row["im_mono"] == pytest.approx(planted_precursor_anchor["dt_center"])
    assert row["ab_cluster_total"] == pytest.approx(1.5e5)
    assert row["rt_sigma_min"] == pytest.approx(planted_precursor_anchor["rt_sigma"])
    assert row["dt_sigma_bins"] == pytest.approx(planted_precursor_anchor["dt_sigma"])
    assert row["n_fragments"] == 2
    assert row["fragments"] == fragments
    assert set(row.keys()) == set(LIBRARY_SCHEMA)


def test_write_library_parquet_roundtrips_fragments(
    tmp_path, planted_precursor_record, planted_precursor_anchor,
):
    fragments = [dict(mz=250.0, intensity=1.0e4, rho=0.99)]
    row = build_library_row(
        sample="x", anchor=planted_precursor_anchor,
        precursor=planted_precursor_record, fragments=fragments,
    )
    out = tmp_path / "library.parquet"
    write_library_parquet([row], str(out))
    loaded = pd.read_parquet(out)
    assert len(loaded) == 1
    loaded_frags = list(loaded.iloc[0]["fragments"])
    assert len(loaded_frags) == 1
    assert float(loaded_frags[0]["mz"]) == pytest.approx(250.0)
    assert float(loaded_frags[0]["rho"]) == pytest.approx(0.99)

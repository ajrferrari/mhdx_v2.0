# tests/test_hdmse_library.py
"""Unit tests for src/hdmse_library.py — pure-numpy library functions.

The tests run on ARM (no Waters SDK) using fixtures from tests/conftest.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from hdmse_library import extract_anchor


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

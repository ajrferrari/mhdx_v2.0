import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
import pytest
from pipeline import complete_charge_families

PROTON = 1.007276


def _row(mz, z, rt, dt, bpi, rt_r2=0.92, dt_r2=0.92, k=0):
    mw = mz * z - z * PROTON
    return dict(
        monoisotopic_mz=mz, charge=z, rt_center=rt, dt_center=dt,
        monoisotopic_mass_da=mw, cluster_bpi=bpi,
        rt_gaussian_r2=rt_r2, dt_gaussian_r2=dt_r2,
        is_best=True, cosine_similarity=0.9, k=k,
        sample="test", rt_lo=rt - 0.5, rt_hi=rt + 0.5,
        dt_lo=dt - 25, dt_hi=dt + 25,
    )


def test_recovers_sibling_charge_state():
    """z=6 passes filters; z=7 has rt_r2=0.78 (fails 0.80 post-filter) but same MW and RT."""
    filtered = pd.DataFrame([_row(584.78, 6, 5.0, 50.0, 1e5)])
    unfiltered = pd.DataFrame([
        _row(584.78, 6, 5.0, 50.0, 1e5),
        _row(501.39, 7, 5.0, 50.0, 8e4, rt_r2=0.78),   # fails post-filter
    ])
    result = complete_charge_families(filtered, unfiltered, mw_ppm=20.0, rt_tol=0.3)
    charges = set(result["charge"].values)
    assert 7 in charges, "z=7 sibling should be recovered"
    sib = result[result["charge"] == 7]
    assert sib["is_family_completion"].iloc[0] == True


def test_no_duplication_if_already_present():
    """z=7 is already in filtered; should not be added again."""
    filtered = pd.DataFrame([
        _row(584.78, 6, 5.0, 50.0, 1e5),
        _row(501.39, 7, 5.0, 50.0, 8e4),
    ])
    unfiltered = pd.DataFrame([
        _row(584.78, 6, 5.0, 50.0, 1e5),
        _row(501.39, 7, 5.0, 50.0, 8e4),
    ])
    result = complete_charge_families(filtered, unfiltered, mw_ppm=20.0, rt_tol=0.3)
    assert len(result[result["charge"] == 7]) == 1


def test_no_completion_without_anchor():
    """Orphan z=7 in unfiltered but no z=6 anchor in filtered — should not be added."""
    filtered = pd.DataFrame([_row(438.88, 8, 5.0, 50.0, 1e5)])  # different MW
    unfiltered = pd.DataFrame([
        _row(438.88, 8, 5.0, 50.0, 1e5),
        _row(501.39, 7, 5.0, 50.0, 8e4),  # different MW from anchor
    ])
    result = complete_charge_families(filtered, unfiltered, mw_ppm=20.0, rt_tol=0.3)
    # z=7 should not be added because it doesn't share MW with the z=8 anchor
    assert 7 not in result["charge"].values or len(result[result["charge"] == 7]) == 0


def test_rt_mismatch_not_added():
    """z=7 sibling at very different RT should not be added."""
    filtered = pd.DataFrame([_row(584.78, 6, 5.0, 50.0, 1e5)])
    unfiltered = pd.DataFrame([
        _row(584.78, 6, 5.0, 50.0, 1e5),
        _row(501.39, 7, 10.0, 50.0, 8e4),  # RT 5 min away
    ])
    result = complete_charge_families(filtered, unfiltered, mw_ppm=20.0, rt_tol=0.3)
    assert 7 not in result["charge"].values or len(result[result["charge"] == 7]) == 0


def test_is_family_completion_flag_false_for_original_rows():
    """Original filtered rows must have is_family_completion=False."""
    filtered = pd.DataFrame([_row(584.78, 6, 5.0, 50.0, 1e5)])
    unfiltered = pd.DataFrame([
        _row(584.78, 6, 5.0, 50.0, 1e5),
        _row(501.39, 7, 5.0, 50.0, 8e4, rt_r2=0.78),
    ])
    result = complete_charge_families(filtered, unfiltered, mw_ppm=20.0, rt_tol=0.3)
    original = result[result["charge"] == 6]
    assert original["is_family_completion"].iloc[0] == False

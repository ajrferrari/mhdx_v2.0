import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest
from diagnostic_gap_analysis import match_signals, charge_family_gaps


def _ref_row(name, rt, im, mw, z, obs_mz):
    return dict(name=name, RT=rt, im_mono=im, ab_cluster_total=1e6,
                MW=mw, charge=z, expect_mz=obs_mz, obs_mz=obs_mz,
                ppm=0.0, abs_ppm=0.0, cluster=1)


def _pipe_row(rt, z, mz, dt):
    return dict(rt_center=rt, charge=z, monoisotopic_mz=mz, dt_center=dt,
                cluster_bpi=1e5, rt_gaussian_r2=0.95, dt_gaussian_r2=0.95,
                monoisotopic_mass_da=(mz * z - z * 1.00728),
                is_best=True, k=0)


def test_exact_match():
    ref = pd.DataFrame([_ref_row("P1", 5.0, 50.0, 3500.0, 6, 584.78)])
    pipe = pd.DataFrame([_pipe_row(5.0, 6, 584.78, 50.0)])
    matched, unmatched = match_signals(ref, pipe, mz_ppm=10.0, rt_tol=0.3)
    assert len(matched) == 1
    assert len(unmatched) == 0


def test_ppm_miss():
    ref = pd.DataFrame([_ref_row("P1", 5.0, 50.0, 3500.0, 6, 584.78)])
    pipe = pd.DataFrame([_pipe_row(5.0, 6, 584.80, 50.0)])  # ~34 ppm off
    matched, unmatched = match_signals(ref, pipe, mz_ppm=10.0, rt_tol=0.3)
    assert len(matched) == 0
    assert len(unmatched) == 1


def test_rt_tolerance():
    ref = pd.DataFrame([_ref_row("P1", 5.0, 50.0, 3500.0, 6, 584.78)])
    # RT 0.29 min off — within tolerance
    pipe_near = pd.DataFrame([_pipe_row(5.29, 6, 584.78, 50.0)])
    matched, _ = match_signals(ref, pipe_near, mz_ppm=10.0, rt_tol=0.3)
    assert len(matched) == 1

    # RT 0.35 min off — outside tolerance
    pipe_far = pd.DataFrame([_pipe_row(5.35, 6, 584.78, 50.0)])
    matched, unmatched = match_signals(ref, pipe_far, mz_ppm=10.0, rt_tol=0.3)
    assert len(matched) == 0
    assert len(unmatched) == 1


def test_charge_mismatch():
    ref = pd.DataFrame([_ref_row("P1", 5.0, 50.0, 3500.0, 6, 584.78)])
    pipe = pd.DataFrame([_pipe_row(5.0, 7, 584.78, 50.0)])  # wrong charge
    matched, unmatched = match_signals(ref, pipe, mz_ppm=10.0, rt_tol=0.3)
    assert len(matched) == 0
    assert len(unmatched) == 1


def test_charge_family_gaps():
    # P1 detected at z=6 and z=7 in ref; pipeline only recovered z=6
    ref = pd.DataFrame([
        _ref_row("P1", 5.0, 50.0, 3500.0, 6, 584.78),
        _ref_row("P1", 5.0, 50.0, 3500.0, 7, 501.39),
    ])
    matched_names = {"P1"}
    gaps = charge_family_gaps(ref, matched_names, recovered_charges={"P1": {6}})
    assert "P1" in gaps
    assert gaps["P1"]["missing"] == {7}
    assert gaps["P1"]["found"] == {6}
    assert gaps["P1"]["all_ref"] == {6, 7}


def test_charge_family_no_gaps():
    # All charge states recovered
    ref = pd.DataFrame([
        _ref_row("P1", 5.0, 50.0, 3500.0, 6, 584.78),
        _ref_row("P1", 5.0, 50.0, 3500.0, 7, 501.39),
    ])
    matched_names = {"P1"}
    gaps = charge_family_gaps(ref, matched_names, recovered_charges={"P1": {6, 7}})
    assert "P1" not in gaps


def test_multiple_signals_one_match():
    ref = pd.DataFrame([
        _ref_row("P1", 5.0, 50.0, 3500.0, 6, 584.78),
        _ref_row("P2", 8.0, 60.0, 4000.0, 7, 572.15),
    ])
    pipe = pd.DataFrame([_pipe_row(5.0, 6, 584.78, 50.0)])
    matched, unmatched = match_signals(ref, pipe, mz_ppm=10.0, rt_tol=0.3)
    assert len(matched) == 1
    assert len(unmatched) == 1
    assert matched.iloc[0]["name"] == "P1"
    assert unmatched.iloc[0]["name"] == "P2"

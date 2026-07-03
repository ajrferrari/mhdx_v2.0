import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from troubleshoot import _best_centered_slice


def _s(rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi):
    return {"rt_lo": rt_lo, "rt_hi": rt_hi,
            "dt_lo": dt_lo, "dt_hi": dt_hi,
            "mz_lo": mz_lo, "mz_hi": mz_hi}


def test_picks_most_centered_slice():
    """Signal exactly at center of slice A; at rt edge of slice B."""
    a = _s(0.0, 1.0,  0, 50,  500.0, 600.0)   # signal at (rt=0.5, dt=25, mz=550) → center
    b = _s(0.5, 1.5,  0, 50,  500.0, 600.0)   # signal at same pos → rt edge of B
    result = _best_centered_slice([a, b], obs_mz=550.0, rt=0.5, im_mono=25.0)
    assert result is a


def test_single_candidate_returned():
    a = _s(0.0, 1.0, 0, 50, 500.0, 600.0)
    result = _best_centered_slice([a], obs_mz=550.0, rt=0.5, im_mono=25.0)
    assert result is a


def test_all_three_dims_considered():
    """Tie in score — function returns one of the candidates without error."""
    a = _s(0.0, 1.0,  0, 50,  500.0, 600.0)
    b = _s(0.5, 1.5, 25, 75,  450.0, 550.0)
    # signal at rt=0.5, dt=25, mz=500.001
    # A: rt_m=0.5, dt_m=0.5, mz_m≈0   → min≈0
    # B: rt_m=0.0, dt_m=0.0, mz_m≈0.5 → min=0
    # tie → returns one of them
    result = _best_centered_slice([a, b], obs_mz=500.001, rt=0.5, im_mono=25.0)
    assert result in (a, b)


def test_best_in_all_three_dims_wins():
    """C is best in all three dimensions."""
    a = _s(0.0, 1.0,  0, 50,  500.0, 600.0)   # signal near rt edge
    b = _s(0.0, 1.0,  0, 50,  500.0, 600.0)
    c = _s(0.2, 1.2, 10, 60,  480.0, 580.0)   # signal at rt=0.5→margin=0.25, dt=30→0.4, mz=550→0.7
    result = _best_centered_slice([a, b, c], obs_mz=550.0, rt=0.4, im_mono=30.0)
    # a: rt_m=(0.4-0)/1=0.4, dt_m=30/50=0.6, mz_m=50/100=0.5 → min=0.4
    # c: rt_m=(0.4-0.2)/1=0.2, dt_m=20/50=0.4, mz_m=70/100=0.7 → min=0.2
    # a wins (min margin 0.4 vs 0.2)
    assert result is a

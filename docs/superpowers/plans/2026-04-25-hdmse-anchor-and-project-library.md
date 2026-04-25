# HDMS^E Anchor-and-Project Pseudo-MS2 Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pseudo-MS2 spectral library from `data/260424_AF2501_04_0s.raw` (HDMS^E, Synapt) by reusing the existing LCE NTF decomposition (Function 0) as immutable RT/IM anchors, projecting those anchors onto the HCE tensor (Function 1), and writing fragments that satisfy a Pearson correlation threshold to a Parquet library compatible with the existing target-decoy identification protocol.

**Architecture:** Two new modules. `src/hdmse_library.py` holds the pure-NumPy/Pandas anchor extraction, HCE projection, correlation, fragment selection, and library DataFrame assembly — fully unit-testable on ARM (no SDK). `src/hdmse_pipeline.py` is the SDK-facing orchestrator: it opens a `WatersRawReader`, iterates 50 Da LCE windows through `tensor_analysis.analyze_chunk` and `isotope_analysis.process_all_factors` to obtain precursor characterizations, then loops HCE m/z slabs through `hdmse_library` to assemble per-precursor fragment lists, and writes a single Parquet library file. The orchestrator is only runnable inside the Singularity container (x86-64 + Waters SDK).

**Tech Stack:** Python 3.10+, NumPy, SciPy, Pandas, PyArrow (for Parquet), pytest. Reuses `tensor_analysis`, `isotope_analysis`, `waters_reader`. No new external dependencies beyond `pyarrow` (already a Pandas optional dep).

---

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `src/hdmse_library.py` | **create** | Pure functions: `extract_anchor`, `project_anchor_onto_hce`, `extract_fragments`, `build_library_row`, `write_library_parquet`, `LIBRARY_SCHEMA`. No SDK calls. |
| `src/hdmse_pipeline.py` | **create** | SDK-facing orchestrator: `process_lce_window`, `process_raw_file`, CLI entry points (`process_window`, `process_raw`). |
| `tests/__init__.py` | **create** | Empty marker. |
| `tests/conftest.py` | **create** | Synthetic LCE/HCE tensor fixtures with planted anchors and chimeric interference. |
| `tests/test_hdmse_library.py` | **create** | TDD coverage of every public function in `hdmse_library`. |
| `src/config.yaml` | **modify** (append `hdmse:` block) | New config section: `lce_function`, `hce_function`, `lce_mz_window`, `hce_mz_slab`, `rho_threshold`, `min_fragment_intensity`, `output_path`. |
| `CLAUDE.md` | **modify** (append "HDMS^E pseudo-MS2 library" section) | Document the new pipeline + invocation. |

**Why these splits:**
- The library logic is pure math on tensors — splitting it from the SDK orchestration lets us run real unit tests on the developer's ARM laptop. The SDK side is integration-tested only on the cluster.
- A single `hdmse_library.py` (rather than four micro-files) keeps the projection math readable end-to-end; total expected size ~350 LOC, well within the existing module sizing in this repo (`isotope_analysis.py` is 1841 LOC).

---

## Task 0: Test infrastructure with synthetic LCE/HCE fixtures

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Modify: nothing

The plan needs deterministic synthetic data because the only real `.raw` file (`data/260424_AF2501_04_0s.raw`) is unreadable on ARM. The fixtures plant a known precursor anchor in LCE and three fragments in HCE — two that share the precursor's RT/DT distribution exactly (correlated, should be kept) and one that has a shifted RT distribution (chimeric interference, should be rejected).

- [ ] **Step 1: Create empty package marker**

```python
# tests/__init__.py
```

- [ ] **Step 2: Write `conftest.py` with synthetic-tensor fixtures**

```python
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
```

- [ ] **Step 3: Verify pytest collects the fixtures (no tests yet, so collection is the success signal)**

Run: `cd /Users/allan/Work/Claude/mhdx_v2.0 && python -m pytest tests/ --collect-only -q`
Expected: `0 tests collected` (no tests yet) — and **no** import or syntax errors. If the user does not have pytest installed yet: `pip install pytest pyarrow`.

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: add synthetic LCE/HCE fixtures for hdmse pipeline TDD"
```

---

## Task 1: `extract_anchor` — RT/DT centroids and sigmas from one LCE factor

**Files:**
- Create: `src/hdmse_library.py` (initial skeleton + `extract_anchor` only)
- Test: `tests/test_hdmse_library.py`

The anchor is the immutable coordinate-prior on which HCE will be projected. We need the centroid and standard deviation of the factor's RT (a_vec) and DT (b_vec) profiles. Because the factors are already non-negative, we use the intensity-weighted moments — they are exact, robust to noise, and faster than re-fitting a Gaussian. No external optimization needed.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run the test — confirm it fails**

Run: `cd /Users/allan/Work/Claude/mhdx_v2.0 && PYTHONPATH=src python -m pytest tests/test_hdmse_library.py::test_extract_anchor_recovers_planted_centroids -v`
Expected: `ImportError: cannot import name 'extract_anchor' from 'hdmse_library'` (or `ModuleNotFoundError`).

- [ ] **Step 3: Implement `extract_anchor` in a new `src/hdmse_library.py`**

```python
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
```

- [ ] **Step 4: Run the test — confirm it passes**

Run: `cd /Users/allan/Work/Claude/mhdx_v2.0 && PYTHONPATH=src python -m pytest tests/test_hdmse_library.py::test_extract_anchor_recovers_planted_centroids -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hdmse_library.py tests/test_hdmse_library.py
git commit -m "feat: add extract_anchor for HDMS^E LCE factor centroid/sigma"
```

---

## Task 2: `project_anchor_onto_hce` — Pearson rho per HCE m/z bin

**Files:**
- Modify: `src/hdmse_library.py` (append `project_anchor_onto_hce`)
- Test: `tests/test_hdmse_library.py`

For one anchor and an HCE tensor `H` of shape `(n_rt, n_dt, n_mz)`, we compute two arrays of length `n_mz`:

1. **Projected intensity** `I_proj[k] = Σ_{i,j} H[i,j,k] · a_norm[i] · b_norm[j]` — the weighted intensity that survives the anchor's RT×DT footprint.
2. **Pearson rho** `ρ[k] = corr( vec(H[:,:,k]), vec(a_norm ⊗ b_norm) )` — how closely the HCE RT×DT slice at m/z bin k matches the LCE anchor's rank-1 shape.

Both are vectorised over `k` using a single matrix multiply by reshaping `H` to `(n_rt*n_dt, n_mz)`.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_hdmse_library.py
from hdmse_library import project_anchor_onto_hce


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
    """For all-zero HCE bins rho must be 0 (defined as "no co-elution")
    rather than NaN, so downstream peak finding is well-behaved."""
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
```

- [ ] **Step 2: Run tests — confirm they fail**

Run: `cd /Users/allan/Work/Claude/mhdx_v2.0 && PYTHONPATH=src python -m pytest tests/test_hdmse_library.py -v -k "project_anchor"`
Expected: FAIL with `ImportError: cannot import name 'project_anchor_onto_hce'`.

- [ ] **Step 3: Implement `project_anchor_onto_hce`**

Append to `src/hdmse_library.py`:

```python
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

    Implementation uses a single ``(n_rt*n_dt, n_mz)`` reshape and a single
    matrix multiplication for the intensity, plus three reductions for the
    Pearson numerator/denominator. Memory cost: one float32 copy of the
    tensor (already shared with the caller).

    Pearson is defined as 0 (not NaN) for HCE m/z bins whose RT×DT slice is
    constant (e.g. all-zero), so downstream peak finding is well-behaved.

    Parameters
    ----------
    hce_tensor : float32 (n_rt, n_dt, n_mz)
    anchor : dict with ``a_norm`` (n_rt,) and ``b_norm`` (n_dt,) — must be
             L2-normalized; ``extract_anchor`` already normalizes.

    Returns
    -------
    rho : float32 (n_mz,)        — Pearson correlation per m/z bin
    intensity : float32 (n_mz,)  — anchor-projected intensity per m/z bin
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

    # Flatten anchor outer product to length N = n_rt * n_dt
    anchor_flat = np.outer(a, b).reshape(-1).astype(np.float32)        # (N,)
    H_flat = H.reshape(n_rt * n_dt, n_mz)                              # (N, n_mz)
    n = float(anchor_flat.size)

    # Intensity is the dot product anchor_flat . H_flat (one GEMV per m/z)
    intensity = (anchor_flat @ H_flat).astype(np.float32)              # (n_mz,)

    # Pearson rho — vectorised over m/z. Standard formula:
    #     rho = (n*Σxy − Σx*Σy) / sqrt((n*Σx² − (Σx)²)(n*Σy² − (Σy)²))
    sum_a = float(anchor_flat.sum())
    sum_a2 = float((anchor_flat * anchor_flat).sum())
    sum_h = H_flat.sum(axis=0)                       # (n_mz,)
    sum_h2 = (H_flat * H_flat).sum(axis=0)           # (n_mz,)
    sum_ah = anchor_flat @ H_flat                    # (n_mz,)

    num = n * sum_ah - sum_a * sum_h
    den_a = n * sum_a2 - sum_a * sum_a
    den_h = n * sum_h2 - sum_h * sum_h
    den = np.sqrt(np.maximum(den_a * den_h, 0.0))

    rho = np.zeros(n_mz, dtype=np.float32)
    nonzero = den > 0
    rho[nonzero] = (num[nonzero] / den[nonzero]).astype(np.float32)

    return rho, intensity
```

- [ ] **Step 4: Run tests — confirm they pass**

Run: `cd /Users/allan/Work/Claude/mhdx_v2.0 && PYTHONPATH=src python -m pytest tests/test_hdmse_library.py -v -k "project_anchor"`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hdmse_library.py tests/test_hdmse_library.py
git commit -m "feat: add project_anchor_onto_hce with vectorised Pearson rho"
```

---

## Task 3: `extract_fragments` — peak finding gated by rho

**Files:**
- Modify: `src/hdmse_library.py`
- Test: `tests/test_hdmse_library.py`

Given the per-m/z `rho` and `intensity` arrays from Task 2, return a list of fragment peaks: only m/z bins where `rho >= rho_threshold` AND `intensity >= min_intensity`. Use SciPy `find_peaks` to keep local maxima only — no broad-shoulder smearing — and reject any peak whose **maximum local rho** within ±tolerance is below threshold.

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_hdmse_library.py
from hdmse_library import extract_fragments


def test_extract_fragments_keeps_correlated_rejects_chimera(
    axes, planted_precursor_anchor, hce_tensor_with_three_fragments,
):
    anchor = dict(
        a_norm=planted_precursor_anchor["a_vec"] /
               np.linalg.norm(planted_precursor_anchor["a_vec"]),
        b_norm=planted_precursor_anchor["b_vec"] /
               np.linalg.norm(planted_precursor_anchor["b_vec"]),
    )
    from hdmse_library import project_anchor_onto_hce
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
    # Two true fragments kept; chimera at 900 Da rejected by rho gate
    mz_kept = sorted(f["mz"] for f in fragments)
    assert len(fragments) == 2
    assert mz_kept[0] == pytest.approx(250.000, abs=0.05)
    assert mz_kept[1] == pytest.approx(600.000, abs=0.05)
    # Each fragment carries its rho and intensity
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
    from hdmse_library import project_anchor_onto_hce
    rho, intensity = project_anchor_onto_hce(
        hce_tensor=hce_tensor_with_three_fragments,
        anchor=anchor,
    )
    fragments = extract_fragments(
        mz_axis=axes["mz_hce"],
        rho=rho,
        intensity=intensity,
        rho_threshold=0.999,         # impossibly high
        min_intensity=10.0,
        peak_distance_da=0.5,
    )
    assert fragments == []
```

- [ ] **Step 2: Run tests — confirm they fail**

Run: `cd /Users/allan/Work/Claude/mhdx_v2.0 && PYTHONPATH=src python -m pytest tests/test_hdmse_library.py -v -k "extract_fragments"`
Expected: FAIL with `ImportError: cannot import name 'extract_fragments'`.

- [ ] **Step 3: Implement `extract_fragments`**

Append to `src/hdmse_library.py`:

```python
from typing import List

from scipy.signal import find_peaks as _find_peaks


def extract_fragments(
    mz_axis: np.ndarray,
    rho: np.ndarray,
    intensity: np.ndarray,
    rho_threshold: float = 0.85,
    min_intensity: float = 0.0,
    peak_distance_da: float = 0.5,
) -> List[Dict[str, float]]:
    """Detect fragment peaks on the anchor-projected HCE intensity profile.

    A fragment peak is accepted only if BOTH:
      * its projected intensity is a local maximum at least *min_intensity*
        tall and separated from any kept stronger peak by *peak_distance_da*
      * the maximum Pearson rho within ±*peak_distance_da*/2 around the peak
        m/z is at least *rho_threshold*

    The rho check uses the maximum rho within a small neighborhood (rather
    than the rho at the peak's exact bin) because the intensity peak and
    the rho peak can be 1-2 bins apart due to noise.

    Parameters
    ----------
    mz_axis : float32 (n_mz,)
    rho : float32 (n_mz,)        — output of ``project_anchor_onto_hce``
    intensity : float32 (n_mz,)  — output of ``project_anchor_onto_hce``
    rho_threshold : minimum Pearson rho for a peak to be kept (default 0.85)
    min_intensity : minimum projected intensity for a peak (default 0)
    peak_distance_da : minimum m/z separation between accepted peaks (Da)

    Returns
    -------
    fragments : list of dicts with keys ``mz``, ``intensity``, ``rho``,
                sorted by descending intensity.
    """
    mz = np.asarray(mz_axis, dtype=np.float64)
    rho_a = np.asarray(rho, dtype=np.float64)
    inten = np.asarray(intensity, dtype=np.float64)
    if mz.shape != rho_a.shape or mz.shape != inten.shape:
        raise ValueError("mz_axis, rho, and intensity must have identical shape")

    if inten.max() <= 0:
        return []

    # Convert peak_distance_da into bin units (axis is uniform for this caller's
    # use-case but we tolerate non-uniform by using median spacing).
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
```

- [ ] **Step 4: Run tests — confirm they pass**

Run: `cd /Users/allan/Work/Claude/mhdx_v2.0 && PYTHONPATH=src python -m pytest tests/test_hdmse_library.py -v -k "extract_fragments"`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hdmse_library.py tests/test_hdmse_library.py
git commit -m "feat: add extract_fragments with rho-gated peak detection"
```

---

## Task 4: Library schema, `build_library_row`, and Parquet writer

**Files:**
- Modify: `src/hdmse_library.py`
- Test: `tests/test_hdmse_library.py`

The library is a single Pandas DataFrame with one row per (precursor, factor) and a `fragments` column that holds a JSON-serializable list of `{mz, intensity, rho}` dicts. We store the per-fragment data nested rather than exploded so the file remains compact and a single row maps directly to one MS2 spectrum during target-decoy search. Parquet preserves nested lists via PyArrow.

The schema is designed for direct reuse by `protein_identification.py`: the precursor columns mirror its expected `obs_mz`, `charge`, `MW`, `RT`, `im_mono`, and `ab_cluster_total` inputs, so a thin adapter is enough to feed the library to the existing identification + decoy + FDR machinery.

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_hdmse_library.py
import pandas as pd

from hdmse_library import (
    LIBRARY_SCHEMA,
    build_library_row,
    write_library_parquet,
)


def test_library_schema_includes_identification_compatible_columns():
    cols = set(LIBRARY_SCHEMA)
    # Columns required by protein_identification.identify()
    assert {"obs_mz", "charge", "MW", "RT", "im_mono",
            "ab_cluster_total"}.issubset(cols)
    # New HDMS^E-specific columns
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
    # Precursor projections
    assert row["obs_mz"] == pytest.approx(812.345)
    assert row["charge"] == 8
    assert row["MW"] == pytest.approx(6490.708)
    assert row["RT"] == pytest.approx(planted_precursor_anchor["rt_center"])
    assert row["im_mono"] == pytest.approx(planted_precursor_anchor["dt_center"])
    assert row["ab_cluster_total"] == pytest.approx(1.5e5)
    # Anchor sigmas exposed for downstream QC
    assert row["rt_sigma_min"] == pytest.approx(planted_precursor_anchor["rt_sigma"])
    assert row["dt_sigma_bins"] == pytest.approx(planted_precursor_anchor["dt_sigma"])
    # Fragment payload preserved verbatim
    assert row["n_fragments"] == 2
    assert row["fragments"] == fragments
    # Schema completeness
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
    # PyArrow stores list-of-struct as a list of dicts (or list of pandas
    # records depending on engine); compare via list normalization.
    loaded_frags = list(loaded.iloc[0]["fragments"])
    assert len(loaded_frags) == 1
    assert float(loaded_frags[0]["mz"]) == pytest.approx(250.0)
    assert float(loaded_frags[0]["rho"]) == pytest.approx(0.99)
```

- [ ] **Step 2: Run tests — confirm they fail**

Run: `cd /Users/allan/Work/Claude/mhdx_v2.0 && PYTHONPATH=src python -m pytest tests/test_hdmse_library.py -v -k "library or write_library"`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement schema, `build_library_row`, and `write_library_parquet`**

Append to `src/hdmse_library.py`:

```python
# ---------------------------------------------------------------------------
# Phase 4 — pseudo-MS2 library assembly
# ---------------------------------------------------------------------------

# Column order is the on-disk schema for the Parquet library.
# Naming aligns with protein_identification.identify() so the same library
# can be fed through the existing target-decoy search with a minimal adapter.
LIBRARY_SCHEMA: tuple = (
    # Precursor identity (consumed by protein_identification)
    "sample",
    "obs_mz",                # observed monoisotopic m/z [Da]
    "charge",                # integer charge state
    "MW",                    # neutral monoisotopic mass [Da]
    "RT",                    # retention time at anchor centroid [min]
    "im_mono",               # ion-mobility centroid (drift bin or ms)
    "ab_cluster_total",      # cluster_intensity from isotope_analysis
    # Anchor provenance (kept for QC and rho re-thresholding downstream)
    "lce_factor_idx",
    "lce_cluster_idx",
    "lce_cosine_similarity",
    "rt_sigma_min",
    "dt_sigma_bins",
    # Fragment payload — list of {mz, intensity, rho}
    "n_fragments",
    "fragments",
)


def build_library_row(
    sample: str,
    anchor: Dict[str, object],
    precursor: Dict[str, object],
    fragments: List[Dict[str, float]],
) -> Dict[str, object]:
    """Assemble one library row from an anchor + precursor + fragment list.

    Parameters
    ----------
    sample :
        Source raw-file stem.
    anchor :
        Output of ``extract_anchor`` (uses ``rt_center``, ``dt_center``,
        ``rt_sigma``, ``dt_sigma``).
    precursor :
        Per-factor record from ``isotope_analysis.process_all_factors``.
        Required keys: ``factor_idx``, ``cluster_idx``, ``charge``,
        ``monoisotopic_mz``, ``monoisotopic_mass_da``, ``cluster_intensity``,
        ``cosine_similarity``.
    fragments :
        List of ``{mz, intensity, rho}`` dicts from ``extract_fragments``.

    Returns
    -------
    dict whose keys are exactly LIBRARY_SCHEMA.
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
    # Defensive check — keep the schema and the row dict in lock-step
    if set(row.keys()) != set(LIBRARY_SCHEMA):
        raise RuntimeError(
            f"build_library_row schema mismatch: {set(row.keys())} vs {set(LIBRARY_SCHEMA)}"
        )
    return row


def write_library_parquet(rows: List[Dict[str, object]], output_path: str) -> None:
    """Write a list of library rows to a Parquet file via PyArrow.

    Columns are emitted in ``LIBRARY_SCHEMA`` order; the ``fragments`` column
    is stored as a list-of-struct (preserves per-fragment ``mz``,
    ``intensity``, ``rho`` without exploding rows). Rows is allowed to be
    empty — an empty Parquet with the right schema is produced.
    """
    import pandas as pd
    df = pd.DataFrame(rows, columns=list(LIBRARY_SCHEMA))
    df.to_parquet(output_path, engine="pyarrow", index=False)
```

- [ ] **Step 4: Run tests — confirm they pass**

Run: `cd /Users/allan/Work/Claude/mhdx_v2.0 && PYTHONPATH=src python -m pytest tests/test_hdmse_library.py -v`
Expected: ALL tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hdmse_library.py tests/test_hdmse_library.py
git commit -m "feat: add LIBRARY_SCHEMA, build_library_row, write_library_parquet"
```

---

## Task 5: Pipeline orchestrator `hdmse_pipeline.py` — `process_lce_window`

**Files:**
- Create: `src/hdmse_pipeline.py`
- Test: none (this code calls the SDK; tested manually in Task 8)

This is the SDK-facing glue. It runs **inside** the Singularity container only. One function, `process_lce_window`, takes an open `WatersRawReader` plus an LCE m/z window and returns a list of library rows. The next task wires it into a CLI that loops m/z windows for the whole file.

For each LCE 50 Da window:
1. Build LCE tensor via `analyze_chunk(reader, function=lce_function, ...)` over full RT × full DT × this LCE m/z window.
2. Use `process_all_factors` to get the precursor records (one per detected isotopic cluster).
3. For each kept LCE factor, `extract_anchor`.
4. Iterate HCE m/z slabs (e.g. 100 Da each) covering the full HCE m/z range:
   - Build HCE tensor at the anchor's RT range × full DT × HCE slab m/z, via `tensor_analysis.build_tensor`.
   - `project_anchor_onto_hce` → `extract_fragments`. Concatenate fragments across slabs.
5. Build library rows via `build_library_row`.

Note on RT axis alignment: the HCE tensor must use the **same RT scan range** as the LCE tensor (both come from the file's RT axis at the same scan indices) so the anchor's `a_norm` aligns 1-to-1 with the HCE tensor's RT axis. `build_tensor` already clips by `rt_lo, rt_hi` against the SDK's per-function RT axis — for HDMS^E, both functions are acquired interleaved on the same instrument clock, so their per-scan RT axes are identical in length. Confirm this in Task 8's smoke test (assert `meta.n_scans[lce] == meta.n_scans[hce]`).

- [ ] **Step 1: Create `src/hdmse_pipeline.py`**

```python
# src/hdmse_pipeline.py
"""
hdmse_pipeline.py
=================
SDK-facing orchestrator for the HDMS^E anchor-and-project library pipeline.

This module imports the Waters SDK transitively via ``waters_reader`` and
therefore only runs on x86-64 Linux/Windows inside the Singularity
container. The pure-numpy library logic lives in ``hdmse_library`` and is
unit-tested separately.

Public functions
----------------
* ``process_lce_window`` — process one LCE m/z window for one open reader,
  returns a list of library rows.
* ``process_raw_file``   — iterate all LCE m/z windows for one .raw file
  and write a single Parquet library.

Both functions mutate no global state and write only the file specified in
``output_path``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from hdmse_library import (
    LIBRARY_SCHEMA,
    build_library_row,
    extract_anchor,
    extract_fragments,
    project_anchor_onto_hce,
    write_library_parquet,
)


def _full_dt_range(reader, function: int) -> Tuple[int, int]:
    """Return (dt_lo, dt_hi) covering every drift bin of *function*."""
    meta = reader.metadata()
    n_drift = meta.n_drift_bins[function]
    return 0, int(n_drift - 1)


def _full_rt_range(reader, function: int) -> Tuple[float, float]:
    rt_lo, rt_hi = reader.metadata().rt_range[function]
    return float(rt_lo), float(rt_hi)


def process_lce_window(
    reader,
    *,
    sample: str,
    lce_function: int,
    hce_function: int,
    lce_mz_lo: float,
    lce_mz_hi: float,
    hce_mz_lo: float,
    hce_mz_hi: float,
    hce_mz_slab: float,
    rho_threshold: float,
    min_fragment_intensity: float,
    peak_distance_da: float,
    # NTF / isotope params — passed through to existing modules
    rank_init: int,
    rank_max: int,
    n_restarts: int,
    rt_r2_min: float,
    dt_r2_min: float,
    charge_range: Tuple[int, int],
    min_cosine: float,
    min_peaks_per_cluster: int,
    # tensor build params
    mz_bin: float,
    gauss_sigma_rt: float,
    gauss_sigma_dt: float,
    intensity_floor: float,
    verbose: bool = False,
) -> List[Dict[str, object]]:
    """Run the anchor-and-project pipeline on one LCE m/z window.

    Steps
    -----
    1. ``tensor_analysis.analyze_chunk`` on (full RT × full DT × LCE window).
    2. ``isotope_analysis.process_all_factors`` → per-factor precursor rows.
    3. For each kept factor: ``extract_anchor`` from A[:, k], B[:, k].
    4. Iterate HCE m/z slabs: ``tensor_analysis.build_tensor`` →
       ``project_anchor_onto_hce`` → ``extract_fragments``. Concatenate.
    5. ``build_library_row`` per (factor, cluster) pair.

    Returns
    -------
    list of dicts in LIBRARY_SCHEMA shape (may be empty).
    """
    from tensor_analysis import analyze_chunk, build_tensor
    from isotope_analysis import process_all_factors

    rt_lo, rt_hi = _full_rt_range(reader, lce_function)
    dt_lo, dt_hi = _full_dt_range(reader, lce_function)

    # ---- Step 1: NTF on the LCE window -----------------------------------
    if verbose:
        print(f"[hdmse] LCE NTF  m/z={lce_mz_lo:.1f}-{lce_mz_hi:.1f}  RT={rt_lo:.2f}-{rt_hi:.2f}")
    lce = analyze_chunk(
        reader, lce_function,
        rt_lo, rt_hi, dt_lo, dt_hi, lce_mz_lo, lce_mz_hi,
        mz_bin=mz_bin,
        gauss_sigma_rt=gauss_sigma_rt, gauss_sigma_dt=gauss_sigma_dt,
        intensity_floor=intensity_floor,
        rank_init=rank_init, rank_max=rank_max, n_restarts=n_restarts,
        rt_r2_min=rt_r2_min, dt_r2_min=dt_r2_min,
        apply_quality_filter=True,
        plot=False,
        verbose=verbose,
    )
    A, B, C = lce.get("A"), lce.get("B"), lce.get("C")
    if A is None or A.shape[1] == 0:
        if verbose:
            print(f"[hdmse]   no LCE factors survived quality filter")
        return []

    # ---- Step 2: precursor characterisations from LCE factors ------------
    precursors_df = process_all_factors(
        A, B, C,
        mz_axis=lce["mz_axis_ntf"],
        rt_axis=lce["rt_axis_ntf"],
        dt_axis=lce["dt_axis_ntf"],
        charge_range=charge_range,
        min_cosine=min_cosine,
        min_peaks_per_cluster=min_peaks_per_cluster,
        output_dir=None,
        verbose=verbose,
        mz_axis_full=lce.get("mz_axis"),
        mask_mz=lce.get("mask_mz"),
    )
    # Keep only k=0 rows — one row per cluster (k=1 is the +1 isotope partner)
    precursors_df = precursors_df[precursors_df["k"] == 0].copy()
    if len(precursors_df) == 0:
        return []

    # ---- Step 3: anchors per surviving factor (NTF-axis A and B) ---------
    rt_axis_ntf = lce["rt_axis_ntf"]
    dt_axis_ntf = lce["dt_axis_ntf"]
    anchors_by_factor: Dict[int, Dict[str, object]] = {}
    for factor_idx in precursors_df["factor_idx"].astype(int).unique():
        anchors_by_factor[int(factor_idx)] = extract_anchor(
            a_vec=A[:, int(factor_idx)],
            b_vec=B[:, int(factor_idx)],
            rt_axis=rt_axis_ntf,
            dt_axis=dt_axis_ntf,
        )

    # ---- Step 4: HCE projection in m/z slabs -----------------------------
    # Build HCE tensors at the SAME RT and DT range used by LCE so the
    # anchor vectors align 1-to-1 with the HCE tensor axes.
    rt_axis_full = reader.metadata().rt_axis[lce_function]
    rt_idx_lo = int(np.searchsorted(rt_axis_full, rt_lo, side="left"))
    rt_idx_hi = int(np.searchsorted(rt_axis_full, rt_hi, side="right")) - 1
    n_rt_lce = rt_idx_hi - rt_idx_lo + 1

    if A.shape[0] != len(rt_axis_ntf):
        raise RuntimeError(
            "Anchor a_vec shape and rt_axis_ntf shape diverge — masking layer "
            "broke axis alignment."
        )

    # Pre-compute HCE m/z slab boundaries
    slab_starts = np.arange(hce_mz_lo, hce_mz_hi, hce_mz_slab, dtype=np.float64)
    slabs = [(float(s), float(min(s + hce_mz_slab, hce_mz_hi))) for s in slab_starts]
    if verbose:
        print(f"[hdmse]   {len(precursors_df)} precursors × {len(slabs)} HCE slabs")

    fragments_by_factor: Dict[int, List[Dict[str, float]]] = {
        f: [] for f in anchors_by_factor
    }

    for slab_lo, slab_hi in slabs:
        hce_tensor, hce_rt_axis, hce_dt_axis, hce_mz_axis = build_tensor(
            reader, hce_function,
            rt_lo, rt_hi,
            dt_lo, dt_hi,
            slab_lo, slab_hi,
            mz_bin=mz_bin,
            gauss_sigma_rt=gauss_sigma_rt, gauss_sigma_dt=gauss_sigma_dt,
            intensity_floor=intensity_floor,
        )
        if hce_tensor.size == 0 or float(hce_tensor.max()) == 0.0:
            continue

        # Sanity-check that HCE RT and DT axes align with LCE NTF (masked) axes.
        # When mask_empty=True (the default in analyze_chunk) the LCE NTF axes
        # are a subset of the full LCE RT axis; we need to subset HCE the same
        # way before projection.
        mask_rt = lce.get("mask_rt")
        mask_dt = lce.get("mask_dt")
        if mask_rt is not None:
            hce_tensor = hce_tensor[mask_rt, :, :]
        if mask_dt is not None:
            hce_tensor = hce_tensor[:, mask_dt, :]
        # Now hce_tensor is (n_rt_ntf, n_dt_ntf, n_mz_slab) and aligns with
        # every anchor's a_norm and b_norm.

        for factor_idx, anchor in anchors_by_factor.items():
            rho, intensity = project_anchor_onto_hce(hce_tensor, anchor)
            slab_fragments = extract_fragments(
                mz_axis=hce_mz_axis,
                rho=rho,
                intensity=intensity,
                rho_threshold=rho_threshold,
                min_intensity=min_fragment_intensity,
                peak_distance_da=peak_distance_da,
            )
            fragments_by_factor[factor_idx].extend(slab_fragments)

    # ---- Step 5: build library rows --------------------------------------
    rows: List[Dict[str, object]] = []
    for _, prec in precursors_df.iterrows():
        factor_idx = int(prec["factor_idx"])
        anchor = anchors_by_factor.get(factor_idx)
        if anchor is None:
            continue
        rows.append(build_library_row(
            sample=sample,
            anchor=anchor,
            precursor=dict(
                factor_idx=factor_idx,
                cluster_idx=int(prec["cluster_idx"]),
                charge=int(prec["charge"]),
                monoisotopic_mz=float(prec["monoisotopic_mz"]),
                monoisotopic_mass_da=float(prec["monoisotopic_mass_da"]),
                cluster_intensity=float(prec["cluster_intensity"]),
                cosine_similarity=float(prec["cosine_similarity"]),
            ),
            fragments=fragments_by_factor.get(factor_idx, []),
        ))
    return rows


def process_raw_file(
    raw_path: str,
    license_path: str,
    output_path: str,
    *,
    lce_function: int = 0,
    hce_function: int = 1,
    lce_mz_window: float = 50.0,
    lce_mz_step: float = 50.0,         # non-overlapping by default
    hce_mz_lo: float = 100.0,
    hce_mz_hi: float = 2000.0,
    hce_mz_slab: float = 100.0,
    rho_threshold: float = 0.85,
    min_fragment_intensity: float = 50.0,
    peak_distance_da: float = 0.5,
    rank_init: int = 5,
    rank_max: int = 15,
    n_restarts: int = 3,
    rt_r2_min: float = 0.85,
    dt_r2_min: float = 0.85,
    charge_range: Tuple[int, int] = (3, 15),
    min_cosine: float = 0.5,
    min_peaks_per_cluster: int = 3,
    mz_bin: float = 0.001,
    gauss_sigma_rt: float = 1.0,
    gauss_sigma_dt: float = 1.0,
    intensity_floor: float = 10.0,
    verbose: bool = False,
) -> None:
    """Iterate LCE m/z windows for one raw file and write a Parquet library."""
    from waters_reader import WatersRawReader, read_license

    sample = Path(raw_path).stem
    license_key = read_license(license_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    with WatersRawReader(raw_path, license=license_key) as reader:
        meta = reader.metadata()
        # Confirm both functions exist with matching scan counts
        if hce_function >= meta.n_functions:
            raise ValueError(
                f"hce_function={hce_function} not present "
                f"(file has {meta.n_functions} functions)"
            )
        if meta.n_scans[lce_function] != meta.n_scans[hce_function]:
            raise ValueError(
                f"LCE/HCE scan counts differ "
                f"({meta.n_scans[lce_function]} vs {meta.n_scans[hce_function]}) "
                "— interleaved acquisition assumption violated."
            )

        lce_mz_min, lce_mz_max = meta.mass_range[lce_function]
        starts = np.arange(lce_mz_min, lce_mz_max, lce_mz_step, dtype=np.float64)
        for s in starts:
            lo = float(s)
            hi = float(min(s + lce_mz_window, lce_mz_max))
            try:
                rows = process_lce_window(
                    reader,
                    sample=sample,
                    lce_function=lce_function, hce_function=hce_function,
                    lce_mz_lo=lo, lce_mz_hi=hi,
                    hce_mz_lo=hce_mz_lo, hce_mz_hi=hce_mz_hi,
                    hce_mz_slab=hce_mz_slab,
                    rho_threshold=rho_threshold,
                    min_fragment_intensity=min_fragment_intensity,
                    peak_distance_da=peak_distance_da,
                    rank_init=rank_init, rank_max=rank_max,
                    n_restarts=n_restarts,
                    rt_r2_min=rt_r2_min, dt_r2_min=dt_r2_min,
                    charge_range=charge_range,
                    min_cosine=min_cosine,
                    min_peaks_per_cluster=min_peaks_per_cluster,
                    mz_bin=mz_bin,
                    gauss_sigma_rt=gauss_sigma_rt,
                    gauss_sigma_dt=gauss_sigma_dt,
                    intensity_floor=intensity_floor,
                    verbose=verbose,
                )
                all_rows.extend(rows)
            except Exception as exc:
                import traceback
                print(f"[hdmse] WARNING: window m/z={lo:.1f}-{hi:.1f} failed: {exc}")
                traceback.print_exc()

    write_library_parquet(all_rows, output_path)
    print(f"[hdmse] wrote {len(all_rows)} library rows → {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HDMS^E anchor-and-project pseudo-MS2 library builder"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("process_raw", help="Build library from one .raw file")
    p.add_argument("raw_path")
    p.add_argument("license_path")
    p.add_argument("--output", required=True, help="Output Parquet path")
    p.add_argument("--lce_function",  type=int,   default=0)
    p.add_argument("--hce_function",  type=int,   default=1)
    p.add_argument("--lce_mz_window", type=float, default=50.0)
    p.add_argument("--lce_mz_step",   type=float, default=50.0)
    p.add_argument("--hce_mz_lo",     type=float, default=100.0)
    p.add_argument("--hce_mz_hi",     type=float, default=2000.0)
    p.add_argument("--hce_mz_slab",   type=float, default=100.0)
    p.add_argument("--rho_threshold", type=float, default=0.85)
    p.add_argument("--min_fragment_intensity", type=float, default=50.0)
    p.add_argument("--peak_distance_da", type=float, default=0.5)
    p.add_argument("--rank_init",  type=int, default=5)
    p.add_argument("--rank_max",   type=int, default=15)
    p.add_argument("--n_restarts", type=int, default=3)
    p.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.cmd == "process_raw":
        process_raw_file(
            raw_path=args.raw_path,
            license_path=args.license_path,
            output_path=args.output,
            lce_function=args.lce_function,
            hce_function=args.hce_function,
            lce_mz_window=args.lce_mz_window,
            lce_mz_step=args.lce_mz_step,
            hce_mz_lo=args.hce_mz_lo,
            hce_mz_hi=args.hce_mz_hi,
            hce_mz_slab=args.hce_mz_slab,
            rho_threshold=args.rho_threshold,
            min_fragment_intensity=args.min_fragment_intensity,
            peak_distance_da=args.peak_distance_da,
            rank_init=args.rank_init,
            rank_max=args.rank_max,
            n_restarts=args.n_restarts,
            verbose=args.verbose,
        )
```

- [ ] **Step 2: Static syntax check (no SDK on ARM, but Python should still parse)**

Run: `cd /Users/allan/Work/Claude/mhdx_v2.0 && python -c "import ast; ast.parse(open('src/hdmse_pipeline.py').read()); print('OK')"`
Expected: prints `OK` (no SyntaxError). The actual import will fail on ARM because `waters_reader` only loads under x86-64 — that is exercised on the cluster in Task 8.

- [ ] **Step 3: Commit**

```bash
git add src/hdmse_pipeline.py
git commit -m "feat: add hdmse_pipeline orchestrator with process_lce_window + CLI"
```

---

## Task 6: Add `hdmse:` config block to `src/config.yaml`

**Files:**
- Modify: `src/config.yaml`

Append a new top-level `hdmse:` block with the parameters surfaced by the CLI in Task 5. Existing pipelines (the main HDX-MS isotope pipeline driven by `Snakefile` + `pipeline.py`) do not read this section, so adding it is non-disruptive.

- [ ] **Step 1: Append `hdmse:` block at the bottom of `src/config.yaml`**

Add the following to the end of the file (after the `resources:` block, with one leading blank line):

```yaml

# ---------------------------------------------------------------------------
# HDMS^E anchor-and-project library  (consumed by src/hdmse_pipeline.py)
# ---------------------------------------------------------------------------
# Builds a pseudo-MS2 fragment library from interleaved low/high collision
# energy data (Function 0 = LCE precursors, Function 1 = HCE fragments).
# Fragmentation occurs post-mobility, so genuine fragments share their
# precursor's RT and IM distributions — quantified here by Pearson rho.

hdmse:
  lce_function: 0          # 0-based function index for LCE (precursors)
  hce_function: 1          # 0-based function index for HCE (fragments)
  lce_mz_window: 50.0      # width of each LCE NTF window [Da]
  lce_mz_step: 50.0        # step between LCE windows [Da] — 50 = no overlap
  hce_mz_lo: 100.0         # lowest HCE m/z to search for fragments [Da]
  hce_mz_hi: 2000.0        # highest HCE m/z to search for fragments [Da]
  hce_mz_slab: 100.0       # HCE m/z slab size for chunked tensor build [Da]
  rho_threshold: 0.85      # Pearson rho cutoff for keeping a fragment
  min_fragment_intensity: 50.0   # minimum projected intensity per fragment
  peak_distance_da: 0.5    # minimum m/z separation between accepted peaks
  output_path: "results/hdmse/{sample}_library.parquet"
```

- [ ] **Step 2: Verify YAML still parses cleanly**

Run: `cd /Users/allan/Work/Claude/mhdx_v2.0 && python -c "import yaml; print(list(yaml.safe_load(open('src/config.yaml'))['hdmse'].keys()))"`
Expected: `['lce_function', 'hce_function', 'lce_mz_window', 'lce_mz_step', 'hce_mz_lo', 'hce_mz_hi', 'hce_mz_slab', 'rho_threshold', 'min_fragment_intensity', 'peak_distance_da', 'output_path']`

- [ ] **Step 3: Commit**

```bash
git add src/config.yaml
git commit -m "config: add hdmse block for anchor-and-project library"
```

---

## Task 7: Full test sweep on ARM (the ground-truth gate)

**Files:** none modified — verification only.

This is the moment to prove the pure-numpy pipeline works end-to-end on synthetic data before we ever touch the real raw file. Every test added in Tasks 0–4 must pass.

- [ ] **Step 1: Run the entire test suite**

Run: `cd /Users/allan/Work/Claude/mhdx_v2.0 && PYTHONPATH=src python -m pytest tests/ -v`
Expected: `7 passed` (1 from Task 1, 2 from Task 2, 2 from Task 3, 2 from Task 4) and zero failures, errors, or warnings related to our code.

- [ ] **Step 2: If any test fails, fix the underlying code (NOT the test) and re-run until green. Do not commit.**

---

## Task 8: Smoke test against `data/260424_AF2501_04_0s.raw` (cluster only)

**Files:** none modified — runtime verification only.

This step **must run inside the Singularity container** on a node with x86-64 + Waters SDK. The user runs it manually; this plan documents the exact incantation.

- [ ] **Step 1: Confirm the file exposes both functions**

Run inside the container:

```bash
python -c "
from waters_reader import WatersRawReader, read_license
key = read_license('../software/MassLynxSDKDownload_v5.0.0/license.key')
with WatersRawReader('data/260424_AF2501_04_0s.raw', license=key) as r:
    m = r.metadata()
    print(f'n_functions={m.n_functions}')
    for fn in range(m.n_functions):
        print(f'  fn={fn}: scans={m.n_scans[fn]} drift={m.n_drift_bins[fn]} '
              f'RT={m.rt_range[fn]} m/z={m.mass_range[fn]}')
"
```

Expected: at least `n_functions=2` with matching `scans` between fn=0 (LCE) and fn=1 (HCE). If they differ, the assumption that LCE/HCE share an RT axis is violated and Task 5's anchor-axis alignment must be revisited before continuing.

- [ ] **Step 2: Run the orchestrator on a single LCE m/z window first**

Use Python (not the CLI) for the single-window test so we can inspect the rows in-process:

```bash
python -c "
from waters_reader import WatersRawReader, read_license
from hdmse_pipeline import process_lce_window
key = read_license('../software/MassLynxSDKDownload_v5.0.0/license.key')
with WatersRawReader('data/260424_AF2501_04_0s.raw', license=key) as r:
    rows = process_lce_window(
        r, sample='260424_AF2501_04_0s',
        lce_function=0, hce_function=1,
        lce_mz_lo=800.0, lce_mz_hi=850.0,
        hce_mz_lo=100.0, hce_mz_hi=2000.0,
        hce_mz_slab=100.0,
        rho_threshold=0.85, min_fragment_intensity=50.0, peak_distance_da=0.5,
        rank_init=5, rank_max=15, n_restarts=3,
        rt_r2_min=0.85, dt_r2_min=0.85,
        charge_range=(3, 15), min_cosine=0.5, min_peaks_per_cluster=3,
        mz_bin=0.001, gauss_sigma_rt=1.0, gauss_sigma_dt=1.0,
        intensity_floor=10.0, verbose=True,
    )
    print(f'rows={len(rows)}')
    for row in rows[:3]:
        print(f\"  precursor m/z={row['obs_mz']:.4f} z={row['charge']} \"
              f\"RT={row['RT']:.3f} im={row['im_mono']:.1f} \"
              f\"n_frag={row['n_fragments']}\")
        for f in row['fragments'][:5]:
            print(f\"      frag m/z={f['mz']:.4f}  I={f['intensity']:.1e}  rho={f['rho']:.3f}\")
"
```

Expected: at least one row, each with `n_fragments >= 1` and every fragment's `rho >= 0.85`. If `rows == []`, lower `min_cosine` (0.5 → 0.3) and re-run; the LCE window may simply not contain a precursor in this slice.

- [ ] **Step 3: Run the full file through the CLI**

```bash
python src/hdmse_pipeline.py process_raw \
  data/260424_AF2501_04_0s.raw \
  ../software/MassLynxSDKDownload_v5.0.0/license.key \
  --output results/hdmse/260424_AF2501_04_0s_library.parquet \
  --lce_mz_window 50.0 --lce_mz_step 50.0 \
  --hce_mz_lo 100.0 --hce_mz_hi 2000.0 --hce_mz_slab 100.0 \
  --rho_threshold 0.85 \
  --verbose
```

Expected: a single `*_library.parquet` file is written. Confirm with:

```bash
python -c "
import pandas as pd
df = pd.read_parquet('results/hdmse/260424_AF2501_04_0s_library.parquet')
print(df[['obs_mz', 'charge', 'RT', 'im_mono', 'n_fragments']].describe())
print(df['n_fragments'].value_counts().head(10))
"
```

- [ ] **Step 4: Record cluster-side observations in the PR description (no commit needed). Track:**
  - Total library rows
  - Median/mean `n_fragments` per precursor
  - Distribution of `rho` across all kept fragments (should be tightly above 0.85)

---

## Task 9: Document the new pipeline in `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

Add a self-contained section so future Claude sessions discover the pipeline without re-deriving it from source. Keep the section short — one paragraph of context, one CLI invocation, and a pointer to `src/hdmse_library.py` and `src/hdmse_pipeline.py`.

- [ ] **Step 1: Insert a new section in `CLAUDE.md`**

Locate the line that begins with `### Module responsibilities` (around line 50 of the existing CLAUDE.md). **Above** that table — after the `### Protein identification pipeline` block and its closing `mkdir` line, but **before** `### Running individual pipeline steps manually (inside the container)` — insert the following block:

```markdown
### HDMS^E anchor-and-project pseudo-MS2 library

Builds a fragment library from interleaved LCE (Function 0, precursors) and
HCE (Function 1, post-mobility fragments) acquisitions. Reuses the existing
LCE NTF decomposition as immutable RT/IM anchors, then projects each anchor
onto the HCE tensor and keeps fragments whose Pearson correlation with the
anchor's rank-1 RT×DT shape is above `rho_threshold` (default 0.85).

```bash
# Inside the Singularity container:
python src/hdmse_pipeline.py process_raw \
  data/260424_AF2501_04_0s.raw \
  ../software/MassLynxSDKDownload_v5.0.0/license.key \
  --output results/hdmse/260424_AF2501_04_0s_library.parquet \
  --lce_mz_window 50.0 --hce_mz_slab 100.0 \
  --rho_threshold 0.85 --verbose
```

Configuration lives in `src/config.yaml` under the top-level `hdmse:` block.
The output Parquet schema (`hdmse_library.LIBRARY_SCHEMA`) is designed to
feed `protein_identification.identify()` directly via the `obs_mz`,
`charge`, `MW`, `RT`, `im_mono`, `ab_cluster_total` columns; per-precursor
fragment lists are kept nested in the `fragments` column.
```

Then extend the **Module responsibilities** table by inserting two new rows directly after the `pipeline.py` row:

```markdown
| `hdmse_library.py` | Pure-NumPy anchor-and-project core: `extract_anchor`, `project_anchor_onto_hce`, `extract_fragments`, library schema and Parquet writer. SDK-free → unit-tested on ARM. |
| `hdmse_pipeline.py` | SDK-facing orchestrator: iterates LCE m/z windows, calls `analyze_chunk` + `process_all_factors`, projects anchors onto HCE m/z slabs, writes the library Parquet. Container-only. |
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document HDMS^E anchor-and-project library pipeline"
```

---

## Self-Review Checklist (run before declaring the plan ready)

- **Spec coverage**
  - Phase 1 (LCE 50 Da NTF) → Task 5 step 1 calls `analyze_chunk` with `lce_mz_window=50.0`. ✓
  - Phase 2 (anchor extraction) → Task 1. ✓
  - Phase 3a (targeted HCE projection) → Task 2 + Task 5 step 4. ✓
  - Phase 3b (Pearson correlation, ρ > 0.85 gate) → Task 2 (rho) + Task 3 (gate). ✓
  - Phase 4 (DataFrame schema, Parquet, target-decoy compatibility) → Task 4 + alignment with `protein_identification` columns documented in Task 9. ✓

- **Placeholder scan**: every step shows complete code; no "TBD", "TODO", "similar to above", or unbound symbols. ✓

- **Type/name consistency**: `extract_anchor` returns dict with keys `rt_center`, `rt_sigma`, `dt_center`, `dt_sigma`, `a_norm`, `b_norm`. `project_anchor_onto_hce` consumes that dict (`a_norm`, `b_norm`). `extract_fragments` returns dicts with `mz`, `intensity`, `rho`. `build_library_row` emits exactly `LIBRARY_SCHEMA` keys. `process_lce_window` calls `extract_anchor` → `project_anchor_onto_hce` → `extract_fragments` → `build_library_row` in that order with matching shapes. ✓

- **ARM/SDK separation**: every pure-numpy function is tested in `tests/`, every SDK call lives in `hdmse_pipeline.py` and is exercised only inside the container in Task 8. ✓

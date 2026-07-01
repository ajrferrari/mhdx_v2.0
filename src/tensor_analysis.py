"""
tensor_analysis.py
==================
Efficient 3D tensor construction from Waters IMS-MS .raw data, followed by
non-negative tensor factorization (NTF / CP decomposition) for resolving
co-eluting species in HDX-MS experiments.

Workflow
--------
1. ``build_tensor``      — build an (n_rt, n_dt, n_mz) float32 tensor from a
                           WatersRawReader chunk; all per-scan m/z axes are
                           tent-interpolated onto a common 0.001 Da grid.
2. ``mask_empty_slices`` — remove all-zero RT, DT, and m/z slices to reduce
                           tensor size before factorization.
3. ``factorize``         — NTF with automatic rank selection: start from
                           ``rank_init`` and iterate until the minimum
                           pairwise correlation across all three modes (RT,
                           DT, m/z) falls below ``corr_threshold``.
4. ``filter_factors``    — keep only factors whose RT and DT profiles fit a
                           Gaussian with R² ≥ threshold.
5. ``factor_mz_spectra`` — recover intensity vs m/z for each factor via the
                           outer-product integral over the RT and DT modes.
6. ``plot_raw``          — RT×DT heatmap + m/z spectrum of the raw tensor.
7. ``plot_factors``      — per-factor RT profile, DT profile, and m/z spectrum.
8. ``plot_correlations`` — RT, DT, m/z, and min-correlation heatmaps.
9. ``analyze_chunk``     — convenience wrapper that runs steps 1–8 in order.

Design notes
------------
* NTF (CP / PARAFAC form) is used rather than 2D NMF because it gives
  independent RT, DT, and m/z vectors for each component, enabling
  per-mode Gaussian quality checks and physically interpretable factors.
  mhdx_tools uses the same decomposition via nn_fac.ntf.ntf(); this
  module reimplements it from scratch with no external NTF library.
* Multiplicative update rules (Lee–Seung generalized to 3-way tensors)
  with a small floor ε guarantee non-negativity and numerical stability.
* Tent interpolation (linear two-bin spreading) is used when reprofiling
  onto the common m/z grid to avoid empty-bin artefacts at 0.001 Da.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Optional: only imported at runtime inside build_tensor to avoid hard dep
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from waters_reader import WatersRawReader

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MZ_BIN_DEFAULT: float = 0.001    # Da per reprofiling bin
SIZE_WARN_GB: float = 1.0        # warn if tensor > this many GB

# Recommended per-chunk limits (warnings issued when exceeded)
RT_WARN_MIN: float = 1.0         # min — maximum RT window before warning
DT_WARN_BINS: int = 50           # bins — maximum DT window before warning
MZ_WARN_DA: float = 40.0         # Da  — maximum m/z window before warning

NTF_EPS: float = 1e-10           # multiplicative-update floor (prevents /0)
GAUSS_R2_DEFAULT: float = 0.85   # minimum Gaussian fit R² for factor to pass
CORR_THRESHOLD: float = 0.17     # maximum allowed min-pairwise correlation


# ===========================================================================
# Section 1 — Tensor construction
# ===========================================================================

def _tent_reproject(
    mz_data: np.ndarray,
    inten_data: np.ndarray,
    mz_lo: float,
    n_mz: int,
    mz_bin: float,
    out: np.ndarray,
) -> None:
    """Nearest-bin reprofiling: each centroid deposits its full intensity into
    the single closest bin on the *mz_bin*-Da grid.

    Replaces the previous tent interpolation, which split each centroid across
    two adjacent bins.  That pairing created artificial doublets in the non-zero
    bin mask (two ~0.001 Da apart bins per ~12.5 mDa centroid) that caused a
    zigzag appearance when matplotlib connected them — obscuring the true
    gaussian peak shape.  With nearest-bin assignment every non-zero bin
    corresponds to exactly one native centroid, so the line plot recovers the
    expected peak profile.
    """
    idx = np.round((mz_data - mz_lo) / mz_bin).astype(np.int32)
    valid = (idx >= 0) & (idx < n_mz)
    np.add.at(out, idx[valid], inten_data[valid])


def build_tensor(
    reader: "WatersRawReader",
    function: int,
    rt_lo: float,
    rt_hi: float,
    dt_lo: int,
    dt_hi: int,
    mz_lo: float,
    mz_hi: float,
    mz_bin: float = MZ_BIN_DEFAULT,
    size_warn_gb: float = SIZE_WARN_GB,
    gauss_sigma_rt: float = 0.0,
    gauss_sigma_dt: float = 0.0,
    intensity_floor: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build an (n_rt, n_dt, n_mz) tensor from a Waters .raw chunk.

    All per-(scan, drift) spectra are tent-interpolated onto a common m/z
    axis at *mz_bin* Da resolution.  Optional Gaussian smoothing can be
    applied in the RT and DT dimensions after accumulation (sigma=0 skips it).
    An optional intensity floor then zeroes out bins below *intensity_floor*,
    which reduces effective tensor density and makes subsequent
    ``mask_empty_slices`` more aggressive.

    Parameters
    ----------
    reader :
        Open ``WatersRawReader`` (``metadata()`` already called).
    function :
        0-based function index (0 = main IMS function).
    rt_lo, rt_hi :
        Retention time window [min], inclusive.
    dt_lo, dt_hi :
        Drift bin indices [0-based], inclusive.
    mz_lo, mz_hi :
        m/z window [Da], inclusive.
    mz_bin :
        Reprofiling resolution [Da]. Default 0.001.
    size_warn_gb :
        Issue a ``RuntimeWarning`` if the estimated tensor size exceeds
        this many GB. Default 0.5.
    gauss_sigma_rt :
        Gaussian smoothing sigma in RT bins (0 = off).
    gauss_sigma_dt :
        Gaussian smoothing sigma in DT bins (0 = off).
    intensity_floor :
        Any bin with intensity strictly below this value is set to zero
        after reprofiling and smoothing.  Set to 0 or None to disable.
        Default 10.

    Returns
    -------
    tensor : float32 (n_rt, n_dt, n_mz)
    rt_axis : float32 (n_rt,)  — retention times [min]
    dt_axis : float32 (n_dt,)  — drift bin indices (or ms if loaded)
    mz_axis : float32 (n_mz,)  — m/z values [Da]
    """
    meta = reader.metadata()
    rt_axis_full = meta.rt_axis[function]
    dt_axis_full = meta.dt_axis[function]
    n_drift_max = meta.n_drift_bins[function] - 1

    # ---- window sanity warnings ----------------------------------------
    rt_span = rt_hi - rt_lo
    dt_span = dt_hi - dt_lo + 1
    mz_span = mz_hi - mz_lo

    if rt_span > RT_WARN_MIN:
        warnings.warn(
            f"RT window {rt_span:.2f} min exceeds recommended maximum "
            f"{RT_WARN_MIN} min. Factorization quality may degrade.",
            RuntimeWarning, stacklevel=2,
        )
    if dt_span > DT_WARN_BINS:
        warnings.warn(
            f"DT window {dt_span} bins exceeds recommended maximum "
            f"{DT_WARN_BINS} bins.",
            RuntimeWarning, stacklevel=2,
        )
    if mz_span > MZ_WARN_DA:
        warnings.warn(
            f"m/z window {mz_span:.1f} Da exceeds recommended maximum "
            f"{MZ_WARN_DA} Da.",
            RuntimeWarning, stacklevel=2,
        )

    # ---- find RT frame indices in the full axis -------------------------
    rt_idx_lo = max(0, int(np.searchsorted(rt_axis_full, rt_lo, side="left")))
    rt_idx_hi = min(
        len(rt_axis_full) - 1,
        int(np.searchsorted(rt_axis_full, rt_hi, side="right")) - 1,
    )
    dt_lo = max(0, int(dt_lo))
    dt_hi = min(int(n_drift_max), int(dt_hi))

    n_rt = rt_idx_hi - rt_idx_lo + 1
    n_dt = dt_hi - dt_lo + 1
    n_mz = int(round((mz_hi - mz_lo) / mz_bin)) + 1

    if n_rt <= 0 or n_dt <= 0:
        raise ValueError(
            f"Empty chunk: n_rt={n_rt}, n_dt={n_dt}. "
            "Check that rt_lo/rt_hi and dt_lo/dt_hi are within file range."
        )

    # ---- size warning ---------------------------------------------------
    tensor_gb = n_rt * n_dt * n_mz * 4 / 1024 ** 3
    if tensor_gb > size_warn_gb:
        warnings.warn(
            f"Estimated tensor size {tensor_gb:.2f} GB exceeds "
            f"threshold {size_warn_gb:.1f} GB. "
            f"Shape: ({n_rt}, {n_dt}, {n_mz}). "
            "Consider narrowing the RT, DT, or m/z window.",
            RuntimeWarning, stacklevel=2,
        )

    # ---- build axes -----------------------------------------------------
    rt_axis = rt_axis_full[rt_idx_lo : rt_idx_hi + 1]
    dt_axis = dt_axis_full[dt_lo : dt_hi + 1]
    mz_axis = (
        mz_lo + np.arange(n_mz, dtype=np.float32) * mz_bin
    ).astype(np.float32)

    # ---- accumulate tensor ----------------------------------------------
    tensor = np.zeros((n_rt, n_dt, n_mz), dtype=np.float32)
    mz_lo_f = float(mz_lo)
    mz_hi_f = float(mz_hi)

    for i_rt, scan_idx in enumerate(range(rt_idx_lo, rt_idx_hi + 1)):
        for i_dt, drift_idx in enumerate(range(dt_lo, dt_hi + 1)):
            mz_data, inten_data = reader.read_drift_scan(
                function, scan_idx, drift_idx
            )
            if len(mz_data) == 0:
                continue
            mz_data = np.asarray(mz_data, dtype=np.float64)
            inten_data = np.asarray(inten_data, dtype=np.float32)

            # Keep only points within the m/z window (plus one bin margin
            # so tent interpolation at the edges still works correctly).
            mask = (mz_data >= mz_lo_f - mz_bin) & (mz_data <= mz_hi_f + mz_bin)
            if not np.any(mask):
                continue

            _tent_reproject(
                mz_data[mask], inten_data[mask],
                mz_lo_f, n_mz, mz_bin,
                tensor[i_rt, i_dt],
            )

    # ---- optional Gaussian smoothing in RT and DT -----------------------
    if gauss_sigma_rt > 0 or gauss_sigma_dt > 0:
        sigma = [
            gauss_sigma_rt if gauss_sigma_rt > 0 else 0,
            gauss_sigma_dt if gauss_sigma_dt > 0 else 0,
            0,   # no smoothing along m/z
        ]
        tensor = gaussian_filter(
            tensor.astype(np.float64), sigma=sigma
        ).astype(np.float32)

    # ---- intensity floor ------------------------------------------------
    # Applied after smoothing so that smoothed tails just below the
    # threshold are also zeroed, making mask_empty_slices more effective.
    if intensity_floor:
        tensor[tensor < intensity_floor] = 0.0

    return tensor, rt_axis, dt_axis, mz_axis


# ===========================================================================
# Section 2 — Pre-processing: mask empty slices
# ===========================================================================

def mask_empty_slices(
    tensor: np.ndarray,
    rt_axis: np.ndarray,
    dt_axis: np.ndarray,
    mz_axis: np.ndarray,
    threshold: float = 0.0,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """Remove RT, DT, and m/z slices whose maximum equals *threshold* or below.

    This reduces the tensor size before NTF and prevents empty slices from
    diluting the factorization.

    Returns
    -------
    tensor_masked : float32
    rt_axis_masked, dt_axis_masked, mz_axis_masked : float32
    mask_rt, mask_dt, mask_mz : bool (n_rt,), (n_dt,), (n_mz,)
        True where the slice was kept.  Store these to map factor
        vectors back to the original full axis for plotting.
    """
    mask_rt = tensor.max(axis=(1, 2)) > threshold
    mask_dt = tensor.max(axis=(0, 2)) > threshold
    mask_mz = tensor.max(axis=(0, 1)) > threshold

    if not (np.any(mask_rt) and np.any(mask_dt) and np.any(mask_mz)):
        warnings.warn(
            "Tensor is all zeros after masking — returning original tensor.",
            RuntimeWarning, stacklevel=2,
        )
        return (
            tensor, rt_axis, dt_axis, mz_axis,
            np.ones(len(rt_axis), dtype=bool),
            np.ones(len(dt_axis), dtype=bool),
            np.ones(len(mz_axis), dtype=bool),
        )

    tensor_masked = tensor[np.ix_(mask_rt, mask_dt, mask_mz)]
    return (
        tensor_masked,
        rt_axis[mask_rt], dt_axis[mask_dt], mz_axis[mask_mz],
        mask_rt, mask_dt, mask_mz,
    )


def _expand_to_full(
    vec: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Insert zeros into *vec* at positions where *mask* is False."""
    out = np.zeros(len(mask), dtype=vec.dtype)
    out[mask] = vec
    return out


# ===========================================================================
# Section 3 — NTF (CP / PARAFAC) from scratch
# ===========================================================================

def _khatri_rao(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Column-wise Kronecker (Khatri–Rao) product.

    A : (I, R), B : (J, R)  →  (I*J, R)
    Row i*J + j of the output equals A[i, :] * B[j, :].
    """
    return (A[:, None, :] * B[None, :, :]).reshape(-1, A.shape[1])


def _ntf(
    tensor: np.ndarray,
    rank: int,
    n_iter_max: int = 10_000,
    tol: float = 1e-6,
    eps: float = NTF_EPS,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """Non-negative CP tensor factorization via HALS (Hierarchical ALS).

    Decomposes tensor T ≈ [[A, B, C]] in CP/PARAFAC form:

        T[i, j, k] ≈ Σ_r  A[i, r] * B[j, r] * C[k, r]

    All factor matrices are constrained non-negative throughout.

    Update rules (HALS — projected gradient with exact per-component step):

        For each r:
            A[:, r] ← max(ε, A[:, r] + (Nₐ[:, r] − A @ Gₐ[:, r]) / Gₐ[r,r])

        where  Nₐ = T_(1) @ KR(B,C)  and  Gₐ = (BᵀB) ⊙ (CᵀC)

        B and C follow the same pattern with successively updated factors.

    HALS converges 10–50× faster than Lee–Seung multiplicative updates
    because the per-component step size 1/Gₐ[r,r] is the exact inverse of
    the Lipschitz constant, and because successively updated factors are
    used immediately (Gauss–Seidel ordering).

    Parameters
    ----------
    tensor : float32 or float64 (I, J, K)
    rank :
        Number of components.
    n_iter_max :
        Maximum iterations.
    tol :
        Convergence: stop when the relative change in reconstruction
        error between consecutive checks (every 10 iterations) is < tol.
    eps :
        Non-negativity floor (prevents /0 and keeps factors strictly > 0).
    rng :
        NumPy random Generator for reproducible initialization.
    verbose :
        Print reconstruction error every 100 iterations.

    Returns
    -------
    A : float32 (I, rank) — RT factor matrix
    B : float32 (J, rank) — DT factor matrix
    C : float32 (K, rank) — m/z factor matrix
    errors : list of float — relative Frobenius error, sampled every 10 iters

    Memory notes
    ------------
    All computation stays in float32.  T is reshaped to the zero-copy view
    T_flat = (I*J, K) shared across all mode updates.  Convergence uses the
    identity  ‖T−recon‖² = ‖T‖² − 2⟨T,recon⟩ + ‖recon‖²  so the full
    reconstruction is never materialised.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    T = tensor.astype(np.float32, copy=False)
    I, J, K = T.shape
    R = rank

    # Zero-copy mode-1 unfolding T_(1) = (I*J, K).
    T_flat = T.reshape(I * J, K)

    # Precompute ‖T‖² once.
    T_norm_sq = float(np.dot(T_flat.ravel(), T_flat.ravel()))
    T_norm = np.sqrt(T_norm_sq) + eps

    # Random non-negative initialisation scaled by signal level.
    scale = max(float((T.mean() / R) ** (1.0 / 3.0)), eps)
    A = (rng.random((I, R)) * scale + eps).astype(np.float32)
    B = (rng.random((J, R)) * scale + eps).astype(np.float32)
    C = (rng.random((K, R)) * scale + eps).astype(np.float32)

    errors: List[float] = []
    prev_error = np.inf

    for iteration in range(n_iter_max):
        # ------------------------------------------------------------------
        # Shared: TC_flat = T_flat @ C  →  (I*J, R)
        # Dominant cost: (I*J) × K × R  FLOPs.  Reused for A and B updates.
        # ------------------------------------------------------------------
        TC_flat = T_flat @ C                   # (I*J, R)
        TC = TC_flat.reshape(I, J, R)          # zero-copy view

        # --- HALS update for A ---
        # Nₐ[i,r] = Σ_{j,k} T[i,j,k]*B[j,r]*C[k,r]  =  einsum(TC, B)
        num_A = np.einsum("ijr,jr->ir", TC, B)  # (I, R)
        GrA = (B.T @ B) * (C.T @ C)             # (R, R) Hadamard Gram
        for r in range(R):
            d = float(GrA[r, r])
            if d > eps:
                A[:, r] = np.maximum(
                    A[:, r] + (num_A[:, r] - A @ GrA[:, r]) / d, eps
                )

        # --- HALS update for B (uses freshly updated A) ---
        num_B = np.einsum("ijr,ir->jr", TC, A)  # (J, R);  TC uses old C — fine
        GrB = (A.T @ A) * (C.T @ C)
        for r in range(R):
            d = float(GrB[r, r])
            if d > eps:
                B[:, r] = np.maximum(
                    B[:, r] + (num_B[:, r] - B @ GrB[:, r]) / d, eps
                )

        # --- HALS update for C (uses freshly updated A and B) ---
        AB_flat = _khatri_rao(A, B)             # (I*J, R)
        num_C = (AB_flat.T @ T_flat).T          # (K, R); direct, no T_(3) copy
        GrC = (A.T @ A) * (B.T @ B)
        for r in range(R):
            d = float(GrC[r, r])
            if d > eps:
                C[:, r] = np.maximum(
                    C[:, r] + (num_C[:, r] - C @ GrC[:, r]) / d, eps
                )

        # --- convergence check every 10 iterations ---
        if iteration % 10 == 0 or iteration == n_iter_max - 1:
            # ‖T−recon‖² = ‖T‖² − 2⟨T,recon⟩ + ‖recon‖²
            # Inner product ⟨T,recon⟩ uses TC_flat (old C) — slight lag, fine
            inner  = float(np.sum(AB_flat * TC_flat))
            rec_sq = float(np.sum((A.T @ A) * (B.T @ B) * (C.T @ C)))
            error  = np.sqrt(max(0.0, T_norm_sq - 2.0 * inner + rec_sq)) / T_norm
            errors.append(error)
            if verbose and iteration % 100 == 0:
                print(f"  NTF iter {iteration:6d}  rel_err={error:.6e}")
            if abs(prev_error - error) < tol:
                if verbose:
                    print(f"  NTF converged at iteration {iteration}")
                break
            prev_error = error

    return A, B, C, errors


def _factor_correlations(
    A: np.ndarray, B: np.ndarray, C: np.ndarray
) -> float:
    """Max off-diagonal element of element-wise minimum correlation matrix.

    For each pair of factors (r, s) the score is
    min(corr_A[r,s], corr_B[r,s], corr_C[r,s]).
    The return value is the largest such score over all off-diagonal pairs.
    A single-factor solution always returns 0 (no pairs).
    """
    R = A.shape[1]
    if R == 1:
        return 0.0

    def _corrcoef(M: np.ndarray) -> np.ndarray:
        # Protect against constant columns (std = 0)
        std = M.std(axis=0)
        std[std == 0] = 1.0
        M_normed = (M - M.mean(axis=0)) / std
        return M_normed.T @ M_normed / max(len(M) - 1, 1)

    corrA = np.corrcoef(A.T)
    corrB = np.corrcoef(B.T)
    corrC = np.corrcoef(C.T)

    min_corr = np.minimum(np.minimum(corrA, corrB), corrC)
    off_diag = ~np.eye(R, dtype=bool)
    return float(np.max(min_corr[off_diag]))


# ===========================================================================
# Section 4 — Auto-rank NTF
# ===========================================================================

def factorize(
    tensor: np.ndarray,
    rank_init: int = 5,
    corr_threshold: float = CORR_THRESHOLD,
    n_iter_max: int = 10_000,
    tol: float = 1e-6,
    rank_max: int = 15,
    min_intensity: float = 0.0,
    n_restarts: int = 3,
    verbose: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    int,
    float,
]:
    """NTF with automatic rank selection via inter-factor correlation.

    Mirrors mhdx_tools' ``gen_factors_with_corr_check()``:

    * If the initial ``max_corr < corr_threshold`` → factors are still
      well-separated → increase rank until ``max_corr ≥ corr_threshold``
      and keep the last rank that was still below threshold.
    * If the initial ``max_corr ≥ corr_threshold`` → too many factors →
      decrease rank until ``max_corr < corr_threshold``.

    Parameters
    ----------
    tensor : float32 (n_rt, n_dt, n_mz)
    rank_init :
        Starting number of components.
    corr_threshold :
        Boundary for the min-pairwise-correlation metric (default 0.17).
    n_iter_max :
        Max NTF iterations per rank attempt.
    tol :
        NTF convergence tolerance.
    rank_max :
        Upper bound on rank.
    min_intensity :
        If the total tensor intensity is below this value the factorization
        is skipped and ``(None, None, None, 0, nan)`` is returned.
    n_restarts :
        Number of independent random restarts per rank trial.  The restart
        with the lowest final reconstruction error is kept.  Increasing this
        makes auto-rank more robust against poor initialisations (default 3).
    verbose :
        Print rank exploration steps.
    rng :
        NumPy random Generator used to seed independent child RNGs for each
        restart; results are fully reproducible for a given seed.

    Returns
    -------
    A : (n_rt, rank) or None
    B : (n_dt, rank) or None
    C : (n_mz, rank) or None
    final_rank : int
    max_corr : float — min-pairwise-correlation of the final factors
    """
    total_intensity = float(tensor.sum())
    if min_intensity > 0 and total_intensity < min_intensity:
        warnings.warn(
            f"Total tensor intensity {total_intensity:.3e} < "
            f"min_intensity {min_intensity:.3e}. Skipping factorization.",
            RuntimeWarning, stacklevel=2,
        )
        return None, None, None, 0, float("nan")

    if rng is None:
        rng = np.random.default_rng(42)

    def _run(rank: int):
        """Run NTF *n_restarts* times with independent seeds; keep the best."""
        best_A = best_B = best_C = best_errs = None
        best_final_err = np.inf
        for _ in range(n_restarts):
            # Derive an independent child RNG from the parent so that:
            # (a) results are reproducible for a given parent seed, and
            # (b) the advancing state of one rank trial does not influence
            #     the initialization of the next rank trial.
            child_rng = np.random.default_rng(rng.integers(2 ** 63))
            A, B, C, errs = _ntf(
                tensor, rank, n_iter_max, tol, rng=child_rng, verbose=False
            )
            final_err = errs[-1] if errs else np.inf
            if final_err < best_final_err:
                best_final_err = final_err
                best_A, best_B, best_C, best_errs = A, B, C, errs
        return best_A, best_B, best_C, best_errs

    if verbose:
        print(f"[factorize] initial rank={rank_init}")
    A, B, C, _ = _run(rank_init)
    max_corr = _factor_correlations(A, B, C)
    if verbose:
        print(f"  max_corr={max_corr:.4f}  (threshold={corr_threshold})")

    best_A, best_B, best_C, best_rank, best_corr = A, B, C, rank_init, max_corr

    if max_corr < corr_threshold:
        # Under-factorized: try increasing rank
        rank = rank_init
        while rank < rank_max:
            rank += 1
            if verbose:
                print(f"[factorize] increasing rank → {rank}")
            A, B, C, _ = _run(rank)
            max_corr = _factor_correlations(A, B, C)
            if verbose:
                print(f"  max_corr={max_corr:.4f}")
            if max_corr < corr_threshold:
                best_A, best_B, best_C, best_rank, best_corr = (
                    A, B, C, rank, max_corr
                )
            else:
                break  # one step too far; keep previous best
    else:
        # Over-factorized: try decreasing rank
        rank = rank_init
        while rank > 1:
            rank -= 1
            if verbose:
                print(f"[factorize] decreasing rank → {rank}")
            A, B, C, _ = _run(rank)
            if rank == 1:
                best_A, best_B, best_C, best_rank, best_corr = A, B, C, 1, 0.0
                break
            max_corr = _factor_correlations(A, B, C)
            if verbose:
                print(f"  max_corr={max_corr:.4f}")
            if max_corr < corr_threshold:
                best_A, best_B, best_C, best_rank, best_corr = (
                    A, B, C, rank, max_corr
                )
                break  # first acceptable rank found

    if verbose:
        print(
            f"[factorize] final rank={best_rank}, "
            f"max_corr={best_corr:.4f}"
        )
    return best_A, best_B, best_C, best_rank, best_corr


# ===========================================================================
# Section 5 — Gaussian quality filter
# ===========================================================================

def _gauss_func(
    x: np.ndarray, baseline: float, amplitude: float, center: float, sigma: float
) -> np.ndarray:
    """Gaussian with baseline: baseline + amplitude * exp(-0.5*((x-center)/sigma)²)."""
    return baseline + amplitude * np.exp(
        -0.5 * ((x - center) / (np.abs(sigma) + 1e-12)) ** 2
    )


def _fit_gauss_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Fit a Gaussian to *y(x)* and return the R² of the fit.

    Returns 0.0 if fewer than 3 points, all-zero, or fit fails.
    For exactly 3 points the fit is fully determined; R² is capped at 0.95 to
    avoid a perfect-fit false positive.
    """
    if len(y) < 3 or np.max(y) <= 0:
        return 0.0
    y_norm = y / (np.max(y) + 1e-12)   # normalize for numerical stability
    x0_guess = float(x[np.argmax(y_norm)])
    x_range = float(x[-1] - x[0])
    try:
        popt, _ = curve_fit(
            _gauss_func, x, y_norm,
            p0=[0.0, 1.0, x0_guess, x_range / 4.0],
            bounds=(
                [0.0, 0.0,  float(x[0]),  0.0],
                [1.0, 2.0,  float(x[-1]), x_range],
            ),
            maxfev=3000,
        )
        y_fit = _gauss_func(x, *popt)
        ss_res = float(np.sum((y_norm - y_fit) ** 2))
        ss_tot = float(np.sum((y_norm - y_norm.mean()) ** 2))
        if ss_tot < 1e-12:
            return 0.0
        r2 = max(0.0, 1.0 - ss_res / ss_tot)
        if len(y) == 3:
            r2 = min(r2, 0.95)
        return r2
    except Exception:
        return 0.0


def filter_factors(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    rt_axis: np.ndarray,
    dt_axis: np.ndarray,
    rt_r2_min: float = GAUSS_R2_DEFAULT,
    dt_r2_min: float = GAUSS_R2_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], Dict[int, Dict]]:
    """Filter factors whose RT and DT profiles fit a Gaussian well.

    Parameters
    ----------
    A : (n_rt, R)  — RT mode matrix
    B : (n_dt, R)  — DT mode matrix
    C : (n_mz, R)  — m/z mode matrix
    rt_axis : (n_rt,)  — retention time [min]
    dt_axis : (n_dt,)  — drift bin indices or ms
    rt_r2_min :
        Minimum R² of Gaussian fit to the RT profile. Default 0.85.
    dt_r2_min :
        Minimum R² of Gaussian fit to the DT profile. Default 0.85.

    Returns
    -------
    A_f, B_f, C_f :
        Filtered factor matrices (columns = kept factors).
    kept_indices :
        List of original factor column indices that passed.
    quality_info :
        Dict mapping original factor index → ``{'rt_r2': float, 'dt_r2': float}``.
    """
    R = A.shape[1]
    kept: List[int] = []
    quality: Dict[int, Dict] = {}

    rt_x = rt_axis.astype(np.float64)
    dt_x = dt_axis.astype(np.float64)

    for r in range(R):
        rt_r2 = _fit_gauss_r2(rt_x, A[:, r])
        dt_r2 = _fit_gauss_r2(dt_x, B[:, r])
        quality[r] = {"rt_r2": rt_r2, "dt_r2": dt_r2}
        if rt_r2 >= rt_r2_min and dt_r2 >= dt_r2_min:
            kept.append(r)

    if len(kept) == 0:
        warnings.warn(
            f"No factors passed the Gaussian R² filter "
            f"(rt_r2_min={rt_r2_min}, dt_r2_min={dt_r2_min}). "
            "Returning all factors unfiltered.",
            RuntimeWarning, stacklevel=2,
        )
        kept = list(range(R))

    return A[:, kept], B[:, kept], C[:, kept], kept, quality


# ===========================================================================
# Section 6 — m/z spectrum recovery
# ===========================================================================

def factor_mz_spectra(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Recover intensity vs m/z for each factor.

    For component r the m/z intensity is C[:, r] scaled by the total
    weight of its RT × DT outer product:

        spectrum_r = C[:, r]  *  Σ_{i,j} A[i,r] * B[j,r]

    This is the integral of the rank-1 tensor over the RT and DT modes,
    leaving only the m/z dimension.

    Parameters
    ----------
    A : (n_rt, R)
    B : (n_dt, R)
    C : (n_mz, R)
    normalize :
        If True, normalize each spectrum to its maximum.

    Returns
    -------
    spectra : float64 (R, n_mz)
        Row r is the m/z spectrum of factor r.
    """
    R = A.shape[1]
    n_mz = C.shape[0]
    spectra = np.zeros((R, n_mz), dtype=np.float64)

    for r in range(R):
        rt_dt_weight = float(A[:, r].sum() * B[:, r].sum())
        spectra[r] = C[:, r] * rt_dt_weight
        if normalize and spectra[r].max() > 0:
            spectra[r] /= spectra[r].max()

    return spectra


# ===========================================================================
# Section 7 — Residual
# ===========================================================================

def compute_residual(
    tensor: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the CP reconstruction and its residual.

    Parameters
    ----------
    tensor : float32 (n_rt, n_dt, n_mz) — the original (masked) tensor
    A : (n_rt, R), B : (n_dt, R), C : (n_mz, R) — factor matrices

    Returns
    -------
    residual : float32 (n_rt, n_dt, n_mz)
        tensor − reconstruction.  Positive values are unexplained signal;
        negative values mean the reconstruction slightly overshoots.
    reconstruction : float32 (n_rt, n_dt, n_mz)
        Σ_r A[:,r] ⊗ B[:,r] ⊗ C[:,r].
    """
    recon = np.einsum("ir,jr,kr->ijk", A, B, C).astype(np.float32)
    residual = tensor.astype(np.float32) - recon
    return residual, recon


# ===========================================================================
# Section 8 — Visualization
# ===========================================================================

def plot_raw(
    tensor: np.ndarray,
    rt_axis: np.ndarray,
    dt_axis: np.ndarray,
    mz_axis: np.ndarray,
    title: str = "Raw Tensor",
    gauss_sigma: Tuple[float, float] = (1.0, 1.0),
    figsize: Tuple[float, float] = (12, 4),
) -> plt.Figure:
    """RT×DT heatmap and m/z spectrum of the raw tensor.

    Parameters
    ----------
    gauss_sigma :
        (sigma_rt, sigma_dt) for display smoothing of the heatmap only.
        Does not modify the tensor returned to the caller.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    rtdt = tensor.sum(axis=2)          # (n_rt, n_dt), sum over m/z
    mz_spec = tensor.sum(axis=(0, 1))  # (n_mz,),      sum over RT and DT

    if gauss_sigma[0] > 0 or gauss_sigma[1] > 0:
        rtdt_disp = gaussian_filter(rtdt.astype(np.float64), sigma=list(gauss_sigma))
    else:
        rtdt_disp = rtdt.astype(np.float64)

    ax = axes[0]
    im = ax.pcolormesh(
        rt_axis, dt_axis, rtdt_disp.T, cmap="Blues", shading="auto"
    )
    ax.set_xlabel("Retention time (min)")
    ax.set_ylabel("Drift bin")
    ax.set_title("RT × DT  (Σ over m/z)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    ax.plot(mz_axis, mz_spec, linewidth=0.7, color="steelblue")
    ax.set_xlabel("m/z (Da)")
    ax.set_ylabel("Intensity")
    ax.set_title("m/z spectrum  (Σ over RT × DT)")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    return fig


def plot_factors(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    rt_axis: np.ndarray,
    dt_axis: np.ndarray,
    mz_axis: np.ndarray,
    tensor: Optional[np.ndarray] = None,
    residual: Optional[np.ndarray] = None,
    quality_info: Optional[Dict[int, Dict]] = None,
    mask_rt: Optional[np.ndarray] = None,
    mask_dt: Optional[np.ndarray] = None,
    mask_mz: Optional[np.ndarray] = None,
    rt_axis_full: Optional[np.ndarray] = None,
    dt_axis_full: Optional[np.ndarray] = None,
    mz_axis_full: Optional[np.ndarray] = None,
    title: str = "",
    row_height: float = 2.5,
    gauss_sigma: Tuple[float, float] = (1.0, 1.0),
) -> plt.Figure:
    """Per-factor RT×DT heatmap, RT profile, DT profile, and m/z spectrum.

    Row layout:
        Row 0          : raw tensor projections (if *tensor* is given)
        Rows 1 … R     : one row per NTF factor
        Last row       : residual projections (if *residual* is given)

    Each row has four panels:
        Col 0: RT×DT heatmap (outer product of factor A[:,r] and B[:,r],
               or tensor/residual marginal for the raw/residual rows)
        Col 1: RT profile
        Col 2: DT profile
        Col 3: m/z spectrum

    Expansion to full axes
    ----------------------
    Factor vectors live in the masked (compressed) coordinate space.  If
    *rt_axis_full*, *dt_axis_full*, and *mz_axis_full* are ALL provided,
    factor vectors are zero-padded back to the original axis lengths for
    display.  When the full axes are not provided (the common case), all
    panels are plotted directly in the masked-space axes — which is correct
    because *tensor* and *residual* are also in that same masked space.
    """
    R = A.shape[1]
    n_rows = R + (1 if tensor is not None else 0) + (1 if residual is not None else 0)

    fig = plt.figure(figsize=(14, row_height * n_rows + 0.5))
    gs = gridspec.GridSpec(n_rows, 4, figure=fig, hspace=0.55, wspace=0.3)

    # Decide whether to expand factor vectors to the original (pre-masking) axes.
    # We only expand when ALL three full axes are provided — otherwise the
    # factor axes and the tensor/residual axes would be inconsistent lengths.
    use_full = (
        rt_axis_full is not None
        and dt_axis_full is not None
        and mz_axis_full is not None
    )
    rt_ax_plt = rt_axis_full if use_full else rt_axis
    dt_ax_plt = dt_axis_full if use_full else dt_axis
    mz_ax_plt = mz_axis_full if use_full else mz_axis

    # Precompute % contribution of each factor relative to the tensor total.
    # Each factor's total signal = A[:,r].sum() * B[:,r].sum() * C[:,r].sum()
    # (integral of the rank-1 outer-product tensor over all three modes).
    tensor_total = float(tensor.sum()) if tensor is not None else None
    factor_totals = [
        float(A[:, r].sum()) * float(B[:, r].sum()) * float(C[:, r].sum())
        for r in range(R)
    ]
    recon_total = sum(factor_totals) + 1e-12
    # Guard against a zero tensor (can occur when the slice is all-zero after
    # masking but NTF still ran).  Without the epsilon ref_total=0 causes a
    # ZeroDivisionError in the factor percentage computation below.
    ref_total = max(tensor_total if tensor_total is not None else recon_total, 1e-12)

    def _expand_vec(vec: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """Expand *vec* from masked space to full space, or return as-is."""
        v = vec.astype(np.float64)
        if use_full and mask is not None:
            return _expand_to_full(v, mask)
        return v

    # Precompute global scale denominators for factor rows so that relative
    # intensity differences between factors are preserved visually.
    # For each axis the denominator is the maximum marginal projection across
    # all factors; the dominant factor reaches 1.0, others scale accordingly.
    _ga_list, _gb_list, _gc_list = [], [], []
    for _r in range(R):
        _ar = _expand_vec(A[:, _r], mask_rt)
        _br = _expand_vec(B[:, _r], mask_dt)
        _cr = _expand_vec(C[:, _r], mask_mz)
        _bs, _cs, _as = float(_br.sum()), float(_cr.sum()), float(_ar.sum())
        _ga_list.append(float((_ar * _bs * _cs).max()))
        _gb_list.append(float((_br * _as * _cs).max()))
        _gc_list.append(float((_cr * _as * _bs).max()))
    _global_scale_a = max(_ga_list) + 1e-12
    _global_scale_b = max(_gb_list) + 1e-12
    _global_scale_c = max(_gc_list) + 1e-12

    def _plot_row(
        row: int,
        a_vec: np.ndarray,
        b_vec: np.ndarray,
        c_vec: np.ndarray,
        label: str,
        q: Optional[Dict],
        is_raw_or_residual: bool = False,
        scale_a: float = 0.0,
        scale_b: float = 0.0,
        scale_c: float = 0.0,
    ) -> None:
        # For factor rows: project the factor's rank-1 tensor onto each axis
        # by multiplying by the sum of the other two modes. This makes the
        # displayed profile represent the same quantity as the raw marginals
        # (i.e. RT profile = Σ_{DT,mz} factor_contribution).
        # For raw/residual rows the vectors are already marginal projections.
        if is_raw_or_residual:
            a_full = a_vec.astype(np.float64)
            b_full = b_vec.astype(np.float64)
            c_full = c_vec.astype(np.float64)
        else:
            a_raw = _expand_vec(a_vec, mask_rt)
            b_raw = _expand_vec(b_vec, mask_dt)
            c_raw = _expand_vec(c_vec, mask_mz)
            # Scale each mode vector by the integral over the other two modes
            b_sum = float(b_raw.sum())
            c_sum = float(c_raw.sum())
            a_sum = float(a_raw.sum())
            a_full = a_raw * b_sum * c_sum   # Σ_{DT,mz} factor[i,j,k]
            b_full = b_raw * a_sum * c_sum   # Σ_{RT,mz} factor[i,j,k]
            c_full = c_raw * a_sum * b_sum   # Σ_{RT,DT} factor[i,j,k]

        # --- Col 0: RT × DT heatmap ---
        # Use raw factor vectors for the heatmap (outer product looks right)
        if is_raw_or_residual:
            hm_a, hm_b = a_full, b_full
        else:
            hm_a = a_raw / (a_raw.max() + 1e-12)  # type: ignore[possibly-undefined]
            hm_b = b_raw / (b_raw.max() + 1e-12)  # type: ignore[possibly-undefined]
        rtdt_mat = np.outer(hm_a, hm_b)
        if gauss_sigma[0] > 0 or gauss_sigma[1] > 0:
            rtdt_mat = gaussian_filter(rtdt_mat, sigma=list(gauss_sigma))
        ax = fig.add_subplot(gs[row, 0])
        cmap = "RdBu_r" if label == "Residual" else "Blues"
        ax.pcolormesh(rt_ax_plt, dt_ax_plt, rtdt_mat.T, cmap=cmap, shading="auto")
        ax.set_title(f"{label} – RT×DT", fontsize=8)
        ax.set_xlabel("RT (min)", fontsize=7)
        ax.set_ylabel("DT bin", fontsize=7)

        # --- Col 1: RT profile (Σ over DT and m/z) ---
        ax = fig.add_subplot(gs[row, 1])
        denom_a = scale_a if scale_a > 0 else (np.abs(a_full).max() + 1e-12)
        a_norm = a_full / denom_a
        ax.plot(rt_ax_plt, a_norm, color="steelblue", linewidth=1.0)
        r2_str = f"  R²={q['rt_r2']:.2f}" if q else ""
        ax.set_title(f"RT  (Σ DT,m/z){r2_str}", fontsize=8)
        ax.set_xlabel("RT (min)", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)

        # --- Col 2: DT profile (Σ over RT and m/z) ---
        ax = fig.add_subplot(gs[row, 2])
        denom_b = scale_b if scale_b > 0 else (np.abs(b_full).max() + 1e-12)
        b_norm = b_full / denom_b
        ax.plot(dt_ax_plt, b_norm, color="darkorange", linewidth=1.0)
        r2_str = f"  R²={q['dt_r2']:.2f}" if q else ""
        ax.set_title(f"DT  (Σ RT,m/z){r2_str}", fontsize=8)
        ax.set_xlabel("DT bin", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)

        # --- Col 3: m/z spectrum (Σ over RT and DT) ---
        ax = fig.add_subplot(gs[row, 3])
        denom_c = scale_c if scale_c > 0 else (np.abs(c_full).max() + 1e-12)
        c_norm = c_full / denom_c
        color = "firebrick" if label == "Residual" else "seagreen"
        ax.plot(mz_ax_plt, c_norm, color=color, linewidth=0.7)
        if label == "Residual":
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title("m/z  (Σ RT,DT)", fontsize=8)
        ax.set_xlabel("m/z (Da)", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    row = 0

    # Raw data row
    if tensor is not None:
        _plot_row(
            row,
            tensor.sum(axis=(1, 2)),   # Σ over DT, m/z  → (n_rt,)
            tensor.sum(axis=(0, 2)),   # Σ over RT, m/z  → (n_dt,)
            tensor.sum(axis=(0, 1)),   # Σ over RT, DT   → (n_mz,)
            "Raw", q=None, is_raw_or_residual=True,
        )
        row += 1

    # Factor rows — label includes % of total tensor intensity.
    # All factor rows are normalised by the same global scale so that visual
    # bar heights reflect relative intensity (dominant factor reaches 1.0).
    for r in range(R):
        q = quality_info.get(r) if quality_info else None
        pct = 100.0 * factor_totals[r] / ref_total
        _plot_row(row, A[:, r], B[:, r], C[:, r],
                  f"Factor {r}  ({pct:.1f}%)", q=q,
                  scale_a=_global_scale_a,
                  scale_b=_global_scale_b,
                  scale_c=_global_scale_c)
        row += 1

    # Residual row — the residual lives in NTF (masked) coordinate space so its
    # marginals must be expanded to the full axes before plotting, just like the
    # factor vectors are expanded in _plot_row via _expand_vec.
    if residual is not None:
        res_a = residual.sum(axis=(1, 2)).astype(np.float64)
        res_b = residual.sum(axis=(0, 2)).astype(np.float64)
        res_c = residual.sum(axis=(0, 1)).astype(np.float64)
        if use_full:
            if mask_rt is not None:
                res_a = _expand_to_full(res_a, mask_rt)
            if mask_dt is not None:
                res_b = _expand_to_full(res_b, mask_dt)
            if mask_mz is not None:
                res_c = _expand_to_full(res_c, mask_mz)
        _plot_row(
            row,
            res_a, res_b, res_c,
            "Residual", q=None, is_raw_or_residual=True,
        )

    fig.suptitle(title, fontsize=10)
    return fig


def plot_correlations(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    title: str = "Factor correlations",
    figsize: Tuple[float, float] = (14, 3.5),
) -> plt.Figure:
    """RT, DT, m/z, and min-correlation heatmaps between all factor pairs.

    Mirrors mhdx_tools' three correlation panels plus the element-wise
    minimum panel with the max off-diagonal value annotated.
    """
    R = A.shape[1]
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    labels = [str(r) for r in range(R)]
    kw = dict(
        annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
        cbar=False, xticklabels=labels, yticklabels=labels,
        annot_kws={"size": 9},
    )

    if R > 1:
        corrA = np.corrcoef(A.T)
        corrB = np.corrcoef(B.T)
        corrC = np.corrcoef(C.T)
    else:
        corrA = corrB = corrC = np.array([[1.0]])

    min_corr = np.minimum(np.minimum(corrA, corrB), corrC)

    sns.heatmap(corrA, ax=axes[0], **kw)
    axes[0].set_title("RT correlation")

    sns.heatmap(corrB, ax=axes[1], **kw)
    axes[1].set_title("DT correlation")

    sns.heatmap(corrC, ax=axes[2], **kw)
    axes[2].set_title("m/z correlation")

    sns.heatmap(min_corr, ax=axes[3], **kw)
    if R > 1:
        off_diag = ~np.eye(R, dtype=bool)
        mc = float(np.max(min_corr[off_diag]))
        axes[3].set_title(f"min(RT,DT,m/z)  max={mc:.2f}")
    else:
        axes[3].set_title("min(RT,DT,m/z)")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    return fig


# ===========================================================================
# Section 9 — Full pipeline wrapper
# ===========================================================================

def analyze_chunk(
    reader: "WatersRawReader",
    function: int,
    rt_lo: float,
    rt_hi: float,
    dt_lo: int,
    dt_hi: int,
    mz_lo: float,
    mz_hi: float,
    *,
    mz_bin: float = MZ_BIN_DEFAULT,
    gauss_sigma_rt: float = 1.0,
    gauss_sigma_dt: float = 1.0,
    intensity_floor: float = 10.0,
    mask_empty: bool = True,
    rank_init: int = 5,
    corr_threshold: float = CORR_THRESHOLD,
    n_iter_max: int = 10_000,
    tol: float = 1e-6,
    rank_max: int = 15,
    min_intensity: float = 0.0,
    n_restarts: int = 3,
    rt_r2_min: float = GAUSS_R2_DEFAULT,
    dt_r2_min: float = GAUSS_R2_DEFAULT,
    apply_quality_filter: bool = True,
    plot: bool = True,
    verbose: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """End-to-end pipeline: build tensor → NTF → filter → (plot).

    Parameters
    ----------
    reader :
        Open WatersRawReader.
    function :
        0-based function index.
    rt_lo, rt_hi : float
        Retention time window [min].
    dt_lo, dt_hi : int
        Drift bin window [0-based].
    mz_lo, mz_hi : float
        m/z window [Da].
    mz_bin :
        Reprofiling resolution [Da]. Default 0.001.
    gauss_sigma_rt, gauss_sigma_dt :
        Gaussian smoothing applied to the tensor before NTF (0 = off).
    intensity_floor :
        Bins with intensity below this value are zeroed after smoothing,
        before masking and NTF.  Default 10.  Set to 0 to disable.
    mask_empty :
        Remove all-zero RT/DT/m/z slices before NTF.
    rank_init, corr_threshold, n_iter_max, tol, rank_max, n_restarts :
        NTF / rank-selection parameters.
    min_intensity :
        Skip if total intensity < this value.
    rt_r2_min, dt_r2_min :
        Gaussian R² thresholds for the quality filter.
    apply_quality_filter :
        If False, skip the Gaussian quality filter step.
    plot :
        If True, generate and return all three figures.
    verbose :
        Print factorization progress.

    Returns
    -------
    result : dict with keys
        ``tensor``          — raw (n_rt, n_dt, n_mz) float32 tensor
        ``rt_axis``         — (n_rt,) float32
        ``dt_axis``         — (n_dt,) float32
        ``mz_axis``         — (n_mz,) float32
        ``A``, ``B``, ``C`` — factor matrices after quality filter
        ``kept_indices``    — original factor indices that survived filter
        ``quality_info``    — dict of Gaussian R² scores per factor
        ``final_rank``      — int
        ``max_corr``        — float
        ``spectra``         — (n_kept, n_mz) m/z spectra, one row per factor
        ``fig_raw``         — matplotlib Figure (or None if plot=False)
        ``fig_factors``     — matplotlib Figure (or None)
        ``fig_corr``        — matplotlib Figure (or None)
        ``mask_rt``, ``mask_dt``, ``mask_mz`` — bool masks (if mask_empty)
    """
    # 1. Build tensor
    if verbose:
        print("[analyze_chunk] Building tensor …")
    tensor, rt_axis, dt_axis, mz_axis = build_tensor(
        reader, function,
        rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi,
        mz_bin=mz_bin,
        gauss_sigma_rt=gauss_sigma_rt,
        gauss_sigma_dt=gauss_sigma_dt,
        intensity_floor=intensity_floor,
    )

    mask_rt = mask_dt = mask_mz = None
    rt_axis_ntf = rt_axis
    dt_axis_ntf = dt_axis
    mz_axis_ntf = mz_axis
    tensor_ntf = tensor

    # 2. Mask empty slices
    if mask_empty:
        if verbose:
            print("[analyze_chunk] Masking empty slices …")
        (tensor_ntf, rt_axis_ntf, dt_axis_ntf, mz_axis_ntf,
         mask_rt, mask_dt, mask_mz) = mask_empty_slices(
            tensor, rt_axis, dt_axis, mz_axis
        )
        if verbose:
            print(
                f"  After masking: "
                f"({tensor_ntf.shape[0]}, {tensor_ntf.shape[1]}, {tensor_ntf.shape[2]})"
                f"  vs raw ({tensor.shape[0]}, {tensor.shape[1]}, {tensor.shape[2]})"
            )

    # 3. Factorize
    if verbose:
        print(f"[analyze_chunk] Factorizing (rank_init={rank_init}) …")
    A, B, C, final_rank, max_corr = factorize(
        tensor_ntf,
        rank_init=rank_init,
        corr_threshold=corr_threshold,
        n_iter_max=n_iter_max,
        tol=tol,
        rank_max=rank_max,
        min_intensity=min_intensity,
        n_restarts=n_restarts,
        verbose=verbose,
        rng=rng,
    )

    if A is None:
        return dict(
            tensor=tensor, rt_axis=rt_axis, dt_axis=dt_axis, mz_axis=mz_axis,
            rt_axis_ntf=rt_axis_ntf, dt_axis_ntf=dt_axis_ntf, mz_axis_ntf=mz_axis_ntf,
            A=None, B=None, C=None, kept_indices=[], quality_info={},
            final_rank=0, max_corr=float("nan"), spectra=None,
            fig_raw=None, fig_factors=None, fig_corr=None,
            mask_rt=mask_rt, mask_dt=mask_dt, mask_mz=mask_mz,
        )

    quality_info: Dict[int, Dict] = {}
    kept_indices = list(range(final_rank))

    # 4. Quality filter
    if apply_quality_filter and final_rank > 0:
        if verbose:
            print("[analyze_chunk] Applying Gaussian quality filter …")
        A, B, C, kept_indices, quality_info = filter_factors(
            A, B, C,
            rt_axis_ntf, dt_axis_ntf,
            rt_r2_min=rt_r2_min,
            dt_r2_min=dt_r2_min,
        )
        if verbose:
            print(f"  {len(kept_indices)}/{final_rank} factors passed.")

    # 5. Recover m/z spectra (using masked mz axis, full length)
    if mask_mz is not None:
        # Expand C back to full m/z axis for spectrum recovery
        C_full = np.zeros((len(mz_axis), C.shape[1]), dtype=C.dtype)
        C_full[mask_mz] = C
        spectra = factor_mz_spectra(A, B, C_full, normalize=True)
    else:
        spectra = factor_mz_spectra(A, B, C, normalize=True)

    # 6. Plots
    fig_raw = fig_factors = fig_corr = None
    if plot:
        if verbose:
            print("[analyze_chunk] Generating plots …")
        chunk_label = (
            f"RT {rt_lo:.2f}–{rt_hi:.2f} min | "
            f"DT {dt_lo}–{dt_hi} | "
            f"m/z {mz_lo:.1f}–{mz_hi:.1f} Da"
        )
        fig_raw = plot_raw(
            tensor, rt_axis, dt_axis, mz_axis,
            title=f"Raw  –  {chunk_label}",
        )
        # Compute NTF reconstruction and residual in the masked (NTF) coordinate
        # space.  reconstruction[i,j,k] = Σ_r  A[i,r] * B[j,r] * C[k,r].
        # The residual stays in NTF space; plot_factors expands its marginals to
        # the full axes internally using the mask parameters.
        reconstruction_ntf = np.einsum("ir,jr,kr->ijk", A, B, C)
        residual_ntf = tensor_ntf.astype(np.float64) - reconstruction_ntf
        fig_factors = plot_factors(
            A, B, C,
            rt_axis_ntf, dt_axis_ntf, mz_axis_ntf,
            tensor=tensor,          # full tensor — marginals match full axes
            residual=residual_ntf,  # NTF-space; marginals expanded via masks inside plot_factors
            quality_info=quality_info if quality_info else None,
            mask_rt=mask_rt, mask_dt=mask_dt, mask_mz=mask_mz,
            rt_axis_full=rt_axis, dt_axis_full=dt_axis, mz_axis_full=mz_axis,
            title=f"Factors (rank={len(kept_indices)}, max_corr={max_corr:.2f})  –  {chunk_label}",
        )
        if len(kept_indices) > 1:
            fig_corr = plot_correlations(
                A, B, C,
                title=f"Factor correlations  –  {chunk_label}",
            )

    return dict(
        tensor=tensor,
        rt_axis=rt_axis,
        dt_axis=dt_axis,
        mz_axis=mz_axis,
        # --- axes that match the shape of A, B, C directly ---
        # C has shape (len(mz_axis_ntf), rank); use mz_axis_ntf (not mz_axis)
        # when calling isotope_analysis.process_all_factors.
        rt_axis_ntf=rt_axis_ntf,
        dt_axis_ntf=dt_axis_ntf,
        mz_axis_ntf=mz_axis_ntf,
        A=A, B=B, C=C,
        kept_indices=kept_indices,
        quality_info=quality_info,
        final_rank=final_rank,
        max_corr=max_corr,
        spectra=spectra,
        fig_raw=fig_raw,
        fig_factors=fig_factors,
        fig_corr=fig_corr,
        mask_rt=mask_rt,
        mask_dt=mask_dt,
        mask_mz=mask_mz,
    )

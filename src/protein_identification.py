"""
protein_identification.py
=========================
Two-stage protein identification from HDX-MS isotope tables.

Overview
--------
Stage 1 — Wide search (50 ppm default)
    For every row in the isotope CSV (k=0, is_best=True recommended), compute
    the neutral mass and search against the target+decoy database within the
    specified ppm window.

Stage 2 — Polyfit re-calibration + narrow search (10 ppm default)
    Build a polynomial correction from Stage 1 target hits (obs_mz vs
    expect_mz residuals as a function of expect_mz), apply it to all observed
    m/z values, then repeat the search with the tighter ppm window.

Decoy generation
----------------
Two random-sequence decoys are generated for every target protein.  Each
decoy sequence is guaranteed to be >50 ppm from all target MWs and all
previously generated decoy MWs.

FDR estimation
--------------
    FDR_sequences = (n_unique_decoy_seqs / 2) / n_unique_target_seqs
    FDR_signals   = (n_decoy_signals    / 2) / n_target_signals

Usage (command line)
--------------------
    python protein_identification.py identify \
        --isotopes   results/final/sample_isotopes_filtered.csv \
        --database   proteins.csv \
        --output     results/id/sample/database_identifications.csv \
        --ppm_stage1 50.0 \
        --ppm_stage2 10.0 \
        --decoy_size 2

    # database.csv must have at minimum two columns:
    #   name     — protein name
    #   sequence — amino acid sequence (standard one-letter codes)
    # MW is computed automatically if the column is absent.

Database CSV format
-------------------
Required columns: ``name``, ``sequence``
Optional columns: ``MW`` (Da, monoisotopic) — computed from BioPython if missing.

Output columns
--------------
``name, sequence, RT, im_mono, ab_cluster_total, MW, charge, expect_mz,
obs_mz, ppm, abs_ppm``
"""

from __future__ import annotations

import argparse
import math
import os
import random
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import molmass as _molmass
except ImportError:
    _molmass = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROTON_MASS = 1.007276   # Da — unified atomic mass convention (NIST)
# The legacy mhdx_tools code uses 1.007825 (H atom mass incl. electron), but
# the IUPAC / UniMod convention for ESI charge-state calculation is the proton
# mass 1.007276 Da.  We use the proton mass throughout for consistency with
# standard proteomics software.

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# ---------------------------------------------------------------------------
# Database utilities
# ---------------------------------------------------------------------------


def _monoisotopic_mass(sequence: str) -> float:
    """Compute monoisotopic MW from an AA sequence using BioPython."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    return float(ProteinAnalysis(sequence, monoisotopic=True).molecular_weight())


def load_database(csv_path: str) -> pd.DataFrame:
    """Load protein database CSV and compute MW if missing.

    Parameters
    ----------
    csv_path :
        Path to a CSV with at minimum ``name`` and ``sequence`` columns.
        If a ``MW`` column is present it is used directly; otherwise MW is
        computed from the sequence using BioPython's monoisotopic molecular
        weight (which includes the water molecule for a full protein chain).

    Returns
    -------
    DataFrame with columns: ``name, sequence, MW`` (plus any extras in the
    input file).
    """
    df = pd.read_csv(csv_path)
    required = {"name", "sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Database CSV is missing columns: {missing}")

    if "MW" not in df.columns:
        print("[load_database] Computing monoisotopic MW from sequences …")
        df["MW"] = df["sequence"].apply(_monoisotopic_mass)

    return df[["name", "sequence", "MW"]].reset_index(drop=True)


def generate_decoys(
    df: pd.DataFrame,
    n_per_target: int = 2,
    ppm_tol: float = 50.0,
    length_min: int = 50,
    length_max: int = 2000,
    max_attempts: int = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate random decoy sequences for each target protein.

    Each decoy:
    - Has a random length drawn uniformly from [length_min, length_max],
      clipped to [len(target)-10%, len(target)+10%] so decoys mirror the
      target mass range.
    - Is composed of random standard amino acids.
    - Has a monoisotopic MW that is > ppm_tol ppm away from every target MW
      and every previously accepted decoy MW.
    - Is labelled ``decoy_<n>`` in the ``name`` column.

    Returns
    -------
    DataFrame with the same columns as the input (``name, sequence, MW``),
    containing BOTH targets and decoys.
    """
    rng = random.Random(seed)

    all_mw = df["MW"].values.tolist()
    decoy_rows: List[Dict] = []

    for _, row in df.iterrows():
        target_mw = row["MW"]
        target_len = len(row["sequence"])
        lo = max(length_min, int(target_len * 0.9))
        hi = min(length_max, int(target_len * 1.1))
        if lo > hi:
            lo, hi = min(lo, hi), max(lo, hi)

        n_accepted = 0
        n_tried = 0
        while n_accepted < n_per_target and n_tried < max_attempts:
            n_tried += 1
            L = rng.randint(lo, hi)
            seq = "".join(rng.choices(AMINO_ACIDS, k=L))
            try:
                mw = _monoisotopic_mass(seq)
            except Exception:
                continue

            # Reject if within ppm_tol ppm of any existing mass (target or decoy)
            too_close = any(
                abs(mw - known) / (known + 1e-12) * 1e6 < ppm_tol
                for known in all_mw
            )
            if too_close:
                continue

            decoy_name = f"decoy_{(len(decoy_rows) + 1):04d}"
            decoy_rows.append({"name": decoy_name, "sequence": seq, "MW": mw})
            all_mw.append(mw)
            n_accepted += 1

        if n_accepted < n_per_target:
            print(
                f"[generate_decoys] WARNING: only {n_accepted}/{n_per_target} "
                f"decoys generated for {row['name']} after {max_attempts} attempts."
            )

    print(
        f"[generate_decoys] Generated {len(decoy_rows)} decoys "
        f"for {len(df)} targets."
    )

    decoy_df = pd.DataFrame(decoy_rows, columns=["name", "sequence", "MW"])
    combined = pd.concat([df, decoy_df], ignore_index=True)
    return combined


# ---------------------------------------------------------------------------
# Mass matching
# ---------------------------------------------------------------------------


def _neutral_mass(obs_mz: float, charge: int) -> float:
    """Convert observed m/z to neutral mass.

    M = z × m/z − z × PROTON
    """
    return obs_mz * charge - charge * PROTON_MASS


def _expect_mz(neutral_mass: float, charge: int) -> float:
    """Convert neutral mass to expected m/z for a given charge state."""
    return (neutral_mass + charge * PROTON_MASS) / charge


def _ppm_error(obs_mz: float, calc_mz: float) -> float:
    """Signed ppm error: (obs − calc) / calc × 1e6."""
    return (obs_mz - calc_mz) / (calc_mz + 1e-12) * 1e6


def search_one(
    obs_mz: float,
    charge: int,
    protein_df: pd.DataFrame,
    ppm: float,
) -> pd.DataFrame:
    """Return rows of *protein_df* within *ppm* of a single observed m/z.

    Parameters
    ----------
    obs_mz :
        Observed monoisotopic m/z.
    charge :
        Assigned charge state.
    protein_df :
        DataFrame with a ``MW`` column (monoisotopic neutral mass).
    ppm :
        Search window in parts-per-million.

    Returns
    -------
    Subset of *protein_df* with ``expect_mz``, ``ppm``, ``abs_ppm`` added.
    """
    neutral = _neutral_mass(obs_mz, charge)
    low  = neutral * (1.0 - ppm / 1e6)
    high = neutral * (1.0 + ppm / 1e6)
    hits = protein_df[(protein_df["MW"] >= low) & (protein_df["MW"] <= high)].copy()
    if hits.empty:
        return hits
    hits["expect_mz"] = hits["MW"].apply(lambda mw: _expect_mz(mw, charge))
    hits["ppm"]     = hits["expect_mz"].apply(lambda e: _ppm_error(obs_mz, e))
    hits["abs_ppm"] = hits["ppm"].abs()
    hits["obs_mz"]  = obs_mz
    hits["charge"]  = charge
    return hits


def _compute_idotp(sequence: str, peak_intensities_str: str) -> float:
    """Isotope dot product between sequence-theoretical and observed envelopes.

    Returns cosine similarity in [0, 1], or NaN if computation is not possible.
    """
    if _molmass is None:
        return float("nan")
    if not isinstance(peak_intensities_str, str) or not peak_intensities_str.strip():
        return float("nan")
    try:
        emp = np.array([float(x) for x in peak_intensities_str.split("|")], dtype=np.float64)
    except ValueError:
        return float("nan")
    if emp.size == 0 or emp.max() <= 0:
        return float("nan")
    emp = emp / emp.max()
    try:
        formula = _molmass.Formula(sequence)
        theo = np.array([v[1] for v in formula.spectrum().values()], dtype=np.float64)
    except Exception:
        return float("nan")
    if theo.size == 0 or theo.max() <= 0:
        return float("nan")
    theo = theo / theo.max()
    n = min(len(theo), len(emp))
    if n == 0:
        return float("nan")
    t, e = theo[:n], emp[:n]
    nt, ne = np.linalg.norm(t), np.linalg.norm(e)
    if nt < 1e-12 or ne < 1e-12:
        return float("nan")
    return float(np.dot(t, e) / (nt * ne))


def identify_at_ppm(
    isotopes_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    ppm: float,
    obs_mz_col: str = "monoisotopic_mz",
    charge_col: str = "charge",
    calibration_fn=None,
) -> pd.DataFrame:
    """Search all rows of *isotopes_df* against *protein_df*.

    Parameters
    ----------
    isotopes_df :
        Isotope table (at minimum ``monoisotopic_mz``, ``charge``, ``RT``,
        ``dt_center``, ``cluster_tic`` columns; the filtered is_best k=0
        subset is recommended).
    protein_df :
        Target+decoy database with ``name, sequence, MW`` columns.
    ppm :
        Search window.
    obs_mz_col :
        Column in *isotopes_df* containing the observed m/z.  Default
        ``"monoisotopic_mz"`` (k=0 value, optionally calibrated).
    charge_col :
        Column containing the charge state.  Default ``"charge"``.
    calibration_fn :
        Optional callable ``f(obs_mz_array) → corrected_mz_array`` applied
        to all observed m/z values before searching.

    Returns
    -------
    DataFrame with one row per (isotope signal × protein hit), containing the
    output columns defined in the module docstring.
    """
    obs_mz_arr = isotopes_df[obs_mz_col].values.astype(np.float64)
    if calibration_fn is not None:
        obs_mz_arr = calibration_fn(obs_mz_arr)

    result_rows: List[Dict] = []

    for i, iso_row in enumerate(isotopes_df.itertuples(index=False)):
        obs_mz = float(obs_mz_arr[i])
        charge  = int(getattr(iso_row, charge_col))
        hits = search_one(obs_mz, charge, protein_df, ppm)
        if hits.empty:
            continue

        # Gather isotope metadata to attach
        rt_val     = float(getattr(iso_row, "rt_center",       float("nan")))
        im_val     = float(getattr(iso_row, "dt_center",       float("nan")))
        ab_total   = float(getattr(iso_row, "cluster_tic",     float("nan")))

        pi_val          = str(getattr(iso_row, "peak_intensities",  "") or "")
        k_val           = int(getattr(iso_row, "k", 0))
        cluster_bpi_val = float(getattr(iso_row, "cluster_bpi",       float("nan")))
        cos_sim_val     = float(getattr(iso_row, "cosine_similarity",  float("nan")))
        rt_r2_val       = float(getattr(iso_row, "rt_gaussian_r2",    float("nan")))
        dt_r2_val       = float(getattr(iso_row, "dt_gaussian_r2",    float("nan")))

        for _, h in hits.iterrows():
            result_rows.append({
                "name":             h["name"],
                "sequence":         h["sequence"],
                "RT":               rt_val,
                "im_mono":          im_val,
                "ab_cluster_total": ab_total,
                "cluster_bpi":      cluster_bpi_val,
                "cosine_similarity": cos_sim_val,
                "rt_gaussian_r2":   rt_r2_val,
                "dt_gaussian_r2":   dt_r2_val,
                "MW":               float(h["MW"]),
                "charge":           charge,
                "expect_mz":        float(h["expect_mz"]),
                "obs_mz":           obs_mz,
                "ppm":              float(h["ppm"]),
                "abs_ppm":          float(h["abs_ppm"]),
                "peak_intensities": pi_val,
                "k":                k_val,
            })

    if not result_rows:
        return pd.DataFrame(columns=[
            "name", "sequence", "RT", "im_mono", "ab_cluster_total",
            "cluster_bpi", "cosine_similarity", "rt_gaussian_r2", "dt_gaussian_r2",
            "MW", "charge", "expect_mz", "obs_mz", "ppm", "abs_ppm",
            "peak_intensities", "k",
        ])

    out = pd.DataFrame(result_rows)
    # Keep best (smallest abs_ppm) hit per signal when the same protein has
    # multiple matching charge states (shouldn't happen within one search, but
    # guard against duplicates from identical MW entries in the DB).
    out = out.sort_values("abs_ppm")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage 2: polyfit re-calibration
# ---------------------------------------------------------------------------


def polyfit_calibration(
    ids_df: pd.DataFrame,
    deg: int = 2,
    n_sigma: float = 3.0,
    min_tic_bpi_ratio: float = 10.0,
    min_cosine_sim: float = 0.98,
    min_rt_r2: float = 0.9,
    min_dt_r2: float = 0.9,
) -> Tuple[np.ndarray, float]:
    """Fit a polynomial m/z correction from Stage 1 target hits.

    Uses only *target* rows (name does not contain "decoy") and performs one
    round of sigma-clipping to remove gross outliers.

    Parameters
    ----------
    ids_df :
        Stage 1 identification table (output of :func:`identify_at_ppm`).
    deg :
        Polynomial degree.  Default 2 (quadratic).
    n_sigma :
        Outlier rejection threshold.  Default 3.0.

    Returns
    -------
    (coeffs, rms_ppm) where *coeffs* are numpy polynomial coefficients
    (highest degree first, as returned by ``np.polyfit``) and *rms_ppm* is
    the residual RMS in ppm after fitting.
    """
    targets = ids_df[~ids_df["name"].str.contains("decoy", case=False, na=False)].copy()
    n_before = len(targets)
    if "cluster_bpi" in targets.columns and "ab_cluster_total" in targets.columns:
        bpi = targets["cluster_bpi"].clip(lower=1e-12)
        targets = targets[(targets["ab_cluster_total"] / bpi) > min_tic_bpi_ratio]
    if "cosine_similarity" in targets.columns:
        targets = targets[targets["cosine_similarity"] > min_cosine_sim]
    if "rt_gaussian_r2" in targets.columns:
        targets = targets[targets["rt_gaussian_r2"] > min_rt_r2]
    if "dt_gaussian_r2" in targets.columns:
        targets = targets[targets["dt_gaussian_r2"] > min_dt_r2]
    print(
        f"[polyfit_calibration] {len(targets)}/{n_before} Stage 1 targets pass "
        f"quality filters (tic/bpi>{min_tic_bpi_ratio}, "
        f"cos_sim>{min_cosine_sim}, rt_r2>{min_rt_r2}, dt_r2>{min_dt_r2})"
    )
    if len(targets) < deg + 2:
        print(
            f"[polyfit_calibration] Only {len(targets)} target hits — "
            f"not enough to fit degree-{deg} polynomial.  Returning identity."
        )
        return np.zeros(deg + 1), float("nan")

    x = targets["expect_mz"].values.astype(np.float64)
    # residual in Da: obs − expect
    y = (targets["obs_mz"].values - targets["expect_mz"].values).astype(np.float64)

    coeffs = np.polyfit(x, y, deg)
    pred   = np.polyval(coeffs, x)
    resid  = y - pred
    sigma  = np.std(resid)

    keep   = np.abs(resid) <= n_sigma * sigma
    n_keep = int(keep.sum())
    if n_keep < deg + 2:
        print(
            f"[polyfit_calibration] After sigma-clipping only {n_keep} hits remain. "
            f"Using pre-clip fit."
        )
    else:
        coeffs = np.polyfit(x[keep], y[keep], deg)
        pred   = np.polyval(coeffs, x[keep])
        resid  = y[keep] - pred

    rms_ppm = float(
        np.sqrt(np.mean((resid / (x[keep] + 1e-12) * 1e6) ** 2))
        if n_keep >= deg + 2
        else float("nan")
    )
    print(
        f"[polyfit_calibration] Fit deg={deg}, {n_keep}/{len(targets)} targets, "
        f"RMS={rms_ppm:.2f} ppm"
    )
    return coeffs, rms_ppm


# ---------------------------------------------------------------------------
# FDR estimation
# ---------------------------------------------------------------------------


def fdr_sequences(ids_df: pd.DataFrame, n_decoy_per_target: int = 2) -> float:
    """FDR estimate based on unique identified sequences.

    FDR = (n_unique_decoy_seqs / n_decoy_per_target) / n_unique_target_seqs
    """
    is_decoy   = ids_df["name"].str.contains("decoy", case=False, na=False)
    n_decoys   = len(set(ids_df.loc[is_decoy,  "sequence"]))
    n_targets  = len(set(ids_df.loc[~is_decoy, "sequence"]))
    if n_targets == 0:
        return float("nan")
    return n_decoys / n_decoy_per_target / n_targets


def fdr_signals(ids_df: pd.DataFrame, n_decoy_per_target: int = 2) -> float:
    """FDR estimate based on total signal counts.

    FDR = (n_decoy_rows / n_decoy_per_target) / n_target_rows
    """
    is_decoy  = ids_df["name"].str.contains("decoy", case=False, na=False)
    n_decoys  = int(is_decoy.sum())
    n_targets = int((~is_decoy).sum())
    if n_targets == 0:
        return float("nan")
    return n_decoys / n_decoy_per_target / n_targets


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------


def _ensure_matplotlib() -> None:
    import matplotlib
    matplotlib.use("Agg")


def plot_identification_diagnostics(
    ids_stage1: pd.DataFrame,
    ids_stage2: pd.DataFrame,
    output_pdf: str,
    n_decoy_per_target: int = 2,
    idotp_min: float = 0.0,
) -> None:
    """Save a multi-page PDF with identification quality diagnostics.

    Pages
    -----
    1. m/z error distributions — Stage 1 histogram (1 ppm bins), Stage 2
       histogram (1 ppm bins), and a KDE overlay of both stages (targets vs
       decoys) in the same panel for direct comparison.
    2. FDR curve vs ppm threshold (sequences and signals, Stage 1 and Stage 2).
    3. Count of identified targets and decoys vs ppm threshold (Stage 1 and
       Stage 2), analogous to the FDR curve.
    4. RT and DT distributions of identified targets vs decoys (Stage 2).
    5. ab_cluster_total distributions (targets vs decoys) — log-scale.
    6. Charge state distribution of identified targets (Stage 2).
    7. idotp distribution (targets vs decoys, Stage 2), with threshold line.
    8. Count vs ppm threshold for idotp-filtered Stage 2 subset.
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.stats import gaussian_kde

    def _is_decoy(df):
        return df["name"].str.contains("decoy", case=False, na=False)

    def _ppm_bins(df1, df2):
        """Fixed 1 ppm bins spanning the union of both dataframes."""
        all_ppm = []
        for df in (df1, df2):
            if len(df) > 0 and "ppm" in df.columns:
                vals = df["ppm"].dropna()
                if len(vals):
                    all_ppm.extend([vals.min(), vals.max()])
        if not all_ppm:
            return np.arange(-55, 56, 1.0)
        lo = math.floor(min(all_ppm))
        hi = math.ceil(max(all_ppm)) + 1
        return np.arange(lo, hi, 1.0)

    with PdfPages(output_pdf) as pdf:
        # ── Page 1: ppm error distributions ──────────────────────────────
        # Three panels: Stage 1 histogram | Stage 2 histogram | KDE overlay
        bins_shared = _ppm_bins(ids_stage1, ids_stage2)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        # Panels 0 and 1: histograms with 1 ppm bins
        for ax, df, title in zip(
            axes[:2],
            [ids_stage1, ids_stage2],
            ["Stage 1 (wide search)", "Stage 2 (polyfit-calibrated)"],
        ):
            dec_mask = _is_decoy(df)
            ppm_bins = _ppm_bins(df, df)
            ax.hist(df.loc[~dec_mask, "ppm"].dropna(), bins=ppm_bins,
                    color="steelblue", alpha=0.7, label="Target")
            ax.hist(df.loc[dec_mask,  "ppm"].dropna(), bins=ppm_bins,
                    color="firebrick", alpha=0.7, label="Decoy")
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("ppm error", fontsize=9)
            ax.set_ylabel("Count",     fontsize=9)
            ax.legend(fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)

        # Panel 2: KDE overlay of Stage 1 and Stage 2
        ax_kde = axes[2]
        kde_styles = [
            (ids_stage1, False, "steelblue", "-",  "S1 Target"),
            (ids_stage1, True,  "firebrick", "-",  "S1 Decoy"),
            (ids_stage2, False, "steelblue", "--", "S2 Target"),
            (ids_stage2, True,  "firebrick", "--", "S2 Decoy"),
        ]
        x_kde = np.linspace(bins_shared[0], bins_shared[-1], 500)
        for df, use_decoy, color, ls, label in kde_styles:
            mask = _is_decoy(df) if use_decoy else ~_is_decoy(df)
            vals = df.loc[mask, "ppm"].dropna().values
            if len(vals) >= 2:
                try:
                    kde = gaussian_kde(vals, bw_method="scott")
                    y_kde = kde(x_kde)
                    # Scale to counts for comparability
                    bin_width = 1.0
                    ax_kde.plot(x_kde, y_kde * len(vals) * bin_width,
                                color=color, linestyle=ls, linewidth=1.2,
                                label=label)
                except Exception:
                    pass
        ax_kde.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax_kde.set_title("KDE overlay — Stage 1 vs Stage 2", fontsize=10)
        ax_kde.set_xlabel("ppm error", fontsize=9)
        ax_kde.set_ylabel("Density × N × bin", fontsize=9)
        ax_kde.legend(fontsize=7, ncol=2)
        ax_kde.spines[["top", "right"]].set_visible(False)

        fig.suptitle("m/z error distributions", fontsize=11)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 2: FDR curves vs ppm threshold ──────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, df, title in zip(
            axes,
            [ids_stage1, ids_stage2],
            ["Stage 1", "Stage 2"],
        ):
            ppms = np.arange(1, 51, 1)
            fdr_seq_vals, fdr_sig_vals = [], []
            for ppm_cut in ppms:
                sub = df[df["abs_ppm"] <= ppm_cut]
                fdr_seq_vals.append(fdr_sequences(sub, n_decoy_per_target) * 100)
                fdr_sig_vals.append(fdr_signals(sub, n_decoy_per_target) * 100)
            ax.plot(ppms, fdr_seq_vals, label="FDR (sequences)", color="steelblue")
            ax.plot(ppms, fdr_sig_vals, label="FDR (signals)",   color="firebrick", linestyle="--")
            ax.axhline(1.0, color="gray", linewidth=0.7, linestyle=":", label="1 % FDR")
            ax.set_title(f"FDR curve — {title}", fontsize=10)
            ax.set_xlabel("ppm threshold", fontsize=9)
            ax.set_ylabel("FDR (%)",       fontsize=9)
            ax.set_ylim(0, min(50, max(fdr_seq_vals + fdr_sig_vals + [5])))
            ax.legend(fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 3: Count vs ppm threshold ───────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, df, title in zip(
            axes,
            [ids_stage1, ids_stage2],
            ["Stage 1", "Stage 2"],
        ):
            ppms = np.arange(1, 51, 1)
            tgt_counts, dec_counts = [], []
            dec_mask = _is_decoy(df)
            for ppm_cut in ppms:
                sub = df[df["abs_ppm"] <= ppm_cut]
                sub_dec = _is_decoy(sub)
                # Count unique protein names (sequences) to avoid double-counting
                tgt_counts.append(len(set(sub.loc[~sub_dec, "name"])))
                dec_counts.append(len(set(sub.loc[sub_dec,  "name"])))
            ax.plot(ppms, tgt_counts, label="Targets",
                    color="steelblue", linewidth=1.5)
            ax.plot(ppms, dec_counts, label="Decoys",
                    color="firebrick", linewidth=1.5, linestyle="--")
            ax.set_title(f"Identifications vs ppm threshold — {title}", fontsize=10)
            ax.set_xlabel("ppm threshold", fontsize=9)
            ax.set_ylabel("Unique proteins identified", fontsize=9)
            ax.legend(fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Count of identified proteins vs ppm threshold", fontsize=11)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if ids_stage2.empty:
            return

        # ── Page 4: RT and DT distributions ──────────────────────────────
        dec_mask = _is_decoy(ids_stage2)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, col, xlabel in zip(
            axes,
            ["RT", "im_mono"],
            ["Retention time (min)", "Drift time (bins)"],
        ):
            if col not in ids_stage2.columns:
                continue
            ax.hist(ids_stage2.loc[~dec_mask, col].dropna(), bins=40,
                    color="steelblue", alpha=0.7, label="Target")
            ax.hist(ids_stage2.loc[dec_mask,  col].dropna(), bins=40,
                    color="firebrick", alpha=0.7, label="Decoy")
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.legend(fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("RT and DT distributions (Stage 2)", fontsize=11)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 5: abundance distribution ───────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 4))
        col = "ab_cluster_total"
        if col in ids_stage2.columns:
            tgt = ids_stage2.loc[~dec_mask, col].dropna()
            dec = ids_stage2.loc[dec_mask,  col].dropna()
            log_min = np.log10(max(min(tgt.min() if len(tgt) else 1,
                                       dec.min() if len(dec) else 1), 1))
            log_max = np.log10(max(tgt.max() if len(tgt) else 10,
                                   dec.max() if len(dec) else 10) + 1)
            bins = np.logspace(log_min, log_max, 40)
            ax.hist(tgt, bins=bins, color="steelblue", alpha=0.7, label="Target")
            ax.hist(dec, bins=bins, color="firebrick", alpha=0.7, label="Decoy")
            ax.set_xscale("log")
            ax.set_xlabel("ab_cluster_total (counts)", fontsize=9)
            ax.set_ylabel("Count",                     fontsize=9)
            ax.legend(fontsize=8)
            ax.set_title("Signal abundance — targets vs decoys", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 6: charge state distribution ────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 4))
        tgt_charges = ids_stage2.loc[~dec_mask, "charge"].value_counts().sort_index()
        dec_charges = ids_stage2.loc[dec_mask,  "charge"].value_counts().sort_index()
        all_z = sorted(set(tgt_charges.index) | set(dec_charges.index))
        x = np.arange(len(all_z))
        w = 0.35
        ax.bar(x - w/2, [tgt_charges.get(z, 0) for z in all_z],
               width=w, color="steelblue", alpha=0.8, label="Target")
        ax.bar(x + w/2, [dec_charges.get(z, 0) for z in all_z],
               width=w, color="firebrick", alpha=0.8, label="Decoy")
        ax.set_xticks(x)
        ax.set_xticklabels([str(z) for z in all_z])
        ax.set_xlabel("Charge state", fontsize=9)
        ax.set_ylabel("Count",        fontsize=9)
        ax.legend(fontsize=8)
        ax.set_title("Charge state distribution (Stage 2 targets)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Page 7: idotp distribution ────────────────────────────────────
        if "idotp" in ids_stage2.columns:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for ax, use_decoy, color, label in [
                (axes[0], False, "steelblue", "Targets"),
                (axes[1], True,  "firebrick", "Decoys"),
            ]:
                mask = _is_decoy(ids_stage2) if use_decoy else ~_is_decoy(ids_stage2)
                vals = ids_stage2.loc[mask, "idotp"].dropna()
                ax.hist(vals, bins=np.linspace(0, 1, 41), color=color, alpha=0.8)
                if idotp_min > 0:
                    ax.axvline(idotp_min, color="black", linewidth=1.2,
                               linestyle="--", label=f"threshold = {idotp_min:.2f}")
                    ax.legend(fontsize=8)
                ax.set_xlim(0, 1)
                ax.set_xlabel("idotp", fontsize=9)
                ax.set_ylabel("Count", fontsize=9)
                ax.set_title(f"idotp distribution — Stage 2 {label}", fontsize=10)
                ax.spines[["top", "right"]].set_visible(False)
            fig.suptitle("Stage 3 — sequence-theoretical idotp", fontsize=11)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # ── Page 8: Count vs ppm, idotp-filtered subset ───────────────
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            idotp_filter = (
                ids_stage2["idotp"].notna() & (ids_stage2["idotp"] >= idotp_min)
                if idotp_min > 0 else pd.Series(True, index=ids_stage2.index)
            )
            ids2_idotp = ids_stage2[idotp_filter]
            for ax, use_decoy, color, label in [
                (axes[0], False, "steelblue", "Targets"),
                (axes[1], True,  "firebrick", "Decoys"),
            ]:
                mask = _is_decoy(ids2_idotp) if use_decoy else ~_is_decoy(ids2_idotp)
                sub_df = ids2_idotp[mask]
                ppms = np.arange(1, 51, 1)
                counts = [
                    len(set(sub_df.loc[sub_df["abs_ppm"] <= p, "name"]))
                    for p in ppms
                ]
                ax.plot(ppms, counts, color=color, linewidth=1.5)
                ax.set_xlabel("ppm threshold", fontsize=9)
                ax.set_ylabel("Unique proteins identified", fontsize=9)
                ax.set_title(
                    f"Identifications vs ppm — idotp-filtered {label}", fontsize=10
                )
                ax.spines[["top", "right"]].set_visible(False)
            thresh_label = (
                f"idotp ≥ {idotp_min:.2f}" if idotp_min > 0 else "idotp unfiltered"
            )
            fig.suptitle(
                f"Count vs ppm threshold — Stage 2 ({thresh_label})", fontsize=11
            )
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"[plot_identification_diagnostics] Saved → {output_pdf}")


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------

_ID_COLUMNS = [
    "name", "sequence", "RT", "im_mono", "ab_cluster_total",
    "MW", "charge", "expect_mz", "obs_mz", "ppm", "abs_ppm", "idotp",
]


def run_identification(
    isotopes_path: str,
    database_path: str,
    output_path: str,
    output_pdf: Optional[str] = None,
    ppm_stage1: float = 50.0,
    ppm_stage2: float = 10.0,
    polyfit_deg: int = 2,
    decoy_size: int = 2,
    best_only: bool = True,
    k0_only: bool = True,
    database_prebuilt: bool = False,
    idotp_min: float = 0.0,
    calib_min_tic_bpi_ratio: float = 10.0,
    calib_min_cosine_sim: float = 0.98,
    calib_min_rt_r2: float = 0.9,
    calib_min_dt_r2: float = 0.9,
) -> pd.DataFrame:
    """Full two-stage identification pipeline.

    Parameters
    ----------
    isotopes_path :
        Path to the isotope CSV (e.g. ``*_isotopes_filtered.csv`` or
        ``*_isotopes_cal.csv``).
    database_path :
        Path to the protein database CSV (``name``, ``sequence``,
        optionally ``MW``).  If *database_prebuilt* is True this file is
        expected to already contain both targets and decoys (i.e. the output
        of the ``generate_decoys`` subcommand or the Snakemake
        ``generate_decoys`` rule) and is loaded directly without generating
        additional decoys.
    output_path :
        Destination CSV for the final identification table.
    output_pdf :
        Optional path for the diagnostic PDF.  If None, no PDF is saved.
    ppm_stage1 :
        Stage 1 ppm search window.  Default 50.
    ppm_stage2 :
        Stage 2 ppm search window (after polyfit re-calibration).  Default 10.
    polyfit_deg :
        Polynomial degree for the m/z re-calibration fit.  Default 2.
    decoy_size :
        Number of decoys per target.  Default 2.  Ignored when
        *database_prebuilt* is True.
    best_only :
        If True (default), restrict the isotope table to rows where
        ``is_best == True`` before searching.
    k0_only :
        If True (default), restrict to k=0 rows (the monoisotopic peak row).
    database_prebuilt :
        If True, *database_path* is treated as an already-merged
        target+decoy CSV (e.g. from the Snakemake ``generate_decoys`` rule)
        and no decoy generation is performed.

    Returns
    -------
    Stage 2 identification DataFrame.
    """
    # Load and filter isotope table
    print(f"[run_identification] Loading isotopes: {isotopes_path}")
    iso = pd.read_csv(isotopes_path)
    print(f"  {len(iso)} rows loaded")

    if k0_only and "k" in iso.columns:
        iso = iso[iso["k"] == 0].copy()
        print(f"  {len(iso)} rows after k=0 filter")

    if best_only and "is_best" in iso.columns:
        iso = iso[iso["is_best"]].copy()
        print(f"  {len(iso)} rows after is_best filter")

    # Load database (and optionally generate decoys)
    print(f"[run_identification] Loading database: {database_path}")
    if database_prebuilt:
        # Pre-built target+decoy CSV — load directly without MW recalculation
        db_with_decoys = pd.read_csv(database_path)
        if not {"name", "sequence", "MW"}.issubset(db_with_decoys.columns):
            raise ValueError(
                "Pre-built database CSV must contain 'name', 'sequence', 'MW' columns."
            )
        n_targets = int((~db_with_decoys["name"].str.contains("decoy", case=False, na=False)).sum())
        n_decoys  = int(db_with_decoys["name"].str.contains("decoy", case=False, na=False).sum())
        print(f"  {n_targets} targets + {n_decoys} decoys loaded (pre-built)")
    else:
        db = load_database(database_path)
        print(f"  {len(db)} target proteins")
        db_with_decoys = generate_decoys(db, n_per_target=decoy_size)
        print(f"  {len(db_with_decoys)} entries (targets + decoys)")

    # Stage 1: wide search
    print(f"\n[run_identification] Stage 1 — {ppm_stage1} ppm search …")
    ids1 = identify_at_ppm(iso, db_with_decoys, ppm=ppm_stage1)
    n_tgt1 = int((~ids1["name"].str.contains("decoy", case=False, na=False)).sum())
    n_dec1 = int(ids1["name"].str.contains("decoy", case=False, na=False).sum())
    fdr1_seq = fdr_sequences(ids1, decoy_size)
    fdr1_sig = fdr_signals(ids1, decoy_size)
    print(
        f"  {n_tgt1} target hits, {n_dec1} decoy hits | "
        f"FDR seq={fdr1_seq*100:.1f}%, sig={fdr1_sig*100:.1f}%"
    )

    # Stage 2: polyfit re-calibration + narrow search
    print(f"\n[run_identification] Stage 2 — polyfit calibration + {ppm_stage2} ppm search …")
    coeffs, rms_ppm = polyfit_calibration(
        ids1,
        deg=polyfit_deg,
        min_tic_bpi_ratio=calib_min_tic_bpi_ratio,
        min_cosine_sim=calib_min_cosine_sim,
        min_rt_r2=calib_min_rt_r2,
        min_dt_r2=calib_min_dt_r2,
    )

    def _calibration_fn(obs_mz_arr: np.ndarray) -> np.ndarray:
        """Apply the polyfit correction: corrected = obs - polynomial(obs)."""
        correction = np.polyval(coeffs, obs_mz_arr)
        return obs_mz_arr - correction

    ids2 = identify_at_ppm(
        iso, db_with_decoys, ppm=ppm_stage2, calibration_fn=_calibration_fn
    )
    n_tgt2 = int((~ids2["name"].str.contains("decoy", case=False, na=False)).sum())
    n_dec2 = int(ids2["name"].str.contains("decoy", case=False, na=False).sum())
    fdr2_seq = fdr_sequences(ids2, decoy_size)
    fdr2_sig = fdr_signals(ids2, decoy_size)
    print(
        f"  {n_tgt2} target hits, {n_dec2} decoy hits | "
        f"FDR seq={fdr2_seq*100:.1f}%, sig={fdr2_sig*100:.1f}%"
    )

    # Stage 3: idotp filtering (sequence-theoretical isotope dot product)
    ids2["idotp"] = [
        _compute_idotp(row["sequence"], row.get("peak_intensities", ""))
        for _, row in ids2.iterrows()
    ]
    n_with_idotp = int(ids2["idotp"].notna().sum())
    print(f"\n[run_identification] Stage 3 — idotp computed for {n_with_idotp}/{len(ids2)} hits")
    if idotp_min > 0:
        idotp_valid = ids2["idotp"].notna()
        ids2 = ids2[~idotp_valid | (ids2["idotp"] >= idotp_min)].copy()
        n_tgt3 = int((~ids2["name"].str.contains("decoy", case=False, na=False)).sum())
        n_dec3 = int(ids2["name"].str.contains("decoy", case=False, na=False).sum())
        print(f"  idotp ≥ {idotp_min}: {n_tgt3} target hits, {n_dec3} decoy hits remaining")

    # Ensure canonical columns and write output
    for col in _ID_COLUMNS:
        if col not in ids2.columns:
            ids2[col] = float("nan")
    ids2 = ids2[_ID_COLUMNS]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ids2.to_csv(output_path, index=False)
    print(f"\n[run_identification] Final table → {output_path}")

    # Diagnostic plots
    if output_pdf is not None:
        plot_identification_diagnostics(
            ids1, ids2, output_pdf,
            n_decoy_per_target=decoy_size,
            idotp_min=idotp_min,
        )

    return ids2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="HDX-MS protein identification — two-stage search with FDR estimation"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- identify ----
    p_id = sub.add_parser(
        "identify",
        help="Run the full two-stage identification pipeline"
    )
    p_id.add_argument("--isotopes",   required=True,
                      help="Path to the isotope CSV (filtered or calibrated)")
    p_id.add_argument("--database",   required=True,
                      help="Path to the protein database CSV (name, sequence[, MW])")
    p_id.add_argument("--output",     required=True,
                      help="Output path for the identification CSV")
    p_id.add_argument("--output_pdf", default=None,
                      help="Optional path for the diagnostic PDF")
    p_id.add_argument("--ppm_stage1", type=float, default=50.0,
                      help="Stage 1 search window in ppm (default: 50)")
    p_id.add_argument("--ppm_stage2", type=float, default=10.0,
                      help="Stage 2 search window in ppm after polyfit (default: 10)")
    p_id.add_argument("--polyfit_deg", type=int,  default=2,
                      help="Polynomial degree for m/z re-calibration (default: 2)")
    p_id.add_argument("--decoy_size",  type=int,  default=2,
                      help="Number of decoys per target protein (default: 2)")
    p_id.add_argument("--no_best_filter", action="store_true",
                      help="Search all rows, not just is_best=True")
    p_id.add_argument("--all_k",          action="store_true",
                      help="Search both k=0 and k=1 rows (default: k=0 only)")
    p_id.add_argument("--database_prebuilt", action="store_true",
                      help="Treat --database as a pre-built target+decoy CSV "
                           "(skip internal decoy generation — used by Snakemake workflow)")
    p_id.add_argument("--idotp_min", type=float, default=0.0,
                      help="Minimum idotp (sequence-theoretical) to accept a Stage 2 hit "
                           "(0 = disabled, reported in output but not filtered)")
    p_id.add_argument("--calib_min_tic_bpi_ratio", type=float, default=10.0,
                      help="Minimum cluster_tic/cluster_bpi ratio for calibration anchors (default: 10)")
    p_id.add_argument("--calib_min_cosine_sim", type=float, default=0.98,
                      help="Minimum cosine_similarity for calibration anchors (default: 0.98)")
    p_id.add_argument("--calib_min_rt_r2", type=float, default=0.9,
                      help="Minimum rt_gaussian_r2 for calibration anchors (default: 0.9)")
    p_id.add_argument("--calib_min_dt_r2", type=float, default=0.9,
                      help="Minimum dt_gaussian_r2 for calibration anchors (default: 0.9)")

    # ---- generate_decoys ----
    p_dec = sub.add_parser(
        "generate_decoys",
        help="Pre-generate and save the target+decoy database to a CSV"
    )
    p_dec.add_argument("--database",  required=True)
    p_dec.add_argument("--output",    required=True)
    p_dec.add_argument("--decoy_size", type=int, default=2)
    p_dec.add_argument("--ppm_tol",   type=float, default=50.0)
    p_dec.add_argument("--seed",      type=int, default=42)

    args = parser.parse_args()

    if args.cmd == "identify":
        run_identification(
            isotopes_path=args.isotopes,
            database_path=args.database,
            output_path=args.output,
            output_pdf=args.output_pdf,
            ppm_stage1=args.ppm_stage1,
            ppm_stage2=args.ppm_stage2,
            polyfit_deg=args.polyfit_deg,
            decoy_size=args.decoy_size,
            best_only=not args.no_best_filter,
            k0_only=not args.all_k,
            database_prebuilt=args.database_prebuilt,
            idotp_min=args.idotp_min,
            calib_min_tic_bpi_ratio=args.calib_min_tic_bpi_ratio,
            calib_min_cosine_sim=args.calib_min_cosine_sim,
            calib_min_rt_r2=args.calib_min_rt_r2,
            calib_min_dt_r2=args.calib_min_dt_r2,
        )

    elif args.cmd == "generate_decoys":
        db = load_database(args.database)
        combined = generate_decoys(
            db,
            n_per_target=args.decoy_size,
            ppm_tol=args.ppm_tol,
            seed=args.seed,
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        combined.to_csv(args.output, index=False)
        print(f"Saved {len(combined)} entries (targets + decoys) → {args.output}")

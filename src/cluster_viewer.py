"""
cluster_viewer.py
=================
Helper functions for visualising isotopic cluster PNGs from the isotopes
dataframe inside a Jupyter notebook.

Usage:

    from cluster_viewer import show_cluster, show_clusters

The PNG filename is built by the pipeline using the averagine envelope bounds,
not the slice m/z bounds:

    {base_path}/{sample}/
        RT{rt_lo:.1f}-{rt_hi:.1f}
        _DT{dt_lo}-{dt_hi}
        _mz{monoisotopic_mz:.3f}-{last_envelope_peak:.3f}
        _Factor{factor_idx:02d}
        _cluster{cluster_idx:02d}
        _z{charge}
        _k{0|1}.png           ← 1 if left_shifted else 0

Because the last envelope peak is not stored in the dataframe, _png_path
uses a glob with monoisotopic_mz as a prefix and a wildcard for the upper
bound, which uniquely identifies the file given the other known fields.

The trailing k in the filename refers to whether the cluster's best
monoisotopic assignment is the left-shifted candidate (k=1) or the
primary position (k=0).  Both k=0 and k=1 dataframe rows for the same
cluster share the same PNG.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd


# ---------------------------------------------------------------------------
# Path reconstruction
# ---------------------------------------------------------------------------

def _png_path(row: pd.Series, base_path: Union[str, Path]) -> Optional[Path]:
    """Return the PNG path for one dataframe row, or None if not found.

    Uses a glob on monoisotopic_mz (the known lower bound of the mz segment)
    with a wildcard for the upper bound, since the last averagine envelope
    peak is not stored in the dataframe.
    """
    k_png = 1 if row["left_shifted"] else 0

    pattern = (
        f"RT{row['rt_lo']:.1f}-{row['rt_hi']:.1f}"
        f"_DT{int(row['dt_lo'])}-{int(row['dt_hi'])}"
        f"_mz{row['monoisotopic_mz']:.3f}-*"
        f"_Factor{int(row['factor_idx']):02d}"
        f"_cluster{int(row['cluster_idx']):02d}"
        f"_z{int(row['charge'])}"
        f"_k{k_png}.png"
    )
    matches = glob.glob(str(Path(base_path) / str(row["sample"]) / pattern))
    if not matches:
        return None
    if len(matches) > 1:
        # Shouldn't happen in practice; take the first and warn
        print(f"[cluster_viewer] Warning: {len(matches)} files matched, using first:\n  {matches[0]}")
    return Path(matches[0])


# ---------------------------------------------------------------------------
# Single-row viewer
# ---------------------------------------------------------------------------

def show_cluster(
    row: pd.Series,
    base_path: Union[str, Path],
    width: int = 900,
) -> None:
    """Display the PNG for a single row from the isotopes dataframe.

    Parameters
    ----------
    row : pd.Series
        One row from the isotopes CSV (k=0 or k=1 — both point to the same PNG).
    base_path : str or Path
        Root directory where per-sample PNG folders live.
        Example: "/scratch/ajf4103/hdx/plots"
    width : int
        Display width in pixels (notebook only, default 900).

    Example
    -------
    >>> df = pd.read_csv("260330_AF2501_01_isotopes_cal.csv")
    >>> row = df[df["is_best"] & (df["k"] == 0)].iloc[0]
    >>> show_cluster(row, base_path="/scratch/ajf4103/hdx/plots")
    """
    from IPython.display import display, Image

    path = _png_path(row, base_path)
    if path is None:
        print(f"[show_cluster] PNG not found for row:")
        _print_row_summary(row)
        return

    print(_row_label(row))
    display(Image(filename=str(path), width=width))


# ---------------------------------------------------------------------------
# Multi-row viewer
# ---------------------------------------------------------------------------

def show_clusters(
    df: pd.DataFrame,
    base_path: Union[str, Path],
    max_rows: int = 10,
    width: int = 900,
    missing: str = "warn",
) -> None:
    """Display PNGs for every row in a (filtered) dataframe slice.

    Parameters
    ----------
    df : pd.DataFrame
        Subset of the isotopes dataframe to display.  It is recommended to
        filter first, e.g.:
            df[df["is_best"] & (df["k"] == 0) & (df["fit_quality"] == "good")]
    base_path : str or Path
        Root directory where per-sample PNG folders live.
    max_rows : int
        Safety cap — raises if more than this many rows are passed (default 10).
        Set to None to disable.
    width : int
        Display width in pixels (default 900).
    missing : "warn" | "error" | "skip"
        What to do when a PNG file is not found.

    Example
    -------
    >>> good = df[(df["is_best"]) & (df["k"] == 0) & (df["fit_quality"] == "good")]
    >>> show_clusters(good.head(5), base_path="/scratch/ajf4103/hdx/plots")
    """
    from IPython.display import display, Image

    if max_rows is not None and len(df) > max_rows:
        raise ValueError(
            f"show_clusters received {len(df)} rows but max_rows={max_rows}. "
            f"Slice the dataframe further or pass max_rows=None."
        )

    for _, row in df.iterrows():
        path = _png_path(row, base_path)
        print(_row_label(row))

        if path is None:
            msg = "  [missing] no PNG found for this row"
            if missing == "error":
                raise FileNotFoundError(msg)
            elif missing == "warn":
                print(msg)
            continue  # "skip" or after warning

        display(Image(filename=str(path), width=width))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_png_path(row: pd.Series, base_path: Union[str, Path]) -> Optional[str]:
    """Return the PNG path as a string without displaying it, or None if not found.

    Useful for checking whether a file exists or passing the path to
    another function (e.g. PIL, cv2, shutil.copy).

    Example
    -------
    >>> path = get_png_path(row, base_path)
    >>> if path:
    ...     print(path)
    """
    p = _png_path(row, base_path)
    return str(p) if p is not None else None


def _row_label(row: pd.Series) -> str:
    return (
        f"sample={row['sample']}  "
        f"RT={row['rt_lo']:.2f}–{row['rt_hi']:.2f} min  "
        f"DT={int(row['dt_lo'])}–{int(row['dt_hi'])}  "
        f"mz={row['mz_lo']:.1f}–{row['mz_hi']:.1f}  "
        f"Factor={int(row['factor_idx'])}  "
        f"cluster={int(row['cluster_idx'])}  "
        f"z={int(row['charge'])}  "
        f"mono={row['monoisotopic_mz']:.4f} Da  "
        f"cos={row['cosine_similarity']:.4f}  "
        f"fit={row['fit_quality']}"
    )


def _print_row_summary(row: pd.Series) -> None:
    """Print key columns from a row to help diagnose a missing PNG."""
    for col in ["sample", "rt_lo", "rt_hi", "dt_lo", "dt_hi",
                "mz_lo", "mz_hi", "factor_idx", "cluster_idx",
                "charge", "left_shifted", "k"]:
        if col in row.index:
            print(f"  {col} = {row[col]}")

"""
plot_viewer.py
==============
Helper functions for displaying isotopic cluster and factor diagnostic plots
from the HDX pipeline inside a Jupyter notebook.

Usage::

    from plot_viewer import show_cluster, show_clusters, show_factor_plots

Cluster images
--------------
One image is saved per accepted isotopic cluster using the filename pattern::

    {base_path}/{sample}/clusters/
        RT{rt_center:.1f}
        _DT{dt_center:.1f}
        _mz{monoisotopic_mz:.3f}
        _Factor{factor_idx:02d}
        _cluster{cluster_idx:02d}
        _charge{charge}
        .{ext}        ← "png" or "pdf"

Both k=0 and k=1 dataframe rows for the same cluster share a single image
(the monoisotopic_mz used in the filename is always the k=0 value).

Factor diagnostic images
------------------------
Three images are saved per slice when ``save_factors: true`` in config.yaml::

    {base_path}/{sample}/factors/
        RT{rt_lo:.1f}-{rt_hi:.1f}_DT{dt_lo}-{dt_hi}_mz{mz_lo:.1f}-{mz_hi:.1f}_raw.{ext}
        RT{rt_lo:.1f}-{rt_hi:.1f}_DT{dt_lo}-{dt_hi}_mz{mz_lo:.1f}-{mz_hi:.1f}_factors.{ext}
        RT{rt_lo:.1f}-{rt_hi:.1f}_DT{dt_lo}-{dt_hi}_mz{mz_lo:.1f}-{mz_hi:.1f}_corr.{ext}

``show_factor_plots`` uses a glob pattern so it works regardless of extension.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd


# ===========================================================================
# Cluster helpers
# ===========================================================================

def _cluster_plot_path(row: pd.Series, base_path: Union[str, Path]) -> Optional[Path]:
    """Return the image path for one cluster dataframe row, or None if not found.

    Uses the k=0 monoisotopic_mz (same image for k=0 and k=1 rows).
    A glob wildcard on the extension handles both png and pdf output.

    Parameters
    ----------
    row : one row from the isotopes CSV (any k).
    base_path : root directory containing per-sample ``clusters/`` folders.
    """
    # Use a wildcard for the mz portion of the filename.  Before calibration,
    # the image is saved with the raw monoisotopic_mz; after apply_calibration
    # the DataFrame stores the corrected value, which no longer matches the
    # filename exactly.  Since RT, DT, Factor, cluster, and charge are all
    # stable across calibration, we match on those fields only and let glob
    # handle the mz portion.
    pattern = (
        f"RT{row['rt_center']:.1f}"
        f"_DT{row['dt_center']:.1f}"
        f"_mz*"
        f"_Factor{int(row['factor_idx']):02d}"
        f"_cluster{int(row['cluster_idx']):02d}"
        f"_charge{int(row['charge'])}"
        f".*"
    )

    clusters_dir = Path(base_path) / str(row["sample"]) / "clusters"
    matches = glob.glob(str(clusters_dir / pattern))

    if not matches:
        return None
    if len(matches) > 1:
        print(
            f"[plot_viewer] Warning: {len(matches)} files matched, using first:\n"
            f"  {matches[0]}"
        )
    return Path(matches[0])


def show_cluster(
    row: pd.Series,
    base_path: Union[str, Path],
    width: int = 900,
) -> None:
    """Display the cluster plot for a single dataframe row.

    Parameters
    ----------
    row : pd.Series
        One row from the isotopes CSV (k=0 or k=1 — both point to the same image).
    base_path : str or Path
        Root directory containing per-sample ``clusters/`` sub-folders.
        Example: ``"/scratch/ajf4103/hdx/plots"``
    width : int
        Display width in pixels (notebook only, default 900).

    Example
    -------
    >>> df = pd.read_csv("260330_AF2501_01_isotopes_cal.csv")
    >>> row = df[df["is_best"] & (df["k"] == 0)].iloc[0]
    >>> show_cluster(row, base_path="/scratch/ajf4103/hdx/plots")
    """
    from IPython.display import display, Image

    path = _cluster_plot_path(row, base_path)
    if path is None:
        print("[show_cluster] Image not found for row:")
        _print_row_summary(row)
        return

    print(_cluster_row_label(row))
    display(Image(filename=str(path), width=width))


def show_clusters(
    df: pd.DataFrame,
    base_path: Union[str, Path],
    max_rows: int = 10,
    width: int = 900,
    missing: str = "warn",
) -> None:
    """Display cluster plots for every row in a (filtered) dataframe slice.

    Parameters
    ----------
    df : pd.DataFrame
        Subset of the isotopes dataframe to display.  Recommend filtering first::

            df[df["is_best"] & (df["k"] == 0) & (df["fit_quality"] == "good")]

    base_path : str or Path
        Root directory containing per-sample ``clusters/`` sub-folders.
    max_rows : int
        Safety cap — raises if more than this many rows are passed (default 10).
        Pass ``None`` to disable.
    width : int
        Display width in pixels (default 900).
    missing : "warn" | "error" | "skip"
        Behaviour when an image file is not found.

    Example
    -------
    >>> good = df[df["is_best"] & (df["k"] == 0) & (df["fit_quality"] == "good")]
    >>> show_clusters(good.head(5), base_path="/scratch/ajf4103/hdx/plots")
    """
    from IPython.display import display, Image

    if max_rows is not None and len(df) > max_rows:
        raise ValueError(
            f"show_clusters received {len(df)} rows but max_rows={max_rows}. "
            f"Slice the dataframe further or pass max_rows=None."
        )

    for _, row in df.iterrows():
        path = _cluster_plot_path(row, base_path)
        print(_cluster_row_label(row))

        if path is None:
            msg = "  [missing] no image found for this row"
            if missing == "error":
                raise FileNotFoundError(msg)
            elif missing == "warn":
                print(msg)
            continue  # "skip" or after warning

        display(Image(filename=str(path), width=width))


def get_cluster_path(row: pd.Series, base_path: Union[str, Path]) -> Optional[str]:
    """Return the cluster image path as a string without displaying it.

    Useful for checking whether a file exists or passing the path to
    another function (e.g. PIL, cv2, shutil.copy).

    Returns None if the file cannot be found.

    Example
    -------
    >>> path = get_cluster_path(row, base_path)
    >>> if path:
    ...     print(path)
    """
    p = _cluster_plot_path(row, base_path)
    return str(p) if p is not None else None


# ===========================================================================
# Factor plot helpers
# ===========================================================================

def _factor_plot_paths(
    row: pd.Series,
    base_path: Union[str, Path],
    plot_types: tuple = ("raw", "factors", "corr"),
) -> dict:
    """Return a dict {plot_type: Path or None} for the factor plots of a slice.

    The slice is identified from the row's rt_lo/rt_hi/dt_lo/dt_hi/mz_lo/mz_hi
    columns.  A glob wildcard on the extension matches both png and pdf outputs.

    Parameters
    ----------
    row : one row from the isotopes CSV (or any object with the slice coordinate
          columns).
    base_path : root directory containing per-sample ``factors/`` sub-folders.
    plot_types : which plot types to look up (default all three).
    """
    factors_dir = Path(base_path) / str(row["sample"]) / "factors"
    slice_tag = (
        f"RT{float(row['rt_lo']):.1f}-{float(row['rt_hi']):.1f}"
        f"_DT{int(row['dt_lo'])}-{int(row['dt_hi'])}"
        f"_mz{float(row['mz_lo']):.1f}-{float(row['mz_hi']):.1f}"
    )

    result = {}
    for pt in plot_types:
        matches = glob.glob(str(factors_dir / f"{slice_tag}_{pt}.*"))
        result[pt] = Path(matches[0]) if matches else None

    return result


def show_factor_plots(
    row: pd.Series,
    base_path: Union[str, Path],
    plot_types: tuple = ("raw", "factors", "corr"),
    width: int = 1000,
    missing: str = "warn",
) -> None:
    """Display the factor diagnostic plots for the slice that produced *row*.

    Looks in ``{base_path}/{sample}/factors/`` for the raw-tensor, NTF-factors,
    and factor-correlation plots belonging to the slice whose coordinates are
    encoded in *row* (rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi).

    Parameters
    ----------
    row : pd.Series
        Any row from the isotopes dataframe — only the slice coordinate columns
        (rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi, sample) are used.
    base_path : str or Path
        Root directory containing per-sample ``factors/`` sub-folders.
    plot_types : tuple of str
        Which plot types to show.  Defaults to all three:
        ``("raw", "factors", "corr")``.
    width : int
        Display width in pixels (default 1000).
    missing : "warn" | "error" | "skip"
        Behaviour when a plot file is not found.

    Example
    -------
    >>> row = df[df["is_best"] & (df["k"] == 0)].iloc[0]
    >>> show_factor_plots(row, base_path="/scratch/ajf4103/hdx/plots")
    """
    from IPython.display import display, Image

    paths = _factor_plot_paths(row, base_path, plot_types=plot_types)
    print(
        f"sample={row['sample']}  "
        f"RT={float(row['rt_lo']):.2f}–{float(row['rt_hi']):.2f} min  "
        f"DT={int(row['dt_lo'])}–{int(row['dt_hi'])}  "
        f"mz={float(row['mz_lo']):.1f}–{float(row['mz_hi']):.1f}"
    )

    for pt, path in paths.items():
        print(f"  [{pt}]", end=" ")
        if path is None:
            msg = f"not found in {Path(base_path) / str(row['sample']) / 'factors'}"
            if missing == "error":
                raise FileNotFoundError(msg)
            elif missing == "warn":
                print(msg)
            else:  # skip
                print("(skipped)")
        else:
            print(str(path))
            display(Image(filename=str(path), width=width))


def show_factor_plots_for_clusters(
    df: pd.DataFrame,
    base_path: Union[str, Path],
    plot_types: tuple = ("raw", "factors", "corr"),
    width: int = 1000,
    max_slices: int = 5,
) -> None:
    """Show factor diagnostic plots for each unique slice in a filtered dataframe.

    Deduplicates by (sample, rt_lo, rt_hi, dt_lo, dt_hi, mz_lo, mz_hi) so
    each slice's factor plots are shown only once, even if *df* contains
    multiple clusters from that slice.

    Parameters
    ----------
    df : pd.DataFrame
        Subset of the isotopes dataframe.
    base_path : str or Path
        Root directory containing per-sample ``factors/`` sub-folders.
    plot_types : tuple of str
        Which plot types to show.
    width : int
        Display width in pixels (default 1000).
    max_slices : int
        Safety cap on unique slices shown (default 5).

    Example
    -------
    >>> good = df[df["is_best"] & (df["k"] == 0) & (df["fit_quality"] == "good")]
    >>> show_factor_plots_for_clusters(good.head(20), base_path="/scratch/ajf4103/hdx/plots")
    """
    slice_cols = ["sample", "rt_lo", "rt_hi", "dt_lo", "dt_hi", "mz_lo", "mz_hi"]
    cols_present = [c for c in slice_cols if c in df.columns]
    unique_df = df.drop_duplicates(subset=cols_present)
    if len(unique_df) > max_slices:
        print(
            f"[show_factor_plots_for_clusters] Reached max_slices={max_slices}. "
            f"Pass a larger value to see more."
        )
        unique_df = unique_df.head(max_slices)
    for _, row in unique_df.iterrows():
        show_factor_plots(row, base_path, plot_types=plot_types, width=width)


def get_factor_paths(
    row: pd.Series,
    base_path: Union[str, Path],
    plot_types: tuple = ("raw", "factors", "corr"),
) -> dict:
    """Return a dict {plot_type: path_str_or_None} for factor diagnostic plots.

    Useful for checking existence or passing paths to external tools.

    Example
    -------
    >>> paths = get_factor_paths(row, base_path)
    >>> for k, v in paths.items():
    ...     print(k, v)
    """
    raw = _factor_plot_paths(row, base_path, plot_types=plot_types)
    return {k: str(v) if v is not None else None for k, v in raw.items()}


# ===========================================================================
# Shared utilities
# ===========================================================================

def _cluster_row_label(row: pd.Series) -> str:
    return (
        f"sample={row['sample']}  "
        f"RT={float(row['rt_lo']):.2f}–{float(row['rt_hi']):.2f} min  "
        f"DT={int(row['dt_lo'])}–{int(row['dt_hi'])}  "
        f"mz={float(row['mz_lo']):.1f}–{float(row['mz_hi']):.1f}  "
        f"Factor={int(row['factor_idx'])}  "
        f"cluster={int(row['cluster_idx'])}  "
        f"z={int(row['charge'])}  "
        f"mono={float(row['monoisotopic_mz']):.4f} Da  "
        f"cos={float(row['cosine_similarity']):.4f}  "
        f"fit={row['fit_quality']}"
    )


def _print_row_summary(row: pd.Series) -> None:
    """Print key columns from a row to help diagnose a missing image."""
    for col in ["sample", "rt_lo", "rt_hi", "dt_lo", "dt_hi",
                "mz_lo", "mz_hi", "rt_center", "dt_center",
                "factor_idx", "cluster_idx", "charge",
                "monoisotopic_mz", "k"]:
        if col in row.index:
            print(f"  {col} = {row[col]}")

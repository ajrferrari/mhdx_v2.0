"""
waters_reader.py
================
High-level Python wrapper around the Waters MassLynx SDK v5.0.0 for efficient
access to .raw files from Synapt HDMS/HDMSE instruments.

Provides chunked, numpy-native access to the full RT × DT × m/z data cube,
along with metadata helpers, TIC/BPI retrieval, and mobilogram extraction.

Requirements
------------
- masslynxsdk 5.0.0 (x86-64 Linux or Windows; install the bundled wheel)
- numpy >= 1.21

Architecture note
-----------------
The underlying ``libMassLynxRaw.so`` / ``MassLynxRaw.dll`` is x86-64 only.
This module cannot be used on ARM systems (e.g. Apple M-series or ARM cloud VMs)
but can be used on any x86-64 Linux or Windows workstation.

Usage
-----
>>> from waters_reader import WatersRawReader, read_license
>>> license_key = read_license("/path/to/license.key")
>>> with WatersRawReader("/path/to/sample.raw", license=license_key) as r:
...     meta = r.metadata()
...     rt, tic = r.read_tic(function=0)
...     mz, inten = r.read_drift_scan(function=0, scan=5, drift=10)
...     cube = r.extract_3d_chunk(
...         function=0,
...         rt_start=4.0, rt_end=7.0,
...         dt_start=3.0, dt_end=9.0,
...         mz_start=700.0, mz_end=900.0,
...     )
"""

from __future__ import annotations

import functools
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# SDK import — graceful failure on unsupported platforms
# ---------------------------------------------------------------------------
try:
    from masslynxsdk import (
        MassLynxRawInfoReader,
        MassLynxRawScanReaderEx,
        MassLynxRawChromatogramReaderEx,
    )
    from masslynxsdk.Providers.MassLynxProvider import MassLynxProvider as _MassLynxProvider
    _SDK_AVAILABLE = True
except (OSError, ModuleNotFoundError) as _e:
    warnings.warn(
        f"MassLynx SDK native library could not be loaded ({_e}). "
        "WatersRawReader requires x86-64 Linux or Windows with masslynxsdk installed. "
        "The module is importable but all reader methods will raise RuntimeError.",
        RuntimeWarning,
        stacklevel=2,
    )
    _SDK_AVAILABLE = False


# ---------------------------------------------------------------------------
# SDK v5.0.0 indexing conventions
#   Functions  : 0-based  (fn=0 → main IMS function, fn=1 → lock-mass, …)
#   RT scans   : 0-based for GetRetentionTime; GetScansInFunction returns count
#   Drift bins : 0-based  (valid range 0 .. GetDriftScanCount(fn) - 1)
# ---------------------------------------------------------------------------
import ctypes as _ctypes

def _get_drift_time_direct(info_reader, function: int, drift: int) -> float:
    """Call getDriftTime(reader, function, drift, &result) directly.

    Workaround for SDK v5.0.0 bug where the Python wrapper omits the
    function parameter, causing MassLynxException('Invalid drift scan number').
    """
    dll = _MassLynxProvider.MassLynxDll
    fn  = dll.getDriftTime
    fn.argtypes = [
        _ctypes.c_void_p,
        _ctypes.c_int,
        _ctypes.c_int,
        _ctypes.POINTER(_ctypes.c_float),
    ]
    fn.restype = _ctypes.c_int
    result = _ctypes.c_float(0.0)
    code = fn(
        info_reader._provider._getReader(),
        _ctypes.c_int(function),
        _ctypes.c_int(drift),
        _ctypes.byref(result),
    )
    if code != 0:
        raise RuntimeError(
            f"getDriftTime failed (fn={function}, drift={drift}): code={code}"
        )
    return float(result.value)


# ---------------------------------------------------------------------------
# License helper
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def read_license(path: str | Path) -> str:
    """Read the Waters SDK license key from *path* and return it as a string.

    The result is cached so repeated calls with the same path (e.g. once per
    slice in a batch run) incur only a single disk read.

    Parameters
    ----------
    path:
        Path to ``license.key`` (shipped with MassLynxSDKDownload_v5.0.0).

    Returns
    -------
    str
        License string, whitespace-stripped.
    """
    return Path(path).read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Data classes for structured returns
# ---------------------------------------------------------------------------

@dataclass
class RawMetadata:
    """Metadata describing a .raw file.

    Attributes
    ----------
    n_functions:
        Total number of acquisition functions (1-indexed in SDK calls).
    n_scans:
        Dict mapping function → number of RT scans.
    n_drift_bins:
        Dict mapping function → number of drift-time bins (0 if not IMS).
    rt_range:
        Dict mapping function → (rt_start_min, rt_end_min).
    mass_range:
        Dict mapping function → (mz_low, mz_high).
    rt_axis:
        Dict mapping function → float32 array of RT values [min] per scan.
    dt_axis:
        Dict mapping function → float32 array of drift-time values [ms] per bin.
    is_continuum:
        Dict mapping function → bool (True = profile mode).
    """
    n_functions: int
    n_scans: Dict[int, int] = field(default_factory=dict)
    n_drift_bins: Dict[int, int] = field(default_factory=dict)
    rt_range: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    mass_range: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    rt_axis: Dict[int, np.ndarray] = field(default_factory=dict)
    dt_axis: Dict[int, np.ndarray] = field(default_factory=dict)
    is_continuum: Dict[int, bool] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        lines = [f"RawMetadata(n_functions={self.n_functions})"]
        for fn in range(self.n_functions):
            rt_lo, rt_hi = self.rt_range.get(fn, (float("nan"), float("nan")))
            mz_lo, mz_hi = self.mass_range.get(fn, (float("nan"), float("nan")))
            ns = self.n_scans.get(fn, 0)
            nd = self.n_drift_bins.get(fn, 0)
            lines.append(
                f"  fn={fn}: {ns} scans × {nd} drift bins | "
                f"RT {rt_lo:.2f}–{rt_hi:.2f} min | m/z {mz_lo:.1f}–{mz_hi:.1f}"
            )
        return "\n".join(lines)


@dataclass
class Chunk3D:
    """Sparse 3D data chunk returned by :meth:`WatersRawReader.extract_3d_chunk`.

    The data are stored in COO (coordinate) sparse format — only non-zero
    voxels are kept — to minimise memory for IMS data.

    Attributes
    ----------
    scan_idx:
        int32 array — RT scan index (0-based, relative to requested range).
    drift_idx:
        int32 array — drift-time bin index (0-based, relative to requested range).
    mz:
        float32 array — m/z values for each non-zero voxel.
    intensity:
        float32 array — intensity values.
    rt_axis:
        float32 array of RT values [min] for the chunk.
    dt_axis:
        float32 array of drift times [ms] for the chunk.
    mz_start:
        Low m/z boundary of the chunk.
    mz_end:
        High m/z boundary of the chunk.

    Methods
    -------
    to_rt_profile():
        Sum intensities along the RT axis → (rt_axis, intensities).
    to_dt_profile():
        Sum intensities along the DT axis → (dt_axis, intensities).
    to_spectrum():
        Sum all RT and DT → (mz, intensities) — averaged spectrum.
    """
    scan_idx: np.ndarray
    drift_idx: np.ndarray
    mz: np.ndarray
    intensity: np.ndarray
    rt_axis: np.ndarray
    dt_axis: np.ndarray
    mz_start: float
    mz_end: float

    def to_rt_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """Collapse DT and m/z → RT profile (rt_axis [min], intensity)."""
        n_rt = len(self.rt_axis)
        rt_inten = np.zeros(n_rt, dtype=np.float64)
        np.add.at(rt_inten, self.scan_idx, self.intensity)
        return self.rt_axis, rt_inten.astype(np.float32)

    def to_dt_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """Collapse RT and m/z → DT profile (dt_axis [ms], intensity)."""
        n_dt = len(self.dt_axis)
        dt_inten = np.zeros(n_dt, dtype=np.float64)
        np.add.at(dt_inten, self.drift_idx, self.intensity)
        return self.dt_axis, dt_inten.astype(np.float32)

    def to_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """Collapse RT and DT → m/z spectrum summed over the chunk."""
        order = np.argsort(self.mz)
        return self.mz[order], self.intensity[order]

    def to_rt_dt_matrix(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return RT × DT intensity matrix.

        Returns
        -------
        rt_axis: float32 (n_rt,)
        dt_axis: float32 (n_dt,)
        matrix: float32 (n_rt, n_dt)
        """
        n_rt = len(self.rt_axis)
        n_dt = len(self.dt_axis)
        mat = np.zeros((n_rt, n_dt), dtype=np.float64)
        np.add.at(mat, (self.scan_idx, self.drift_idx), self.intensity)
        return self.rt_axis, self.dt_axis, mat.astype(np.float32)


# ---------------------------------------------------------------------------
# Main reader class
# ---------------------------------------------------------------------------

class WatersRawReader:
    """High-level reader for Waters .raw directories.

    Opens the MassLynx SDK readers lazily on first use and closes them via
    the context-manager protocol.  All array outputs are numpy arrays.

    Parameters
    ----------
    raw_path:
        Path to a Waters ``.raw`` directory (e.g. ``sample_01.raw``).
    license:
        License key string.  Use :func:`read_license` to load it from file.
        Pass ``""`` for unlicensed demo mode (limited functionality).

    Examples
    --------
    >>> with WatersRawReader("sample.raw", license=read_license("license.key")) as r:
    ...     meta = r.metadata()
    ...     print(meta)
    ...     rt, tic = r.read_tic(function=0)
    """

    def __init__(self, raw_path: str | Path, license: str = "") -> None:
        if not _SDK_AVAILABLE:
            raise RuntimeError(
                "MassLynx SDK native library is unavailable on this platform. "
                "Run on x86-64 Linux or Windows with masslynxsdk installed."
            )
        self._path = str(Path(raw_path).resolve())
        if not os.path.isdir(self._path):
            raise FileNotFoundError(f"Not a directory: {self._path}")
        self._license = license

        # Readers are opened lazily
        self._info: Optional[MassLynxRawInfoReader] = None
        self._scan: Optional[MassLynxRawScanReaderEx] = None
        self._chrom: Optional[MassLynxRawChromatogramReaderEx] = None
        self._meta_cache: Optional[RawMetadata] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "WatersRawReader":
        self._open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _open(self) -> None:
        """Open all SDK readers from the same raw path.

        Only the info reader is opened eagerly; scan and chromatogram readers
        are opened lazily on first use to minimise memory pressure at open time.
        """
        self._info  = MassLynxRawInfoReader(self._path, self._license)
        self._scan  = None   # opened lazily
        self._chrom = None   # opened lazily

    def _require_scan_reader(self) -> None:
        if self._scan is None:
            if self._info is None:
                self._open()
            self._scan = MassLynxRawScanReaderEx(self._info, self._license)

    def _require_chrom_reader(self) -> None:
        if self._chrom is None:
            if self._info is None:
                self._open()
            self._chrom = MassLynxRawChromatogramReaderEx(self._info, self._license)

    def close(self) -> None:
        """Release all SDK resources."""
        # SDK readers do not expose explicit close/destroy — Python GC handles it
        self._info = None
        self._scan = None
        self._chrom = None
        self._meta_cache = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_open(self) -> None:
        """Ensure at minimum the info reader is open."""
        if self._info is None:
            self._open()

    def _to_np32(self, seq) -> np.ndarray:
        """Convert a ctypes/list sequence to float32 numpy array."""
        return np.array(seq, dtype=np.float32)

    def _to_npi32(self, seq) -> np.ndarray:
        """Convert a ctypes/list sequence to int32 numpy array."""
        return np.array(seq, dtype=np.int32)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def metadata(self, force_refresh: bool = False) -> RawMetadata:
        """Return a :class:`RawMetadata` describing all functions in the file.

        Results are cached after the first call.

        Parameters
        ----------
        force_refresh:
            Re-query the SDK even if cached.

        Returns
        -------
        RawMetadata
        """
        self._require_open()
        if self._meta_cache is not None and not force_refresh:
            return self._meta_cache

        n_fn = self._info.GetNumberofFunctions()
        meta = RawMetadata(n_functions=n_fn)

        for fn in range(n_fn):
            try:
                n_total = self._info.GetScansInFunction(fn)
            except Exception as e:
                warnings.warn(
                    f"Function {fn} is not accessible ({e}) — skipping.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            try:
                n_drift = self._info.GetDriftScanCount(fn)
            except Exception:
                n_drift = 0
            meta.n_drift_bins[fn] = n_drift

            # GetScansInFunction returns the number of RT frames directly,
            # NOT the total IMS spectrum count (n_rt × n_drift).
            n_rt = n_total
            meta.n_scans[fn] = n_rt

            rt_lo, rt_hi = self._info.GetAcquisitionTimeRange(fn)
            meta.rt_range[fn] = (float(rt_lo), float(rt_hi))

            mz_lo, mz_hi = self._info.GetAcquisitionMassRange(fn)
            meta.mass_range[fn] = (float(mz_lo), float(mz_hi))

            try:
                meta.is_continuum[fn] = bool(self._info.IsContinuum(fn))
            except Exception:
                meta.is_continuum[fn] = False

            # ── RT axis ───────────────────────────────────────────────
            # Built from GetAcquisitionTimeRange only — no per-scan calls.
            # Waters acquisitions run at fixed scan rate so linspace is exact.
            meta.rt_axis[fn] = np.linspace(
                float(rt_lo), float(rt_hi), n_rt, dtype=np.float32
            )

            # ── DT axis ───────────────────────────────────────────────
            # Populated as bin indices (0 .. n_drift-1) here.
            # Call load_dt_axis_ms(function) after metadata() to replace
            # these with actual drift-time values in milliseconds.
            if n_drift > 0:
                meta.dt_axis[fn] = np.arange(n_drift, dtype=np.float32)
            else:
                meta.dt_axis[fn] = np.array([], dtype=np.float32)

        self._meta_cache = meta
        return meta

    def rt_axis(self, function: int) -> np.ndarray:
        """Return the RT axis [min] as a float32 array for *function*."""
        return self.metadata().rt_axis[function]

    def dt_axis(self, function: int) -> np.ndarray:
        """Return the drift-time axis for *function*.

        Returns bin indices (0-based) unless :meth:`load_dt_axis_ms` has been
        called, after which it returns actual drift times in milliseconds.
        """
        return self.metadata().dt_axis[function]

    def load_dt_axis_ms(self, function: int) -> np.ndarray:
        """Load the actual drift-time axis in milliseconds via the DLL.

        ``metadata()`` populates ``dt_axis`` with 0-based bin indices to avoid
        crashing.  Call this method once after ``metadata()`` to replace those
        indices with real ms values using ``getDriftTime``.

        Parameters
        ----------
        function:
            1-based function index.

        Returns
        -------
        dt_axis_ms: float32 (n_drift,)
            Drift times in milliseconds.  Also stored in ``metadata().dt_axis``.
        """
        self._require_open()
        meta = self.metadata()
        n_drift = meta.n_drift_bins.get(function, 0)
        if n_drift == 0:
            return np.array([], dtype=np.float32)

        # Query first and last bin only; linspace fills the rest.
        # Waters IMS sweeps the wave velocity linearly so the axis is uniform.
        # SDK v5.0.0 wrapper: GetDriftTime(drift_bin) — 0-based, raises
        # MassLynxException instead of segfaulting, so safe to call.
        try:
            dt_first = float(self._info.GetDriftTime(0))
            dt_last  = float(self._info.GetDriftTime(n_drift - 1))
        except Exception as e:
            warnings.warn(
                f"GetDriftTime failed for function {function} ({e}); "
                "dt_axis remains as bin indices.",
                RuntimeWarning,
                stacklevel=2,
            )
            return meta.dt_axis[function]
        axis = np.linspace(dt_first, dt_last, n_drift, dtype=np.float32)
        meta.dt_axis[function] = axis   # update cache in-place
        return axis

    def diagnose(self) -> None:
        """Print the result of each SDK call one at a time to isolate crashes.

        Run this in a cell by itself if ``metadata()`` kills the kernel.  Each
        line is flushed immediately so the last printed line before the crash
        identifies the failing call.
        """
        import sys
        p = lambda *a: (print(*a), sys.stdout.flush())

        p("=== WatersRawReader.diagnose() ===")
        p(f"Raw path : {self._path}")

        p("Step 1: open MassLynxRawInfoReader ...")
        from masslynxsdk import MassLynxRawInfoReader as _IR
        info = _IR(self._path, self._license)
        p("  OK")

        p("Step 2: GetNumberofFunctions ...")
        n_fn = info.GetNumberofFunctions()
        p(f"  n_functions = {n_fn}")

        for fn in range(n_fn):
            p(f"\n--- Function {fn} ---")

            p(f"  GetScansInFunction({fn}) ...")
            n_total = info.GetScansInFunction(fn)
            p(f"    n_total_spectra = {n_total}")

            p(f"  GetDriftScanCount({fn}) ...")
            try:
                n_drift = info.GetDriftScanCount(fn)
                p(f"    n_drift_bins = {n_drift}")
            except Exception as e:
                p(f"    FAILED: {e}")
                n_drift = 0

            p(f"  GetAcquisitionTimeRange({fn}) ...")
            rt_lo, rt_hi = info.GetAcquisitionTimeRange(fn)
            p(f"    RT = {rt_lo:.3f} – {rt_hi:.3f} min")

            p(f"  GetAcquisitionMassRange({fn}) ...")
            mz_lo, mz_hi = info.GetAcquisitionMassRange(fn)
            p(f"    m/z = {mz_lo:.1f} – {mz_hi:.1f} Da")

            # GetScansInFunction returns RT frame count directly
            n_rt = n_total
            p(f"  n_rt_frames = {n_rt}  (= GetScansInFunction directly)")

            if n_drift > 0:
                p(f"  info.GetDriftTime(0) ...")
                try:
                    dt0 = float(info.GetDriftTime(0))
                    p(f"    dt[0] = {dt0:.4f} ms")
                except Exception as e:
                    p(f"    FAILED: {e}")

                p(f"  info.GetDriftTime({n_drift-1}) ...")
                try:
                    dt_last = float(info.GetDriftTime(n_drift - 1))
                    p(f"    dt[{n_drift-1}] = {dt_last:.4f} ms")
                except Exception as e:
                    p(f"    FAILED: {e}")

        p("\nStep 3: open MassLynxRawScanReaderEx from info ...")
        from masslynxsdk import MassLynxRawScanReaderEx as _SR
        scan = _SR(info, self._license)
        p("  OK")

        p("Step 4: open MassLynxRawChromatogramReaderEx from info ...")
        from masslynxsdk import MassLynxRawChromatogramReaderEx as _CR
        chrom = _CR(info, self._license)
        p("  OK")

        p("\n=== diagnose() complete — no crash ===")

    # ------------------------------------------------------------------
    # TIC / BPI
    # ------------------------------------------------------------------

    def read_tic(self, function: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Read the Total Ion Chromatogram for *function*.

        Parameters
        ----------
        function:
            1-based function index.

        Returns
        -------
        rt: float32 (n_scans,)
            Retention times [min].
        intensity: float32 (n_scans,)
            TIC intensity per scan.
        """
        self._require_chrom_reader()
        times, intensities = self._chrom.ReadTIC(function)
        return self._to_np32(times), self._to_np32(intensities)

    def read_bpi(self, function: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Read the Base Peak Intensity chromatogram for *function*.

        SDK v5.0.0 bug: MassLynxRawChromatogramProvider.ReadBPI calls
        ``readTICChromatogram`` instead of ``readBPIChromatogram`` (copy-paste
        error in the provider).  We bypass it and call the DLL directly.

        Returns
        -------
        rt: float32 (n_scans,)
        intensity: float32 (n_scans,)
        """
        self._require_chrom_reader()
        from masslynxsdk.Providers.MassLynxProvider import MassLynxProvider
        import ctypes as _ct
        dll = MassLynxProvider.MassLynxDll
        readBPI = dll.readBPIChromatogram
        readBPI.argtypes = [
            _ct.c_void_p, _ct.c_int,
            _ct.POINTER(_ct.c_void_p), _ct.POINTER(_ct.c_void_p), _ct.POINTER(_ct.c_int),
        ]
        size = _ct.c_int(0)
        pTimes = _ct.c_void_p()
        pIntensities = _ct.c_void_p()
        code = readBPI(
            self._chrom._provider._getReader(),
            function,
            pTimes, pIntensities, size,
        )
        self._chrom.CheckReturnCode(code)
        times = _ct.cast(pTimes, _ct.POINTER(_ct.c_float))[0:size.value]
        intens = _ct.cast(pIntensities, _ct.POINTER(_ct.c_float))[0:size.value]
        self._chrom.ReleaseMemory(pTimes)
        self._chrom.ReleaseMemory(pIntensities)
        return self._to_np32(times), self._to_np32(intens)

    # ------------------------------------------------------------------
    # Mass chromatograms
    # ------------------------------------------------------------------

    def read_xic(
        self,
        function: int,
        mz: float,
        mz_window: float = 0.02,
        products: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extracted Ion Chromatogram (XIC) for a single m/z value.

        Parameters
        ----------
        function:
            1-based function index.
        mz:
            Target m/z [Da].
        mz_window:
            Half-width of m/z extraction window [Da].
        products:
            If True, search product-ion spectra instead.

        Returns
        -------
        rt: float32 (n_scans,)
        intensity: float32 (n_scans,)
        """
        self._require_chrom_reader()
        times, intensities = self._chrom.ReadMassChromatogram(
            function, mz, mz_window, products
        )
        return self._to_np32(times), self._to_np32(intensities)

    def read_xics(
        self,
        function: int,
        mz_list: List[float],
        mz_window: float = 0.02,
        products: bool = False,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Extracted Ion Chromatograms for multiple m/z values simultaneously.

        Parameters
        ----------
        function:
            1-based function index.
        mz_list:
            List of target m/z values [Da].
        mz_window:
            Half-width of m/z extraction window [Da].
        products:
            If True, search product-ion spectra.

        Returns
        -------
        rt: float32 (n_scans,)
        intensities: list of float32 arrays, one per m/z in *mz_list*.
        """
        self._require_chrom_reader()
        times, intensities_list = self._chrom.ReadMassChromatograms(
            function, mz_list, mz_window, products
        )
        rt = self._to_np32(times)
        return rt, [self._to_np32(ints) for ints in intensities_list]

    # ------------------------------------------------------------------
    # Mobilogram
    # ------------------------------------------------------------------

    def read_mobilogram(
        self,
        function: int,
        scan_start: int,
        scan_end: int,
        mz_start: float,
        mz_end: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Drift-time (mobilogram) projection over an RT × m/z window.

        Integrates all scans from *scan_start* to *scan_end* (inclusive,
        0-based) within *mz_start*–*mz_end* and returns the DT profile.

        Parameters
        ----------
        function:
            1-based function index.
        scan_start, scan_end:
            0-based RT scan range (inclusive). Use :meth:`read_mobilogram_rt_range`
            to supply RT times in minutes instead.
        mz_start, mz_end:
            m/z extraction window [Da].

        Returns
        -------
        dt_bins: int32 (n_nonzero,)
            0-based drift bin indices with signal.
        intensity: float32 (n_nonzero,)
            Integrated intensity per drift bin.
        """
        self._require_chrom_reader()
        bins, intensities = self._chrom.ReadMobillogram(
            function, scan_start, scan_end, mz_start, mz_end
        )
        return self._to_npi32(bins), self._to_np32(intensities)

    def read_mobilogram_rt_range(
        self,
        function: int,
        rt_start: float,
        rt_end: float,
        mz_start: float,
        mz_end: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mobilogram using RT range in minutes (auto-converts to scan indices).

        Parameters
        ----------
        function:
            1-based function index.
        rt_start, rt_end:
            RT window [min].
        mz_start, mz_end:
            m/z window [Da].

        Returns
        -------
        dt_axis: float32 (n_dt_bins,)
            Drift times [ms] for *all* bins (zeros where no signal).
        intensity: float32 (n_dt_bins,)
            Integrated intensity.
        """
        self._require_open()
        # GetScanRange returns 0-based indices; ReadMobillogram accepts them directly.
        s_start, s_end = self._info.GetScanRange(function, rt_start, rt_end)
        bins_raw, ints_raw = self.read_mobilogram(function, s_start, s_end, mz_start, mz_end)

        meta = self.metadata()
        n_dt = meta.n_drift_bins[function]
        dt_ax = meta.dt_axis[function]

        out = np.zeros(n_dt, dtype=np.float32)
        if len(bins_raw) > 0:
            # ReadMobillogram returns 0-based drift bin indices
            idx = np.asarray(bins_raw, dtype=np.int32)
            valid = (idx >= 0) & (idx < n_dt)
            np.add.at(out, idx[valid], np.asarray(ints_raw)[valid])

        return dt_ax, out

    # ------------------------------------------------------------------
    # Individual scan / drift-scan access
    # ------------------------------------------------------------------

    def read_scan(
        self, function: int, scan: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a single RT scan (summed over all drift bins).

        Parameters
        ----------
        function:
            1-based function index.
        scan:
            1-based scan index.

        Returns
        -------
        mz: float32
        intensity: float32
        """
        self._require_scan_reader()
        mz, ints = self._scan.ReadScan(function, scan)
        return self._to_np32(mz), self._to_np32(ints)

    def read_drift_scan(
        self, function: int, scan: int, drift: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a single (RT scan, drift bin) cell.

        Parameters
        ----------
        function:
            1-based function index.
        scan:
            1-based RT scan index.
        drift:
            1-based drift bin index.

        Returns
        -------
        mz: float32 (n_peaks,)
        intensity: float32 (n_peaks,)
        """
        self._require_scan_reader()
        mz, ints = self._scan.ReadDriftScan(function, scan, drift)
        return self._to_np32(mz), self._to_np32(ints)

    def read_drift_scan_index(
        self, function: int, scan: int, drift: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a (scan, drift) cell returning integer mass *bin* indices.

        This is faster than :meth:`read_drift_scan` because no float
        conversion happens in the SDK.  Use :meth:`bins_to_mz` to convert
        indices to m/z values.

        Returns
        -------
        bin_indices: int32 (n_peaks,)
            Integer m/z bin indices.
        intensity: float32 (n_peaks,)
        """
        self._require_scan_reader()
        bins, ints = self._scan.ReadDriftScanIndex(function, scan, drift)
        return self._to_npi32(bins), self._to_np32(ints)

    def bins_to_mz(
        self, function: int, scan: int, bin_indices: np.ndarray
    ) -> np.ndarray:
        """Convert integer m/z bin indices to float m/z values.

        Uses ``GetMassScale`` to obtain the calibrated m/z scale for the given
        scan, then indexes into it.

        Parameters
        ----------
        function, scan:
            Identify which calibration to use.
        bin_indices:
            int32 array of bin indices (from :meth:`read_drift_scan_index`).

        Returns
        -------
        mz: float32 (n_peaks,)
        """
        self._require_scan_reader()
        scale_raw, start_idx = self._scan.GetMassScale(function, scan)
        scale = self._to_np32(scale_raw)
        adjusted = np.asarray(bin_indices, dtype=np.int32) - int(start_idx)
        clipped = np.clip(adjusted, 0, len(scale) - 1)
        return scale[clipped]

    # ------------------------------------------------------------------
    # Core 3D extraction — ExtractByBins
    # ------------------------------------------------------------------

    def extract_3d_chunk(
        self,
        function: int,
        rt_start: float,
        rt_end: float,
        dt_start: float,
        dt_end: float,
        mz_start: float,
        mz_end: float,
    ) -> Chunk3D:
        """Extract a 3D (RT × DT × m/z) data chunk via ``ExtractByBins``.

        This is the fastest access path for IMS data — a single SDK call
        retrieves all non-zero voxels within the requested window.

        .. note::
            The SDK's ``ExtractByBins`` returns m/z-*integrated* intensity
            (all peaks within [mz_start, mz_end] summed per voxel).  The
            returned ``Chunk3D.mz`` field therefore contains the bin-centre
            m/z rather than per-peak m/z.  Use :meth:`accumulate_3d` when
            you need exact per-peak m/z values.

        .. note::
            The exact binary layout of the ``ExtractByBins`` output is not
            publicly documented.  The current decoding assumes a dense
            (n_blocks × n_dt_bins) layout when the returned length matches
            that shape, and falls back to a heuristic sparse decode
            otherwise.  Validate against :meth:`accumulate_3d` on first use
            with a new instrument/firmware version.

        Parameters
        ----------
        function:
            1-based function index.
        rt_start, rt_end:
            Retention-time window [min].
        dt_start, dt_end:
            Drift-time window [ms].
        mz_start, mz_end:
            m/z window [Da].

        Returns
        -------
        Chunk3D
            Sparse COO representation of the chunk with helper methods.
        """
        self._require_open()
        meta = self.metadata()

        # GetScanRange/GetDriftRange return 0-based indices.
        scan_lo, scan_hi = self._info.GetScanRange(function, rt_start, rt_end)
        scan_lo = max(0, int(scan_lo))
        scan_hi = min(meta.n_scans[function] - 1, int(scan_hi))

        dt_lo_idx, dt_hi_idx = self._info.GetDriftRange(function, dt_start, dt_end)
        dt_lo_idx = max(0, int(dt_lo_idx))
        dt_hi_idx = min(meta.n_drift_bins[function] - 1, int(dt_hi_idx))

        # Slice axes (both axes are 0-based numpy arrays)
        rt_ax = meta.rt_axis[function][scan_lo : scan_hi + 1]
        dt_ax = meta.dt_axis[function][dt_lo_idx : dt_hi_idx + 1]

        # Call SDK — ExtractByBins takes 0-based scan and drift indices.
        self._require_chrom_reader()
        raw_bins, raw_ints = self._chrom.ExtractByBins(
            function,
            float(mz_start),
            float(mz_end),
            int(scan_lo),
            int(scan_hi),
            int(dt_lo_idx),
            int(dt_hi_idx),
        )

        if len(raw_bins) == 0:
            return Chunk3D(
                scan_idx=np.array([], dtype=np.int32),
                drift_idx=np.array([], dtype=np.int32),
                mz=np.array([], dtype=np.float32),
                intensity=np.array([], dtype=np.float32),
                rt_axis=rt_ax,
                dt_axis=dt_ax,
                mz_start=mz_start,
                mz_end=mz_end,
            )

        # ExtractByBins encodes both scan and drift info in the bin values:
        # The SDK returns interleaved data where each element of raw_bins
        # encodes (scan_block, drift_bin) — decode via the SDK's encoding scheme.
        # Based on SDK source: bins are drift bin indices; data is structured
        # as n_scans × n_drift blocks.
        raw_bins_arr = self._to_npi32(raw_bins)
        raw_ints_arr = self._to_np32(raw_ints)

        n_blocks  = scan_hi - scan_lo + 1
        n_dt_bins = dt_hi_idx - dt_lo_idx + 1

        # Decode: SDK lays out data as consecutive blocks of drift bins per scan
        total = len(raw_bins_arr)
        if total == n_blocks * n_dt_bins:
            # Dense layout: reshape directly
            intensity_mat = raw_ints_arr.reshape(n_blocks, n_dt_bins)
            scan_idx_full, drift_idx_full = np.where(intensity_mat > 0)
            intensity_out = intensity_mat[scan_idx_full, drift_idx_full]
            mz_out = np.full(len(intensity_out), (mz_start + mz_end) / 2.0, dtype=np.float32)
        else:
            # Sparse layout: raw_bins are drift-time encoded indices
            # We unpack by block structure
            scan_idx_list, drift_idx_list, mz_list, int_list = [], [], [], []
            pos = 0
            for bi in range(n_blocks):
                if pos >= total:
                    break
                # count entries for this block = number of non-zero drifts
                # SDK puts all dt entries consecutively per scan block
                # Heuristic: read until we hit a negative or wrap-around sentinel
                block_start = pos
                while pos < total and raw_bins_arr[pos] >= dt_lo_idx:
                    scan_idx_list.append(bi)
                    drift_idx_list.append(int(raw_bins_arr[pos]) - dt_lo_idx)  # 0-based within chunk
                    int_list.append(raw_ints_arr[pos])
                    mz_list.append((mz_start + mz_end) / 2.0)
                    pos += 1

            scan_idx_full = np.array(scan_idx_list, dtype=np.int32)
            drift_idx_full = np.array(drift_idx_list, dtype=np.int32)
            mz_out = np.array(mz_list, dtype=np.float32)
            intensity_out = np.array(int_list, dtype=np.float32)

        return Chunk3D(
            scan_idx=scan_idx_full,
            drift_idx=drift_idx_full,
            mz=mz_out,
            intensity=intensity_out,
            rt_axis=rt_ax,
            dt_axis=dt_ax,
            mz_start=mz_start,
            mz_end=mz_end,
        )

    # ------------------------------------------------------------------
    # Full 3D accumulation via scan iteration (slower but exact)
    # ------------------------------------------------------------------

    def accumulate_3d(
        self,
        function: int,
        rt_start: float,
        rt_end: float,
        mz_start: float,
        mz_end: float,
        dt_start: Optional[float] = None,
        dt_end: Optional[float] = None,
        use_bin_index: bool = True,
    ) -> Chunk3D:
        """Accumulate a 3D chunk by iterating over individual drift scans.

        Slower than :meth:`extract_3d_chunk` but gives exact m/z values for
        each peak (rather than the bin-averaged m/z from ``ExtractByBins``).
        Useful for species-specific extraction where precise m/z matters.

        Parameters
        ----------
        function:
            1-based function index.
        rt_start, rt_end:
            RT window [min].
        mz_start, mz_end:
            m/z window [Da].
        dt_start, dt_end:
            Optional DT window [ms].  If None, all drift bins are used.
        use_bin_index:
            If True, use the faster ``ReadDriftScanIndex`` + ``GetMassScale``
            path instead of ``ReadDriftScan``.

        Returns
        -------
        Chunk3D
            Sparse COO with exact m/z values per peak.
        """
        self._require_open()
        meta = self.metadata()

        # GetScanRange / GetDriftRange return 0-based indices.
        scan_lo, scan_hi = self._info.GetScanRange(function, rt_start, rt_end)
        scan_lo = max(0, int(scan_lo))
        scan_hi = min(meta.n_scans[function] - 1, int(scan_hi))

        n_dt = meta.n_drift_bins[function]
        dt_ax_full = meta.dt_axis[function]

        if dt_start is not None and dt_end is not None and n_dt > 0:
            d_lo, d_hi = self._info.GetDriftRange(function, dt_start, dt_end)
            d_lo = max(0, int(d_lo))
            d_hi = min(n_dt - 1, int(d_hi))
        else:
            d_lo, d_hi = 0, n_dt - 1

        rt_ax = meta.rt_axis[function][scan_lo : scan_hi + 1]
        dt_ax = dt_ax_full[d_lo : d_hi + 1] if n_dt > 0 else np.array([], dtype=np.float32)

        scan_idx_list: List[int] = []
        drift_idx_list: List[int] = []
        mz_list: List[np.ndarray] = []
        int_list: List[np.ndarray] = []

        self._require_scan_reader()
        for si, scan in enumerate(range(scan_lo, scan_hi + 1)):
            for di, drift in enumerate(range(d_lo, d_hi + 1)):
                if use_bin_index:
                    bins, ints = self._scan.ReadDriftScanIndex(function, scan, drift)
                    bins_np = self._to_npi32(bins)
                    ints_np = self._to_np32(ints)
                    if len(bins_np) == 0:
                        continue
                    mz_scale, start_idx = self._scan.GetMassScale(function, scan)
                    scale = self._to_np32(mz_scale)
                    adj = bins_np - int(start_idx)
                    adj_clipped = np.clip(adj, 0, len(scale) - 1)
                    mz_np = scale[adj_clipped]
                else:
                    mz_np, ints_np = self._scan.ReadDriftScan(function, scan, drift)
                    mz_np = self._to_np32(mz_np)
                    ints_np = self._to_np32(ints_np)
                    if len(mz_np) == 0:
                        continue

                # Apply m/z filter
                mask = (mz_np >= mz_start) & (mz_np <= mz_end) & (ints_np > 0)
                if not np.any(mask):
                    continue

                n_hit = int(np.sum(mask))
                scan_idx_list.extend([si] * n_hit)
                drift_idx_list.extend([di] * n_hit)
                mz_list.append(mz_np[mask])
                int_list.append(ints_np[mask])

        if len(scan_idx_list) == 0:
            return Chunk3D(
                scan_idx=np.array([], dtype=np.int32),
                drift_idx=np.array([], dtype=np.int32),
                mz=np.array([], dtype=np.float32),
                intensity=np.array([], dtype=np.float32),
                rt_axis=rt_ax,
                dt_axis=dt_ax,
                mz_start=mz_start,
                mz_end=mz_end,
            )

        return Chunk3D(
            scan_idx=np.array(scan_idx_list, dtype=np.int32),
            drift_idx=np.array(drift_idx_list, dtype=np.int32),
            mz=np.concatenate(mz_list),
            intensity=np.concatenate(int_list),
            rt_axis=rt_ax,
            dt_axis=dt_ax,
            mz_start=mz_start,
            mz_end=mz_end,
        )

    # ------------------------------------------------------------------
    # HDX convenience: per-species RT/DT profiles
    # ------------------------------------------------------------------

    def extract_species_profiles(
        self,
        function: int,
        mz_center: float,
        mz_window: float,
        rt_start: Optional[float] = None,
        rt_end: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract RT and DT intensity profiles for a species by m/z.

        This convenience method integrates over the full RT and DT axes
        (or a specified RT window) for a given m/z band, returning profiles
        suitable for centroid/Gaussian fitting in HDX workflows.

        Parameters
        ----------
        function:
            1-based function index.
        mz_center:
            Central m/z [Da].
        mz_window:
            Half-width of the m/z extraction window [Da].
        rt_start, rt_end:
            Optional RT window [min].  Defaults to the full run.

        Returns
        -------
        dict with keys:
            ``"rt_axis"`` float32 (n_scans,) — RT axis [min]
            ``"dt_axis"`` float32 (n_dt,) — DT axis [ms]
            ``"rt_profile"`` float32 (n_scans,) — intensity vs RT
            ``"dt_profile"`` float32 (n_dt,) — intensity vs DT
            ``"total_intensity"`` float — summed intensity
        """
        self._require_open()
        meta = self.metadata()

        if rt_start is None:
            rt_start = meta.rt_range[function][0]
        if rt_end is None:
            rt_end = meta.rt_range[function][1]

        mz_lo = mz_center - mz_window
        mz_hi = mz_center + mz_window

        # RT profile via XIC
        rt_axis_xic, rt_profile = self.read_xic(function, mz_center, mz_window)

        # DT profile via mobilogram over the RT window
        dt_axis, dt_profile = self.read_mobilogram_rt_range(
            function, rt_start, rt_end, mz_lo, mz_hi
        )

        total = float(rt_profile.sum())

        return {
            "rt_axis": rt_axis_xic,
            "dt_axis": dt_axis,
            "rt_profile": rt_profile,
            "dt_profile": dt_profile,
            "total_intensity": total,
        }

    # ------------------------------------------------------------------
    # CCS / drift time conversion
    # ------------------------------------------------------------------

    def ccs_to_drift_time(self, ccs: float, mass: float, charge: int) -> float:
        """Convert a CCS value [Å²] to drift time [ms].

        Parameters
        ----------
        ccs:
            Collisional cross-section [Å²].
        mass:
            Neutral mass [Da].
        charge:
            Charge state.

        Returns
        -------
        float
            Drift time [ms].
        """
        self._require_open()
        return float(self._info.GetDriftTimeFromCCS(ccs, mass, charge))

    def drift_time_to_ccs(self, drift_time: float, mass: float, charge: int) -> float:
        """Convert a drift time [ms] to CCS [Å²].

        Parameters
        ----------
        drift_time:
            Drift time [ms].
        mass:
            Neutral mass [Da].
        charge:
            Charge state.

        Returns
        -------
        float
            CCS [Å²].
        """
        self._require_open()
        return float(self._info.GetCollisionalCrossSection(drift_time, mass, charge))

    # ------------------------------------------------------------------
    # Convenience: scan header info
    # ------------------------------------------------------------------

    def acquisition_info(self) -> dict:
        """Return acquisition parameters as a dict (sample name, date, etc.)."""
        self._require_open()
        return dict(self._info.GetAcquisitionInfo())

    def __repr__(self) -> str:  # pragma: no cover
        return f"WatersRawReader(path={self._path!r}, open={self._info is not None})"


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def open_raw(
    raw_path: str | Path,
    license_path: Optional[str | Path] = None,
    license_str: str = "",
) -> WatersRawReader:
    """Open a Waters .raw file and return a :class:`WatersRawReader`.

    Convenience wrapper that loads the license key from *license_path* if
    supplied.  The returned reader must be closed explicitly (or used as a
    context manager).

    Parameters
    ----------
    raw_path:
        Path to ``*.raw`` directory.
    license_path:
        Path to ``license.key``.  Takes precedence over *license_str*.
    license_str:
        License key as a string (used if *license_path* is None).

    Returns
    -------
    WatersRawReader
        Already opened (``_open`` has been called).
    """
    if license_path is not None:
        license_str = read_license(license_path)
    reader = WatersRawReader(raw_path, license=license_str)
    reader._open()
    return reader

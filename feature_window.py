from __future__ import annotations

"""
feature_window.py — Sliding-window feature extractor for DriftObserver.

Produces one 6-dim feature vector φ_k every hop_sec seconds, computed
over the last window_sec seconds of multimodal data:

    φ = [HR (BPM),  ln(RMSSD) (ms),  SCL (V),  SCR rate (ev/min),
         PERCLOS (0-1),  IrisVar (variance)]

Design notes
------------
* Physio (ECG/EDA) and ocular streams use absolute time.time() timestamps so
  the two data sources can be windowed together without clock-skew.
* push_physio_chunk() is called every timer tick with the full rolling buffer
  from NIMultiReader.get_data(); it deduplicates against the last seen
  timestamp so no sample is counted twice.
* push_ocular() is called once per tracker frame.
* maybe_emit() returns a φ vector (or None) based on the hop schedule.
* For calibration, calibration_samples() computes several overlapping vectors
  from whatever data has been collected so far.
"""

import time
from collections import deque
from typing import Optional

import numpy as np
from scipy.signal import find_peaks


class FeatureWindow:

    N_FEATURES = 6   # [HR, lnRMSSD, SCL, SCRrate, PERCLOS, IrisVar]

    def __init__(
        self,
        window_sec: float = 60.0,
        hop_sec:    float = 30.0,
        fs_physio:  float = 500.0,
        fps_ocular: float = 30.0,
    ):
        self.window_sec = float(window_sec)
        self.hop_sec    = float(hop_sec)
        self.fs         = float(fs_physio)
        self.fps_oc     = float(fps_ocular)

        # ── Physio buffers (time.time() timestamps) ───────────────────────────
        maxlen_ph = int(self.fs * (self.window_sec + 10.0))
        self._ecg_t: deque = deque(maxlen=maxlen_ph)
        self._ecg_v: deque = deque(maxlen=maxlen_ph)
        self._eda_t: deque = deque(maxlen=maxlen_ph)
        self._eda_v: deque = deque(maxlen=maxlen_ph)
        self._last_ecg_t: float = -np.inf
        self._last_eda_t: float = -np.inf

        # ── Ocular buffers (time.time() timestamps) ───────────────────────────
        maxlen_oc = int(self.fps_oc * (self.window_sec + 10.0))
        self._oc_t:      deque = deque(maxlen=maxlen_oc)
        self._perclos:   deque = deque(maxlen=maxlen_oc)
        self._iris_var:  deque = deque(maxlen=maxlen_oc)

        # ── Hop schedule ─────────────────────────────────────────────────────
        self._next_emit_t: Optional[float] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Data ingestion
    # ─────────────────────────────────────────────────────────────────────────

    def push_physio_chunk(
        self,
        ecg_t: np.ndarray, ecg_v: np.ndarray,
        eda_t: np.ndarray, eda_v: np.ndarray,
    ) -> None:
        """
        Append only the samples newer than the last seen timestamp.
        ecg_t / eda_t are absolute time.time() arrays from NIMultiReader.
        """
        if len(ecg_t) > 0:
            new_mask = ecg_t > self._last_ecg_t
            if new_mask.any():
                for t, v in zip(ecg_t[new_mask], ecg_v[new_mask]):
                    self._ecg_t.append(float(t))
                    self._ecg_v.append(float(v))
                self._last_ecg_t = float(ecg_t[new_mask][-1])

        if len(eda_t) > 0:
            new_mask = eda_t > self._last_eda_t
            if new_mask.any():
                for t, v in zip(eda_t[new_mask], eda_v[new_mask]):
                    self._eda_t.append(float(t))
                    self._eda_v.append(float(v))
                self._last_eda_t = float(eda_t[new_mask][-1])

    def push_ocular(self, t_abs: float, perclos: float, iris_var: float) -> None:
        """
        Push one ocular frame.  t_abs is time.time() at capture.
        NaN values are accepted and handled gracefully during feature computation.
        """
        self._oc_t.append(float(t_abs))
        self._perclos.append(float(perclos) if np.isfinite(perclos) else np.nan)
        self._iris_var.append(float(iris_var) if np.isfinite(iris_var) else np.nan)

    # ─────────────────────────────────────────────────────────────────────────
    # Emission
    # ─────────────────────────────────────────────────────────────────────────

    def maybe_emit(self, t_abs: float) -> Optional[np.ndarray]:
        """
        Return a (6,) feature vector if the hop interval has elapsed, else None.
        """
        if self._next_emit_t is None:
            self._next_emit_t = t_abs + self.hop_sec
            return None

        if t_abs < self._next_emit_t:
            return None

        self._next_emit_t = t_abs + self.hop_sec
        return self._compute(t_abs)

    def calibration_samples(self, sub_window_sec: float = 15.0, sub_hop_sec: float = 5.0) -> np.ndarray:
        """
        Extract multiple overlapping feature vectors from the data currently
        in the buffers, using shorter sub-windows suitable for calibration.

        Returns an (N, 6) array; rows with any NaN are kept (the observer will
        filter finite rows internally).
        """
        if len(self._oc_t) == 0:
            return np.zeros((0, self.N_FEATURES))

        t_end   = float(self._oc_t[-1]) if len(self._oc_t) > 0 else time.time()
        t_start = t_end - self.window_sec

        samples = []
        t = t_start + sub_window_sec
        while t <= t_end + 1e-3:
            phi = self._compute_window(t_center=t, win=sub_window_sec)
            samples.append(phi)
            t += sub_hop_sec

        return np.array(samples, dtype=float) if samples else np.zeros((0, self.N_FEATURES))

    # ─────────────────────────────────────────────────────────────────────────
    # Feature computation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute(self, t_now: float) -> np.ndarray:
        return self._compute_window(t_center=t_now, win=self.window_sec)

    def _compute_window(self, t_center: float, win: float) -> np.ndarray:
        phi = np.full(self.N_FEATURES, np.nan)
        t0  = t_center - win

        # ── ECG features ─────────────────────────────────────────────────────
        if len(self._ecg_t) > 0:
            ecg_t = np.array(self._ecg_t)
            ecg_v = np.array(self._ecg_v)
            mask  = (ecg_t >= t0) & (ecg_t <= t_center)
            seg   = ecg_v[mask]
            if len(seg) >= int(self.fs * 2.0):
                hr, ln_rmssd = self._ecg_features(seg)
                phi[0] = hr
                phi[1] = ln_rmssd

        # ── EDA features ─────────────────────────────────────────────────────
        if len(self._eda_t) > 0:
            eda_t = np.array(self._eda_t)
            eda_v = np.array(self._eda_v)
            mask  = (eda_t >= t0) & (eda_t <= t_center)
            seg   = eda_v[mask]
            if len(seg) >= int(self.fs * 1.0):
                phi[2], phi[3] = self._eda_features(seg)

        # ── Ocular features ──────────────────────────────────────────────────
        if len(self._oc_t) > 0:
            oc_t  = np.array(self._oc_t)
            per   = np.array(self._perclos, dtype=float)
            ivar  = np.array(self._iris_var, dtype=float)
            mask  = (oc_t >= t0) & (oc_t <= t_center)
            if mask.sum() >= 5:
                phi[4] = float(np.nanmean(per[mask]))
                phi[5] = float(np.nanmean(ivar[mask]))

        return phi

    # ─────────────────────────────────────────────────────────────────────────
    # Per-modality feature extractors
    # ─────────────────────────────────────────────────────────────────────────

    def _ecg_features(self, seg: np.ndarray) -> tuple[float, float]:
        """
        Returns (HR in BPM, ln(RMSSD) in ln-ms) from a filtered ECG segment.
        """
        amp = float(np.max(np.abs(seg)))
        if amp < 1e-6:
            return np.nan, np.nan

        # R-peak detection
        peaks, _ = find_peaks(seg, height=0.40 * amp, distance=int(0.30 * self.fs))
        if len(peaks) < 3:
            return np.nan, np.nan

        # RR intervals (ms), physiology-plausible range
        rr_ms = np.diff(peaks) / self.fs * 1000.0
        rr_ms = rr_ms[(rr_ms > 300.0) & (rr_ms < 2000.0)]
        if len(rr_ms) < 2:
            return np.nan, np.nan

        hr      = float(60000.0 / np.mean(rr_ms))
        rmssd   = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
        ln_rmssd = float(np.log(rmssd + 1.0))   # +1 to avoid log(0)
        return hr, ln_rmssd

    def _eda_features(self, seg: np.ndarray) -> tuple[float, float]:
        """
        Returns (SCL mean in volts, SCR rate in events/min).
        """
        scl = float(np.mean(seg))

        diff = np.diff(seg)
        threshold = max(float(np.std(diff)) * 2.5, 1e-9)
        scr_peaks, _ = find_peaks(diff, height=threshold, distance=int(self.fs))

        duration_min = max(len(seg) / self.fs / 60.0, 1e-6)
        scr_rate = float(len(scr_peaks) / duration_min)
        return scl, scr_rate

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._ecg_t.clear(); self._ecg_v.clear()
        self._eda_t.clear(); self._eda_v.clear()
        self._oc_t.clear();  self._perclos.clear(); self._iris_var.clear()
        self._last_ecg_t   = -np.inf
        self._last_eda_t   = -np.inf
        self._next_emit_t  = None

    @property
    def has_enough_data(self) -> bool:
        """True if at least window_sec of data has been buffered."""
        if len(self._oc_t) < 2:
            return False
        span = float(self._oc_t[-1]) - float(self._oc_t[0])
        return span >= self.window_sec

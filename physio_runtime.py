from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

from physio_ni import NIMultiReader
from physio_pipeline import apply_notch, bandpass_ecg
from physio_types import PhysioConfig, PhysioViewData


class PhysioRuntime:
    def __init__(self, cfg: PhysioConfig, save_dir: Path):
        self.cfg = cfg
        self.save_dir = save_dir
        self.reader = NIMultiReader(cfg, save_dir)
        self.running = False
        self.paused = False
        self.session_context = {"session_id": "", "subject_id": "", "block_name": ""}
        self.last_view = PhysioViewData(status_text="No conectado")
        self._hr_bpm: float = 0.0
        self._eda_tonic: float = 0.0
        self._scr_count: int = 0

    def set_session_context(self, session_id: str, subject_id: str, block_name: str):
        self.session_context = {
            "session_id": session_id,
            "subject_id": subject_id,
            "block_name": block_name,
        }
        # Propagate immediately to the reader so CSV rows are labelled correctly
        self.reader.set_session_context(session_id, subject_id, block_name)

    def update_config(self, cfg: PhysioConfig):
        self.cfg = cfg
        self.reader.cfg = cfg

    def start(self):
        if self.running:
            return
        self.reader.start()
        self.running = True
        self.paused = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.running = False
        try:
            self.reader.stop()
        except Exception:
            pass

    def get_hardware_status(self) -> dict:
        return {
            "ecg": self.reader.ecg.status,
            "eda": self.reader.eda.status,
            "ecg_running": self.reader.ecg.running,
            "eda_running": self.reader.eda.running,
        }

    def _estimate_hr(self, ecg_vis: np.ndarray, fs: float) -> float:
        if len(ecg_vis) < int(fs * 1.5):
            return self._hr_bpm
        amp = float(np.max(np.abs(ecg_vis)))
        if amp < 1e-6:
            return 0.0
        threshold = 0.45 * amp
        min_dist = int(0.30 * fs)
        peaks, _ = find_peaks(ecg_vis, height=threshold, distance=min_dist)
        if len(peaks) < 2:
            return self._hr_bpm
        peak_times = peaks[-min(8, len(peaks)):] / fs
        rr = np.diff(peak_times)
        rr = rr[(rr > 0.3) & (rr < 2.0)]
        if len(rr) == 0:
            return self._hr_bpm
        return float(60.0 / np.mean(rr))

    def _estimate_eda_features(self, eda_clean: np.ndarray, fs: float) -> tuple[float, int]:
        if len(eda_clean) == 0:
            return self._eda_tonic, self._scr_count
        tonic = float(np.mean(eda_clean))
        if len(eda_clean) > int(fs):
            diff = np.diff(eda_clean)
            threshold = max(float(np.std(diff)) * 2.5, 1e-6)
            scr_peaks, _ = find_peaks(diff, height=threshold, distance=int(fs))
            scr_count = len(scr_peaks)
        else:
            scr_count = self._scr_count
        return tonic, scr_count

    def pop_pending_window_rows(self):
        return []

    def step(self):
        if not self.running or self.paused:
            return self.last_view

        ecg_t, ecg_v, eda_t, eda_v = self.reader.get_data()
        if len(ecg_t) < 20 or len(eda_t) < 20:
            return self.last_view

        # Independent visible window per stream
        ecg_now = ecg_t[-1]
        eda_now = eda_t[-1]
        ecg_mask = ecg_t >= (ecg_now - self.cfg.visible_window_sec)
        eda_mask = eda_t >= (eda_now - self.cfg.visible_window_sec)
        ecg_raw = ecg_v[ecg_mask]
        eda_raw = eda_v[eda_mask]

        if self.cfg.invert_ecg:
            ecg_raw = -ecg_raw
        if self.cfg.invert_eda:
            eda_raw = -eda_raw

        ecg_vis = bandpass_ecg(
            apply_notch(ecg_raw, self.cfg.fs, self.cfg.notch_mode),
            self.cfg.fs, self.cfg.ecg_vis_hp, self.cfg.ecg_vis_lp,
        )
        eda_clean = apply_notch(eda_raw, self.cfg.fs, self.cfg.notch_mode)

        n = min(len(ecg_raw), len(ecg_vis), len(eda_raw), len(eda_clean))
        if n < 20:
            return self.last_view
        ecg_raw = ecg_raw[-n:]
        ecg_vis = ecg_vis[-n:]
        eda_raw = eda_raw[-n:]
        eda_clean = eda_clean[-n:]
        xp = np.linspace(-self.cfg.visible_window_sec, 0.0, n)

        self._hr_bpm = self._estimate_hr(ecg_vis, self.cfg.fs)
        self._eda_tonic, self._scr_count = self._estimate_eda_features(eda_clean, self.cfg.fs)

        self.last_view = PhysioViewData(
            ok=True,
            master_time_s=float(max(ecg_now, eda_now)),
            xp=xp,
            ecg_raw=ecg_raw,
            ecg_vis=ecg_vis,
            eda_raw=eda_raw,
            eda_clean=eda_clean,
            status_text=(
                f"ECG {self.cfg.ecg_device}/{self.cfg.ecg_channel} | "
                f"EDA {self.cfg.eda_device}/{self.cfg.eda_channel}"
            ),
            hr_bpm=self._hr_bpm,
            eda_tonic_mean=self._eda_tonic,
            scr_count_window=self._scr_count,
            ecg_hw_status=self.reader.ecg.status,
            eda_hw_status=self.reader.eda.status,
        )
        return self.last_view

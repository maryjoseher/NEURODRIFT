from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def apply_notch(x: np.ndarray, fs: float, notch_mode: str, q: float = 30.0) -> np.ndarray:
    if len(x) < 20 or notch_mode == "Off":
        return x.copy()
    f0 = 50.0 if notch_mode == "50 Hz" else 60.0
    w0 = f0 / (fs / 2.0)
    if not (0 < w0 < 1):
        return x.copy()
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, x)


def bandpass_ecg(x: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0, order: int = 3) -> np.ndarray:
    if len(x) < 20:
        return x.copy()
    nyq = fs / 2.0
    low_n = max(low / nyq, 1e-6)
    high_n = min(high / nyq, 0.999999)
    if low_n >= high_n:
        return x.copy()
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, x)

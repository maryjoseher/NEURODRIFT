from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PhysioConfig:
    ecg_device: str = "Dev1"
    ecg_channel: str = "ai0"
    ecg_vmin: float = 0.0
    ecg_vmax: float = 3.3

    eda_device: str = "Dev2"
    eda_channel: str = "ai1"
    eda_vmin: float = -0.1
    eda_vmax: float = 0.1

    fs: float = 500.0
    raw_buffer_sec: float = 600.0
    visible_window_sec: float = 20.0
    common_delay_sec: float = 1.0
    notch_mode: str = "60 Hz"

    invert_ecg: bool = False
    invert_eda: bool = False
    ecg_vis_hp: float = 0.5
    ecg_vis_lp: float = 40.0


@dataclass
class PhysioViewData:
    ok: bool = False
    master_time_s: float = 0.0
    xp: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    ecg_raw: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    ecg_vis: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    eda_raw: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    eda_clean: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    status_text: str = "No conectado"
    hr_bpm: float = 0.0
    eda_tonic_mean: float = 0.0
    scr_count_window: int = 0
    ecg_hw_status: str = "Detenido"
    eda_hw_status: str = "Detenido"

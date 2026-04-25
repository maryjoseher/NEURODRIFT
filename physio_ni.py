from __future__ import annotations

"""
physio_ni.py — Dual-NI acquisition with master-clock synchronization.

Master clock strategy
---------------------
All three subsystems (ocular/FaceMesh, ECG, EDA) use time.time() as their
epoch reference.  FaceMeshTracker already stamps every frame with time.time()
at the moment the frame arrives.  Here we record a single shared anchor:

    master_clock_t0 = time.time()

at the moment NIMultiReader.start() is called.  Every sample then gets:

    pc_timestamp  — absolute wall-clock epoch (time.time())
    master_rel_s  — seconds since master_clock_t0

This lets post-processing align ECG/EDA samples with ocular rows simply by
comparing pc_timestamp values across all CSV files — no drift, no offset.

Enriched CSV columns
--------------------
pc_timestamp, master_rel_s, signal, device, channel, fs_hz,
sample_index, value_volts, block_name, subject_id, session_id
"""

import csv
import threading
import time
from collections import deque
from pathlib import Path

import nidaqmx
import numpy as np
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
from nidaqmx.stream_readers import AnalogMultiChannelReader

from physio_types import PhysioConfig


class _DAQStream:
    def __init__(self, name: str, device: str, channel: str, fs: float, window_sec: float,
                 vmin: float, vmax: float, initial_value: float = 0.0):
        self.name = name
        self.device = device
        self.channel = channel
        self.fs = fs
        self.vmin = vmin
        self.vmax = vmax
        self.buffer_size = int(fs * window_sec)
        self.chunk = max(1, int(fs * 0.020))
        self.lock = threading.Lock()
        self.signal_buffer = deque([initial_value] * self.buffer_size, maxlen=self.buffer_size)
        self.time_buffer = deque([0.0] * self.buffer_size, maxlen=self.buffer_size)
        self.running = False
        self.status = "Detenido"
        self.thread = None
        self.task = None
        self.sample_index = 0

    def start(self, sample_callback=None):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, args=(sample_callback,), daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        try:
            if self.task is not None:
                self.task.stop()
                self.task.close()
        except Exception as e:
            self.status = f"Error al cerrar tarea {self.name}: {e}"

    def _loop(self, sample_callback=None):
        try:
            self.task = nidaqmx.Task()
            self.task.ai_channels.add_ai_voltage_chan(
                f"{self.device}/{self.channel}",
                terminal_config=TerminalConfiguration.RSE,
                min_val=self.vmin,
                max_val=self.vmax,
            )
            self.task.timing.cfg_samp_clk_timing(
                rate=self.fs,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.chunk * 10,
            )
            reader = AnalogMultiChannelReader(self.task.in_stream)
            chunk_buffer = np.zeros((1, self.chunk), dtype=np.float64)
            self.task.start()
            self.status = f"OK — {self.name}: {self.device}/{self.channel} @ {self.fs:.0f} Hz"
            while self.running:
                try:
                    n = reader.read_many_sample(chunk_buffer, number_of_samples_per_channel=self.chunk, timeout=2.0)
                    now = time.time()   # wall-clock epoch, consistent with FaceMeshTracker
                    dt = 1.0 / self.fs
                    t0 = now - (n - 1) * dt
                    with self.lock:
                        for i in range(n):
                            ts = t0 + i * dt
                            val = float(chunk_buffer[0, i])
                            idx = self.sample_index
                            self.sample_index += 1
                            self.signal_buffer.append(val)
                            self.time_buffer.append(ts)
                            if sample_callback is not None:
                                sample_callback(
                                    signal_name=self.name,
                                    pc_timestamp=ts,
                                    device=self.device,
                                    channel=self.channel,
                                    sample_index=idx,
                                    value_volts=val,
                                )
                except Exception as e:
                    self.status = f"Error lectura {self.name}: {e}"
                    time.sleep(0.05)
        except Exception as e:
            self.status = f"Error NI {self.name}: {e}"
            self.running = False

    def get_signal(self):
        with self.lock:
            return np.asarray(self.signal_buffer, dtype=np.float64)

    def get_time(self):
        with self.lock:
            return np.asarray(self.time_buffer, dtype=np.float64)


class NIMultiReader:
    """
    Manages two independent NI tasks (ECG on Dev1, EDA on Dev2).

    Master-clock anchor
    -------------------
    self.master_clock_t0  — time.time() captured at start(); used to compute
    master_rel_s for every sample so all modalities share the same origin.

    Session context
    ---------------
    Call set_session_context(session_id, subject_id, block_name) before or
    after start(); values are written into every CSV row for traceability.
    """

    # CSV header — enriched for multimodal post-processing
    _CSV_HEADER = [
        "pc_timestamp",       # absolute wall-clock (time.time()), epoch seconds — aligns with ocular rows
        "master_rel_s",       # seconds since master_clock_t0 — relative block time
        "signal",             # "ECG" | "EDA"
        "device",             # NI device string, e.g. "Dev1"
        "channel",            # NI channel string, e.g. "ai0"
        "fs_hz",              # nominal sample rate
        "sample_index",       # monotonic counter per stream (reset on start)
        "value_volts",        # raw ADC value in volts
        "block_name",         # experimental block label set by operator
        "subject_id",         # participant identifier
        "session_id",         # session UUID
    ]

    def __init__(self, cfg: PhysioConfig, save_dir: Path):
        self.cfg = cfg
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.raw_csv = self.save_dir / f"physio_raw_dual_{stamp}.csv"
        self.raw_file = open(self.raw_csv, "w", newline="", encoding="utf-8")
        self.raw_writer = csv.writer(self.raw_file)
        self.raw_writer.writerow(self._CSV_HEADER)

        self.write_lock = threading.Lock()

        # Master clock anchor — set when start() is called
        self.master_clock_t0: float = 0.0

        # Session context — updated by PhysioRuntime.set_session_context()
        self._session_id: str = ""
        self._subject_id: str = ""
        self._block_name: str = ""

        self.ecg = _DAQStream(
            "ECG", cfg.ecg_device, cfg.ecg_channel,
            cfg.fs, cfg.raw_buffer_sec, cfg.ecg_vmin, cfg.ecg_vmax, 1.65,
        )
        self.eda = _DAQStream(
            "EDA", cfg.eda_device, cfg.eda_channel,
            cfg.fs, cfg.raw_buffer_sec, cfg.eda_vmin, cfg.eda_vmax, 0.0,
        )

    def set_session_context(self, session_id: str, subject_id: str, block_name: str):
        """Update metadata written into every subsequent CSV row."""
        self._session_id = session_id or ""
        self._subject_id = subject_id or ""
        self._block_name = block_name or ""

    def _sample_callback(self, signal_name: str, pc_timestamp: float, device: str, channel: str,
                         sample_index: int, value_volts: float):
        master_rel = pc_timestamp - self.master_clock_t0 if self.master_clock_t0 > 0 else float("nan")
        fs = self.ecg.fs if signal_name == "ECG" else self.eda.fs
        with self.write_lock:
            self.raw_writer.writerow([
                f"{pc_timestamp:.9f}",
                f"{master_rel:.9f}",
                signal_name,
                device,
                channel,
                f"{fs:.1f}",
                int(sample_index),
                f"{value_volts:.9f}",
                self._block_name,
                self._subject_id,
                self._session_id,
            ])

    def start(self):
        # Capture master clock anchor — same epoch as time.time() used in FaceMeshTracker
        self.master_clock_t0 = time.time()
        self.ecg.start(self._sample_callback)
        self.eda.start(self._sample_callback)

    def stop(self):
        self.ecg.stop()
        self.eda.stop()
        try:
            self.raw_file.flush()
            self.raw_file.close()
        except Exception as e:
            print(f"[NIMultiReader] Advertencia al cerrar archivo CSV: {e}")

    def get_data(self):
        ecg_t = self.ecg.get_time()
        ecg_v = self.ecg.get_signal()
        eda_t = self.eda.get_time()
        eda_v = self.eda.get_signal()
        return ecg_t, ecg_v, eda_t, eda_v

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np

from config import AppConfig


@dataclass
class Baseline:
    mu_ear: float = np.nan
    sd_ear: float = np.nan
    mu_var: float = np.nan
    sd_var: float = np.nan


class OcularEngine:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.idx = {
            "rightEAR": np.array([33, 160, 158, 133, 153, 144], dtype=int),
            "leftEAR": np.array([362, 385, 387, 263, 373, 380], dtype=int),
            "rightIris": np.array([469, 470, 471, 472], dtype=int),
            "leftIris": np.array([474, 475, 476, 477], dtype=int),
            "rightCorners": np.array([33, 133], dtype=int),
            "leftCorners": np.array([362, 263], dtype=int),
        }
        self.reset_all()

    def set_session_context(self, session_id: str, subject_id: str, block_name: str) -> None:
        self.session_id = session_id or ""
        self.subject_id = subject_id or ""
        self.block_name = block_name or ""

    def set_phase(self, phase: str) -> None:
        self.phase = phase or "UNSPECIFIED"

    def reset_all(self) -> None:
        self.phase = "IDLE"
        self.baseline_ready = False
        self.baseline = Baseline()

        self.session_id = ""
        self.subject_id = ""
        self.block_name = ""

        self.frame_id = 0

        self.time: List[float] = []

        self.ear_l: List[float] = []
        self.ear_r: List[float] = []
        self.ear_mean: List[float] = []
        self.ear_filt: List[float] = []
        self.ear_plot: List[float] = []

        self.blink: List[int] = []
        self.closed_eye: List[int] = []
        self.closed_eye_perclos: List[int] = []

        self.xn_raw: List[float] = []
        self.yn_raw: List[float] = []
        self.xn: List[float] = []
        self.yn: List[float] = []

        self.var_iris_raw: List[float] = []
        self.var_iris_clean: List[float] = []
        self.perclos: List[float] = []

        self.valid_raw: List[int] = []
        self.valid_used: List[int] = []
        self.pose_ok: List[int] = []
        self.filled: List[int] = []

        self.yaw_asym_hist: List[float] = []
        self.left_eye_width_hist: List[float] = []
        self.right_eye_width_hist: List[float] = []
        self.jump_iris_hist: List[float] = []
        self.valid_reason: List[str] = []

        self.z_ear: List[float] = []
        self.z_var: List[float] = []
        self.state: List[str] = []
        self.drift_index: List[float] = []

        self.rows: List[Dict[str, Any]] = []
        self._phase_hist: List[str] = []

        self.last_valid_xn = np.nan
        self.last_valid_yn = np.nan
        self.last_valid_ear_plot = np.nan

        self.eye_closed = False
        self.t_close_start = np.nan
        self.last_blink_time = -np.inf
        self.blink_count = 0
        self.ear_filt_last = np.nan

        self.calibration_mode = False
        self.running_mode = False

    def start_calibration(self) -> None:
        self.clear_block_only()
        self.phase = "CALIB_A"
        self.calibration_mode = True
        self.running_mode = False

    def start_running(self) -> None:
        self.clear_block_only()
        self.phase = "RUNNING"
        self.calibration_mode = False
        self.running_mode = True

    def clear_block_only(self) -> None:
        self.frame_id = 0

        self.time.clear()

        self.ear_l.clear()
        self.ear_r.clear()
        self.ear_mean.clear()
        self.ear_filt.clear()
        self.ear_plot.clear()

        self.blink.clear()
        self.closed_eye.clear()
        self.closed_eye_perclos.clear()

        self.xn_raw.clear()
        self.yn_raw.clear()
        self.xn.clear()
        self.yn.clear()

        self.var_iris_raw.clear()
        self.var_iris_clean.clear()
        self.perclos.clear()

        self.valid_raw.clear()
        self.valid_used.clear()
        self.pose_ok.clear()
        self.filled.clear()

        self.yaw_asym_hist.clear()
        self.left_eye_width_hist.clear()
        self.right_eye_width_hist.clear()
        self.jump_iris_hist.clear()
        self.valid_reason.clear()

        self.z_ear.clear()
        self.z_var.clear()
        self.state.clear()
        self.drift_index.clear()
        self.rows.clear()
        self._phase_hist.clear()

        self.last_valid_xn = np.nan
        self.last_valid_yn = np.nan
        self.last_valid_ear_plot = np.nan

        self.eye_closed = False
        self.t_close_start = np.nan
        self.last_blink_time = -np.inf
        self.blink_count = 0
        self.ear_filt_last = np.nan

    def finalize_calibration(self) -> bool:
        t = np.array(self.time, dtype=float)
        valid = np.array(self.valid_used, dtype=int)
        blink = np.array(self.blink, dtype=int)
        closed = np.array(self.closed_eye_perclos, dtype=int)
        ear = np.array(self.ear_mean, dtype=float)
        var_clean = np.array(self.var_iris_clean, dtype=float)
        interp = np.array(self.filled, dtype=int)
        phase_arr = np.array(self._phase_hist, dtype=object)

        mask_phase = phase_arr == "CALIB_D"

        mask = (
            mask_phase
            & (valid == 1)
            & (blink == 0)
            & (closed == 0)
            & (interp == 0)
            & np.isfinite(ear)
        )

        if int(mask.sum()) < self.cfg.min_calibration_frames:
            self.phase = "CALIBRATION_FAILED"
            self.baseline_ready = False
            return False

        mu_ear = float(np.nanmean(ear[mask]))
        sd_ear = float(np.nanstd(ear[mask]))
        if sd_ear < 1e-6:
            sd_ear = 1e-3

        mask_var = mask & np.isfinite(var_clean)
        if int(mask_var.sum()) >= 10:
            mu_var = float(np.nanmean(var_clean[mask_var]))
            sd_var = float(np.nanstd(var_clean[mask_var]))
        else:
            mu_var = 1e-3
            sd_var = 1e-3

        if sd_var < 1e-6:
            sd_var = 1e-3

        self.baseline = Baseline(
            mu_ear=mu_ear,
            sd_ear=sd_ear,
            mu_var=mu_var,
            sd_var=sd_var,
        )
        self.baseline_ready = True
        self.phase = "READY"
        return True

    def step(self, frame_id: int, tnow: float, landmarks: Optional[np.ndarray]) -> Dict[str, Any]:
        self.frame_id += 1

        ear_l = np.nan
        ear_r = np.nan
        ear_mean = np.nan
        ear_plot = np.nan

        xn_raw = np.nan
        yn_raw = np.nan

        xn = self.last_valid_xn
        yn = self.last_valid_yn
        if np.isnan(xn):
            xn = 0.0
        if np.isnan(yn):
            yn = 0.0

        blink_evt = False
        closed_eye = False
        closed_eye_perclos = False

        valid_raw = False
        valid_used = False
        interpolated = 1
        pose_ok = False

        z_ear = np.nan
        z_var = np.nan

        yaw_asym = np.nan
        left_eye_width = np.nan
        right_eye_width = np.nan
        jump = np.nan

        valid_reason = "NO_FACE"

        if landmarks is not None and landmarks.shape[0] >= 478 and np.isfinite(landmarks).all():
            geom = landmarks.copy()
            geom[:, 0] *= self.cfg.frame_w
            geom[:, 1] *= self.cfg.frame_h
            geom[:, 2] *= self.cfg.frame_w

            pose_ok, yaw_asym, left_eye_width, right_eye_width = self._check_pose_quality(geom)

            ear_l = self._compute_ear(geom, self.idx["leftEAR"])
            ear_r = self._compute_ear(geom, self.idx["rightEAR"])
            ear_mean = float(np.nanmean([ear_l, ear_r]))

            xn_l, yn_l = self._iris_local_coords(geom, self.idx["leftCorners"], self.idx["leftIris"])
            xn_r, yn_r = self._iris_local_coords(geom, self.idx["rightCorners"], self.idx["rightIris"])
            xn_raw = float(np.nanmean([xn_l, xn_r]))
            yn_raw = float(np.nanmean([yn_l, yn_r]))

            valid_raw = bool(
                np.isfinite(ear_mean)
                and np.isfinite(xn_raw)
                and np.isfinite(yn_raw)
                and ear_mean > 0
            )
            valid_used = bool(valid_raw and pose_ok)

            if np.isfinite(ear_mean):
                if np.isnan(self.ear_filt_last):
                    self.ear_filt_last = ear_mean
                else:
                    a = self.cfg.ear_smooth_alpha
                    self.ear_filt_last = a * ear_mean + (1.0 - a) * self.ear_filt_last

            if self.baseline_ready:
                th_close = min(
                    self.cfg.blink_ear_thresh_init,
                    self.baseline.mu_ear - self.cfg.blink_close_offset * self.baseline.sd_ear,
                )
                th_open = th_close + self.cfg.blink_hysteresis
            else:
                th_close = self.cfg.blink_ear_thresh_init
                th_open = th_close + self.cfg.blink_hysteresis

            # Blink NO debe depender de valid_used estricto
            blink_valid = bool(np.isfinite(ear_mean) and ear_mean > 0 and valid_raw)

            if blink_valid:
                if (not self.eye_closed) and ear_mean < th_close:
                    self.eye_closed = True
                    self.t_close_start = tnow

                if self.eye_closed and ear_mean > th_open:
                    dur_close = tnow - self.t_close_start

                    if (
                        dur_close >= self.cfg.blink_min_dur_sec
                        and dur_close <= self.cfg.blink_max_dur_sec
                        and (tnow - self.last_blink_time) >= self.cfg.blink_refractory_sec
                    ):
                        blink_evt = True
                        self.blink_count += 1

                    self.last_blink_time = tnow
                    self.eye_closed = False
                    self.t_close_start = np.nan

            closed_eye = self.eye_closed

            if self.eye_closed and np.isfinite(self.t_close_start):
                dur_now = tnow - self.t_close_start
                closed_eye_perclos = dur_now >= self.cfg.perclos_min_closure_sec
            else:
                closed_eye_perclos = False

            if np.isfinite(xn_raw) and np.isfinite(yn_raw) and np.isfinite(self.last_valid_xn) and np.isfinite(self.last_valid_yn):
                jump = math.hypot(xn_raw - self.last_valid_xn, yn_raw - self.last_valid_yn)
            elif np.isfinite(xn_raw) and np.isfinite(yn_raw):
                jump = 0.0

            can_use_sample = bool(
                valid_used
                and (not closed_eye)
                and (not blink_evt)
                and np.isfinite(jump)
                and jump <= self.cfg.max_iris_jump
            )

            if can_use_sample:
                xn = xn_raw
                yn = yn_raw
                ear_plot = ear_mean

                self.last_valid_xn = xn
                self.last_valid_yn = yn
                self.last_valid_ear_plot = ear_plot
                interpolated = 0
                valid_reason = "OK"
            else:
                if np.isnan(self.last_valid_xn):
                    self.last_valid_xn = 0.0
                if np.isnan(self.last_valid_yn):
                    self.last_valid_yn = 0.0
                if np.isnan(self.last_valid_ear_plot):
                    self.last_valid_ear_plot = ear_mean if np.isfinite(ear_mean) else np.nan

                xn = self.last_valid_xn
                yn = self.last_valid_yn
                ear_plot = self.last_valid_ear_plot
                interpolated = 1

                if not valid_raw:
                    valid_reason = "LANDMARKS_INVALIDOS"
                elif not pose_ok:
                    valid_reason = "POSE"
                elif closed_eye:
                    valid_reason = "EYE_CLOSED"
                elif blink_evt:
                    valid_reason = "BLINK"
                elif np.isfinite(jump) and jump > self.cfg.max_iris_jump:
                    valid_reason = "IRIS_JUMP"
                else:
                    valid_reason = "HELD_SAMPLE"

        self.time.append(float(tnow))

        self.ear_l.append(float(ear_l) if np.isfinite(ear_l) else np.nan)
        self.ear_r.append(float(ear_r) if np.isfinite(ear_r) else np.nan)
        self.ear_mean.append(float(ear_mean) if np.isfinite(ear_mean) else np.nan)
        self.ear_filt.append(float(self.ear_filt_last) if np.isfinite(self.ear_filt_last) else np.nan)
        self.ear_plot.append(float(ear_plot) if np.isfinite(ear_plot) else np.nan)

        self.blink.append(int(blink_evt))
        self.closed_eye.append(int(closed_eye))
        self.closed_eye_perclos.append(int(closed_eye_perclos))

        self.xn_raw.append(float(xn_raw) if np.isfinite(xn_raw) else np.nan)
        self.yn_raw.append(float(yn_raw) if np.isfinite(yn_raw) else np.nan)

        self.xn.append(float(xn) if np.isfinite(xn) else np.nan)
        self.yn.append(float(yn) if np.isfinite(yn) else np.nan)

        self.valid_raw.append(int(valid_raw))
        self.valid_used.append(int(valid_used))
        self.pose_ok.append(int(pose_ok))
        self.filled.append(int(interpolated))

        self.yaw_asym_hist.append(float(yaw_asym) if np.isfinite(yaw_asym) else np.nan)
        self.left_eye_width_hist.append(float(left_eye_width) if np.isfinite(left_eye_width) else np.nan)
        self.right_eye_width_hist.append(float(right_eye_width) if np.isfinite(right_eye_width) else np.nan)
        self.jump_iris_hist.append(float(jump) if np.isfinite(jump) else np.nan)
        self.valid_reason.append(valid_reason)

        var_raw = self._compute_var_window(mode="raw")
        var_clean = self._compute_var_window(mode="clean")
        perclos = self._compute_perclos()

        self.var_iris_raw.append(var_raw)
        self.var_iris_clean.append(var_clean)
        self.perclos.append(perclos)

        if self.baseline_ready and np.isfinite(ear_mean) and np.isfinite(var_clean):
            z_ear = (ear_mean - self.baseline.mu_ear) / self.baseline.sd_ear
            z_var = (var_clean - self.baseline.mu_var) / self.baseline.sd_var

        self.z_ear.append(float(z_ear) if np.isfinite(z_ear) else np.nan)
        self.z_var.append(float(z_var) if np.isfinite(z_var) else np.nan)

        state = self._classify_state(valid_used, blink_evt, z_ear, z_var, perclos)
        self.state.append(state)

        drift_index = np.nan
        if np.isfinite(perclos) and np.isfinite(z_var) and self.baseline_ready:
            drift_index = float(np.clip(
                self.cfg.deriva_perclos_weight * perclos
                + self.cfg.deriva_var_weight * self._z_to_unit(z_var),
                0.0,
                1.0
            ))

        row = {
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "block_name": self.block_name,
            "phase": self.phase,

            "frame_id": int(frame_id),
            "time_s": float(tnow),

            "valid_raw": int(valid_raw),
            "valid_used": int(valid_used),
            "valid_reason": valid_reason,
            "pose_ok": int(pose_ok),
            "yaw_asym": float(yaw_asym) if np.isfinite(yaw_asym) else np.nan,
            "left_eye_width": float(left_eye_width) if np.isfinite(left_eye_width) else np.nan,
            "right_eye_width": float(right_eye_width) if np.isfinite(right_eye_width) else np.nan,

            "blink": int(blink_evt),
            "blink_count_accum": int(self.blink_count),
            "closed_eye": int(closed_eye),
            "closed_eye_perclos": int(closed_eye_perclos),
            "interpolated": int(interpolated),

            "ear_l": float(ear_l) if np.isfinite(ear_l) else np.nan,
            "ear_r": float(ear_r) if np.isfinite(ear_r) else np.nan,
            "ear_mean": float(ear_mean) if np.isfinite(ear_mean) else np.nan,
            "ear_filt": float(self.ear_filt_last) if np.isfinite(self.ear_filt_last) else np.nan,
            "ear_plot": float(ear_plot) if np.isfinite(ear_plot) else np.nan,

            "xn_raw": float(xn_raw) if np.isfinite(xn_raw) else np.nan,
            "yn_raw": float(yn_raw) if np.isfinite(yn_raw) else np.nan,
            "xn": float(xn) if np.isfinite(xn) else np.nan,
            "yn": float(yn) if np.isfinite(yn) else np.nan,
            "jump_iris": float(jump) if np.isfinite(jump) else np.nan,

            "var_iris_raw": float(var_raw) if np.isfinite(var_raw) else np.nan,
            "var_iris_clean": float(var_clean) if np.isfinite(var_clean) else np.nan,
            "perclos": float(perclos) if np.isfinite(perclos) else np.nan,

            "z_ear": float(z_ear) if np.isfinite(z_ear) else np.nan,
            "z_var": float(z_var) if np.isfinite(z_var) else np.nan,
            "attentional_drift_index": float(drift_index) if np.isfinite(drift_index) else np.nan,

            "state": state,
        }
        self.rows.append(row)
        self._phase_hist.append(self.phase)
        self.drift_index.append(float(drift_index) if np.isfinite(drift_index) else np.nan)

        max_rows = self.cfg.max_ocular_rows_memory
        if max_rows > 0 and len(self.rows) > max_rows:
            excess = len(self.rows) - max_rows
            del self.rows[:excess]
            del self._phase_hist[:excess]

        valid_pct = 100.0 * float(np.mean(self.valid_used)) if len(self.valid_used) > 0 else 0.0

        return {
            "time_s": float(tnow),
            "ear_mean": row["ear_mean"],
            "ear_filt": row["ear_filt"],
            "perclos": row["perclos"],
            "var_iris_raw": row["var_iris_raw"],
            "var_iris_clean": row["var_iris_clean"],
            "xn": row["xn"],
            "yn": row["yn"],
            "blink": bool(blink_evt),
            "blink_count_accum": int(self.blink_count),
            "valid": bool(valid_used),
            "valid_pct": valid_pct,
            "state": state,
            "z_ear": row["z_ear"],
            "z_var": row["z_var"],
            "attentional_drift_index": row["attentional_drift_index"],
            "yaw_asym": row["yaw_asym"],
            "valid_reason": valid_reason,
            "phase": self.phase,
        }

    def get_summary(self) -> Dict[str, Any]:
        duration_s = float(self.time[-1]) if len(self.time) else 0.0
        valid_frames = int(np.sum(np.array(self.valid_used, dtype=int) == 1))
        blink_per_min = float(self.blink_count / (duration_s / 60.0)) if duration_s > 0 else 0.0

        state_arr = np.array(self.state, dtype=object)
        perclos_arr = np.array(self.perclos, dtype=float)
        ear_arr = np.array(self.ear_mean, dtype=float)
        var_arr = np.array(self.var_iris_clean, dtype=float)

        focus_pct = 100.0 * float(np.mean(state_arr == "FOCO")) if len(state_arr) else 0.0
        distraction_pct = 100.0 * float(np.mean(state_arr == "DISTRACCION")) if len(state_arr) else 0.0
        fatigue_pct = 100.0 * float(np.mean(state_arr == "FATIGA")) if len(state_arr) else 0.0
        invalid_pct = 100.0 * float(np.mean(state_arr == "NO_CONFIABLE")) if len(state_arr) else 0.0

        return {
            "duration_s": duration_s,
            "valid_frames": valid_frames,
            "blink_count": int(self.blink_count),
            "blink_per_min": blink_per_min,
            "perclos_mean": float(np.nanmean(perclos_arr)) if len(perclos_arr) else np.nan,
            "perclos_max": float(np.nanmax(perclos_arr)) if len(perclos_arr) and np.any(np.isfinite(perclos_arr)) else np.nan,
            "ear_mean": float(np.nanmean(ear_arr)) if len(ear_arr) else np.nan,
            "ear_std": float(np.nanstd(ear_arr)) if len(ear_arr) else np.nan,
            "var_iris_mean": float(np.nanmean(var_arr)) if len(var_arr) else np.nan,
            "var_iris_std": float(np.nanstd(var_arr)) if len(var_arr) else np.nan,
            "focus_pct": focus_pct,
            "distraction_pct": distraction_pct,
            "fatigue_pct": fatigue_pct,
            "invalid_pct": invalid_pct,
        }

    def _get_col(self, key: str) -> List[Any]:
        return [row.get(key) for row in self.rows]

    def _compute_ear(self, L: np.ndarray, idx6: np.ndarray) -> float:
        p1 = L[idx6[0], :2]
        p2 = L[idx6[1], :2]
        p3 = L[idx6[2], :2]
        p4 = L[idx6[3], :2]
        p5 = L[idx6[4], :2]
        p6 = L[idx6[5], :2]

        num = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        den = 2.0 * np.linalg.norm(p1 - p4) + np.finfo(float).eps
        return float(num / den)

    def _iris_local_coords(self, L: np.ndarray, corner_idx: np.ndarray, iris_idx: np.ndarray) -> tuple[float, float]:
        c1 = L[corner_idx[0], :2]
        c2 = L[corner_idx[1], :2]
        iris_pts = L[iris_idx, :2]

        if not (np.isfinite(c1).all() and np.isfinite(c2).all() and np.isfinite(iris_pts).all()):
            return np.nan, np.nan

        iris_c = np.nanmean(iris_pts, axis=0)

        ex = c2 - c1
        eye_width = np.linalg.norm(ex) + np.finfo(float).eps
        ex = ex / eye_width

        ey = np.array([-ex[1], ex[0]], dtype=float)

        eye_center = (c1 + c2) / 2.0
        rel = iris_c - eye_center

        xn = float(np.dot(rel, ex) / eye_width)
        eye_height = eye_width * 0.35 + np.finfo(float).eps
        yn = float(np.dot(rel, ey) / eye_height)
        return xn, yn

    def _check_pose_quality(self, L: np.ndarray) -> tuple[bool, float, float, float]:
        idx = self.idx

        left_eye_width = float(np.linalg.norm(L[idx["leftCorners"][1], :2] - L[idx["leftCorners"][0], :2]))
        right_eye_width = float(np.linalg.norm(L[idx["rightCorners"][1], :2] - L[idx["rightCorners"][0], :2]))

        yaw_asym = abs(left_eye_width - right_eye_width) / max(
            left_eye_width + right_eye_width,
            np.finfo(float).eps,
        )

        pose_ok = bool(
            np.isfinite(yaw_asym)
            and yaw_asym < self.cfg.max_head_yaw_asym
            and left_eye_width > 5
            and right_eye_width > 5
        )
        return pose_ok, yaw_asym, left_eye_width, right_eye_width

    def _compute_var_window(self, mode: str = "clean") -> float:
        if len(self.time) < 3:
            return np.nan

        t = np.array(self.time, dtype=float)
        xn_raw = np.array(self.xn_raw, dtype=float)
        yn_raw = np.array(self.yn_raw, dtype=float)
        valid = np.array(self.valid_used, dtype=int)
        valid_raw = np.array(self.valid_raw, dtype=int)
        blink = np.array(self.blink, dtype=int)
        closed = np.array(self.closed_eye, dtype=int)
        closed_p = np.array(self.closed_eye_perclos, dtype=int)
        interp = np.array(self.filled, dtype=int)

        t_now = t[-1]
        time_mask = (t >= (t_now - self.cfg.var_window_sec)) & (t <= t_now)

        if mode == "raw":
            mask = time_mask & (valid_raw == 1) & np.isfinite(xn_raw) & np.isfinite(yn_raw)
        else:
            mask = (
                time_mask
                & (valid == 1)
                & (blink == 0)
                & (closed == 0)
                & (closed_p == 0)
                & (interp == 0)
                & np.isfinite(xn_raw)
                & np.isfinite(yn_raw)
            )

        if int(mask.sum()) < 3:
            return np.nan

        vx = float(np.nanvar(xn_raw[mask]))
        vy = float(np.nanvar(yn_raw[mask]))
        return vx + vy

    def _compute_perclos(self) -> float:
        if len(self.time) < 3:
            return np.nan

        t = np.array(self.time, dtype=float)
        closed = np.array(self.closed_eye_perclos, dtype=int)
        valid = np.array(self.valid_used, dtype=int)

        t_now = t[-1]
        mask = (t >= (t_now - self.cfg.perclos_window_sec)) & (t <= t_now) & (valid == 1)
        if int(mask.sum()) < 3:
            return np.nan

        return float(np.nanmean(closed[mask]))

    def _z_to_unit(self, z: float) -> float:
        if not np.isfinite(z):
            return np.nan
        return float(np.clip((z + 2.0) / 4.0, 0.0, 1.0))

    def _classify_state(
        self,
        valid: bool,
        blink: bool,
        z_ear: float,
        z_var: float,
        perclos: float,
    ) -> str:
        if (not valid) or blink or (not np.isfinite(z_ear)) or (not np.isfinite(z_var)) or (not np.isfinite(perclos)):
            return "NO_CONFIABLE"

        if z_ear > self.cfg.z_ear_fatigue and z_var < self.cfg.z_var_low and perclos < 0.20:
            return "FOCO"

        if z_ear > self.cfg.z_ear_fatigue and z_var >= self.cfg.z_var_high:
            return "DISTRACCION"

        if z_ear <= self.cfg.z_ear_fatigue and z_var < self.cfg.z_var_low and perclos >= 0.20:
            return "FATIGA"

        if perclos >= 0.20 and z_ear < 0:
            return "FATIGA"

        if z_var >= self.cfg.z_var_high:
            return "DISTRACCION"

        return "FOCO"
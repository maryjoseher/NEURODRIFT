from __future__ import annotations

"""
attention_estimator.py — Static-gain (Luenberger) attentional drift observer.

Implements "retroalimentación estática de estados estimados":
the steady-state Kalman gain K_∞ is computed ONCE from the Discrete Algebraic
Riccati Equation (DARE) before any data arrives, so every prediction step is a
fixed linear update — no matrix inversions at run-time.

Architecture
------------
Window = 30 s, hop = 15 s.
Effective measurement delay = 1 hop = 1 Kalman step → compact 2-block augmentation.

State x_k ∈ R³   : [a_k (autonomic), o_k (ocular), d_k (global drift)]
Augmented X_k ∈ R⁶: [x_k ; x_{k-1}]
Observation z_k ∈ R⁶: normalised [ΔHR, −Δln(RMSSD), ΔSCL, ΔSCRrate, ΔPERCLOS, ΔIrisVar]

Dynamics:
    Aa = [[A, 0], [I, 0]]   Ha = [0 | H]
    X̂_{k|k} = Aa·X̂_{k-1|k-1} + K_∞·(z_k − Ha·Aa·X̂_{k-1|k-1})

Composite drift index:
    J_k  = c^T · x̂_k         (c = [0.50, 0.15, 0.35])
    I_k  = clip(α·J_k + β·D_k, 0, 1)
    D_k  = Mahalanobis distance from individual baseline (direct, no soft-norm)

Calibration:
    ECG/EDA baseline  → CALIB_D phase only
    Ocular baseline   → CALIB_A + CALIB_D phases

Usage from app.py:
    1. estimator = AttentionEstimator()
    2. estimator.calibrate_from_engine(engine, physio_reader)   # after CALIB_OK
    3. dr = estimator.tick(t_abs, ocular_result, physio_reader)  # in on_timer
    4. t_arr, I_arr = estimator.get_drift_history()              # for plot
    5. estimator.save_csv(path)                                  # on save_action
"""

import csv
import time
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from scipy.signal import find_peaks
from scipy.linalg import solve_discrete_are


# ─────────────────────────────────────────────────────────────────────────────
# Internal sliding-window feature extractor (30 s window / 15 s hop)
# ─────────────────────────────────────────────────────────────────────────────

class _FeatureWindow:
    """Minimal 30-s/15-s sliding window.  Embedded so module stays self-contained."""

    N_FEATURES = 6

    def __init__(self, window_sec: float = 30.0, hop_sec: float = 15.0,
                 fs: float = 500.0, fps: float = 30.0):
        self.window_sec = float(window_sec)
        self.hop_sec    = float(hop_sec)
        self.fs         = float(fs)
        self.fps        = float(fps)

        maxph = int(self.fs * (self.window_sec + 10.0))
        self._ecg_t: deque = deque(maxlen=maxph)
        self._ecg_v: deque = deque(maxlen=maxph)
        self._eda_t: deque = deque(maxlen=maxph)
        self._eda_v: deque = deque(maxlen=maxph)
        self._last_ecg_t: float = -np.inf
        self._last_eda_t: float = -np.inf

        maxoc = int(self.fps * (self.window_sec + 10.0))
        self._oc_t:    deque = deque(maxlen=maxoc)
        self._perclos: deque = deque(maxlen=maxoc)
        self._irisvar: deque = deque(maxlen=maxoc)

        self._next_emit_t: Optional[float] = None

    def push_physio_chunk(self, ecg_t: np.ndarray, ecg_v: np.ndarray,
                          eda_t: np.ndarray, eda_v: np.ndarray) -> None:
        if len(ecg_t) > 0:
            mask = ecg_t > self._last_ecg_t
            if mask.any():
                for t, v in zip(ecg_t[mask], ecg_v[mask]):
                    self._ecg_t.append(float(t))
                    self._ecg_v.append(float(v))
                self._last_ecg_t = float(ecg_t[mask][-1])
        if len(eda_t) > 0:
            mask = eda_t > self._last_eda_t
            if mask.any():
                for t, v in zip(eda_t[mask], eda_v[mask]):
                    self._eda_t.append(float(t))
                    self._eda_v.append(float(v))
                self._last_eda_t = float(eda_t[mask][-1])

    def push_ocular(self, t_abs: float, perclos: float, iris_var: float) -> None:
        self._oc_t.append(float(t_abs))
        self._perclos.append(float(perclos) if np.isfinite(perclos) else np.nan)
        self._irisvar.append(float(iris_var) if np.isfinite(iris_var) else np.nan)

    def maybe_emit(self, t_abs: float) -> Optional[np.ndarray]:
        if self._next_emit_t is None:
            self._next_emit_t = t_abs + self.hop_sec
            return None
        if t_abs < self._next_emit_t:
            return None
        self._next_emit_t = t_abs + self.hop_sec
        return self._compute(t_abs)

    def calibration_samples(self, sub_win: float = 15.0,
                             sub_hop: float = 5.0) -> np.ndarray:
        if not self._oc_t:
            return np.zeros((0, self.N_FEATURES))
        t_end   = float(self._oc_t[-1])
        t_start = t_end - self.window_sec
        samples = []
        t = t_start + sub_win
        while t <= t_end + 1e-3:
            samples.append(self._compute_window(t_center=t, win=sub_win))
            t += sub_hop
        return np.array(samples, dtype=float) if samples else np.zeros((0, self.N_FEATURES))

    def reset(self) -> None:
        self._ecg_t.clear(); self._ecg_v.clear()
        self._eda_t.clear(); self._eda_v.clear()
        self._oc_t.clear();  self._perclos.clear(); self._irisvar.clear()
        self._last_ecg_t  = -np.inf
        self._last_eda_t  = -np.inf
        self._next_emit_t = None

    def _compute(self, t_now: float) -> np.ndarray:
        return self._compute_window(t_center=t_now, win=self.window_sec)

    def _compute_window(self, t_center: float, win: float) -> np.ndarray:
        phi = np.full(self.N_FEATURES, np.nan)
        t0  = t_center - win

        if self._ecg_t:
            et = np.array(self._ecg_t); ev = np.array(self._ecg_v)
            seg = ev[(et >= t0) & (et <= t_center)]
            if len(seg) >= int(self.fs * 2.0):
                phi[0], phi[1] = self._ecg_feats(seg)

        if self._eda_t:
            at = np.array(self._eda_t); av = np.array(self._eda_v)
            seg = av[(at >= t0) & (at <= t_center)]
            if len(seg) >= int(self.fs * 1.0):
                phi[2], phi[3] = self._eda_feats(seg)

        if self._oc_t:
            ot  = np.array(self._oc_t)
            per = np.array(self._perclos, dtype=float)
            iv  = np.array(self._irisvar, dtype=float)
            m   = (ot >= t0) & (ot <= t_center)
            if m.sum() >= 5:
                phi[4] = float(np.nanmean(per[m]))
                phi[5] = float(np.nanmean(iv[m]))

        return phi

    def _ecg_feats(self, seg: np.ndarray) -> Tuple[float, float]:
        amp = float(np.max(np.abs(seg)))
        if amp < 1e-6:
            return np.nan, np.nan
        peaks, _ = find_peaks(seg, height=0.40 * amp, distance=int(0.30 * self.fs))
        if len(peaks) < 3:
            return np.nan, np.nan
        rr = np.diff(peaks) / self.fs * 1000.0
        rr = rr[(rr > 300.0) & (rr < 2000.0)]
        if len(rr) < 2:
            return np.nan, np.nan
        hr      = float(60000.0 / np.mean(rr))
        rmssd   = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
        return hr, float(np.log(rmssd + 1.0))

    def _eda_feats(self, seg: np.ndarray) -> Tuple[float, float]:
        scl  = float(np.mean(seg))
        diff = np.diff(seg)
        thr  = max(float(np.std(diff)) * 2.5, 1e-9)
        pks, _ = find_peaks(diff, height=thr, distance=int(self.fs))
        dur_min = max(len(seg) / self.fs / 60.0, 1e-6)
        return scl, float(len(pks) / dur_min)


# ─────────────────────────────────────────────────────────────────────────────
# Main estimator
# ─────────────────────────────────────────────────────────────────────────────

class AttentionEstimator:
    """
    Static-gain Luenberger observer for attentional drift.

    K_∞ is computed once at construction via DARE.
    Runtime cost per step: two matrix-vector products.
    """

    N_STATE = 3
    N_AUG   = 6
    N_OBS   = 6

    CSV_COLUMNS: List[str] = [
        "t_block_rel_s", "t_abs", "block_name", "step_k",
        "HR", "lnRMSSD", "SCL", "SCRrate", "PERCLOS", "IrisVar",
        "a_hat", "o_hat", "d_hat",
        "J_k", "D_k", "I_k",
        "uncertainty", "event_flag", "innovation_norm",
    ]

    def __init__(
        self,
        A:       Optional[np.ndarray] = None,
        H:       Optional[np.ndarray] = None,
        Q_diag:  Optional[np.ndarray] = None,
        R_diag:  Optional[np.ndarray] = None,
        c:       Optional[np.ndarray] = None,
        alpha:   float = 0.6,
        beta:    float = 0.4,
        T_on:    float = 0.65,
        T_off:   float = 0.45,
        window_sec:  float = 30.0,
        hop_sec:     float = 15.0,
        fs_physio:   float = 500.0,
        fps_ocular:  float = 30.0,
    ):
        # ── State-transition A (3×3) ─────────────────────────────────────────
        self.A = A if A is not None else np.array([
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.10, 0.10, 0.85],
        ], dtype=float)

        # ── Observation H (6×3): bio-marker → latent state ──────────────────
        self.H = H if H is not None else np.array([
            [1.0, 0.0, 0.6],
            [0.9, 0.0, 0.7],
            [0.8, 0.1, 0.5],
            [0.9, 0.0, 0.4],
            [0.0, 0.9, 0.7],
            [0.1, 0.8, 0.6],
        ], dtype=float)

        # ── Process noise Q (3 diag) ─────────────────────────────────────────
        q = Q_diag if Q_diag is not None else np.array([0.05, 0.05, 0.03])
        self.Q = np.diag(q.astype(float))

        # ── Measurement noise R (6 diag) ─────────────────────────────────────
        r = R_diag if R_diag is not None else np.array([0.5, 0.7, 0.5, 0.8, 1.0, 1.2])
        self.R = np.diag(r.astype(float))

        # ── Drift-index vector c  (PDF spec: [0.50, 0.15, 0.35]) ─────────────
        self.c     = c if c is not None else np.array([0.50, 0.15, 0.35], dtype=float)
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.T_on  = float(T_on)
        self.T_off = float(T_off)

        # ── Build augmented matrices ──────────────────────────────────────────
        self._build_augmented()

        # ── Compute steady-state gain K_∞ via DARE ───────────────────────────
        self._compute_static_gain()

        # ── Baseline (set by calibrate_from_engine()) ─────────────────────────
        self.calibrated:      bool             = False
        self.mu_basal:        np.ndarray       = np.zeros(self.N_OBS)
        self.sigma_basal:     np.ndarray       = np.ones(self.N_OBS)
        self.Sigma_basal_inv: Optional[np.ndarray] = None

        # ── Observer state ────────────────────────────────────────────────────
        self._X_hat = np.zeros(self.N_AUG)

        # ── Per-block tracking ────────────────────────────────────────────────
        self._block_name:  str            = ""
        self._block_t0:    Optional[float] = None
        self._step_k:      int            = 0
        self._hyst_count:  int            = 0
        self._drift_hist:  List[Tuple[float, float]] = []   # (t_rel_s, I_k)
        self._csv_rows:    List[Dict[str, Any]]       = []

        # ── Feature window (30 s / 15 s) ──────────────────────────────────────
        self._fw = _FeatureWindow(
            window_sec=window_sec,
            hop_sec=hop_sec,
            fs=fs_physio,
            fps=fps_ocular,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Construction helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_augmented(self) -> None:
        n = self.N_STATE
        z = np.zeros((n, n))
        I = np.eye(n)
        self.Aa = np.block([[self.A, z], [I, z]])
        self.Ha = np.hstack([np.zeros((self.N_OBS, n)), self.H])
        eps     = 1e-4
        self.Qa = np.block([[self.Q, z], [z, eps * I]])

    def _compute_static_gain(self) -> None:
        """
        Solve DARE for steady-state covariance P_∞, then compute K_∞.

        scipy.linalg.solve_discrete_are(a, b, q, r) solves:
            a^T X a - X + q - a^T X b (r + b^T X b)^{-1} b^T X a = 0
        Call with  a=Aa^T, b=Ha^T  to recover the forward DARE:
            Aa X Aa^T - X + Qa - Aa X Ha^T (R + Ha X Ha^T)^{-1} Ha X Aa^T = 0
        """
        try:
            P_inf = solve_discrete_are(
                self.Aa.T, self.Ha.T, self.Qa, self.R
            )
            S_inf = self.Ha @ P_inf @ self.Ha.T + self.R
            self.K_inf   = P_inf @ self.Ha.T @ np.linalg.inv(S_inf)
            self._P_inf  = P_inf
            self._dare_ok = True
        except Exception:
            # Fallback: identity-scaled proportional gain
            self.K_inf    = np.zeros((self.N_AUG, self.N_OBS))
            self._P_inf   = np.eye(self.N_AUG)
            self._dare_ok = False

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration
    # ─────────────────────────────────────────────────────────────────────────

    def calibrate_from_engine(
        self,
        engine_rows: List[Dict[str, Any]],
        physio_reader=None,
    ) -> bool:
        """
        Compute individual baseline from calibration data.

        ECG/EDA baseline  : CALIB_D rows only (clean resting physio).
        Ocular baseline   : CALIB_A + CALIB_D rows (baseline eyes).

        Parameters
        ----------
        engine_rows  : list of ocular-engine row dicts (engine.rows)
        physio_reader: NIMultiReader instance, or None if physio not started

        Returns True on success (≥3 finite calibration windows).
        """
        self._fw.reset()

        # ── Ocular: CALIB_A + CALIB_D ────────────────────────────────────────
        oc_phases = {"CALIB_A", "CALIB_D"}
        oc_rows = [r for r in engine_rows if r.get("phase") in oc_phases]
        if oc_rows:
            # Reconstruct absolute timestamps from relative time_s
            t0_abs = time.time() - oc_rows[-1].get("time_s", 0.0)
            for r in oc_rows:
                t_abs = t0_abs + r.get("time_s", 0.0)
                self._fw.push_ocular(
                    t_abs,
                    r.get("perclos", np.nan),
                    r.get("var_iris_clean", np.nan),
                )

        # ── Physio: CALIB_D only (NI buffer contains full recording) ──────────
        if physio_reader is not None:
            try:
                ecg_t, ecg_v, eda_t, eda_v = physio_reader.get_data()
                # Restrict to the time span of CALIB_D ocular rows
                calib_d = [r for r in engine_rows if r.get("phase") == "CALIB_D"]
                if calib_d and len(ecg_t) > 0:
                    t_rel_start = calib_d[0].get("time_s", 0.0)
                    t_rel_end   = calib_d[-1].get("time_s", 0.0)
                    t0_abs = time.time() - calib_d[-1].get("time_s", 0.0)
                    t_abs_start = t0_abs + t_rel_start
                    t_abs_end   = t0_abs + t_rel_end
                    mask_ecg = (ecg_t >= t_abs_start) & (ecg_t <= t_abs_end)
                    mask_eda = (eda_t >= t_abs_start) & (eda_t <= t_abs_end)
                    self._fw.push_physio_chunk(
                        ecg_t[mask_ecg], ecg_v[mask_ecg],
                        eda_t[mask_eda], eda_v[mask_eda],
                    )
                elif len(ecg_t) > 0:
                    # No timing info: use everything (best-effort)
                    self._fw.push_physio_chunk(ecg_t, ecg_v, eda_t, eda_v)
            except Exception:
                pass

        phi_samples = self._fw.calibration_samples(sub_win=15.0, sub_hop=5.0)
        ok = self._fit_baseline(phi_samples)

        # Reset window for real-time operation
        self._fw.reset()
        self._X_hat     = np.zeros(self.N_AUG)
        self._hyst_count = 0
        return ok

    def _fit_baseline(self, phi_samples: np.ndarray) -> bool:
        if phi_samples.ndim != 2 or phi_samples.shape[1] != self.N_OBS:
            return False
        finite = np.all(np.isfinite(phi_samples), axis=1)
        phi = phi_samples[finite]
        if len(phi) < 3:
            return False

        self.mu_basal    = np.mean(phi, axis=0)
        self.sigma_basal = np.maximum(np.std(phi, axis=0), 1e-6)

        centred = phi - self.mu_basal
        cov     = (centred.T @ centred) / max(len(phi) - 1, 1)
        cov    += np.eye(self.N_OBS) * 1e-4
        try:
            self.Sigma_basal_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self.Sigma_basal_inv = np.diag(1.0 / np.maximum(np.diag(cov), 1e-6))

        self.calibrated = True
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Block lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def set_block(self, block_name: str) -> None:
        """Call when a new experimental block starts."""
        self._block_name  = block_name
        self._block_t0    = time.time()
        self._step_k      = 0
        self._hyst_count  = 0
        self._drift_hist.clear()
        self._X_hat       = np.zeros(self.N_AUG)
        self._fw.reset()

    # ─────────────────────────────────────────────────────────────────────────
    # Real-time tick  (called in on_timer, ~30 ms)
    # ─────────────────────────────────────────────────────────────────────────

    def tick(
        self,
        t_abs:        float,
        ocular_result: Optional[Dict[str, Any]],
        physio_reader=None,
    ) -> Optional[Dict[str, Any]]:
        """
        Ingest one frame of data.  Returns a drift-estimate dict every hop_sec
        seconds, or None between hops.

        Parameters
        ----------
        t_abs         : current time.time()
        ocular_result : dict from OcularEngine.step() (may be None)
        physio_reader : NIMultiReader instance, or None
        """
        if not self.calibrated:
            return None

        # Push ocular
        if ocular_result is not None:
            self._fw.push_ocular(
                t_abs,
                ocular_result.get("perclos", np.nan),
                ocular_result.get("var_iris_clean", np.nan),
            )

        # Push physio
        if physio_reader is not None:
            try:
                ecg_t, ecg_v, eda_t, eda_v = physio_reader.get_data()
                self._fw.push_physio_chunk(ecg_t, ecg_v, eda_t, eda_v)
            except Exception:
                pass

        phi = self._fw.maybe_emit(t_abs)
        if phi is None:
            return None

        return self._step(phi, t_abs)

    # ─────────────────────────────────────────────────────────────────────────
    # Observer step
    # ─────────────────────────────────────────────────────────────────────────

    def _step(self, phi_k: np.ndarray, t_abs: float) -> Dict[str, Any]:
        # NaN → baseline mean (no-information imputation)
        phi_safe = np.where(np.isfinite(phi_k), phi_k, self.mu_basal)

        # Normalise against individual baseline
        z    = (phi_safe - self.mu_basal) / self.sigma_basal
        z[1] = -z[1]   # flip ln(RMSSD): ↓RMSSD → ↑load → positive z

        # Mahalanobis distance (direct, no soft normalisation)
        delta = phi_safe - self.mu_basal
        D2    = float(delta @ self.Sigma_basal_inv @ delta)
        D     = float(np.sqrt(max(D2, 0.0)))

        # ── Luenberger static-gain update ────────────────────────────────────
        # Predict
        X_pred    = self.Aa @ self._X_hat
        # Innovation (observation sees delayed block)
        innovation = z - self.Ha @ X_pred
        # Correct with fixed gain K_∞
        self._X_hat = X_pred + self.K_inf @ innovation

        # Current state block
        x_hat = self._X_hat[:self.N_STATE].copy()

        # Composite drift index  I_k = clip(α·J + β·D, 0, 1)
        J   = float(self.c @ x_hat)
        I_k = float(np.clip(self.alpha * J + self.beta * D, 0.0, 1.0))

        # Hysteresis
        if I_k >= self.T_on:
            self._hyst_count += 1
        elif I_k < self.T_off:
            self._hyst_count = 0
        event_flag = self._hyst_count >= 2
        if event_flag:
            self._hyst_count = 0

        # Time relative to block start
        t_rel = (t_abs - self._block_t0) if self._block_t0 is not None else float("nan")

        # Record for plot and CSV
        self._drift_hist.append((t_rel, I_k))
        self._step_k += 1

        self._csv_rows.append({
            "t_block_rel_s":  round(t_rel, 3),
            "t_abs":          round(t_abs, 6),
            "block_name":     self._block_name,
            "step_k":         self._step_k,
            "HR":             _fmt(phi_k[0]),
            "lnRMSSD":        _fmt(phi_k[1]),
            "SCL":            _fmt(phi_k[2]),
            "SCRrate":        _fmt(phi_k[3]),
            "PERCLOS":        _fmt(phi_k[4]),
            "IrisVar":        _fmt(phi_k[5]),
            "a_hat":          _fmt(x_hat[0]),
            "o_hat":          _fmt(x_hat[1]),
            "d_hat":          _fmt(x_hat[2]),
            "J_k":            _fmt(J),
            "D_k":            _fmt(D),
            "I_k":            _fmt(I_k),
            "uncertainty":    _fmt(self.get_uncertainty()),
            "event_flag":     int(event_flag),
            "innovation_norm": _fmt(float(np.linalg.norm(innovation))),
        })

        return {
            "drift_index":  I_k,
            "mahalanobis":  D,
            "J_kalman":     J,
            "autonomic":    float(x_hat[0]),
            "ocular":       float(x_hat[1]),
            "global_drift": float(x_hat[2]),
            "innovation":   innovation,
            "event_flag":   event_flag,
            "ready":        True,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Outputs
    # ─────────────────────────────────────────────────────────────────────────

    def get_drift_history(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (t_block_rel_arr, I_k_arr) for plot refresh."""
        if not self._drift_hist:
            return np.array([]), np.array([])
        arr = np.array(self._drift_hist, dtype=float)
        return arr[:, 0], arr[:, 1]

    def get_uncertainty(self) -> float:
        """Trace of the steady-state covariance — fixed for a static observer."""
        return float(np.trace(self._P_inf[:self.N_STATE, :self.N_STATE]))

    def get_last_drift_index(self) -> float:
        return self._drift_hist[-1][1] if self._drift_hist else 0.0

    def dare_succeeded(self) -> bool:
        return self._dare_ok

    def save_csv(self, path) -> bool:
        """Write all accumulated drift estimates to a CSV file."""
        if not self._csv_rows:
            return False
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(p, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(self._csv_rows)
            return True
        except Exception:
            return False

    def reset(self) -> None:
        self._X_hat       = np.zeros(self.N_AUG)
        self._hyst_count  = 0
        self._drift_hist.clear()
        self._csv_rows.clear()
        self._step_k      = 0
        self._block_t0    = None
        self._fw.reset()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(x: float) -> str:
    """Format float for CSV; NaN → empty string."""
    if not np.isfinite(x):
        return ""
    return f"{x:.6f}"

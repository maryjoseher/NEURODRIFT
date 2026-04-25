from __future__ import annotations

"""
drift_observer.py — Predictive discrete Kalman observer with augmented state.

System description
------------------
The attentional drift state is not measured directly.  It is *estimated* from
six noisy, delayed biomarcadores.  Because each observation comes from a 60-second
window that is updated every 30 s, every measurement z_k actually describes the
system state from ~one step ago.  A standard Kalman filter applied to z_k as if
it were instantaneous would react one step late.

Solution: augment the state so the filter can predict the current state while
correcting against the delayed observation.

State vector  x_k ∈ R³
    a_k : autonomic activation   (driven mainly by ECG + EDA)
    o_k : ocular instability     (driven mainly by PERCLOS + IrisVar)
    d_k : global attentional drift

Augmented state  X_k = [x_k ; x_{k-1}] ∈ R⁶

Observation  z_k ∈ R⁶  (normalised against individual baseline)
    [ΔHR,  −Δln(RMSSD),  ΔSCL,  ΔSCRrate,  ΔPERCLOS,  ΔIrisVar]

The −sign on ln(RMSSD): physiologically, RMSSD drops under cognitive load /
autonomic stress, so negating it aligns its direction with "more drift".

Augmented dynamics
    X_{k+1} = Aa · X_k + W_k
    z_k      = Ha · X_k + v_k

where
    Aa = [[A, 0], [I, 0]]   (upper block predicts current; lower copies it as x_{k-1})
    Ha = [0 | H]            (observation only sees the delayed half)

Composite drift index
    J_k   = c^T x̂_k              (Kalman-based state projection)
    D_k   = Mahalanobis distance from baseline
    I_k   = clip( α·J_k + β · D_k / (D_k + 3), 0, 1 )

The D/(D+3) soft normalisation keeps I_k bounded even when Mahalanobis is large.
"""

from typing import Optional, Dict, Any

import numpy as np


class DriftObserver:
    N_STATE = 3   # latent state dimension
    N_AUG   = 6   # augmented state dimension  (2 * N_STATE)
    N_OBS   = 6   # observation dimension

    def __init__(
        self,
        A:        Optional[np.ndarray] = None,
        H:        Optional[np.ndarray] = None,
        Q_diag:   Optional[np.ndarray] = None,
        R_diag:   Optional[np.ndarray] = None,
        c:        Optional[np.ndarray] = None,
        alpha:    float = 0.6,
        beta:     float = 0.4,
    ):
        # ── State transition A (3×3) ─────────────────────────────────────────
        self.A = A if A is not None else np.array([
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.10, 0.10, 0.85],
        ], dtype=float)

        # ── Observation matrix H (6×3): biomarcador → state loading ──────────
        # Rows: ΔHR, −Δln(RMSSD), ΔSCL, ΔSCRrate, ΔPERCLOS, ΔIrisVar
        self.H = H if H is not None else np.array([
            [1.0, 0.0, 0.6],
            [0.9, 0.0, 0.7],
            [0.8, 0.1, 0.5],
            [0.9, 0.0, 0.4],
            [0.0, 0.9, 0.7],
            [0.1, 0.8, 0.6],
        ], dtype=float)

        # ── Process noise Q (state-space, 3 diagonal entries) ────────────────
        q = Q_diag if Q_diag is not None else np.array([0.05, 0.05, 0.03])
        self.Q = np.diag(q.astype(float))

        # ── Measurement noise R (6 diagonal entries) ─────────────────────────
        # Higher values → filter trusts that sensor less.
        # ECG/EDA intentionally lower; iris channels higher.
        r = R_diag if R_diag is not None else np.array([0.5, 0.7, 0.5, 0.8, 1.0, 1.2])
        self.R = np.diag(r.astype(float))

        # ── Drift-index projection vector c ──────────────────────────────────
        self.c     = c if c is not None else np.array([0.4, 0.2, 0.4], dtype=float)
        self.alpha = float(alpha)
        self.beta  = float(beta)

        # ── Build augmented system matrices ──────────────────────────────────
        self._build_augmented()

        # ── Baseline (set by calibrate()) ─────────────────────────────────────
        self.calibrated:      bool             = False
        self.mu_basal:        np.ndarray       = np.zeros(self.N_OBS)
        self.sigma_basal:     np.ndarray       = np.ones(self.N_OBS)
        self.Sigma_basal_inv: Optional[np.ndarray] = None

        # ── Filter state ──────────────────────────────────────────────────────
        self._X_hat = np.zeros(self.N_AUG)
        self._P     = np.eye(self.N_AUG)

        # ── Published outputs ─────────────────────────────────────────────────
        self.drift_index:     float      = 0.0
        self.mahalanobis:     float      = 0.0
        self.state_estimate:  np.ndarray = np.zeros(self.N_STATE)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal builders
    # ─────────────────────────────────────────────────────────────────────────

    def _build_augmented(self) -> None:
        n = self.N_STATE
        z = np.zeros((n, n))
        I = np.eye(n)

        # Aa = [[A, 0], [I, 0]]
        self.Aa = np.block([[self.A, z], [I, z]])

        # Ha = [0 | H]  — observation sees the *delayed* half of the state
        self.Ha = np.hstack([np.zeros((self.N_OBS, n)), self.H])

        # Qa: process noise on current block; tiny eps on delayed copy
        eps = 1e-4
        self.Qa = np.block([[self.Q, z], [z, eps * I]])

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration
    # ─────────────────────────────────────────────────────────────────────────

    def calibrate(self, phi_samples: np.ndarray) -> bool:
        """
        Fit the individual baseline from calibration feature vectors.

        phi_samples : (N, 6) array — each row is one feature vector
                      [HR, ln(RMSSD), SCL, SCRrate, PERCLOS, IrisVar]
                      computed from the CALIB_D phase (no sign flip yet).
        Returns True on success.
        """
        if phi_samples.ndim != 2 or phi_samples.shape[1] != self.N_OBS:
            return False

        finite_rows = np.all(np.isfinite(phi_samples), axis=1)
        phi = phi_samples[finite_rows]
        if len(phi) < 3:
            return False

        self.mu_basal    = np.mean(phi, axis=0)
        self.sigma_basal = np.maximum(np.std(phi, axis=0), 1e-6)

        # Regularised sample covariance for Mahalanobis
        centred = phi - self.mu_basal
        cov = (centred.T @ centred) / max(len(phi) - 1, 1)
        cov += np.eye(self.N_OBS) * 1e-4
        try:
            self.Sigma_basal_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self.Sigma_basal_inv = np.diag(1.0 / np.maximum(np.diag(cov), 1e-6))

        # Initialise filter covariance from baseline variability
        p0 = float(np.mean((self.sigma_basal / np.max(np.abs(self.sigma_basal) + 1e-9)) ** 2))
        self._P     = p0 * np.eye(self.N_AUG)
        self._X_hat = np.zeros(self.N_AUG)

        self.calibrated = True
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Real-time step  (called once per 30-second hop)
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, phi_k: np.ndarray) -> Dict[str, Any]:
        """
        Process one feature vector and return the updated drift estimate.

        phi_k : (6,) vector [HR, ln(RMSSD), SCL, SCRrate, PERCLOS, IrisVar]
                Values may be NaN for missing channels — they are replaced by
                the baseline mean so they contribute no information.

        Returns a dict with keys:
            drift_index   : float in [0, 1] — composite attentional drift
            mahalanobis   : float ≥ 0       — raw Mahalanobis distance from baseline
            J_kalman      : float            — Kalman state projection c^T x̂
            autonomic     : float            — a_k estimate
            ocular        : float            — o_k estimate
            global_drift  : float            — d_k estimate
            innovation    : (6,) array       — Kalman residual z_k − Ĥa X̂_{k|k-1}
            ready         : bool
        """
        _nan = {"drift_index": 0.0, "mahalanobis": 0.0, "J_kalman": 0.0,
                "autonomic": 0.0, "ocular": 0.0, "global_drift": 0.0,
                "innovation": np.zeros(self.N_OBS), "ready": False}

        if not self.calibrated:
            return _nan

        # ── Replace NaN with baseline mean (no-info imputation) ──────────────
        phi_safe = np.where(np.isfinite(phi_k), phi_k, self.mu_basal)

        # ── Normalise against individual baseline ────────────────────────────
        z = (phi_safe - self.mu_basal) / self.sigma_basal
        z[1] = -z[1]   # flip ln(RMSSD): ↓RMSSD → ↑load → positive z

        # ── Mahalanobis distance ─────────────────────────────────────────────
        delta = phi_safe - self.mu_basal
        D2    = float(delta @ self.Sigma_basal_inv @ delta)
        D     = float(np.sqrt(max(D2, 0.0)))

        # ── Kalman: predict ───────────────────────────────────────────────────
        X_pred = self.Aa @ self._X_hat
        P_pred = self.Aa @ self._P @ self.Aa.T + self.Qa

        # ── Kalman: innovation ───────────────────────────────────────────────
        r_k = z - self.Ha @ X_pred

        # ── Kalman: gain ─────────────────────────────────────────────────────
        S = self.Ha @ P_pred @ self.Ha.T + self.R
        try:
            K = P_pred @ self.Ha.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = np.zeros((self.N_AUG, self.N_OBS))

        # ── Kalman: correct ──────────────────────────────────────────────────
        self._X_hat = X_pred + K @ r_k
        IKH         = np.eye(self.N_AUG) - K @ self.Ha
        self._P     = IKH @ P_pred @ IKH.T + K @ self.R @ K.T   # Joseph form

        # ── Extract current-state block ───────────────────────────────────────
        x_hat = self._X_hat[:self.N_STATE].copy()
        self.state_estimate = x_hat

        # ── Composite index ───────────────────────────────────────────────────
        J   = float(self.c @ x_hat)
        I_k = float(np.clip(self.alpha * J + self.beta * (D / (D + 3.0)), 0.0, 1.0))

        self.drift_index  = I_k
        self.mahalanobis  = D

        return {
            "drift_index":  I_k,
            "mahalanobis":  D,
            "J_kalman":     J,
            "autonomic":    float(x_hat[0]),
            "ocular":       float(x_hat[1]),
            "global_drift": float(x_hat[2]),
            "innovation":   r_k,
            "ready":        True,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._X_hat          = np.zeros(self.N_AUG)
        self._P              = np.eye(self.N_AUG)
        self.drift_index     = 0.0
        self.mahalanobis     = 0.0
        self.state_estimate  = np.zeros(self.N_STATE)

    def get_uncertainty(self) -> float:
        """Trace of the current-state covariance block — smaller = more confident."""
        return float(np.trace(self._P[:self.N_STATE, :self.N_STATE]))

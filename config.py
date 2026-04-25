from dataclasses import dataclass


@dataclass
class AppConfig:
    camera_index: int = 0
    frame_w: int = 640
    frame_h: int = 480
    fps: int = 30

    # =========================
    # PROTOCOLO DE CALIBRACIÓN
    # =========================
    calib_phase_a_sec: float = 15.0
    calib_phase_b_sec: float = 30.0
    calib_phase_c_sec: float = 15.0
    calib_phase_d_sec: float = 60.0

    min_calibration_frames: int = 60
    validation_loss_threshold: float = 0.20

    # Ventanas de métricas
    var_window_sec: float = 2.0
    perclos_window_sec: float = 30.0

    # Blink
    blink_ear_thresh_init: float = 0.22
    blink_min_dur_sec: float = 0.04
    blink_max_dur_sec: float = 0.50
    blink_refractory_sec: float = 0.05
    blink_close_offset: float = 0.8
    blink_hysteresis: float = 0.02

    # Suavizado
    ear_smooth_alpha: float = 0.35

    # Pose útil basada en proxy geométrico
    max_head_yaw_asym: float = 0.35

    # Calidad de muestra
    max_iris_jump: float = 0.18
    perclos_min_closure_sec: float = 0.15

    # Estados
    z_ear_fatigue: float = -1.0
    z_var_high: float = 1.0
    z_var_low: float = 0.25

    # UI / salida
    ui_update_ms: int = 30
    output_dir: str = "outputs"

    # Visualización
    plot_window_sec: float = 120.0
    var_plot_norm_window_sec: float = 20.0
    deriva_perclos_weight: float = 0.65
    deriva_var_weight: float = 0.35

    # Límite de filas en memoria para OcularEngine (0 = sin límite).
    # A 30 fps, 36 000 ≈ 20 min de datos completos por bloque.
    max_ocular_rows_memory: int = 36000
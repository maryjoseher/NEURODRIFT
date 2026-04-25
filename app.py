from __future__ import annotations

import os
import sys
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from config import AppConfig
from facemesh_tracker import FaceMeshTracker
from ocular_engine import OcularEngine
from participant_view import ParticipantWindow
from session_manager import SessionManager
from session_accumulator import SessionAccumulator
from block1_odd_even import Block1OddEvenTask, Block1Config
from block2_gonogo import Block2GoNoGoTask, Block2Config
from block3_working_memory import Block3WorkingMemoryTask, Block3Config

from physio_runtime import PhysioRuntime
from physio_types import PhysioConfig
from drift_observer import DriftObserver
from feature_window import FeatureWindow
from attention_estimator import AttentionEstimator


class PanelFrame(QtWidgets.QFrame):
    def __init__(self, title: str, parent=None, title_height: int = 28):
        super().__init__(parent)
        self.setObjectName("PanelFrame")
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        self.title_bar = QtWidgets.QFrame()
        self.title_bar.setObjectName("PanelTitleBar")
        self.title_bar.setFixedHeight(title_height)

        title_layout = QtWidgets.QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(10, 0, 10, 0)
        title_layout.setSpacing(0)

        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setObjectName("PanelTitleLabel")
        title_layout.addWidget(self.title_label)

        self.body = QtWidgets.QFrame()
        self.body.setObjectName("PanelBody")

        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(10, 10, 10, 10)
        self.body_layout.setSpacing(8)

        self._layout.addWidget(self.title_bar)
        self._layout.addWidget(self.body)

    def content_layout(self):
        return self.body_layout


class HUDValue(QtWidgets.QLabel):
    def __init__(self, text="---", parent=None):
        super().__init__(text, parent)
        self.setObjectName("HUDValue")
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)


class HUDLabel(QtWidgets.QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setObjectName("HUDLabel")
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)


class MasterBox(QtWidgets.QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("MasterBox")
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Maximum)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.head = QtWidgets.QFrame()
        self.head.setObjectName("MasterBoxHead")
        self.head.setFixedHeight(24)

        h = QtWidgets.QHBoxLayout(self.head)
        h.setContentsMargins(8, 0, 8, 0)

        self.title = QtWidgets.QLabel(title)
        self.title.setObjectName("MasterBoxTitle")
        h.addWidget(self.title)

        self.body = QtWidgets.QFrame()
        self.body.setObjectName("MasterBoxBody")
        self.body.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Maximum)

        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(8, 8, 8, 8)
        self.body_layout.setSpacing(8)

        lay.addWidget(self.head)
        lay.addWidget(self.body)

    def content_layout(self):
        return self.body_layout


# ---------------------------------------------------------------------------
# Mini scene widget: mirrors ParticipantWindow state in the operator panel
# ---------------------------------------------------------------------------
class SceneMirror(QtWidgets.QWidget):
    """Renders whatever the participant window is showing, in miniature."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bg_color = QtGui.QColor(45, 41, 38)
        self.fg_color = QtGui.QColor(220, 220, 220)
        self.mode = "blank"
        self.message = ""
        self.stimulus_text = ""
        self.point_visible = False
        self.point_norm = (0.5, 0.5)
        self.multi_items = []
        self.setMinimumSize(100, 60)

    def sync_from(self, pw: ParticipantWindow):
        """Copy display state from the ParticipantWindow."""
        self.mode = pw.mode
        self.message = pw.message
        self.stimulus_text = pw.stimulus_text
        self.point_visible = pw.point_visible
        self.point_norm = pw.point_norm
        self.multi_items = list(pw.multi_items)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.bg_color)

        w = self.width()
        h = self.height()
        cx = w // 2
        cy = h // 2
        painter.setPen(QtGui.QPen(self.fg_color))

        if self.mode == "instruction":
            font = QtGui.QFont("Segoe UI", max(7, int(h * 0.045)))
            painter.setFont(font)
            rect = self.rect().adjusted(6, 6, -6, -6)
            painter.drawText(
                rect,
                QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap,
                self.message,
            )

        elif self.mode == "fixation":
            pen = QtGui.QPen(self.fg_color)
            pen.setWidth(max(1, int(min(w, h) * 0.012)))
            painter.setPen(pen)
            size = min(w, h) * 0.07
            painter.drawLine(int(cx - size), cy, int(cx + size), cy)
            painter.drawLine(cx, int(cy - size), cx, int(cy + size))

        elif self.mode == "stimulus":
            font = QtGui.QFont("Segoe UI", max(14, int(min(w, h) * 0.35)))
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, self.stimulus_text)

        elif self.mode == "point" and self.point_visible:
            px = int(self.point_norm[0] * w)
            py = int(self.point_norm[1] * h)
            radius = max(5, int(min(w, h) * 0.025))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(self.fg_color)
            painter.drawEllipse(QtCore.QPoint(px, py), radius, radius)

        elif self.mode == "multi_stimuli":
            font = QtGui.QFont("Segoe UI", max(8, int(h * 0.08)))
            font.setBold(True)
            painter.setFont(font)
            for item in self.multi_items:
                txt = str(item.get("text", ""))
                x = float(item.get("x", 0.5)) * w
                y = float(item.get("y", 0.5)) * h
                rect = QtCore.QRectF(x - 30, y - 20, 60, 40)
                painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, txt)
        # blank → already filled with bg


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.cfg = AppConfig()

        self.tracker = FaceMeshTracker(self.cfg)
        self.engine = OcularEngine(self.cfg)
        self.session = SessionManager(self.cfg)
        self.acc = SessionAccumulator()

        self.physio_cfg = PhysioConfig(
            ecg_device="Dev1",
            ecg_channel="ai0",
            ecg_vmin=0.0,
            ecg_vmax=3.3,
            eda_device="Dev2",
            eda_channel="ai1",
            eda_vmin=-0.1,
            eda_vmax=0.1,
            fs=500.0,
        )
        self.physio = PhysioRuntime(
            cfg=self.physio_cfg,
            save_dir=Path("outputs_multimodal_sync")
        )
        self.physio_started = False
        self.current_physio_block_rows = []
        self.current_physio_eda_events = []
        self._physio_hw_ok_logged = False
        self._physio_hw_error_logged: set = set()

        self.drift_obs = DriftObserver()
        self.feat_win  = FeatureWindow(
            window_sec=self.physio_cfg.visible_window_sec * 3.0,   # 60 s
            hop_sec=30.0,
            fs_physio=self.physio_cfg.fs,
        )
        self._last_drift_result: Optional[dict] = None
        self._drift_hysteresis_count: int = 0

        self.attn_est = AttentionEstimator(
            window_sec=30.0,
            hop_sec=15.0,
            fs_physio=self.physio_cfg.fs,
        )

        self.tracker_started = False
        self.current_mode = "IDLE"
        self.current_block = "Calibracion"
        self.mode_before_pause = "IDLE"

        self.block_t0: Optional[float] = None
        self.pause_started_at: Optional[float] = None

        self.last_result = None
        self.participant_window: Optional[ParticipantWindow] = None

        self.event_times = []
        self.event_names = []

        self.calib_5pt = [
            (0.50, 0.50),
            (0.20, 0.20),
            (0.80, 0.20),
            (0.20, 0.80),
            (0.80, 0.80),
        ]
        self.validation_points = []

        self.block1_task: Optional[Block1OddEvenTask] = None
        self.block2_task: Optional[Block2GoNoGoTask] = None
        self.block3_task: Optional[Block3WorkingMemoryTask] = None

        self.session_initialized = False

        self.C = {
            "bg": "#0a0e14",
            "panel": "#101824",
            "edge": "#233750",
            "text": "#ffffff",
            "subtext": "#94b3d1",
            "dimtext": "#556b82",
            "cyan": "#00f2ff",
            "cyanPale": "#98ddff",
            "green": "#16f08c",
            "yellow": "#ffda3a",
            "magenta": "#ff3bb8",
            "red": "#ff4757",
            "orange": "#ff9a30",
            "blueBtn": "#173b77",
            "greenBtn": "#145a2b",
            "yellowBtn": "#6b4a10",
            "redBtn": "#6e151c",
            "slot": "#07111b",
            "slotEdge": "#29415d",
        }

        self.setWindowTitle("NEURODRYFT - Consola de Investigación")
        self.resize(1540, 900)

        self.apply_global_style()
        self.build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(self.cfg.ui_update_ms)

    def apply_global_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {self.C["bg"]};
                color: {self.C["text"]};
                font-family: Segoe UI;
                font-size: 11px;
            }}
            QFrame#HeaderBar {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0b121d,
                    stop:1 #0f1c2c
                );
                border: none;
            }}
            QLabel#HeaderTitle {{
                color: {self.C["subtext"]};
                font-size: 14px;
                font-weight: 500;
            }}
            QLabel#HeaderDivider {{
                color: {self.C["dimtext"]};
                font-size: 20px;
                font-weight: 300;
            }}
            QLabel#LogoText {{
                color: {self.C["cyan"]};
                font-size: 22px;
                font-weight: 700;
            }}
            QFrame#LeftPanel {{
                background: {self.C["panel"]};
                border: none;
            }}
            QFrame#PanelFrame {{
                background: {self.C["panel"]};
                border: none;
            }}
            QFrame#PanelTitleBar {{
                background: #0f1a29;
                border: none;
            }}
            QLabel#PanelTitleLabel {{
                color: {self.C["cyanPale"]};
                font-size: 11px;
                font-weight: 700;
            }}
            QFrame#PanelBody {{
                background: {self.C["panel"]};
                border: none;
            }}
            QLabel#SectionTitle {{
                color: {self.C["cyan"]};
                font-size: 10px;
                font-weight: 700;
                text-transform: uppercase;
            }}
            QLabel#FieldLabel {{
                color: {self.C["subtext"]};
                font-size: 10px;
                font-weight: 700;
            }}
            QLabel#StateName {{
                color: {self.C["subtext"]};
                font-size: 10px;
                font-weight: 700;
            }}
            QLabel#StateValue {{
                color: {self.C["text"]};
                font-size: 12px;
                font-weight: 400;
            }}
            QLabel#HUDLabel {{
                color: {self.C["subtext"]};
                font-size: 10px;
                font-family: Consolas;
                font-weight: 600;
            }}
            QLabel#HUDValue {{
                color: {self.C["text"]};
                font-size: 11px;
                font-family: Consolas;
                font-weight: 700;
            }}
            QLabel#MiniCaption {{
                color: {self.C["dimtext"]};
                font-size: 8px;
                font-weight: 700;
            }}
            QLabel#FixationLabel {{
                color: {self.C["subtext"]};
                font-size: 9px;
            }}
            QLabel#ReservedText, QLabel#ReservedText2 {{
                color: #6d87a3;
                font-size: 11px;
                font-weight: 700;
            }}

            QLineEdit {{
                background: #09111b;
                border: 1px solid #4a5d73;
                color: white;
                padding: 4px 6px;
                min-height: 28px;
                font-size: 11px;
            }}
            QComboBox, QDoubleSpinBox, QSpinBox {{
                background: #09111b;
                border: 1px solid #4a5d73;
                border-radius: 6px;
                color: white;
                padding: 4px 8px;
                min-height: 28px;
                font-size: 12px;
            }}
            QCheckBox {{
                color: {self.C["text"]};
                spacing: 6px;
                font-size: 11px;
            }}
            QPushButton {{
                border-radius: 6px;
                font-size: 12px;
                font-weight: 700;
                color: white;
                min-height: 34px;
                border: 1px solid #5a6f86;
            }}
            QPushButton#BtnStart {{
                background: {self.C["greenBtn"]};
                color: #d4ffe3;
            }}
            QPushButton#BtnPause {{
                background: {self.C["yellowBtn"]};
                color: #ffe9a6;
            }}
            QPushButton#BtnResume {{
                background: {self.C["blueBtn"]};
                color: #b8d8ff;
            }}
            QPushButton#BtnStop {{
                background: {self.C["redBtn"]};
                color: #ffd4d8;
            }}
            QPushButton#SecondaryBtn {{
                background: #112541;
                color: white;
            }}
            QPlainTextEdit {{
                background: #020a14;
                color: #6fe4f0;
                border: 1px solid #29415d;
                font-family: Consolas;
                font-size: 10px;
            }}
            QFrame#Divider {{
                background: {self.C["edge"]};
                min-height: 1px;
                max-height: 1px;
            }}

            QFrame#MasterBox {{
                background: transparent;
                border: 1px solid {self.C["edge"]};
                border-radius: 8px;
            }}
            QFrame#MasterBoxHead {{
                background: #0d1826;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border-bottom: 1px solid {self.C["edge"]};
            }}
            QLabel#MasterBoxTitle {{
                color: {self.C["cyanPale"]};
                font-size: 10px;
                font-weight: 700;
            }}
            QFrame#MasterBoxBody {{
                background: #0b1420;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
            }}
        """)

    def build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        self.build_header(root)
        self.build_body(root)

    def build_header(self, parent_layout):
        header = QtWidgets.QFrame()
        header.setObjectName("HeaderBar")
        header.setFixedHeight(50)

        layout = QtWidgets.QHBoxLayout(header)
        layout.setContentsMargins(14, 6, 14, 6)
        layout.setSpacing(10)

        logo_wrap = QtWidgets.QWidget()
        logo_lay = QtWidgets.QHBoxLayout(logo_wrap)
        logo_lay.setContentsMargins(0, 0, 0, 0)
        logo_lay.setSpacing(8)

        # ── LOGO IMAGE ──────────────────────────────────────────────────────
        logo_label = QtWidgets.QLabel()
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neurodrift_logo.png")
        if os.path.isfile(logo_path):
            pix = QtGui.QPixmap(logo_path)
            # Scale to fit header height (38 px) keeping aspect ratio
            pix = pix.scaledToHeight(38, QtCore.Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(pix)
        else:
            # Fallback text if the image is missing
            logo_label.setText("NEURODRIFT")
            logo_label.setObjectName("LogoText")
        logo_lay.addWidget(logo_label)

        divider = QtWidgets.QLabel("|")
        divider.setObjectName("HeaderDivider")
        title = QtWidgets.QLabel("Consola de Investigacion")
        title.setObjectName("HeaderTitle")

        logo_lay.addWidget(divider)
        logo_lay.addWidget(title)
        logo_lay.addStretch(1)

        layout.addWidget(logo_wrap, 1)

        btn_wrap = QtWidgets.QWidget()
        btn_lay = QtWidgets.QHBoxLayout(btn_wrap)
        btn_lay.setContentsMargins(0, 0, 0, 0)
        btn_lay.setSpacing(8)

        self.btn_start = QtWidgets.QPushButton("Iniciar")
        self.btn_start.setObjectName("BtnStart")
        self.btn_pause = QtWidgets.QPushButton("Pausar")
        self.btn_pause.setObjectName("BtnPause")
        self.btn_resume = QtWidgets.QPushButton("Reanudar")
        self.btn_resume.setObjectName("BtnResume")
        self.btn_stop = QtWidgets.QPushButton("Finalizar")
        self.btn_stop.setObjectName("BtnStop")

        for b in [self.btn_start, self.btn_pause, self.btn_resume, self.btn_stop]:
            b.setFixedSize(108, 34)
            btn_lay.addWidget(b)

        layout.addWidget(btn_wrap, 0, QtCore.Qt.AlignmentFlag.AlignRight)

        self.btn_start.clicked.connect(self.start_action)
        self.btn_pause.clicked.connect(self.pause_action)
        self.btn_resume.clicked.connect(self.resume_action)
        self.btn_stop.clicked.connect(self.stop_action)

        parent_layout.addWidget(header)

    def build_body(self, parent_layout):
        body = QtWidgets.QWidget()
        body_layout = QtWidgets.QHBoxLayout(body)
        body_layout.setContentsMargins(8, 0, 8, 8)
        body_layout.setSpacing(8)

        self.build_left_column(body_layout)
        self.build_right_zone(body_layout)

        parent_layout.addWidget(body, 1)

    def build_left_column(self, parent_layout):
        left = QtWidgets.QFrame()
        left.setObjectName("LeftPanel")
        left.setFixedWidth(360)

        layout = QtWidgets.QVBoxLayout(left)
        layout.setContentsMargins(14, 14, 14, 10)
        layout.setSpacing(10)

        layout.addWidget(self.make_section_title("DATOS DE SESION"))

        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(14)

        col1 = QtWidgets.QVBoxLayout()
        col1.setSpacing(6)
        col1.addWidget(self.make_field_label("ID del sujeto"))
        self.edt_subject = QtWidgets.QLineEdit("S01")
        self.edt_subject.setFixedHeight(28)
        col1.addWidget(self.edt_subject)

        col2 = QtWidgets.QVBoxLayout()
        col2.setSpacing(6)
        col2.addWidget(self.make_field_label("Operador"))
        self.edt_operator = QtWidgets.QLineEdit("Investigador")
        self.edt_operator.setFixedHeight(28)
        col2.addWidget(self.edt_operator)

        row1.addLayout(col1, 1)
        row1.addLayout(col2, 1)
        layout.addLayout(row1)

        layout.addWidget(self.make_field_label("Nombre del Registro"))
        self.edt_record = QtWidgets.QLineEdit("Sesion_Atencional_01")
        self.edt_record.setFixedHeight(28)
        layout.addWidget(self.edt_record)

        layout.addWidget(self.make_divider())

        layout.addWidget(self.make_section_title("BLOQUE EXPERIMENTAL"))
        layout.addWidget(self.make_field_label("Seleccionar bloque"))

        self.dd_block = QtWidgets.QComboBox()
        self.dd_block.addItems(["Calibracion", "Bloque 1", "Bloque 2", "Bloque 3", "Medicion continua"])
        self.dd_block.currentTextChanged.connect(self.on_block_changed)
        layout.addWidget(self.dd_block)

        layout.addWidget(self.make_field_label("Trials entrenamiento"))
        self.edt_train_trials = QtWidgets.QLineEdit("10")
        self.edt_train_trials.setFixedHeight(28)
        layout.addWidget(self.edt_train_trials)

        layout.addWidget(self.make_field_label("Trials principales"))
        self.edt_main_trials = QtWidgets.QLineEdit("180")
        self.edt_main_trials.setFixedHeight(28)
        layout.addWidget(self.edt_main_trials)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(10)

        self.btn_save = QtWidgets.QPushButton("Guardar Sesion")
        self.btn_save.setObjectName("SecondaryBtn")
        self.btn_save.setFixedHeight(32)

        self.btn_participant = QtWidgets.QPushButton("Ventana participante")
        self.btn_participant.setObjectName("SecondaryBtn")
        self.btn_participant.setFixedHeight(32)

        self.btn_save.clicked.connect(self.save_action)
        self.btn_participant.clicked.connect(self.open_participant)

        btn_row.addWidget(self.btn_save, 1)
        btn_row.addWidget(self.btn_participant, 1)
        layout.addLayout(btn_row)

        layout.addWidget(self.make_divider())

        layout.addWidget(self.make_section_title("ESTADO GENERAL"))
        self.lbl_block = self.make_state_line(layout, "Bloque actual:", "Ninguno")
        self.lbl_phase = self.make_state_line(layout, "Fase actual:", "En espera")
        self.lbl_session = self.make_state_line(layout, "Sesion:", "No iniciada")

        layout.addWidget(self.make_divider())

        layout.addWidget(self.make_section_title("CONSOLA DEL SISTEMA"))
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        layout.addWidget(self.console, 1)

        self.log("Sistema inicializado.")
        self.log("Esperando datos...")

        parent_layout.addWidget(left)

    def build_right_zone(self, parent_layout):
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        gap = 8
        top_h = 310

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(gap)
        right_layout.addLayout(top_row, 0)

        top_left = PanelFrame("Tracking Ocular en Vivo")
        top_left.setFixedHeight(top_h)
        top_row.addWidget(top_left, 7)

        top_right = PanelFrame("Indice de Deriva Atencional")
        top_right.setFixedHeight(top_h)
        top_row.addWidget(top_right, 5)

        self.build_tracking_panel(top_left.content_layout())
        self.build_deriva_panel(top_right.content_layout())

        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.setSpacing(gap)
        right_layout.addLayout(bottom_row, 1)

        left_stack = QtWidgets.QVBoxLayout()
        left_stack.setSpacing(gap)
        bottom_row.addLayout(left_stack, 0)

        self.panel_perclos = PanelFrame("PERCLOS / Variabilidad del Iris")
        self.panel_perclos.setFixedWidth(250)
        self.panel_perclos.setFixedHeight(170)
        left_stack.addWidget(self.panel_perclos)
        self.build_perclos_panel(self.panel_perclos.content_layout())

        # ── Vista de Escena: now renders a live mirror of the participant window ──
        self.panel_scene = PanelFrame("Vista de Escena")
        self.panel_scene.setFixedWidth(250)
        left_stack.addWidget(self.panel_scene, 1)
        self.build_scene_panel(self.panel_scene.content_layout())

        # ── ECG / EDA panel: plots only, no control boxes ──
        self.panel_bio_master = PanelFrame("Control ECG y EDA")
        bottom_row.addWidget(self.panel_bio_master, 1)
        self.build_biosignal_master_panel(self.panel_bio_master.content_layout())

        parent_layout.addWidget(right, 1)

    def build_tracking_panel(self, layout):
        self.video_label = QtWidgets.QLabel()
        self.video_label.setMinimumSize(420, 215)
        self.video_label.setStyleSheet(f"""
            background: #010814;
            border: 1px solid {self.C["edge"]};
        """)
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        caption = QtWidgets.QLabel("HUD")
        caption.setObjectName("MiniCaption")
        caption.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(caption)
        layout.addWidget(self.video_label, 1)

        hud_grid = QtWidgets.QGridLayout()
        hud_grid.setHorizontalSpacing(12)
        hud_grid.setVerticalSpacing(6)

        self.hud_state = HUDValue("---")
        self.hud_ear = HUDValue("---")
        self.hud_perclos = HUDValue("---")
        self.hud_variris = HUDValue("---")
        self.hud_blink = HUDValue("---")
        self.hud_valid = HUDValue("---%")
        self.hud_t = HUDValue("--- s")
        self.hud_pose = HUDValue("---")
        self.hud_reason = HUDValue("---")

        items = [
            ("Estado:", self.hud_state),
            ("EAR:", self.hud_ear),
            ("PERCLOS:", self.hud_perclos),
            ("ValIRIS:", self.hud_variris),
            ("Blink:", self.hud_blink),
            ("Validez:", self.hud_valid),
            ("T bloque:", self.hud_t),
            ("Pose:", self.hud_pose),
            ("Motivo:", self.hud_reason),
        ]

        for i, (k, v) in enumerate(items):
            r = i // 3
            c = (i % 3) * 2
            hud_grid.addWidget(HUDLabel(k), r, c)
            hud_grid.addWidget(v, r, c + 1)

        layout.addLayout(hud_grid)

    def build_deriva_panel(self, layout):
        """
        Índice de Deriva Atencional panel.
        The PlotWidget is kept ready for future signals;
        currently no curve is plotted (per user request).
        """
        self.plot_deriva = pg.PlotWidget()
        self.configure_plot(self.plot_deriva, y_range=(0, 1), x_label="Tiempo (s)", y_label="Indice")
        self.plot_deriva.setMinimumHeight(230)

        self.curve_deriva = self.plot_deriva.plot(
            pen=pg.mkPen(color=(0, 230, 255), width=2),
            name="Ocular drift",
        )
        # Static-gain observer (AttentionEstimator): green dashed overlay
        self.curve_drift_obs = self.plot_deriva.plot(
            pen=pg.mkPen(color=(22, 240, 140), width=2, style=QtCore.Qt.PenStyle.DashLine),
            name="Estimador estático",
        )

        layout.addWidget(self.plot_deriva)

    def build_perclos_panel(self, layout):
        self.plot_perclos = pg.PlotWidget()
        self.configure_plot(self.plot_perclos, x_label="Tiempo (s)", y_label=None)
        self.plot_perclos.setMinimumHeight(110)
        self.plot_perclos.setYRange(0.0, 1.0)

        self.plot_perclos.getAxis("left").setLabel("PERCLOS", color=self.C["yellow"])
        self.plot_perclos.getAxis("left").setTextPen(pg.mkPen(self.C["yellow"]))

        self.plot_perclos.showAxis("right")
        self.plot_perclos.getAxis("right").setLabel("ValIRIS norm", color=self.C["magenta"])
        self.plot_perclos.getAxis("right").setTextPen(pg.mkPen(self.C["magenta"]))

        self.curve_perclos = self.plot_perclos.plot(
            pen=pg.mkPen(color=(255, 215, 40), width=2)
        )

        self.variris_view = pg.ViewBox()
        self.plot_perclos.scene().addItem(self.variris_view)
        self.plot_perclos.getAxis("right").linkToView(self.variris_view)
        self.variris_view.setXLink(self.plot_perclos)
        self.variris_view.setYRange(0.0, 1.0)

        self.curve_variris = pg.PlotCurveItem(
            pen=pg.mkPen(color=(255, 60, 180), width=2)
        )
        self.variris_view.addItem(self.curve_variris)

        self.plot_perclos.getViewBox().sigResized.connect(self.sync_perclos_views)
        self.sync_perclos_views()
        layout.addWidget(self.plot_perclos)

    def build_scene_panel(self, layout):
        """
        Live mirror of whatever the ParticipantWindow is displaying.
        Uses SceneMirror widget which is synced every timer tick.
        """
        self.scene_mirror = SceneMirror()
        self.scene_mirror.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        layout.addWidget(self.scene_mirror, 1)

    def build_biosignal_master_panel(self, layout):
        """
        ECG and EDA plots fill the entire panel — control boxes removed.
        Physio status/error messages are now redirected to the main system console.
        """
        plots_col = QtWidgets.QVBoxLayout()
        plots_col.setSpacing(10)
        layout.addLayout(plots_col, 1)

        ecg_box = MasterBox("ECG")
        plots_col.addWidget(ecg_box, 1)
        self.build_ecg_plot_panel(ecg_box.content_layout())

        eda_box = MasterBox("EDA")
        plots_col.addWidget(eda_box, 1)
        self.build_eda_plot_panel(eda_box.content_layout())

    def build_ecg_plot_panel(self, layout):
        self.plot_ecg = pg.PlotWidget()
        self.configure_plot(self.plot_ecg, x_label="Tiempo (s)", y_label="ECG")
        self.plot_ecg.setMinimumHeight(180)

        self.curve_ecg_raw = self.plot_ecg.plot(
            pen=pg.mkPen(color=(80, 190, 255, 90), width=1.0)
        )
        self.curve_ecg_filt = self.plot_ecg.plot(
            pen=pg.mkPen(color=(80, 255, 170), width=1.8)
        )

        layout.addWidget(self.plot_ecg, 1)

        info = QtWidgets.QGridLayout()
        info.setHorizontalSpacing(10)
        info.setVerticalSpacing(4)

        self.lbl_ecg_status = HUDValue("---")
        self.lbl_ecg_mean = HUDValue("---")
        self.lbl_ecg_pp = HUDValue("---")
        self.lbl_ecg_rms = HUDValue("---")
        self.lbl_hr_bpm = HUDValue("--- BPM")

        info.addWidget(HUDLabel("Estado:"), 0, 0)
        info.addWidget(self.lbl_ecg_status, 0, 1, 1, 3)
        info.addWidget(HUDLabel("Mean:"), 1, 0)
        info.addWidget(self.lbl_ecg_mean, 1, 1)
        info.addWidget(HUDLabel("P-P:"), 1, 2)
        info.addWidget(self.lbl_ecg_pp, 1, 3)
        info.addWidget(HUDLabel("RMS:"), 2, 0)
        info.addWidget(self.lbl_ecg_rms, 2, 1)
        info.addWidget(HUDLabel("FC:"), 2, 2)
        info.addWidget(self.lbl_hr_bpm, 2, 3)

        layout.addLayout(info)

    def build_eda_plot_panel(self, layout):
        self.plot_eda = pg.PlotWidget()
        self.configure_plot(self.plot_eda, x_label="Tiempo (s)", y_label="EDA")
        self.plot_eda.setMinimumHeight(180)

        self.curve_eda_raw = self.plot_eda.plot(
            pen=pg.mkPen(color=(255, 160, 60, 90), width=1.0)
        )
        self.curve_eda_filt = self.plot_eda.plot(
            pen=pg.mkPen(color=(255, 80, 120), width=1.8)
        )

        layout.addWidget(self.plot_eda, 1)

        info = QtWidgets.QGridLayout()
        info.setHorizontalSpacing(10)
        info.setVerticalSpacing(4)

        self.lbl_eda_status = HUDValue("---")
        self.lbl_eda_mean = HUDValue("---")
        self.lbl_eda_pp = HUDValue("---")
        self.lbl_eda_rms = HUDValue("---")
        self.lbl_eda_tonic = HUDValue("---")
        self.lbl_scr_count = HUDValue("---")

        info.addWidget(HUDLabel("Estado:"), 0, 0)
        info.addWidget(self.lbl_eda_status, 0, 1, 1, 3)
        info.addWidget(HUDLabel("Mean:"), 1, 0)
        info.addWidget(self.lbl_eda_mean, 1, 1)
        info.addWidget(HUDLabel("P-P:"), 1, 2)
        info.addWidget(self.lbl_eda_pp, 1, 3)
        info.addWidget(HUDLabel("RMS:"), 2, 0)
        info.addWidget(self.lbl_eda_rms, 2, 1)
        info.addWidget(HUDLabel("Tónico:"), 2, 2)
        info.addWidget(self.lbl_eda_tonic, 2, 3)
        info.addWidget(HUDLabel("SCR/vent:"), 3, 0)
        info.addWidget(self.lbl_scr_count, 3, 1)

        layout.addLayout(info)

    # ── Removed: build_common_controls, build_ecg_controls, build_eda_controls ──
    # Those widgets are no longer part of the UI. build_physio_config_from_ui
    # now uses fixed defaults from self.physio_cfg directly.

    def sync_perclos_views(self):
        if not hasattr(self, "variris_view"):
            return
        vb = self.plot_perclos.getViewBox()
        self.variris_view.setGeometry(vb.sceneBoundingRect())
        self.variris_view.linkedViewChanged(vb, self.variris_view.XAxis)

    def configure_plot(self, plot, y_range=None, x_label=None, y_label=None):
        plot.setBackground("#020a14")
        plot.showGrid(x=True, y=True, alpha=0.12)
        plot.getPlotItem().setMenuEnabled(False)
        plot.getPlotItem().hideButtons()
        plot.getViewBox().setMouseEnabled(x=False, y=False)
        plot.setAntialiasing(True)

        axis_pen = pg.mkPen("#8faecc")
        for ax_name in ("bottom", "left", "right", "top"):
            ax = plot.getAxis(ax_name)
            ax.setPen(axis_pen)
            ax.setTextPen(axis_pen)

        if x_label:
            plot.setLabel("bottom", x_label, color="#8faecc", size="9pt")
        if y_label:
            plot.setLabel("left", y_label, color="#8faecc", size="9pt")
        if y_range is not None:
            plot.setYRange(y_range[0], y_range[1])

    def make_section_title(self, text):
        lbl = QtWidgets.QLabel(text)
        lbl.setObjectName("SectionTitle")
        return lbl

    def make_field_label(self, text):
        lbl = QtWidgets.QLabel(text)
        lbl.setObjectName("FieldLabel")
        return lbl

    def make_divider(self):
        div = QtWidgets.QFrame()
        div.setObjectName("Divider")
        div.setFixedHeight(1)
        return div

    def make_state_line(self, parent_layout, label_text, value_text):
        row = QtWidgets.QHBoxLayout()
        lbl_name = QtWidgets.QLabel(label_text)
        lbl_name.setObjectName("StateName")
        lbl_name.setFixedWidth(120)
        lbl_val = QtWidgets.QLabel(value_text)
        lbl_val.setObjectName("StateValue")
        row.addWidget(lbl_name)
        row.addWidget(lbl_val, 1)
        parent_layout.addLayout(row)
        return lbl_val

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.console.appendPlainText(f"[{ts}] {msg}")

    # bio_log now routes to the main system console
    def bio_log(self, msg: str):
        self.log(f"[ECG/EDA] {msg}")

    def build_physio_config_from_ui(self) -> PhysioConfig:
        """
        Returns physio config using fixed defaults — the UI control widgets
        have been removed. Adjust PhysioConfig defaults in physio_types.py
        or self.physio_cfg at startup if you need different values.
        """
        cfg = PhysioConfig()
        cfg.common_delay_sec = self.physio_cfg.common_delay_sec
        cfg.visible_window_sec = self.physio_cfg.visible_window_sec
        cfg.notch_mode = self.physio_cfg.notch_mode
        cfg.invert_ecg = self.physio_cfg.invert_ecg
        cfg.invert_eda = self.physio_cfg.invert_eda
        cfg.ecg_vis_hp = self.physio_cfg.ecg_vis_hp
        cfg.ecg_vis_lp = self.physio_cfg.ecg_vis_lp
        return cfg

    def _calibrate_drift_observer(self) -> None:
        """
        Build the DriftObserver baseline from data collected during CALIB_D.
        Uses calibration_samples() with 15-second sub-windows so we get
        enough φ vectors from the 60-second CALIB_D phase.
        """
        self.feat_win.reset()
        # Re-feed the CALIB_D ocular rows into the feature window using absolute time
        calib_rows = [r for r in self.engine.rows if r.get("phase") == "CALIB_D"]
        if calib_rows:
            t0_abs = time.time() - calib_rows[-1].get("time_s", 0.0)
            for r in calib_rows:
                t_abs = t0_abs + r.get("time_s", 0.0)
                self.feat_win.push_ocular(
                    t_abs,
                    r.get("perclos", float("nan")),
                    r.get("var_iris_clean", float("nan")),
                )

        # Re-feed physio data for the same window
        if self.physio_started:
            ecg_t, ecg_v, eda_t, eda_v = self.physio.reader.get_data()
            self.feat_win.push_physio_chunk(ecg_t, ecg_v, eda_t, eda_v)

        phi_samples = self.feat_win.calibration_samples(sub_window_sec=15.0, sub_hop_sec=5.0)
        ok = self.drift_obs.calibrate(phi_samples)
        if ok:
            self.log(
                f"Observador de deriva calibrado — {len(phi_samples)} muestras basales. "
                f"μ=[HR:{self.drift_obs.mu_basal[0]:.1f} BPM, "
                f"PERCLOS:{self.drift_obs.mu_basal[4]:.3f}, "
                f"IrisVar:{self.drift_obs.mu_basal[5]:.5f}]"
            )
        else:
            self.log(
                "Observador de deriva: calibración insuficiente (pocas ventanas finitas). "
                "El índice de deriva no estará disponible."
            )
        # Reset window for real-time operation
        self.feat_win.reset()

        # ── AttentionEstimator: calibrate with same engine data ───────────────
        physio_reader = self.physio.reader if self.physio_started else None
        ae_ok = self.attn_est.calibrate_from_engine(self.engine.rows, physio_reader)
        if ae_ok:
            self.log(
                f"[AE] Estimador estático calibrado — "
                f"μ=[HR:{self.attn_est.mu_basal[0]:.1f} BPM, "
                f"PERCLOS:{self.attn_est.mu_basal[4]:.3f}]  "
                f"DARE={'OK' if self.attn_est.dare_succeeded() else 'fallback'}"
            )
        else:
            self.log("[AE] Estimador estático: calibración insuficiente.")

    def _on_drift_update(self, dr: dict) -> None:
        """
        Called every 30 s when the observer emits a new drift estimate.
        Implements hysteresis-gated feedback logging.
        """
        I_k = dr.get("drift_index", 0.0)
        T_on  = 0.65
        T_off = 0.45

        if I_k >= T_on:
            self._drift_hysteresis_count += 1
        elif I_k < T_off:
            self._drift_hysteresis_count = 0

        D = dr.get("mahalanobis", 0.0)
        unc = self.drift_obs.get_uncertainty()
        self.log(
            f"[Deriva] I={I_k:.3f}  "
            f"a={dr.get('autonomic', 0.0):.2f}  "
            f"o={dr.get('ocular', 0.0):.2f}  "
            f"d={dr.get('global_drift', 0.0):.2f}  "
            f"D_mah={D:.2f}  "
            f"P_tr={unc:.4f}"
        )

        if self._drift_hysteresis_count >= 2:
            self.log("⚑ ALERTA DERIVA ATENCIONAL sostenida — considera intervención.")
            self._drift_hysteresis_count = 0

    def _check_physio_hw_status(self, phys):
        ecg_st = phys.ecg_hw_status
        eda_st = phys.eda_hw_status
        if not self._physio_hw_ok_logged:
            if ecg_st.startswith("OK") and eda_st.startswith("OK"):
                self._physio_hw_ok_logged = True
                self.log(f"[ECG/EDA] Hardware conectado — {ecg_st}")
        for label, st in (("ECG", ecg_st), ("EDA", eda_st)):
            if "Error" in st and label not in self._physio_hw_error_logged:
                self._physio_hw_error_logged.add(label)
                self.log(f"[ECG/EDA] ADVERTENCIA {label}: {st}")

    def ensure_physio(self) -> bool:
        if self.physio_started:
            return True
        try:
            self.physio.update_config(self.build_physio_config_from_ui())
            self.physio.start()
            self.physio_started = True
            self.log("[ECG/EDA] Módulo ECG/EDA inicializado.")
            return True
        except Exception as e:
            self.log(f"[ECG/EDA] Error al iniciar ECG/EDA: {e}")
            return False

    def parse_trial_fields(self):
        try:
            n_train = int(self.edt_train_trials.text().strip())
            if n_train < 0:
                raise ValueError
        except Exception:
            n_train = 10

        try:
            n_main = int(self.edt_main_trials.text().strip())
            if n_main < 1:
                raise ValueError
        except Exception:
            n_main = 180

        return n_train, n_main

    def shift_block_t0_after_pause(self):
        if self.block_t0 is not None and self.pause_started_at is not None:
            dt = time.perf_counter() - self.pause_started_at
            self.block_t0 += dt
        self.pause_started_at = None

    def on_block_changed(self, value: str):
        self.current_block = value
        self.log(f"Bloque: {value}")

    def ensure_tracker(self) -> bool:
        if self.tracker_started:
            return True
        try:
            self.tracker.start()
            self.tracker_started = True
            self.log("Camara y FaceMesh inicializados.")
            return True
        except Exception as e:
            self.log(f"Error al iniciar camara: {e}")
            return False

    def ensure_session_initialized(self):
        if self.session_initialized:
            return

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        subject_id = self.edt_subject.text().strip() or "S01"
        operator = self.edt_operator.text().strip() or "Investigador"
        record_name = self.edt_record.text().strip() or "Sesion_Atencional_01"

        self.session.start(subject_id, operator, record_name, session_id)
        self.acc.start_session(session_id, subject_id, operator, record_name)

        self.session_initialized = True
        self.lbl_session.setText(f"Activa: {session_id}")
        self.log(f"Sesion experimental iniciada: {session_id}")

    def open_participant(self):
        if self.participant_window is None:
            self.participant_window = ParticipantWindow()
            self.participant_window.set_palette_colors(
                bg_rgb=(45, 41, 38),
                fg_rgb=(220, 220, 220),
            )
            self.participant_window.key_pressed.connect(self.on_participant_key)

        self.participant_window.show()
        self.participant_window.raise_()
        self.participant_window.activateWindow()
        self.participant_window.setFocus(QtCore.Qt.FocusReason.ActiveWindowFocusReason)

    def on_participant_key(self, key_txt: str):
        if self.current_mode == "BLOCK1" and self.block1_task is not None:
            self.block1_task.handle_key(key_txt)
        elif self.current_mode == "BLOCK2" and self.block2_task is not None:
            self.block2_task.handle_key(key_txt)
        elif self.current_mode == "BLOCK3" and self.block3_task is not None:
            self.block3_task.handle_key(key_txt)

    def current_time(self) -> float:
        if self.block_t0 is None:
            return 0.0
        return time.perf_counter() - self.block_t0

    def register_event(self, time_s: float, event_name: str):
        self.session.add_event(time_s, event_name)
        self.acc.append_global_event({
            "session_id": self.session.session_id,
            "time_s": float(time_s),
            "event_name": event_name,
        })

    def finalize_current_block(self, block_name: str):
        summary_rows = []

        if block_name in ("Calibracion", "Medicion continua", "Bloque 1", "Bloque 2", "Bloque 3"):
            ocular_rows = [dict(r) for r in self.engine.rows]
        else:
            ocular_rows = []

        behavior_rows = []
        markers_rows = []

        if block_name == "Bloque 1" and self.block1_task is not None:
            behavior_rows = [dict(r) for r in self.block1_task.behavior_rows]
            markers_rows = [dict(r) for r in self.block1_task.marker_rows]
            summary_rows = [dict(r) for r in self.block1_task.get_summary_rows()]
        elif block_name == "Bloque 2" and self.block2_task is not None:
            behavior_rows = [dict(r) for r in self.block2_task.behavior_rows]
            markers_rows = [dict(r) for r in self.block2_task.marker_rows]
            summary_rows = [dict(r) for r in self.block2_task.get_summary_rows()]
        elif block_name == "Bloque 3" and self.block3_task is not None:
            behavior_rows = [dict(r) for r in self.block3_task.behavior_rows]
            markers_rows = [dict(r) for r in self.block3_task.marker_rows]
            summary_rows = [dict(r) for r in self.block3_task.get_summary_rows()]
        else:
            summary_rows = [self.engine.get_summary()]

        self.acc.append_block_data(
            block_name=block_name,
            ocular_rows=ocular_rows,
            behavior_rows=behavior_rows,
            markers_rows=markers_rows,
            summary_rows=summary_rows,
        )

        if self.current_physio_block_rows:
            self.acc.append_physio_block_data(
                block_name=block_name,
                physio_feature_rows=list(self.current_physio_block_rows),
            )
            self.current_physio_block_rows = []

        block_summary = self.acc.summarize_ocular_block(block_name)

        if behavior_rows:
            dfb = pd.DataFrame(behavior_rows)
            if "is_correct" in dfb:
                block_summary["accuracy_percent"] = 100.0 * float(dfb["is_correct"].mean())
            else:
                block_summary["accuracy_percent"] = np.nan

            if "rt_ms" in dfb:
                rt = pd.to_numeric(dfb["rt_ms"], errors="coerce")
                block_summary["rt_mean_ms"] = float(rt.mean()) if len(rt.dropna()) else np.nan
            else:
                block_summary["rt_mean_ms"] = np.nan
        else:
            block_summary["accuracy_percent"] = np.nan
            block_summary["rt_mean_ms"] = np.nan

        self.acc.add_block_summary(block_summary)
        self.acc.build_trials_long_from_behavior()

    def start_action(self):
        block = self.dd_block.currentText()

        if block != "Calibracion" and block != "Medicion continua" and not self.engine.baseline_ready:
            self.log("Primero debes completar la calibracion.")
            self.lbl_phase.setText("Calibracion requerida")
            return

        if not self.ensure_tracker():
            return

        if not self.ensure_physio():
            return

        self.ensure_session_initialized()

        if self.participant_window is None:
            self.open_participant()

        self.engine.set_session_context(
            session_id=self.session.session_id,
            subject_id=self.session.subject_id,
            block_name=block,
        )

        self.physio.update_config(self.build_physio_config_from_ui())
        self.physio.set_session_context(
            session_id=self.session.session_id,
            subject_id=self.session.subject_id,
            block_name=block,
        )

        self.current_physio_block_rows = []
        self.current_physio_eda_events = []

        self.event_times = [0.0]
        self.event_names = ["INICIO"]
        self.lbl_block.setText(block)
        self.block_t0 = time.perf_counter()

        n_train, n_main = self.parse_trial_fields()

        self.register_event(0.0, f"START_{block.upper().replace(' ', '_')}")

        if block == "Calibracion":
            self.validation_points = random.sample(self.calib_5pt, 2)
            self.engine.start_calibration()
            self.current_mode = "CALIBRATION"
            self.lbl_phase.setText("Calibracion - Fase A")
            self.log("Calibracion iniciada.")
            self.participant_window.activate_input_focus()
            return

        if block == "Bloque 1":
            self.block1_task = Block1OddEvenTask(Block1Config(
                practice_trials=n_train,
                default_main_trials=n_main,
            ))
            self.block1_task.set_practice_trials(n_train)
            self.block1_task.set_main_trials(n_main)
            self.block1_task.start()

            self.engine.start_running()
            self.engine.set_phase("BLOCK1")
            self.current_mode = "BLOCK1"
            self.attn_est.set_block("Bloque 1")
            self.lbl_phase.setText("Bloque 1 - Instrucciones")
            self.log(f"Bloque 1 iniciado. Entrenamiento={n_train}, Principal={n_main}")
            self.participant_window.activate_input_focus()
            return

        if block == "Bloque 2":
            self.block2_task = Block2GoNoGoTask(Block2Config(
                practice_trials=n_train,
                default_main_trials=n_main,
            ))
            self.block2_task.set_practice_trials(n_train)
            self.block2_task.set_main_trials(n_main)
            self.block2_task.start()

            self.engine.start_running()
            self.engine.set_phase("BLOCK2")
            self.current_mode = "BLOCK2"
            self.attn_est.set_block("Bloque 2")
            self.lbl_phase.setText("Bloque 2 - Instrucciones")
            self.log(f"Bloque 2 iniciado. Entrenamiento={n_train}, Principal={n_main}")
            self.participant_window.activate_input_focus()
            return

        if block == "Bloque 3":
            self.block3_task = Block3WorkingMemoryTask(Block3Config(
                practice_trials=n_train,
                default_main_trials=n_main,
            ))
            self.block3_task.set_practice_trials(n_train)
            self.block3_task.set_main_trials(n_main)
            self.block3_task.start()

            self.engine.start_running()
            self.engine.set_phase("BLOCK3")
            self.current_mode = "BLOCK3"
            self.attn_est.set_block("Bloque 3")
            self.lbl_phase.setText("Bloque 3 - Instrucciones")
            self.log(f"Bloque 3 iniciado. Entrenamiento={n_train}, Principal={n_main}")
            self.participant_window.activate_input_focus()
            return

        self.participant_window.show_fixation()
        self.participant_window.activate_input_focus()
        self.engine.start_running()
        self.engine.set_phase("RUNNING")
        self.current_mode = "RUNNING"
        self.attn_est.set_block("Medicion continua")
        self.lbl_phase.setText("Ejecutando")
        self.log("Medicion continua iniciada.")

    def pause_action(self):
        if self.current_mode not in ("CALIBRATION", "RUNNING", "BLOCK1", "BLOCK2", "BLOCK3"):
            return

        self.mode_before_pause = self.current_mode
        self.pause_started_at = time.perf_counter()

        if self.current_mode == "BLOCK1" and self.block1_task is not None:
            self.block1_task.pause()
        elif self.current_mode == "BLOCK2" and self.block2_task is not None:
            self.block2_task.pause()
        elif self.current_mode == "BLOCK3" and self.block3_task is not None:
            self.block3_task.pause()

        if self.physio_started:
            self.physio.pause()

        self.current_mode = "PAUSED"
        self.lbl_phase.setText("Pausado")
        t = self.current_time()
        self.register_event(t, "PAUSE")
        self.add_timeline_event(t, "PAUSE")
        self.log("Sesion pausada.")

    def resume_action(self):
        if self.current_mode != "PAUSED":
            return

        self.shift_block_t0_after_pause()

        if self.mode_before_pause == "BLOCK1" and self.block1_task is not None:
            self.block1_task.resume()
        elif self.mode_before_pause == "BLOCK2" and self.block2_task is not None:
            self.block2_task.resume()
        elif self.mode_before_pause == "BLOCK3" and self.block3_task is not None:
            self.block3_task.resume()

        if self.physio_started:
            self.physio.update_config(self.build_physio_config_from_ui())
            self.physio.resume()

        self.current_mode = self.mode_before_pause
        self.lbl_phase.setText("Ejecutando")
        t = self.current_time()
        self.register_event(t, "RESUME")
        self.add_timeline_event(t, "RESUME")
        self.log("Sesion reanudada.")
        if self.participant_window is not None:
            self.participant_window.activate_input_focus()

    def stop_action(self):
        if self.current_mode == "IDLE":
            return

        block_name = self.dd_block.currentText()
        t = self.current_time()

        self.register_event(t, "STOP")
        self.add_timeline_event(t, "STOP")

        self.finalize_current_block(block_name)

        self.current_mode = "IDLE"
        self.lbl_phase.setText("Finalizado")
        self.log(f"Bloque {block_name} finalizado y agregado a la sesión.")

    def save_action(self):
        if not self.session_initialized:
            self.log("No hay sesión inicializada para guardar.")
            return

        selected_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Selecciona la carpeta donde guardar la sesión",
            os.path.abspath(self.cfg.output_dir)
        )

        if not selected_dir:
            self.log("Guardado cancelado por el usuario.")
            return

        baseline = {
            "mu_ear": self.engine.baseline.mu_ear,
            "sd_ear": self.engine.baseline.sd_ear,
            "mu_var": self.engine.baseline.mu_var,
            "sd_var": self.engine.baseline.sd_var,
        }

        extra_files = []
        try:
            raw_csv_path = getattr(self.physio.reader, "raw_csv", None)
            if raw_csv_path is not None:
                extra_files.append(str(raw_csv_path))
        except Exception as e:
            self.log(f"No se pudo resolver la ruta del CSV crudo: {e}")

        try:
            session_dir = self.session.save_full_session(
                acc=self.acc,
                baseline=baseline,
                base_dir=selected_dir,
                extra_files=extra_files,
            )
            self.log(f"Sesion completa guardada en: {session_dir}")
        except Exception as e:
            self.log(f"Error al guardar: {e}")
            return

        # Save AttentionEstimator drift CSV alongside session data
        try:
            ae_csv_path = Path(session_dir) / "drift_estimates_ae.csv"
            if self.attn_est.save_csv(ae_csv_path):
                self.log(f"[AE] CSV de deriva guardado: {ae_csv_path.name}")
            else:
                self.log("[AE] Sin estimaciones de deriva que guardar.")
        except Exception as e:
            self.log(f"[AE] Error al guardar CSV de deriva: {e}")

    def update_calibration_phase(self, tnow: float):
        a_end = self.cfg.calib_phase_a_sec
        b_end = a_end + self.cfg.calib_phase_b_sec
        c_end = b_end + self.cfg.calib_phase_c_sec
        d_end = c_end + self.cfg.calib_phase_d_sec

        if tnow < a_end:
            self.engine.set_phase("CALIB_A")
            self.lbl_phase.setText(f"Calibracion - Fase A {tnow:.1f}/{self.cfg.calib_phase_a_sec:.1f}s")
            if self.participant_window:
                self.participant_window.show_fixation()
            return None

        if tnow < b_end:
            self.engine.set_phase("CALIB_B")
            local_t = tnow - a_end
            seg = min(4, int(local_t // 6.0))
            self.lbl_phase.setText(f"Calibracion - Fase B punto {seg + 1}/5")
            if self.participant_window:
                self.participant_window.show_point(*self.calib_5pt[seg])
            return None

        if tnow < c_end:
            self.engine.set_phase("CALIB_C")
            local_t = tnow - b_end
            seg = 0 if local_t < (self.cfg.calib_phase_c_sec / 2.0) else 1
            self.lbl_phase.setText(f"Calibracion - Fase C punto {seg + 1}/2")
            if self.participant_window:
                self.participant_window.show_point(*self.validation_points[seg])
            return None

        if tnow < d_end:
            self.engine.set_phase("CALIB_D")
            local_t = tnow - c_end
            self.lbl_phase.setText(f"Calibracion - Fase D {local_t:.1f}/{self.cfg.calib_phase_d_sec:.1f}s")
            if self.participant_window:
                self.participant_window.show_fixation()
            return None

        return "FIN"

    def evaluate_validation_phase(self):
        rows_c = [r for r in self.engine.rows if r.get("phase") == "CALIB_C"]
        if not rows_c:
            return False, 1.0

        valid_raw = np.array([int(r.get("valid_raw", 0)) for r in rows_c], dtype=int)
        loss = float(np.mean(valid_raw == 0))
        ok = loss <= self.cfg.validation_loss_threshold
        return ok, loss

    def update_block1_participant(self, task_out: dict):
        if self.participant_window is None:
            return
        screen = task_out.get("screen", "blank")
        text = task_out.get("text", "")
        if screen == "instruction":
            self.participant_window.show_instruction(text)
        elif screen == "fixation":
            self.participant_window.show_fixation()
        elif screen == "stimulus":
            self.participant_window.show_stimulus(text)
        else:
            self.participant_window.show_blank()

    def update_block2_participant(self, task_out: dict):
        if self.participant_window is None:
            return
        screen = task_out.get("screen", "blank")
        text = task_out.get("text", "")
        if screen == "instruction":
            self.participant_window.show_instruction(text)
        elif screen == "fixation":
            self.participant_window.show_fixation()
        elif screen == "stimulus":
            self.participant_window.show_stimulus(text)
        else:
            self.participant_window.show_blank()

    def update_block3_participant(self, task_out: dict):
        if self.participant_window is None:
            return
        screen = task_out.get("screen", "blank")
        if screen == "instruction":
            self.participant_window.show_instruction(task_out.get("text", ""))
        elif screen == "fixation":
            self.participant_window.show_fixation()
        elif screen == "stimulus":
            self.participant_window.show_stimulus(task_out.get("text", ""))
        elif screen == "multi_stimuli":
            self.participant_window.show_multi_stimuli(task_out.get("items", []))
        else:
            self.participant_window.show_blank()

    def on_timer(self):
        # ── Mirror participant scene every tick ──────────────────────────────
        if self.participant_window is not None:
            self.scene_mirror.sync_from(self.participant_window)
        else:
            # Show blank mirror when no participant window exists
            self.scene_mirror.mode = "blank"
            self.scene_mirror.update()

        if not self.tracker_started:
            return

        try:
            sample = self.tracker.read()
        except Exception as e:
            self.log(f"Error leyendo camara: {e}")
            return

        self.show_frame(sample.frame_bgr, sample.landmarks)

        if self.current_mode not in ("CALIBRATION", "RUNNING", "BLOCK1", "BLOCK2", "BLOCK3"):
            return

        tnow = self.current_time()

        if self.current_mode == "CALIBRATION":
            phase_status = self.update_calibration_phase(tnow)
            if phase_status == "FIN":
                ok_val, loss = self.evaluate_validation_phase()
                if not ok_val:
                    self.current_mode = "IDLE"
                    self.lbl_phase.setText("Calibracion invalida")
                    self.register_event(tnow, "CALIB_FAIL_VALIDATION")
                    self.add_timeline_event(tnow, "CALIB_FAIL")
                    self.log(f"Calibracion invalida. Perdida de tracking en validacion: {loss * 100:.1f}%")
                    self.finalize_current_block("Calibracion")
                    return

        result = self.engine.step(
            frame_id=sample.frame_id,
            tnow=tnow,
            landmarks=sample.landmarks,
        )
        self.last_result = result
        self.update_hud(result)
        self.update_plots()

        if self.physio_started:
            self.physio.update_config(self.build_physio_config_from_ui())
            phys = self.physio.step()
            self.update_physio_panels(phys)
            self._check_physio_hw_status(phys)
            if phys.ok and self.current_mode in ("BLOCK1", "BLOCK2", "BLOCK3", "RUNNING"):
                self.current_physio_block_rows.append({
                    "time_s": phys.master_time_s,
                    "hr_bpm": phys.hr_bpm,
                    "eda_tonic_mean": phys.eda_tonic_mean,
                    "scr_count_window": phys.scr_count_window,
                })

        # ── Feed feature window ──────────────────────────────────────────────
        if self.current_mode in ("CALIBRATION", "BLOCK1", "BLOCK2", "BLOCK3", "RUNNING"):
            t_abs = time.time()
            if result is not None:
                self.feat_win.push_ocular(
                    t_abs,
                    result.get("perclos", float("nan")),
                    result.get("var_iris_clean", float("nan")),
                )
            if self.physio_started:
                ecg_t, ecg_v, eda_t, eda_v = self.physio.reader.get_data()
                self.feat_win.push_physio_chunk(ecg_t, ecg_v, eda_t, eda_v)

            if self.current_mode != "CALIBRATION" and self.drift_obs.calibrated:
                phi = self.feat_win.maybe_emit(t_abs)
                if phi is not None:
                    dr = self.drift_obs.step(phi)
                    self._last_drift_result = dr
                    self._on_drift_update(dr)

            # ── AttentionEstimator (static-gain observer, 30 s/15 s) ─────────
            if self.current_mode != "CALIBRATION" and self.attn_est.calibrated:
                physio_reader = self.physio.reader if self.physio_started else None
                ae_result = self.attn_est.tick(t_abs, result, physio_reader)
                if ae_result is not None:
                    I_k = ae_result["drift_index"]
                    D   = ae_result["mahalanobis"]
                    unc = self.attn_est.get_uncertainty()
                    self.log(
                        f"[AE] I={I_k:.3f}  "
                        f"a={ae_result['autonomic']:.2f}  "
                        f"o={ae_result['ocular']:.2f}  "
                        f"d={ae_result['global_drift']:.2f}  "
                        f"D={D:.2f}  σ²={unc:.4f}"
                    )
                    if ae_result.get("event_flag"):
                        self.log("⚑ [AE] DERIVA SOSTENIDA detectada.")

        if self.current_mode == "CALIBRATION":
            total_calib = (
                self.cfg.calib_phase_a_sec
                + self.cfg.calib_phase_b_sec
                + self.cfg.calib_phase_c_sec
                + self.cfg.calib_phase_d_sec
            )
            if tnow >= total_calib:
                ok = self.engine.finalize_calibration()
                self.finalize_current_block("Calibracion")
                if ok:
                    self.current_mode = "IDLE"
                    self.lbl_phase.setText("Calibracion completada")
                    self.register_event(tnow, "CALIB_OK")
                    self.add_timeline_event(tnow, "CALIB_OK")
                    self.log("Calibracion completada y agregada a la sesión.")
                    self._calibrate_drift_observer()
                else:
                    self.current_mode = "IDLE"
                    self.lbl_phase.setText("Calibracion fallida — repite")
                    self.register_event(tnow, "CALIB_FAIL")
                    self.add_timeline_event(tnow, "CALIB_FAIL")
                    baseline_info = (
                        f"frames válidos insuficientes (mínimo {self.cfg.min_calibration_frames}). "
                        "Revisa iluminación y posición de la cámara."
                    )
                    self.log(f"Calibracion insuficiente: {baseline_info}")
            return

        if self.current_mode == "RUNNING":
            return

        if self.current_mode == "BLOCK1":
            if self.block1_task is None:
                self.current_mode = "IDLE"
                return

            task_out = self.block1_task.update()
            self.update_block1_participant(task_out)
            status = self.block1_task.get_operator_status()
            self.lbl_phase.setText(
                f'Bloque 1 - {status["phase"]} | Trial {status["trial_current"]}/{status["trial_total"]}'
            )

            if self.block1_task.finished:
                self.finalize_current_block("Bloque 1")
                self.current_mode = "IDLE"
                self.lbl_phase.setText("Bloque 1 terminado")
                self.register_event(tnow, "BLOCK1_FINISHED")
                self.add_timeline_event(tnow, "BLOCK1_FIN")
                self.log("Bloque 1 terminado y agregado a la sesión.")
            return

        if self.current_mode == "BLOCK2":
            if self.block2_task is None:
                self.current_mode = "IDLE"
                return

            task_out = self.block2_task.update()
            self.update_block2_participant(task_out)
            status = self.block2_task.get_operator_status()
            self.lbl_phase.setText(
                f'Bloque 2 - {status["phase"]} | Trial {status["trial_current"]}/{status["trial_total"]}'
            )

            if self.block2_task.finished:
                self.finalize_current_block("Bloque 2")
                self.current_mode = "IDLE"
                self.lbl_phase.setText("Bloque 2 terminado")
                self.register_event(tnow, "BLOCK2_FINISHED")
                self.add_timeline_event(tnow, "BLOCK2_FIN")
                self.log("Bloque 2 terminado y agregado a la sesión.")
            return

        if self.current_mode == "BLOCK3":
            if self.block3_task is None:
                self.current_mode = "IDLE"
                return

            task_out = self.block3_task.update()
            self.update_block3_participant(task_out)
            status = self.block3_task.get_operator_status()
            self.lbl_phase.setText(
                f'Bloque 3 - {status["phase"]} | Trial {status["trial_current"]}/{status["trial_total"]}'
            )

            if self.block3_task.finished:
                self.finalize_current_block("Bloque 3")
                self.current_mode = "IDLE"
                self.lbl_phase.setText("Bloque 3 terminado")
                self.register_event(tnow, "BLOCK3_FINISHED")
                self.add_timeline_event(tnow, "BLOCK3_FIN")
                self.log("Bloque 3 terminado y agregado a la sesión.")
            return

    def show_frame(self, frame_bgr: np.ndarray, landmarks: Optional[np.ndarray]):
        img = frame_bgr.copy()
        if landmarks is not None:
            h, w = img.shape[:2]
            for i in range(0, landmarks.shape[0], 6):
                x = int(landmarks[i, 0] * w)
                y = int(landmarks[i, 1] * h)
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(img, (x, y), 1, (0, 230, 255), -1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QtGui.QImage(img_rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pix.scaled(self.video_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
        )

    def update_hud(self, result):
        state = result["state"]

        if state == "FOCO":
            state_color = self.C["green"]
        elif state == "DISTRACCION":
            state_color = self.C["yellow"]
        elif state == "FATIGA":
            state_color = self.C["red"]
        else:
            state_color = self.C["text"]

        self.hud_state.setText(state)
        self.hud_state.setStyleSheet(
            f"color:{state_color}; font-size:11px; font-family:Consolas; font-weight:700;"
        )

        self.hud_ear.setText(f'{result["ear_mean"]:.3f}' if np.isfinite(result["ear_mean"]) else "---")
        self.hud_perclos.setText(f'{result["perclos"]:.3f}' if np.isfinite(result["perclos"]) else "---")
        self.hud_variris.setText(f'{result["var_iris_clean"]:.5f}' if np.isfinite(result["var_iris_clean"]) else "---")
        self.hud_blink.setText(str(result.get("blink_count_accum", 0)))
        self.hud_valid.setText(f'{result["valid_pct"]:.1f}%')
        self.hud_t.setText(f'{result["time_s"]:.1f} s')
        self.hud_pose.setText(f'{result["yaw_asym"]:.3f}' if np.isfinite(result["yaw_asym"]) else "---")
        self.hud_reason.setText(result.get("valid_reason", "---"))

    def robust_normalize_recent(self, t: np.ndarray, y: np.ndarray, window_sec: float) -> np.ndarray:
        out = np.full_like(y, np.nan, dtype=float)
        if len(y) == 0:
            return out

        finite_mask = np.isfinite(y)
        if not np.any(finite_mask):
            return out

        t_now = t[-1]
        recent_mask = finite_mask & (t >= (t_now - window_sec))
        vals = y[recent_mask]

        if vals.size < 5:
            vals = y[finite_mask]
        if vals.size == 0:
            return out

        p5 = float(np.nanpercentile(vals, 5))
        p95 = float(np.nanpercentile(vals, 95))
        den = max(p95 - p5, 1e-6)

        out[finite_mask] = np.clip((y[finite_mask] - p5) / den, 0.0, 1.0)
        return out

    def update_plots(self):
        if not self.engine.time:
            return

        t = np.array(self.engine.time, dtype=float)
        per = np.array(self.engine.perclos, dtype=float)
        var_clean = np.array(self.engine.var_iris_clean, dtype=float)

        var_norm = self.robust_normalize_recent(t, var_clean, self.cfg.var_plot_norm_window_sec)

        self.curve_perclos.setData(t, per)
        self.curve_variris.setData(t, var_norm)

        if self.engine.drift_index:
            drift = np.array(self.engine.drift_index, dtype=float)
            self.curve_deriva.setData(t, drift)

        ae_t, ae_I = self.attn_est.get_drift_history()
        if len(ae_t) > 0:
            self.curve_drift_obs.setData(ae_t, ae_I)

        if len(t) > 0:
            x0 = max(0.0, t[-1] - self.cfg.plot_window_sec)
            x1 = t[-1] + 2.0
            self.plot_perclos.setXRange(x0, x1, padding=0)
            self.variris_view.setXRange(x0, x1, padding=0)
            self.plot_deriva.setXRange(x0, x1, padding=0)

        self.sync_perclos_views()

    def update_physio_panels(self, phys):
        if not phys.ok:
            return

        try:
            self.curve_ecg_raw.setData(phys.xp, phys.ecg_raw)
            self.curve_ecg_filt.setData(phys.xp, phys.ecg_vis)

            self.curve_eda_raw.setData(phys.xp, phys.eda_raw)
            self.curve_eda_filt.setData(phys.xp, phys.eda_clean)

            if len(phys.xp) > 0:
                x0 = float(np.min(phys.xp))
                x1 = float(np.max(phys.xp))
                self.plot_ecg.setXRange(x0, x1, padding=0)
                self.plot_eda.setXRange(x0, x1, padding=0)

            # Autoscale always on (control widget removed)
            if len(phys.ecg_raw) > 0 or len(phys.ecg_vis) > 0:
                ecg_join = np.concatenate([x for x in [phys.ecg_raw, phys.ecg_vis] if len(x) > 0])
                y0, y1 = self.robust_ylim(ecg_join)
                self.plot_ecg.setYRange(y0, y1, padding=0)

            if len(phys.eda_raw) > 0 or len(phys.eda_clean) > 0:
                eda_join = np.concatenate([x for x in [phys.eda_raw, phys.eda_clean] if len(x) > 0])
                y0, y1 = self.robust_ylim(eda_join)
                self.plot_eda.setYRange(y0, y1, padding=0)

            ecg_mean = float(np.mean(phys.ecg_raw)) if len(phys.ecg_raw) else np.nan
            ecg_pp = float(np.ptp(phys.ecg_raw)) if len(phys.ecg_raw) else np.nan
            ecg_rms = float(np.sqrt(np.mean(np.square(phys.ecg_raw)))) if len(phys.ecg_raw) else np.nan

            eda_mean = float(np.mean(phys.eda_raw)) if len(phys.eda_raw) else np.nan
            eda_pp = float(np.ptp(phys.eda_raw)) if len(phys.eda_raw) else np.nan
            eda_rms = float(np.sqrt(np.mean(np.square(phys.eda_raw)))) if len(phys.eda_raw) else np.nan

            ecg_st = phys.ecg_hw_status
            eda_st = phys.eda_hw_status
            ecg_color = self.C["red"] if "Error" in ecg_st else self.C["green"] if ecg_st.startswith("OK") else self.C["subtext"]
            eda_color = self.C["red"] if "Error" in eda_st else self.C["green"] if eda_st.startswith("OK") else self.C["subtext"]

            self.lbl_ecg_status.setText(ecg_st)
            self.lbl_ecg_status.setStyleSheet(f"color:{ecg_color}; font-size:10px;")
            self.lbl_ecg_mean.setText(f"{ecg_mean:.4f}" if np.isfinite(ecg_mean) else "---")
            self.lbl_ecg_pp.setText(f"{ecg_pp:.4f}" if np.isfinite(ecg_pp) else "---")
            self.lbl_ecg_rms.setText(f"{ecg_rms:.4f}" if np.isfinite(ecg_rms) else "---")
            self.lbl_hr_bpm.setText(f"{phys.hr_bpm:.0f} BPM" if phys.hr_bpm > 0 else "--- BPM")

            self.lbl_eda_status.setText(eda_st)
            self.lbl_eda_status.setStyleSheet(f"color:{eda_color}; font-size:10px;")
            self.lbl_eda_mean.setText(f"{eda_mean:.4f}" if np.isfinite(eda_mean) else "---")
            self.lbl_eda_pp.setText(f"{eda_pp:.4f}" if np.isfinite(eda_pp) else "---")
            self.lbl_eda_rms.setText(f"{eda_rms:.4f}" if np.isfinite(eda_rms) else "---")
            self.lbl_eda_tonic.setText(f"{phys.eda_tonic_mean:.5f}")
            self.lbl_scr_count.setText(str(phys.scr_count_window))

        except Exception as e:
            self.bio_log(f"Error actualizando paneles ECG/EDA: {e}")

    def robust_ylim(self, signal: np.ndarray, pad_ratio: float = 0.12, min_pad: float = 0.01):
        if len(signal) == 0:
            return -1.0, 1.0
        smin = float(np.nanmin(signal))
        smax = float(np.nanmax(signal))
        span = smax - smin
        if span < 1e-12:
            return smin - min_pad, smax + min_pad
        pad = max(span * pad_ratio, min_pad)
        return smin - pad, smax + pad

    def add_timeline_event(self, t_evt: float, label: str):
        self.event_times.append(float(t_evt))
        self.event_names.append(str(label))

    def keyPressEvent(self, event):
        if self.current_mode == "BLOCK1" and self.block1_task is not None:
            key_txt = "space" if event.key() == QtCore.Qt.Key.Key_Space else event.text().lower().strip()
            if event.key() == QtCore.Qt.Key.Key_1:
                key_txt = "1"
            elif event.key() == QtCore.Qt.Key.Key_2:
                key_txt = "2"
            if key_txt:
                self.block1_task.handle_key(key_txt)
            return

        if self.current_mode == "BLOCK2" and self.block2_task is not None:
            key_txt = "space" if event.key() == QtCore.Qt.Key.Key_Space else event.text().lower().strip()
            if key_txt:
                self.block2_task.handle_key(key_txt)
            return

        if self.current_mode == "BLOCK3" and self.block3_task is not None:
            if event.key() == QtCore.Qt.Key.Key_Space:
                key_txt = "space"
            elif event.key() == QtCore.Qt.Key.Key_1:
                key_txt = "1"
            elif event.key() == QtCore.Qt.Key.Key_2:
                key_txt = "2"
            elif event.key() == QtCore.Qt.Key.Key_Escape:
                key_txt = "escape"
            else:
                key_txt = event.text().lower().strip()
            if key_txt:
                self.block3_task.handle_key(key_txt)
            return

        super().keyPressEvent(event)

    def closeEvent(self, event):
        try:
            self.tracker.stop()
        except Exception:
            pass

        try:
            self.physio.stop()
        except Exception:
            pass

        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

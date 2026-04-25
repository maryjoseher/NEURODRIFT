from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from config import AppConfig


@dataclass
class TrackerSample:
    frame_id: int
    timestamp: float          # reloj maestro del sistema (epoch, time.time)
    frame_bgr: np.ndarray
    frame_rgb: np.ndarray
    landmarks: Optional[np.ndarray]
    face_detected: bool


class FaceMeshTracker:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.cap = None
        self.mesh = None
        self.frame_id = 0
        self.mp_face_mesh = mp.solutions.face_mesh

    def start(self) -> None:
        if self.cap is not None:
            return

        self.cap = cv2.VideoCapture(self.cfg.camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_h)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)

        if not self.cap.isOpened():
            self.cap = None
            raise RuntimeError("No se pudo abrir la cámara.")

        self.mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.frame_id = 0

    def stop(self) -> None:
        if self.mesh is not None:
            self.mesh.close()
            self.mesh = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read(self) -> TrackerSample:
        if self.cap is None or self.mesh is None:
            raise RuntimeError("Tracker no inicializado.")

        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("No se pudo leer frame de la cámara.")

        # Reloj maestro del sistema: se etiqueta en cuanto llega el frame
        ts_master = time.time()

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mesh.process(frame_rgb)

        landmarks = None
        face_detected = False

        if results.multi_face_landmarks:
            face_detected = True
            face = results.multi_face_landmarks[0]
            coords = [[lm.x, lm.y, lm.z] for lm in face.landmark]
            landmarks = np.array(coords, dtype=np.float64)

        sample = TrackerSample(
            frame_id=self.frame_id,
            timestamp=ts_master,
            frame_bgr=frame,
            frame_rgb=frame_rgb,
            landmarks=landmarks,
            face_detected=face_detected,
        )

        self.frame_id += 1
        return sample
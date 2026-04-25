from PySide6 import QtCore, QtGui, QtWidgets


class ParticipantWindow(QtWidgets.QWidget):
    key_pressed = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ventana participante")
        self.resize(900, 700)

        self.bg_color = QtGui.QColor(45, 41, 38)
        self.fg_color = QtGui.QColor(220, 220, 220)

        self.mode = "fixation"   # instruction / fixation / stimulus / blank / point / multi_stimuli
        self.message = ""
        self.point_visible = False
        self.point_norm = (0.5, 0.5)
        self.stimulus_text = ""
        self.multi_items = []

        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def set_palette_colors(self, bg_rgb=(45, 41, 38), fg_rgb=(220, 220, 220)):
        self.bg_color = QtGui.QColor(*bg_rgb)
        self.fg_color = QtGui.QColor(*fg_rgb)
        self.update()

    def activate_input_focus(self):
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus(QtCore.Qt.ActiveWindowFocusReason)

    def show_instruction(self, text: str):
        self.mode = "instruction"
        self.message = text
        self.stimulus_text = ""
        self.point_visible = False
        self.multi_items = []
        self.update()

    def show_fixation(self):
        self.mode = "fixation"
        self.message = ""
        self.stimulus_text = ""
        self.point_visible = False
        self.multi_items = []
        self.update()

    def show_stimulus(self, text: str):
        self.mode = "stimulus"
        self.stimulus_text = str(text)
        self.message = ""
        self.point_visible = False
        self.multi_items = []
        self.update()

    def show_blank(self):
        self.mode = "blank"
        self.message = ""
        self.stimulus_text = ""
        self.point_visible = False
        self.multi_items = []
        self.update()

    def show_point(self, xn: float, yn: float):
        self.mode = "point"
        self.message = ""
        self.stimulus_text = ""
        self.point_visible = True
        self.point_norm = (float(xn), float(yn))
        self.multi_items = []
        self.update()

    def show_multi_stimuli(self, items):
        self.mode = "multi_stimuli"
        self.message = ""
        self.stimulus_text = ""
        self.point_visible = False
        self.multi_items = items or []
        self.update()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key_Space:
            self.key_pressed.emit("space")
            event.accept()
            return
        if event.key() == QtCore.Qt.Key_1:
            self.key_pressed.emit("1")
            event.accept()
            return
        if event.key() == QtCore.Qt.Key_2:
            self.key_pressed.emit("2")
            event.accept()
            return
        if event.key() == QtCore.Qt.Key_Escape:
            self.key_pressed.emit("escape")
            event.accept()
            return

        txt = event.text().strip().lower()
        if txt:
            self.key_pressed.emit(txt)
            event.accept()
            return

        super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), self.bg_color)

        w = self.width()
        h = self.height()
        cx = w // 2
        cy = h // 2

        painter.setPen(QtGui.QPen(self.fg_color))

        if self.mode == "instruction":
            font = QtGui.QFont("Segoe UI", 22)
            font.setBold(False)
            painter.setFont(font)
            rect = self.rect().adjusted(80, 90, -80, -90)
            painter.drawText(
                rect,
                QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap,
                self.message
            )

        elif self.mode == "fixation":
            pen = QtGui.QPen(self.fg_color)
            pen.setWidth(4)
            painter.setPen(pen)

            size = min(w, h) * 0.06
            painter.drawLine(int(cx - size), cy, int(cx + size), cy)
            painter.drawLine(cx, int(cy - size), int(cx), int(cy + size))

        elif self.mode == "stimulus":
            font = QtGui.QFont("Segoe UI", 96)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, self.stimulus_text)

        elif self.mode == "point" and self.point_visible:
            px = int(self.point_norm[0] * w)
            py = int(self.point_norm[1] * h)
            radius = max(8, int(min(w, h) * 0.012))

            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(self.fg_color)
            painter.drawEllipse(QtCore.QPoint(px, py), radius, radius)

        elif self.mode == "multi_stimuli":
            font = QtGui.QFont("Segoe UI", 28)
            font.setBold(True)
            painter.setFont(font)
            for item in self.multi_items:
                txt = str(item.get("text", ""))
                x = float(item.get("x", 0.5)) * w
                y = float(item.get("y", 0.5)) * h
                rect = QtCore.QRectF(x - 40, y - 30, 80, 60)
                painter.drawText(rect, QtCore.Qt.AlignCenter, txt)
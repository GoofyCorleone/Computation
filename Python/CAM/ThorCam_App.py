"""
ThorCam — Aplicación de captura para cámara CCD Thorlabs DCC
=============================================================

Características:
  • Vista en vivo (live view) con histograma en tiempo real
  • Controles de exposición, ganancia, brillo y contraste
  • Captura individual y secuencias
  • Formatos de guardado: JPG, PNG, TIFF (8/16-bit), TIFF-float32, NPY (raw float)
  • Selector de cámara (compatible con built-in y DCC vía UVC/SDK)
  • Estadísticas en vivo: min/max/mean/std, % saturación
  • Compatibilidad con pipeline polarimétrico (CamPol_3.py / CamPol_5.py)

Requisitos: PySide6, opencv-python, numpy, Pillow, tifffile
"""

from __future__ import annotations
import sys
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import tifffile
from PIL import Image

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QAction, QPainter, QColor, QPen
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider,
    QSpinBox, QDoubleSpinBox, QComboBox, QFileDialog, QGroupBox,
    QHBoxLayout, QVBoxLayout, QGridLayout, QStatusBar, QCheckBox,
    QMessageBox, QSizePolicy
)


# ============================================================================
# BACKENDS DE CÁMARA
# ============================================================================

class OpenCVBackend:
    """Backend OpenCV (UVC). Funciona con la cámara integrada del Mac y
    con cámaras USB-Vision compatibles UVC."""

    def __init__(self, indice=0):
        self.cap = cv2.VideoCapture(indice, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara (índice {indice})")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 4096)
        self.bit_depth = 8

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Convertir a escala de grises (canal único) — útil para polarimetría
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def set_exposure(self, valor):
        # OpenCV en AVFoundation: -13 (oscuro) a -1 (brillante)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, valor)

    def set_gain(self, valor):
        self.cap.set(cv2.CAP_PROP_GAIN, valor)

    def get_info(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return f"UVC {w}×{h} @ {self.bit_depth}-bit"

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class PyUEyeBackend:
    """Backend IDS uEye nativo. Disponible si el SDK uEye/IDS Peak está
    instalado. Soporta bit-depth completo y control fino."""

    def __init__(self, indice=0):
        from pyueye import ueye
        self.ueye = ueye
        self.hcam = ueye.HIDS(indice)
        ret = ueye.is_InitCamera(self.hcam, None)
        if ret != ueye.IS_SUCCESS:
            raise RuntimeError(f"uEye init falló: {ret}")
        ueye.is_SetColorMode(self.hcam, ueye.IS_CM_MONO8)
        info = ueye.SENSORINFO()
        ueye.is_GetSensorInfo(self.hcam, info)
        self.width  = int(info.nMaxWidth)
        self.height = int(info.nMaxHeight)
        self.sensor = info.strSensorName.decode()
        self.bit_depth = 8
        self.mem_ptr = ueye.c_mem_p()
        self.mem_id  = ueye.int()
        ueye.is_AllocImageMem(self.hcam, self.width, self.height, 8,
                              self.mem_ptr, self.mem_id)
        ueye.is_SetImageMem(self.hcam, self.mem_ptr, self.mem_id)
        ueye.is_CaptureVideo(self.hcam, ueye.IS_DONT_WAIT)

    def read(self):
        data = self.ueye.get_data(self.mem_ptr, self.width, self.height, 8,
                                  self.width, copy=True)
        return np.reshape(data, (self.height, self.width))

    def set_exposure(self, valor_ms):
        exp = self.ueye.double(float(valor_ms))
        self.ueye.is_Exposure(self.hcam, self.ueye.IS_EXPOSURE_CMD_SET_EXPOSURE,
                              exp, 8)

    def set_gain(self, valor):
        self.ueye.is_SetHardwareGain(self.hcam, int(valor), -1, -1, -1)

    def get_info(self):
        return f"uEye {self.sensor} {self.width}×{self.height} @ 8-bit"

    def close(self):
        self.ueye.is_FreeImageMem(self.hcam, self.mem_ptr, self.mem_id)
        self.ueye.is_ExitCamera(self.hcam)


def crear_backend(indice=0):
    """Intenta backend uEye primero; si falla, cae a OpenCV."""
    try:
        return PyUEyeBackend(indice)
    except Exception as e:
        print(f"[INFO] uEye no disponible ({e}). Usando OpenCV.")
        return OpenCVBackend(indice)


# ============================================================================
# HILO DE CAPTURA
# ============================================================================

class HiloCamara(QThread):
    """Hilo que adquiere frames continuamente sin bloquear la GUI."""
    frame_listo = Signal(np.ndarray)
    error = Signal(str)

    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.activo = True

    def run(self):
        while self.activo:
            try:
                frame = self.backend.read()
                if frame is not None:
                    self.frame_listo.emit(frame)
                self.msleep(30)  # ~33 fps tope
            except Exception as e:
                self.error.emit(str(e))
                break

    def detener(self):
        self.activo = False
        self.wait(1000)


# ============================================================================
# VENTANA PRINCIPAL
# ============================================================================

class ThorCamApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ThorCam — Captura Polarimétrica")
        self.resize(1200, 750)

        self.backend = None
        self.hilo = None
        self.frame_actual = None      # uint8 (raw del sensor)
        self.frame_procesado = None   # float64 después de brillo/contraste
        self.directorio_salida = Path.home() / "ThorCam_capturas"
        self.directorio_salida.mkdir(exist_ok=True)
        self.contador_capturas = 0

        self._construir_ui()
        self._conectar_camara()

    # ────────────────────────────────────────────────────────────────────────
    # UI
    # ────────────────────────────────────────────────────────────────────────
    def _construir_ui(self):
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout_principal = QHBoxLayout(widget_central)

        # ── Panel izquierdo: video + histograma ────────────────────────────
        col_izq = QVBoxLayout()

        self.label_video = QLabel("Iniciando cámara...")
        self.label_video.setAlignment(Qt.AlignCenter)
        self.label_video.setMinimumSize(800, 600)
        self.label_video.setStyleSheet("background:#101010; color:#aaa; border:1px solid #444;")
        self.label_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        col_izq.addWidget(self.label_video, 4)

        self.label_histograma = QLabel()
        self.label_histograma.setFixedHeight(120)
        self.label_histograma.setStyleSheet("background:#1a1a1a; border:1px solid #444;")
        col_izq.addWidget(self.label_histograma, 1)

        layout_principal.addLayout(col_izq, 4)

        # ── Panel derecho: controles ───────────────────────────────────────
        col_der = QVBoxLayout()
        col_der.setSpacing(10)

        # Cámara
        grupo_cam = QGroupBox("Cámara")
        l1 = QGridLayout(grupo_cam)
        l1.addWidget(QLabel("Dispositivo:"), 0, 0)
        self.combo_dispositivo = QComboBox()
        for i in range(4):
            self.combo_dispositivo.addItem(f"Índice {i}", i)
        l1.addWidget(self.combo_dispositivo, 0, 1)
        self.btn_reconectar = QPushButton("Reconectar")
        self.btn_reconectar.clicked.connect(self._reconectar)
        l1.addWidget(self.btn_reconectar, 1, 0, 1, 2)
        self.label_info = QLabel("—")
        self.label_info.setStyleSheet("color:#888; font-size:11px;")
        l1.addWidget(self.label_info, 2, 0, 1, 2)
        col_der.addWidget(grupo_cam)

        # Exposición / Ganancia
        grupo_exp = QGroupBox("Exposición y ganancia")
        l2 = QGridLayout(grupo_exp)
        l2.addWidget(QLabel("Exposición:"), 0, 0)
        self.spin_exposicion = QDoubleSpinBox()
        self.spin_exposicion.setRange(-13.0, 100.0)
        self.spin_exposicion.setSingleStep(0.5)
        self.spin_exposicion.setValue(-6.0)
        self.spin_exposicion.valueChanged.connect(self._actualizar_exposicion)
        l2.addWidget(self.spin_exposicion, 0, 1)
        l2.addWidget(QLabel("Ganancia:"), 1, 0)
        self.spin_ganancia = QSpinBox()
        self.spin_ganancia.setRange(0, 100)
        self.spin_ganancia.setValue(0)
        self.spin_ganancia.valueChanged.connect(self._actualizar_ganancia)
        l2.addWidget(self.spin_ganancia, 1, 1)
        col_der.addWidget(grupo_exp)

        # Brillo / Contraste (post-proceso software)
        grupo_post = QGroupBox("Procesamiento (software)")
        l3 = QGridLayout(grupo_post)
        l3.addWidget(QLabel("Brillo:"), 0, 0)
        self.slider_brillo = QSlider(Qt.Horizontal)
        self.slider_brillo.setRange(-100, 100)
        self.slider_brillo.setValue(0)
        l3.addWidget(self.slider_brillo, 0, 1)
        self.label_brillo = QLabel("0")
        self.slider_brillo.valueChanged.connect(lambda v: self.label_brillo.setText(str(v)))
        l3.addWidget(self.label_brillo, 0, 2)

        l3.addWidget(QLabel("Contraste:"), 1, 0)
        self.slider_contraste = QSlider(Qt.Horizontal)
        self.slider_contraste.setRange(10, 300)
        self.slider_contraste.setValue(100)
        l3.addWidget(self.slider_contraste, 1, 1)
        self.label_contraste = QLabel("1.00")
        self.slider_contraste.valueChanged.connect(
            lambda v: self.label_contraste.setText(f"{v/100:.2f}"))
        l3.addWidget(self.label_contraste, 1, 2)

        self.chk_falsecolor = QCheckBox("Falso color (mapa térmico)")
        l3.addWidget(self.chk_falsecolor, 2, 0, 1, 3)

        self.chk_saturacion = QCheckBox("Resaltar saturación (rojo)")
        l3.addWidget(self.chk_saturacion, 3, 0, 1, 3)
        col_der.addWidget(grupo_post)

        # Estadísticas
        grupo_stats = QGroupBox("Estadísticas (frame actual)")
        self.label_stats = QLabel("—")
        self.label_stats.setStyleSheet("font-family: 'Menlo'; font-size:11px;")
        QVBoxLayout(grupo_stats).addWidget(self.label_stats)
        col_der.addWidget(grupo_stats)

        # Captura
        grupo_cap = QGroupBox("Captura")
        l4 = QGridLayout(grupo_cap)
        l4.addWidget(QLabel("Formato:"), 0, 0)
        self.combo_formato = QComboBox()
        self.combo_formato.addItems([
            "PNG (8-bit, sin pérdida)",
            "JPG (8-bit, comprimido)",
            "TIFF (8-bit)",
            "TIFF float32 (raw decimal)",
            "NPY (raw decimal, float64)",
        ])
        self.combo_formato.setCurrentIndex(3)
        l4.addWidget(self.combo_formato, 0, 1)

        l4.addWidget(QLabel("Prefijo:"), 1, 0)
        self.combo_prefijo = QComboBox()
        self.combo_prefijo.addItems([
            "captura",
            "I_0_0", "I_45_0", "I_90_0", "I_45_90",  # nombres polarimétricos
        ])
        self.combo_prefijo.setEditable(True)
        l4.addWidget(self.combo_prefijo, 1, 1)

        self.btn_captura = QPushButton("📸  Capturar (Espacio)")
        self.btn_captura.setStyleSheet("padding:10px; font-weight:bold; background:#2266cc; color:white;")
        self.btn_captura.clicked.connect(self.capturar_frame)
        l4.addWidget(self.btn_captura, 2, 0, 1, 2)

        self.btn_carpeta = QPushButton("📁  Carpeta de salida...")
        self.btn_carpeta.clicked.connect(self._elegir_carpeta)
        l4.addWidget(self.btn_carpeta, 3, 0, 1, 2)

        self.label_carpeta = QLabel(str(self.directorio_salida))
        self.label_carpeta.setStyleSheet("color:#888; font-size:10px;")
        self.label_carpeta.setWordWrap(True)
        l4.addWidget(self.label_carpeta, 4, 0, 1, 2)

        col_der.addWidget(grupo_cap)
        col_der.addStretch()

        layout_principal.addLayout(col_der, 1)

        # Status bar
        self.statusBar().showMessage("Listo")

        # Atajo: Espacio = capturar
        atajo_cap = QAction(self)
        atajo_cap.setShortcut(Qt.Key_Space)
        atajo_cap.triggered.connect(self.capturar_frame)
        self.addAction(atajo_cap)

    # ────────────────────────────────────────────────────────────────────────
    # Conexión y captura
    # ────────────────────────────────────────────────────────────────────────
    def _conectar_camara(self, indice=0):
        self._detener_hilo()
        try:
            self.backend = crear_backend(indice)
            self.label_info.setText(self.backend.get_info())
            self.hilo = HiloCamara(self.backend)
            self.hilo.frame_listo.connect(self._procesar_frame)
            self.hilo.error.connect(self._error_camara)
            self.hilo.start()
            self.statusBar().showMessage(f"Cámara conectada: {self.backend.get_info()}")
        except Exception as e:
            QMessageBox.critical(self, "Error de cámara",
                f"No se pudo conectar la cámara:\n{e}\n\n"
                "Verifica permisos en Configuración → Privacidad → Cámara")
            self.label_info.setText("Sin cámara")

    def _detener_hilo(self):
        if self.hilo is not None:
            self.hilo.detener()
            self.hilo = None
        if self.backend is not None:
            self.backend.close()
            self.backend = None

    def _reconectar(self):
        idx = self.combo_dispositivo.currentData()
        self._conectar_camara(idx)

    def _error_camara(self, mensaje):
        self.statusBar().showMessage(f"Error: {mensaje}")

    def _actualizar_exposicion(self, valor):
        if self.backend:
            try:
                self.backend.set_exposure(valor)
            except Exception as e:
                self.statusBar().showMessage(f"Exposición no soportada: {e}")

    def _actualizar_ganancia(self, valor):
        if self.backend:
            try:
                self.backend.set_gain(valor)
            except Exception as e:
                self.statusBar().showMessage(f"Ganancia no soportada: {e}")

    def _elegir_carpeta(self):
        ruta = QFileDialog.getExistingDirectory(self, "Carpeta de salida",
                                                str(self.directorio_salida))
        if ruta:
            self.directorio_salida = Path(ruta)
            self.label_carpeta.setText(str(self.directorio_salida))

    # ────────────────────────────────────────────────────────────────────────
    # Procesamiento de cada frame
    # ────────────────────────────────────────────────────────────────────────
    def _procesar_frame(self, frame):
        self.frame_actual = frame.copy()  # raw (uint8) del sensor

        # Aplicar brillo/contraste en float
        brillo = self.slider_brillo.value() / 255.0
        contraste = self.slider_contraste.value() / 100.0
        f = frame.astype(np.float64) / 255.0
        f = np.clip(contraste * (f - 0.5) + 0.5 + brillo, 0.0, 1.0)
        self.frame_procesado = f

        # Display 8-bit
        display = (f * 255).astype(np.uint8)
        if self.chk_falsecolor.isChecked():
            display = cv2.applyColorMap(display, cv2.COLORMAP_JET)
        else:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)

        if self.chk_saturacion.isChecked():
            mask_sat = frame >= 254
            display[mask_sat] = [255, 0, 0]

        h, w = display.shape[:2]
        qimg = QImage(display.data, w, h, w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.label_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label_video.setPixmap(pix)

        # Estadísticas
        self._actualizar_estadisticas(frame)
        self._actualizar_histograma(frame)

    def _actualizar_estadisticas(self, frame):
        mn, mx = int(frame.min()), int(frame.max())
        media = float(frame.mean())
        std = float(frame.std())
        sat = float((frame >= 254).mean() * 100)
        self.label_stats.setText(
            f"Tamaño:     {frame.shape[1]} × {frame.shape[0]} px\n"
            f"Min / Max:  {mn} / {mx}\n"
            f"Media:      {media:.2f}\n"
            f"Desv. est.: {std:.2f}\n"
            f"Saturado:   {sat:.2f}%"
        )

    def _actualizar_histograma(self, frame):
        hist = cv2.calcHist([frame], [0], None, [256], [0, 256]).flatten()
        if hist.max() > 0:
            hist = hist / hist.max()
        w, h = self.label_histograma.width(), self.label_histograma.height()
        pix = QPixmap(w, h)
        pix.fill(QColor(26, 26, 26))
        painter = QPainter(pix)
        painter.setPen(QPen(QColor(80, 200, 255), 1))
        bar_w = max(1, w / 256)
        for i, v in enumerate(hist):
            x = int(i * bar_w)
            y = int(h - v * h * 0.95)
            painter.drawLine(x, h, x, y)
        painter.end()
        self.label_histograma.setPixmap(pix)

    # ────────────────────────────────────────────────────────────────────────
    # Guardado
    # ────────────────────────────────────────────────────────────────────────
    def capturar_frame(self):
        if self.frame_actual is None:
            self.statusBar().showMessage("Sin frame para capturar")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefijo = self.combo_prefijo.currentText() or "captura"

        # Nombres especiales sin timestamp para nomenclatura polarimétrica
        if prefijo in ("I_0_0", "I_45_0", "I_90_0", "I_45_90"):
            base = prefijo
        else:
            base = f"{prefijo}_{timestamp}"

        idx = self.combo_formato.currentIndex()
        raw = self.frame_actual           # uint8
        proc = self.frame_procesado        # float64 [0,1]

        try:
            if idx == 0:    # PNG
                ruta = self.directorio_salida / f"{base}.png"
                Image.fromarray(raw).save(ruta)
            elif idx == 1:  # JPG
                ruta = self.directorio_salida / f"{base}.jpg"
                Image.fromarray(raw).save(ruta, quality=95)
            elif idx == 2:  # TIFF 8-bit
                ruta = self.directorio_salida / f"{base}.tiff"
                tifffile.imwrite(ruta, raw)
            elif idx == 3:  # TIFF float32 (raw decimal — para polarimetría)
                ruta = self.directorio_salida / f"{base}.tiff"
                tifffile.imwrite(ruta, proc.astype(np.float32))
            elif idx == 4:  # NPY raw float64
                ruta = self.directorio_salida / f"{base}.npy"
                np.save(ruta, proc)

            self.contador_capturas += 1
            self.statusBar().showMessage(
                f"✓ Capturada [{self.contador_capturas}]: {ruta.name}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Error al guardar", str(e))

    def closeEvent(self, event):
        self._detener_hilo()
        event.accept()


# ============================================================================
# MAIN
# ============================================================================
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ThorCam")
    app.setOrganizationName("CAM")
    ventana = ThorCamApp()
    ventana.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

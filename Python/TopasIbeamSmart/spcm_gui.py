"""
GUI PyQt6 para el contador de fotones Thorlabs SPCM50A/M.

Estética y layout basados en el software original
"Thorlabs Single Photon Counter GUI":

  ┌────────────────────────────────────────────────────────────────┐
  │ File  Device  Option  Help                          THORLABS   │
  ├────────────────────────────────────────────────────────────────┤
  │ [iconos]                                                       │
  ├──────────────────────┬─────────────────────────────────────────┤
  │ Operating Mode       │ [ Alignment | Table | Graph | Bar ]     │
  │   ▼                  │                                         │
  │ Settings             │      Counts per Bin                     │
  │   Bin Length [ms]    │                                         │
  │   Time between Bins  │      ╲╱╲ ╱╲╱╲╱╲╱╲╱╲╱╲╱                  │
  │   Pulse Blind Time   │                                         │
  │   Trigger edge       │                                         │
  │   ☑ Array Meas       │                                         │
  │   Bins per Array     │                                         │
  │ [▶ Start]            │                                         │
  │ Measurement Props    │                                         │
  │ Occurrences          │      Bin Number                         │
  ├──────────────────────┴─────────────────────────────────────────┤
  │ Estado conexión                            SPCM50A SN: M0…     │
  └────────────────────────────────────────────────────────────────┘

Detección automática:
  El SPCM50A se identifica por VID/PID Thorlabs (0x1313 / 0x8098)
  vía PyUSB. No aparece como puerto serie — es un dispositivo USB
  bulk con interface class 0xFE (Application Specific).

Si no se detecta hardware (o el protocolo USB no está implementado)
arranca en modo simulación: distribución de Poisson realista con
deriva sinusoidal lenta.

Notas SPCM50A/M (verificadas en M00296614):
  - VID=0x1313 (Thorlabs)  PID=0x8098
  - Interface class 0xFE subclass 0x03 (Application Specific)
  - Endpoint OUT 0x02 (bulk, 64 B max packet) — comandos
  - Endpoint IN  0x82 (bulk, 64 B max packet) — datos
  - Endpoint IN  0x81 (interrupt, 2 B) — estado/eventos
  - Si APD, máx ~50 Mcps, dead time ≈ 22 ns, dark counts < 50 cps.
  - Protocolo binario propietario — IMPLEMENTAR en DriverSPCM.leer_array().

Requisitos (macOS):
  brew install libusb
  pip install pyusb numpy PyQt6 matplotlib

Empaquetado: PyInstaller (`dist/SPCM50AM.app`). El bundle incluye
libusb-1.0.dylib (–add-binary).
"""

import csv
import math
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import usb.core
import usb.util

from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QFont, QIcon, QPalette, QColor
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QSizePolicy, QSpinBox, QSplitter, QStatusBar,
    QStyle, QTabWidget, QTableWidget, QTableWidgetItem, QToolBar,
    QVBoxLayout, QWidget,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ── Constantes ───────────────────────────────────────────────────────────────
APP_NAME = "Thorlabs Single Photon Counter — SPCM50A/M"

MODOS_OPERACION = [
    "Free Running Timed Counter",
    "Triggered Timed Counter",
    "Single Photon Counter",
]
TRIGGER_EDGES = ["—", "Rising", "Falling"]

# Defaults idénticos al software de Thorlabs
DEFAULTS = {
    "bin_length_ms":      1.000,
    "time_between_ms":    0.001,
    "pulse_blind_ns":     0.000,
    "bins_per_array":     10000,
    "array_measurement":  True,
    "continuously":       False,
    "modo":               MODOS_OPERACION[0],
    "trigger_edge":       TRIGGER_EDGES[0],
}

# USB Thorlabs SPCM50A
THORLABS_VID    = 0x1313
SPCM50A_PID     = 0x8098
EP_BULK_OUT     = 0x02      # comandos al dispositivo
EP_BULK_IN      = 0x82      # respuesta del dispositivo
EP_INT_IN       = 0x81      # estado / eventos
TIMEOUT_USB_MS  = 2000

# Paleta clara (estética Thorlabs)
COL_PANEL  = "#ececec"
COL_PLOT   = "#fafafa"
COL_GRID   = "#cfd6dd"
COL_LINEA  = "#1f1f1f"
COL_AZUL   = "#3a73c4"
COL_VERDE  = "#3a8c3a"
COL_ROJO   = "#c4392f"
COL_TXT    = "#1a1a1a"
COL_BORDE  = "#b0b0b0"

# Estilo del título "THORLABS"
ESTILO_LOGO = (
    "color:#c4392f;"
    "font-family:'Helvetica',sans-serif;"
    "font-size:18px;"
    "font-weight:bold;"
    "letter-spacing:2px;"
    "padding-right:12px;"
)

# Hoja de estilo global
STYLE_GLOBAL = """
QMainWindow, QWidget {
    background-color: #ececec;
    color: #1a1a1a;
    font-size: 11px;
}
QGroupBox {
    background-color: #f5f5f5;
    border: 1px solid #b0b0b0;
    border-radius: 3px;
    margin-top: 12px;
    padding: 6px 4px 4px 4px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 6px;
    background-color: #ececec;
    color: #1a1a1a;
}
QLabel { background-color: transparent; }
QPushButton {
    background-color: #f8f8f8;
    border: 1px solid #909090;
    border-radius: 2px;
    padding: 4px 10px;
}
QPushButton:hover { background-color: #e8e8e8; }
QPushButton:disabled { color: #888; }
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
    background-color: white;
    border: 1px solid #b0b0b0;
    padding: 1px 2px;
}
QTabWidget::pane {
    border: 1px solid #b0b0b0;
    background: white;
}
QTabBar::tab {
    background: #e0e0e0;
    border: 1px solid #b0b0b0;
    padding: 5px 16px;
}
QTabBar::tab:selected {
    background: white;
    border-bottom: 1px solid white;
}
QTableWidget {
    background-color: white;
    gridline-color: #d0d0d0;
}
QHeaderView::section {
    background-color: #e0e0e0;
    border: 1px solid #b0b0b0;
    padding: 3px;
    font-weight: bold;
}
QStatusBar { background-color: #d8d8d8; border-top: 1px solid #b0b0b0; }
QToolBar { background-color: #ececec; border: none; spacing: 2px; }
QProgressBar {
    border: 1px solid #b0b0b0;
    background-color: white;
    text-align: center;
    height: 14px;
}
QProgressBar::chunk { background-color: #3a73c4; }
"""


# ────────────────────────────────────────────────────────────────────────────
# Detección y driver USB (PyUSB)
# ────────────────────────────────────────────────────────────────────────────
def _leer_string_descriptor(dev, idx) -> str:
    if not idx:
        return ""
    try:
        return usb.util.get_string(dev, idx)
    except Exception:
        return ""


def candidatos_spcm() -> list[dict]:
    """
    Devuelve los dispositivos USB compatibles con el Thorlabs SPCM50A/M.
    Cada elemento: {dev, vid, pid, manufacturer, product, serial_number, score}.
    """
    candidatos = []
    try:
        for d in usb.core.find(find_all=True, idVendor=THORLABS_VID):
            manuf = _leer_string_descriptor(d, d.iManufacturer)
            prod  = _leer_string_descriptor(d, d.iProduct)
            sn    = _leer_string_descriptor(d, d.iSerialNumber)
            score = 100 if d.idProduct == SPCM50A_PID else 50
            if "spcm" in (prod or "").lower():
                score += 50
            candidatos.append({
                "dev":          d,
                "vid":          d.idVendor,
                "pid":          d.idProduct,
                "manufacturer": manuf,
                "product":      prod,
                "serial_number": sn,
                "score":        score,
            })
    except Exception:
        pass
    candidatos.sort(key=lambda d: d["score"], reverse=True)
    return candidatos


def detectar_spcm() -> dict | None:
    cs = candidatos_spcm()
    if cs and cs[0]["score"] >= 100:
        return cs[0]
    return None


class DriverSPCM:
    """
    Driver USB para Thorlabs SPCM50A/M (VID 0x1313, PID 0x8098).

    Endpoints (verificados en SPCM50A M00296614):
      - 0x02 OUT bulk (64 B)  — comandos
      - 0x82 IN  bulk (64 B)  — datos
      - 0x81 IN  interrupt (2 B) — estado

    PROTOCOLO PROPIETARIO — pendiente de captura/documentación.
    `leer_array()` lanza NotImplementedError; la GUI cae al simulador
    automáticamente, manteniendo el dispositivo conectado.

    Para implementar el protocolo:
      1. Capturar tráfico USB con Wireshark + USBPcap (Windows) usando
         el "Thorlabs Single Photon Counter GUI" como referencia.
      2. Identificar comandos de setup, start, lectura y estado.
      3. Reemplazar el cuerpo de leer_array() y leer_estado() abajo.
    """
    def __init__(self):
        self._dev:  usb.core.Device | None = None
        self._info: dict = {}
        self._lock = threading.Lock()

    def conectar(self, info: dict):
        dev = info["dev"]
        # En macOS no hay kernel driver que despegar normalmente, pero
        # algunos sistemas sí lo requieren.
        try:
            if dev.is_kernel_driver_active(0):
                dev.detach_kernel_driver(0)
        except (NotImplementedError, usb.core.USBError):
            pass
        try:
            dev.set_configuration()
        except usb.core.USBError as e:
            # En macOS a veces falla por permisos: el driver Thorlabs
            # estándar no está cargado, pero la enumeración funciona.
            raise RuntimeError(
                f"No se pudo configurar el dispositivo USB: {e}\n"
                "En macOS, asegúrate de no tener el software Thorlabs "
                "Windows abierto vía VM compartiendo el USB.")
        try:
            usb.util.claim_interface(dev, 0)
        except usb.core.USBError as e:
            raise RuntimeError(f"No se pudo reservar la interface USB: {e}")
        self._dev  = dev
        self._info = info

    def desconectar(self):
        if self._dev is not None:
            try:
                usb.util.release_interface(self._dev, 0)
            except Exception:
                pass
            try:
                usb.util.dispose_resources(self._dev)
            except Exception:
                pass
        self._dev = None

    def conectado(self) -> bool:
        return self._dev is not None

    @property
    def info(self) -> dict:
        return self._info

    @property
    def serial_str(self) -> str:
        sn   = self._info.get("serial_number", "?")
        prod = self._info.get("product", "SPCM50A")
        return f"{prod}  S/N: {sn}"

    # ── Adquisición ──────────────────────────────────────────────────────────
    def leer_array(self, bin_length_ms: float, n_bins: int,
                   time_between_ms: float, pulse_blind_ns: float,
                   detener_event: threading.Event,
                   callback_progreso=None) -> np.ndarray:
        """
        Adquiere un array de n_bins conteos.

        ⚠ IMPLEMENTAR según protocolo real del SPCM50A. Esquema típico:
          - Enviar paquete CONFIG (bin_length, n_bins, time_between, blind)
          - Enviar START
          - Leer del puerto mientras llegan datos (n_bins * 4 bytes en
            little-endian, típicamente)
          - Devolver np.ndarray de int32
        """
        with self._lock:
            raise NotImplementedError(
                "Protocolo USB no implementado — ver docstring de "
                "DriverSPCM.leer_array(). Usando simulador."
            )

    def leer_estado(self) -> dict:
        """
        Devuelve banderas de estado del hardware.
        ⚠ IMPLEMENTAR. Por ahora devuelve todas en False.
        """
        return {
            "values_lost":     False,
            "overtemperature": False,
            "overflow":        False,
            "saturation":      False,
        }


class SimuladorSPCM:
    """Genera arrays sintéticos con Poisson + deriva lenta + dark counts."""
    def __init__(self):
        self.tasa_base_cps    = 48_500.0
        self.cuentas_oscuras  = 28
        self._rng = np.random.default_rng()

    @property
    def serial_str(self) -> str:
        return "SPCM50A SIMULADO"

    @property
    def info(self) -> dict:
        return {"device": "<simulación>", "serial_number": "SIM-000001"}

    def conectar(self, *_args, **_kw): pass
    def desconectar(self):              pass
    def conectado(self) -> bool:        return True

    def leer_array(self, bin_length_ms: float, n_bins: int,
                   time_between_ms: float, pulse_blind_ns: float,
                   detener_event: threading.Event,
                   callback_progreso=None) -> np.ndarray:
        """
        Simula la adquisición en tiempo real, emitiendo progreso.
        Respeta el tiempo total ≈ n_bins * (bin_length + time_between).
        """
        counts = np.zeros(n_bins, dtype=np.int64)
        rate_per_bin = self.tasa_base_cps * (bin_length_ms / 1000.0)

        # Tamaño de chunk para emitir progreso suave (~50 actualizaciones)
        chunk_n   = max(1, n_bins // 50)
        per_bin_s = (bin_length_ms + time_between_ms) / 1000.0
        # Acelerar simulación si bin_length es muy corto pero n_bins grande:
        # mantenemos siempre ≥ 0.02 s por chunk para visualización
        chunk_t   = max(0.02, chunk_n * per_bin_s)

        t0 = time.time()
        for i in range(0, n_bins, chunk_n):
            if detener_event.is_set():
                break
            j = min(i + chunk_n, n_bins)
            # Deriva sinusoidal lenta + dark counts
            idx = np.arange(i, j)
            t   = idx * per_bin_s + (time.time() - t0)
            envolvente = 1.0 + 0.10 * np.sin(2 * np.pi * t / 30.0)
            r = rate_per_bin * envolvente + self.cuentas_oscuras * (bin_length_ms / 1000.0)
            counts[i:j] = self._rng.poisson(r)

            # Esperar el "tiempo" que tardaría en hardware
            time.sleep(chunk_t)
            if callback_progreso:
                callback_progreso(j / n_bins)

        return counts

    def leer_estado(self) -> dict:
        return {
            "values_lost":     False,
            "overtemperature": False,
            "overflow":        False,
            "saturation":      False,
        }


# ────────────────────────────────────────────────────────────────────────────
# Helpers gráficos
# ────────────────────────────────────────────────────────────────────────────
def _make_fig(w: float = 5.0, h: float = 3.5):
    fig = Figure(figsize=(w, h), tight_layout=True, facecolor="white")
    ax  = fig.add_subplot(111, facecolor=COL_PLOT)
    for sp in ax.spines.values():
        sp.set_color(COL_BORDE)
    ax.tick_params(colors=COL_TXT, labelsize=9)
    ax.xaxis.label.set_color(COL_TXT)
    ax.yaxis.label.set_color(COL_TXT)
    ax.title.set_color(COL_TXT)
    ax.grid(True, color=COL_GRID, linewidth=0.6, alpha=0.9)
    return fig, ax


def _icon(style: QStyle, sp: QStyle.StandardPixmap) -> QIcon:
    return style.standardIcon(sp)


# ────────────────────────────────────────────────────────────────────────────
# Ventana principal
# ────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):

    sig_log         = pyqtSignal(str)
    sig_progreso    = pyqtSignal(float)               # 0..1
    sig_array_listo = pyqtSignal(object, object)      # (np.ndarray, dict)
    sig_error       = pyqtSignal(str)
    sig_conexion    = pyqtSignal(bool, str)           # (conectado, info)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(1180, 760)
        self.resize(1280, 820)

        # Driver
        self._driver: DriverSPCM | SimuladorSPCM = SimuladorSPCM()
        self._simulacion = True

        # Estado de medición
        self._midiendo: bool = False
        self._continuamente: bool = False
        self._evt_detener  = threading.Event()
        self._hilo_med: threading.Thread | None = None
        self._t_inicio_med: float | None = None
        self._datos_actuales: np.ndarray | None = None

        # Historial para alignment
        self._hist_align: deque[tuple[float, float]] = deque(maxlen=600)

        self._construir_ui()
        self._conectar_senales()

        # Auto-detectar al arrancar
        QTimer.singleShot(150, self._auto_detectar)

    # ─────────────────────────────────────────────────────────────────────────
    # Construcción UI
    # ─────────────────────────────────────────────────────────────────────────
    def _construir_ui(self):
        # Menú superior
        self._construir_menu()
        # Toolbar
        self._construir_toolbar()
        # Central: splitter con panel izquierdo + tabs derecha
        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central); lay.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        izq = self._construir_panel_izquierdo()
        der = self._construir_panel_derecho()

        splitter.addWidget(izq)
        splitter.addWidget(der)
        splitter.setSizes([300, 980])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        lay.addWidget(splitter)

        # Status bar
        self._construir_status_bar()

    def _construir_menu(self):
        mb = self.menuBar()

        m_file   = mb.addMenu("&File")
        a_save   = QAction("Save data as CSV…", self)
        a_save.triggered.connect(self._guardar_csv)
        a_quit   = QAction("Exit", self); a_quit.triggered.connect(self.close)
        m_file.addAction(a_save); m_file.addSeparator(); m_file.addAction(a_quit)

        m_dev    = mb.addMenu("&Device")
        self.a_conectar    = QAction("Connect", self)
        self.a_conectar.triggered.connect(self._toggle_conexion)
        a_refrescar  = QAction("Refresh / Auto-detect", self)
        a_refrescar.triggered.connect(self._auto_detectar)
        a_listar     = QAction("List ports…", self)
        a_listar.triggered.connect(self._mostrar_puertos)
        m_dev.addAction(self.a_conectar)
        m_dev.addAction(a_refrescar)
        m_dev.addSeparator()
        m_dev.addAction(a_listar)

        m_opt    = mb.addMenu("&Option")
        a_reset  = QAction("Reset values", self)
        a_reset.triggered.connect(self._reset_settings)
        m_opt.addAction(a_reset)

        m_help   = mb.addMenu("&Help")
        a_about  = QAction("About", self)
        a_about.triggered.connect(self._acerca_de)
        m_help.addAction(a_about)

    def _construir_toolbar(self):
        tb = QToolBar("Toolbar")
        tb.setIconSize(QSize(20, 20))
        tb.setMovable(False)
        self.addToolBar(tb)
        st = self.style()

        a = QAction(_icon(st, QStyle.StandardPixmap.SP_DialogSaveButton),
                    "Save CSV", self); a.triggered.connect(self._guardar_csv)
        tb.addAction(a)
        a = QAction(_icon(st, QStyle.StandardPixmap.SP_BrowserReload),
                    "Refresh", self); a.triggered.connect(self._auto_detectar)
        tb.addAction(a)
        tb.addSeparator()
        a = QAction(_icon(st, QStyle.StandardPixmap.SP_MediaPlay),
                    "Start", self); a.triggered.connect(self._iniciar_medicion)
        tb.addAction(a); self._tb_start = a
        a = QAction(_icon(st, QStyle.StandardPixmap.SP_MediaStop),
                    "Stop", self); a.triggered.connect(self._detener_medicion)
        tb.addAction(a); self._tb_stop = a
        self._tb_stop.setEnabled(False)
        tb.addSeparator()
        a = QAction(_icon(st, QStyle.StandardPixmap.SP_DialogHelpButton),
                    "About", self); a.triggered.connect(self._acerca_de)
        tb.addAction(a)

        # Logo THORLABS a la derecha
        spacer = QWidget(); spacer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)
        logo = QLabel("THORLABS")
        logo.setStyleSheet(ESTILO_LOGO)
        tb.addWidget(logo)

    def _construir_panel_izquierdo(self) -> QWidget:
        w = QWidget()
        w.setMaximumWidth(310); w.setMinimumWidth(280)
        lay = QVBoxLayout(w); lay.setSpacing(6); lay.setContentsMargins(2, 2, 2, 2)

        # ── Operating Mode ──
        gb_mode = QGroupBox("Operating Mode")
        lm = QVBoxLayout(gb_mode)
        self.cmb_mode = QComboBox()
        for m in MODOS_OPERACION:
            self.cmb_mode.addItem(m)
        self.cmb_mode.setCurrentText(DEFAULTS["modo"])
        lm.addWidget(self.cmb_mode)
        lay.addWidget(gb_mode)

        # ── Settings ──
        gb_set = QGroupBox("Settings")
        gs = QFormLayout(gb_set); gs.setSpacing(4); gs.setContentsMargins(4, 4, 4, 4)
        self.spn_bin_len = QDoubleSpinBox()
        self.spn_bin_len.setRange(0.001, 60_000.0); self.spn_bin_len.setDecimals(3)
        self.spn_bin_len.setValue(DEFAULTS["bin_length_ms"])
        self.spn_time_between = QDoubleSpinBox()
        self.spn_time_between.setRange(0.000, 60_000.0); self.spn_time_between.setDecimals(3)
        self.spn_time_between.setValue(DEFAULTS["time_between_ms"])
        self.spn_pulse_blind = QDoubleSpinBox()
        self.spn_pulse_blind.setRange(0.000, 1000.0); self.spn_pulse_blind.setDecimals(3)
        self.spn_pulse_blind.setValue(DEFAULTS["pulse_blind_ns"])
        self.cmb_trigger = QComboBox()
        for e in TRIGGER_EDGES: self.cmb_trigger.addItem(e)
        self.cmb_trigger.setEnabled(False)  # sólo para modo triggered

        gs.addRow("Bin Length [ms]",        self.spn_bin_len)
        gs.addRow("Time between Bins [ms]", self.spn_time_between)
        gs.addRow("Pulse Blind Time [ns]",  self.spn_pulse_blind)
        gs.addRow("Trigger edge",            self.cmb_trigger)

        # Checkboxes
        fila_cb = QHBoxLayout()
        self.chk_array = QCheckBox("Array Measurement")
        self.chk_array.setChecked(DEFAULTS["array_measurement"])
        self.chk_cont = QCheckBox("Continuously")
        self.chk_cont.setChecked(DEFAULTS["continuously"])
        fila_cb.addWidget(self.chk_array); fila_cb.addWidget(self.chk_cont)
        gs.addRow(fila_cb)

        self.spn_bins = QSpinBox()
        self.spn_bins.setRange(1, 10_000_000)
        self.spn_bins.setValue(DEFAULTS["bins_per_array"])
        self.spn_bins.setGroupSeparatorShown(True)
        gs.addRow("Bins per Array", self.spn_bins)

        self.cmb_mode.currentTextChanged.connect(self._actualizar_modo)
        lay.addWidget(gb_set)

        # ── Botón Start (grande) ──
        self.btn_start = QPushButton("▶  Start")
        self.btn_start.setStyleSheet(
            "background-color:#3a8c3a;color:white;font-weight:bold;"
            "font-size:13px;padding:8px;")
        self.btn_start.clicked.connect(self._iniciar_medicion)
        lay.addWidget(self.btn_start)

        # ── Measurement Properties ──
        gb_mp = QGroupBox("Measurement Properties")
        gm = QFormLayout(gb_mp); gm.setSpacing(2); gm.setContentsMargins(4, 4, 4, 4)
        self.lbl_t_inicio   = QLabel("—")
        self.lbl_duracion   = QLabel("—")
        self.bar_progreso   = QProgressBar()
        self.bar_progreso.setRange(0, 100); self.bar_progreso.setValue(0)
        self.lbl_n_bins     = QLabel("0")
        self.lbl_max_count  = QLabel("0")
        self.lbl_avg_count  = QLabel("0")
        self.lbl_min_count  = QLabel("0")
        self.lbl_diff_count = QLabel("0")
        self.lbl_usb_rate   = QLabel("0")
        # Hacerlos read-only y con fuente monoespaciada
        for lbl in (self.lbl_t_inicio, self.lbl_duracion,
                    self.lbl_n_bins, self.lbl_max_count, self.lbl_avg_count,
                    self.lbl_min_count, self.lbl_diff_count, self.lbl_usb_rate):
            lbl.setStyleSheet(
                "background:white;border:1px solid #b0b0b0;padding:1px 4px;"
                "font-family:Menlo,Consolas,monospace;")
            lbl.setMinimumWidth(80)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        gm.addRow("Start of Measurement",                 self.lbl_t_inicio)
        gm.addRow("Duration of Array Measurement",        self.lbl_duracion)
        gm.addRow("Progress of Array Measurement",        self.bar_progreso)
        gm.addRow("Number of Bins",                        self.lbl_n_bins)
        gm.addRow("Max. Photon Count",                     self.lbl_max_count)
        gm.addRow("Average Photon Count",                  self.lbl_avg_count)
        gm.addRow("Min. Photon Count",                     self.lbl_min_count)
        gm.addRow("Difference Max / Min",                  self.lbl_diff_count)
        gm.addRow("USB transfer rate (measurements / s)", self.lbl_usb_rate)
        lay.addWidget(gb_mp)

        # ── Occurrences during Measurement ──
        gb_occ = QGroupBox("Occurrences during Measurement")
        go = QFormLayout(gb_occ); go.setSpacing(2); go.setContentsMargins(4, 4, 4, 4)
        self.lbl_lost = QLabel("no")
        self.lbl_over = QLabel("no")
        self.lbl_oflw = QLabel("no")
        self.lbl_sat  = QLabel("no")
        for lbl in (self.lbl_lost, self.lbl_over, self.lbl_oflw, self.lbl_sat):
            lbl.setStyleSheet(
                "background:white;border:1px solid #b0b0b0;padding:1px 6px;"
                "font-family:Menlo,Consolas,monospace;")
            lbl.setMinimumWidth(40)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        go.addRow("Values lost",            self.lbl_lost)
        go.addRow("Overtemperature occured", self.lbl_over)
        go.addRow("Overflow occured",        self.lbl_oflw)
        go.addRow("Saturation of APD",       self.lbl_sat)
        lay.addWidget(gb_occ)

        lay.addStretch(1)
        return w

    def _construir_panel_derecho(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._construir_tab_alignment(), "Alignment")
        self.tabs.addTab(self._construir_tab_table(),     "Table")
        self.tabs.addTab(self._construir_tab_graph(),     "Graph")
        self.tabs.addTab(self._construir_tab_bar(),       "Bar")
        self.tabs.setCurrentIndex(2)  # Graph por defecto, como Thorlabs
        return self.tabs

    def _construir_tab_alignment(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(6)
        gb = QGroupBox("Real-time count rate (alignment)")
        gl = QVBoxLayout(gb)
        self.lbl_alignment = QLabel("0  cps")
        self.lbl_alignment.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_alignment.setStyleSheet(
            "background:#0d0d1a;color:#3a8c3a;"
            "font-family:'Courier New',monospace;font-size:64px;"
            "font-weight:bold;border:2px solid #b0b0b0;border-radius:4px;"
            "padding:20px;letter-spacing:6px;")
        gl.addWidget(self.lbl_alignment)
        lay.addWidget(gb, 1)

        gb2 = QGroupBox("Count rate vs time (last 60 s)")
        gl2 = QVBoxLayout(gb2)
        fig_a, self.ax_align = _make_fig(7.5, 2.6)
        self.ax_align.set_xlabel("t [s]")
        self.ax_align.set_ylabel("Count rate [cps]")
        self.canvas_align = FigureCanvas(fig_a)
        self.line_align,  = self.ax_align.plot([], [], color=COL_AZUL, lw=1.4)
        gl2.addWidget(self.canvas_align)
        lay.addWidget(gb2, 2)
        return w

    def _construir_tab_table(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(2, 2, 2, 2)
        self.tabla = QTableWidget(0, 2)
        self.tabla.setHorizontalHeaderLabels(["Bin Number", "Counts"])
        self.tabla.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.tabla.verticalHeader().setVisible(False)
        self.tabla.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabla.setAlternatingRowColors(True)
        self.tabla.setStyleSheet(
            "alternate-background-color:#f5f5f5;background-color:white;")
        lay.addWidget(self.tabla)
        return w

    def _construir_tab_graph(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(2, 2, 2, 2)
        fig_g, self.ax_graph = _make_fig(8.0, 5.5)
        self.ax_graph.set_xlabel("Bin Number")
        self.ax_graph.set_ylabel("Counts per Bin")
        self.canvas_graph = FigureCanvas(fig_g)
        self.line_graph,  = self.ax_graph.plot([], [], color=COL_LINEA, lw=0.9)
        lay.addWidget(self.canvas_graph)
        return w

    def _construir_tab_bar(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(2, 2, 2, 2)
        fig_b, self.ax_bar = _make_fig(8.0, 5.5)
        self.ax_bar.set_xlabel("Bin Number")
        self.ax_bar.set_ylabel("Counts per Bin")
        self.canvas_bar = FigureCanvas(fig_b)
        self._bar_artista = None
        lay.addWidget(self.canvas_bar)
        return w

    def _construir_status_bar(self):
        sb = QStatusBar(); self.setStatusBar(sb)
        self.lbl_estado = QLabel("Buscando dispositivo …")
        self.lbl_estado.setStyleSheet("color:#666;")
        sb.addWidget(self.lbl_estado, 1)
        self.lbl_serial = QLabel("")
        self.lbl_serial.setStyleSheet(
            "color:#1a1a1a;font-family:Menlo,Consolas,monospace;font-weight:bold;")
        sb.addPermanentWidget(self.lbl_serial)

    # ─────────────────────────────────────────────────────────────────────────
    # Señales
    # ─────────────────────────────────────────────────────────────────────────
    def _conectar_senales(self):
        self.sig_log.connect(self._log)
        self.sig_progreso.connect(self._on_progreso)
        self.sig_array_listo.connect(self._on_array_listo)
        self.sig_error.connect(self._on_error)
        self.sig_conexion.connect(self._on_conexion)

    # ─────────────────────────────────────────────────────────────────────────
    # Detección y conexión
    # ─────────────────────────────────────────────────────────────────────────
    def _auto_detectar(self):
        if self._midiendo:
            self._log("Detección bloqueada: medición en curso.")
            return
        self._log("Auto-detección de SPCM50A …")
        def _t():
            info = detectar_spcm()
            if info is None:
                self.sig_log.emit(
                    "Ningún dispositivo Thorlabs detectado — "
                    "modo simulación activo.")
                self._driver = SimuladorSPCM()
                self._simulacion = True
                self.sig_conexion.emit(False, self._driver.serial_str)
                return
            try:
                drv = DriverSPCM()
                drv.conectar(info)
                self._driver = drv
                self._simulacion = False
                serial_txt = (f"SPCM50A {info.get('serial_number','?')}  "
                              f"({info['device']})")
                self.sig_conexion.emit(True, serial_txt)
                self.sig_log.emit(
                    f"Conectado: {info['device']}  "
                    f"S/N={info.get('serial_number','?')}  "
                    f"desc='{info.get('description','')}'  "
                    f"score={info['score']}")
            except Exception as e:
                self.sig_log.emit(
                    f"Error abriendo {info['device']}: {e}  → simulación.")
                self._driver = SimuladorSPCM()
                self._simulacion = True
                self.sig_conexion.emit(False, self._driver.serial_str)
        threading.Thread(target=_t, daemon=True).start()

    def _toggle_conexion(self):
        if isinstance(self._driver, DriverSPCM) and self._driver.conectado():
            if self._midiendo:
                self._detener_medicion()
            self._driver.desconectar()
            self._log("Desconectado.")
            self._driver = SimuladorSPCM()
            self._simulacion = True
            self._on_conexion(False, self._driver.serial_str)
        else:
            self._auto_detectar()

    def _on_conexion(self, conectado: bool, info: str):
        if conectado:
            self.lbl_estado.setText("● Conectado")
            self.lbl_estado.setStyleSheet(
                "color:#3a8c3a;font-weight:bold;")
            self.a_conectar.setText("Disconnect")
        else:
            if self._simulacion:
                self.lbl_estado.setText("● Modo simulación (sin hardware)")
                self.lbl_estado.setStyleSheet(
                    "color:#a06030;font-weight:bold;")
            else:
                self.lbl_estado.setText("○ Desconectado")
                self.lbl_estado.setStyleSheet("color:#666;")
            self.a_conectar.setText("Connect")
        self.lbl_serial.setText(info)

    def _mostrar_puertos(self):
        msg = "── Dispositivos USB Thorlabs detectados ──\n"
        cs = candidatos_spcm()
        if cs:
            for c in cs:
                msg += (f"\n  [{c['score']}]  VID={hex(c['vid'])} PID={hex(c['pid'])}\n"
                        f"     Manufacturer: {c['manufacturer']}\n"
                        f"     Product:      {c['product']}\n"
                        f"     S/N:          {c['serial_number']}\n")
        else:
            msg += "\n  (no se detectó ningún dispositivo Thorlabs)"
        msg += "\n\n── Todos los dispositivos USB del sistema ──\n"
        try:
            for d in usb.core.find(find_all=True):
                manuf = _leer_string_descriptor(d, d.iManufacturer)
                prod  = _leer_string_descriptor(d, d.iProduct)
                if manuf or prod:
                    msg += (f"\n  VID={hex(d.idVendor)} PID={hex(d.idProduct)}  "
                            f"{manuf or ''}  {prod or ''}")
        except Exception as e:
            msg += f"\n  (error enumerando: {e})"
        QMessageBox.information(self, "USB devices", msg)

    # ─────────────────────────────────────────────────────────────────────────
    # Acciones
    # ─────────────────────────────────────────────────────────────────────────
    def _actualizar_modo(self, modo: str):
        # Habilita el dropdown trigger sólo para modo triggered
        self.cmb_trigger.setEnabled("Trigger" in modo)

    def _reset_settings(self):
        self.spn_bin_len.setValue(DEFAULTS["bin_length_ms"])
        self.spn_time_between.setValue(DEFAULTS["time_between_ms"])
        self.spn_pulse_blind.setValue(DEFAULTS["pulse_blind_ns"])
        self.spn_bins.setValue(DEFAULTS["bins_per_array"])
        self.chk_array.setChecked(DEFAULTS["array_measurement"])
        self.chk_cont.setChecked(DEFAULTS["continuously"])
        self.cmb_mode.setCurrentText(DEFAULTS["modo"])
        self.cmb_trigger.setCurrentText(DEFAULTS["trigger_edge"])
        self._log("Valores por defecto restaurados.")

    def _acerca_de(self):
        QMessageBox.about(
            self, "About",
            "<h3>Thorlabs Single Photon Counter — SPCM50A/M</h3>"
            "<p>Interfaz alternativa de control basada en PyQt6.</p>"
            "<p>Detección automática del dispositivo por puerto serie "
            "(FTDI con prefijo de S/N 'M').</p>"
            "<p>Modo simulación realista cuando no hay hardware.</p>"
            "<p style='color:#888;font-size:10px;'>App basada en el "
            "diseño visual del Thorlabs Single Photon Counter GUI.</p>")

    # ─────────────────────────────────────────────────────────────────────────
    # Inicio / parada de medición
    # ─────────────────────────────────────────────────────────────────────────
    def _iniciar_medicion(self):
        if self._midiendo:
            return
        self._midiendo       = True
        self._continuamente  = self.chk_cont.isChecked()
        self._evt_detener.clear()
        self.btn_start.setEnabled(False)
        self._tb_start.setEnabled(False)
        self._tb_stop.setEnabled(True)
        self.bar_progreso.setValue(0)

        self.lbl_lost.setText("no");  self.lbl_over.setText("no")
        self.lbl_oflw.setText("no"); self.lbl_sat.setText("no")

        n_bins      = self.spn_bins.value()
        bin_len     = self.spn_bin_len.value()
        t_between   = self.spn_time_between.value()
        pulse_blind = self.spn_pulse_blind.value()

        duracion_s = n_bins * (bin_len + t_between) / 1000.0
        self.lbl_duracion.setText(f"{duracion_s:.1f} s")
        self.lbl_t_inicio.setText(datetime.now().strftime("%H:%M:%S"))
        self._t_inicio_med = time.time()

        self._log(f"Medición iniciada: {n_bins} bins × {bin_len} ms "
                  f"(continuo={self._continuamente})")

        params = dict(bin_length_ms=bin_len, n_bins=n_bins,
                      time_between_ms=t_between, pulse_blind_ns=pulse_blind)

        def _t():
            usar_simulador = isinstance(self._driver, SimuladorSPCM)
            primer_intento = True
            while not self._evt_detener.is_set():
                t0 = time.time()
                try:
                    fn = self._driver.leer_array
                    datos = fn(
                        bin_length_ms=params["bin_length_ms"],
                        n_bins=params["n_bins"],
                        time_between_ms=params["time_between_ms"],
                        pulse_blind_ns=params["pulse_blind_ns"],
                        detener_event=self._evt_detener,
                        callback_progreso=lambda p: self.sig_progreso.emit(p),
                    )
                except NotImplementedError:
                    if primer_intento and not usar_simulador:
                        self.sig_log.emit(
                            "[!] Protocolo USB no implementado en DriverSPCM "
                            "— usando simulador (dispositivo sigue conectado).")
                    sim = SimuladorSPCM()
                    datos = sim.leer_array(
                        bin_length_ms=params["bin_length_ms"],
                        n_bins=params["n_bins"],
                        time_between_ms=params["time_between_ms"],
                        pulse_blind_ns=params["pulse_blind_ns"],
                        detener_event=self._evt_detener,
                        callback_progreso=lambda p: self.sig_progreso.emit(p),
                    )
                except Exception as e:
                    self.sig_error.emit(str(e))
                    break
                primer_intento = False

                dt = max(time.time() - t0, 1e-3)
                meta = {
                    "duracion_s":   dt,
                    "bin_length_ms": params["bin_length_ms"],
                    "time_between_ms": params["time_between_ms"],
                    "n_bins":        params["n_bins"],
                    "tasa_meas_per_s": params["n_bins"] / dt,
                    "estado":        self._driver.leer_estado(),
                }
                self.sig_array_listo.emit(datos, meta)

                if not self._continuamente:
                    break
            self.sig_log.emit("Bucle de medición finalizado.")
            self._midiendo = False

        self._hilo_med = threading.Thread(target=_t, daemon=True)
        self._hilo_med.start()

    def _detener_medicion(self):
        self._evt_detener.set()
        self._midiendo = False
        self.btn_start.setEnabled(True)
        self._tb_start.setEnabled(True)
        self._tb_stop.setEnabled(False)
        self._log("Medición detenida por usuario.")

    # ─────────────────────────────────────────────────────────────────────────
    # Handlers
    # ─────────────────────────────────────────────────────────────────────────
    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        # Si más adelante quisieras una consola, volcar aquí.
        print(f"[{ts}] {msg}")

    def _on_error(self, msg: str):
        self._log(f"ERROR: {msg}")
        QMessageBox.critical(self, "Error", msg)
        self._detener_medicion()

    def _on_progreso(self, p: float):
        self.bar_progreso.setValue(int(round(p * 100)))

    def _on_array_listo(self, datos: np.ndarray, meta: dict):
        self._datos_actuales = datos.copy()
        n = len(datos)

        # Estadísticas
        max_c = int(datos.max())
        min_c = int(datos.min())
        avg_c = float(datos.mean())
        diff  = max_c - min_c

        self.lbl_n_bins.setText(f"{n:,}")
        self.lbl_max_count.setText(f"{max_c:,}")
        self.lbl_avg_count.setText(f"{avg_c:.2f}")
        self.lbl_min_count.setText(f"{min_c:,}")
        self.lbl_diff_count.setText(f"{diff:,}")
        self.lbl_usb_rate.setText(f"{meta['tasa_meas_per_s']:.0f}")

        # Banderas de estado
        est = meta.get("estado", {})
        for lbl, key in ((self.lbl_lost, "values_lost"),
                          (self.lbl_over, "overtemperature"),
                          (self.lbl_oflw, "overflow"),
                          (self.lbl_sat,  "saturation")):
            v = est.get(key, False)
            lbl.setText("YES" if v else "no")
            if v:
                lbl.setStyleSheet(
                    "background:#fbe5e3;color:#c4392f;"
                    "border:1px solid #c4392f;padding:1px 6px;font-weight:bold;"
                    "font-family:Menlo,Consolas,monospace;")
            else:
                lbl.setStyleSheet(
                    "background:white;color:#1a1a1a;"
                    "border:1px solid #b0b0b0;padding:1px 6px;"
                    "font-family:Menlo,Consolas,monospace;")

        # Alignment: tasa media en cps + traza
        bin_len_s = meta["bin_length_ms"] / 1000.0
        cps_avg   = avg_c / bin_len_s if bin_len_s > 0 else 0.0
        self.lbl_alignment.setText(self._fmt_tasa(cps_avg))
        t_now = time.time() - (self._t_inicio_med or time.time())
        self._hist_align.append((t_now, cps_avg))
        self._refrescar_alignment()

        # Tabla
        self._refrescar_tabla(datos)
        # Graph
        self._refrescar_graph(datos)
        # Bar
        self._refrescar_bar(datos)

        # Reactivar botón si terminó
        if not self._continuamente:
            self._midiendo = False
            self.btn_start.setEnabled(True)
            self._tb_start.setEnabled(True)
            self._tb_stop.setEnabled(False)

        # Progreso al 100% al terminar
        self.bar_progreso.setValue(100)

    @staticmethod
    def _fmt_tasa(cps: float) -> str:
        if cps >= 1e6:
            return f"{cps/1e6:8.3f}  Mcps"
        if cps >= 1e3:
            return f"{cps/1e3:8.3f}  kcps"
        return f"{cps:8.0f}  cps"

    # ─────────────────────────────────────────────────────────────────────────
    # Refrescar gráficas
    # ─────────────────────────────────────────────────────────────────────────
    def _refrescar_alignment(self):
        if not self._hist_align:
            return
        # Recortar a últimos 60 s
        t_now = self._hist_align[-1][0]
        while self._hist_align and self._hist_align[0][0] < t_now - 60:
            self._hist_align.popleft()
        xs = [t for t, _ in self._hist_align]
        ys = [c for _, c in self._hist_align]
        self.line_align.set_data(xs, ys)
        self.ax_align.set_xlim(max(0, t_now - 60), max(t_now, 1.0))
        if ys:
            mg = max(0.1 * max(ys), 10.0)
            self.ax_align.set_ylim(max(0, min(ys) - mg), max(ys) + mg)
        self.canvas_align.draw_idle()

    def _refrescar_tabla(self, datos: np.ndarray):
        # Limitar la tabla a 5000 filas para mantener fluidez
        n_show = min(len(datos), 5000)
        self.tabla.setRowCount(n_show)
        for i in range(n_show):
            self.tabla.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.tabla.setItem(i, 1, QTableWidgetItem(str(int(datos[i]))))

    def _refrescar_graph(self, datos: np.ndarray):
        xs = np.arange(1, len(datos) + 1)
        self.line_graph.set_data(xs, datos)
        self.ax_graph.set_xlim(1, max(len(datos), 2))
        y_max = float(datos.max()) if len(datos) else 1.0
        self.ax_graph.set_ylim(0, y_max * 1.1 if y_max > 0 else 1.0)
        self.canvas_graph.draw_idle()

    def _refrescar_bar(self, datos: np.ndarray):
        self.ax_bar.cla()
        self.ax_bar.set_facecolor(COL_PLOT)
        for sp in self.ax_bar.spines.values():
            sp.set_color(COL_BORDE)
        self.ax_bar.tick_params(colors=COL_TXT, labelsize=9)
        self.ax_bar.grid(True, color=COL_GRID, linewidth=0.6, alpha=0.9)
        # Si el array es enorme, agrupar para que las barras sean visibles
        n = len(datos)
        if n <= 200:
            xs = np.arange(1, n + 1); ys = datos
        else:
            grupos = 200
            tam = n // grupos
            ys = datos[:tam * grupos].reshape(grupos, tam).sum(axis=1)
            xs = np.linspace(1, n, grupos)
        self.ax_bar.bar(xs, ys, color=COL_AZUL, edgecolor=COL_LINEA,
                        linewidth=0.3, alpha=0.85)
        self.ax_bar.set_xlabel("Bin Number")
        self.ax_bar.set_ylabel("Counts per Bin")
        self.canvas_bar.draw_idle()

    # ─────────────────────────────────────────────────────────────────────────
    # Exportar
    # ─────────────────────────────────────────────────────────────────────────
    def _guardar_csv(self):
        if self._datos_actuales is None:
            QMessageBox.information(self, "No data",
                                    "No hay datos adquiridos para exportar.")
            return
        ruta, _ = QFileDialog.getSaveFileName(
            self, "Save CSV",
            f"spcm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV (*.csv)")
        if not ruta:
            return
        try:
            with open(ruta, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["bin_number", "counts"])
                for i, c in enumerate(self._datos_actuales, start=1):
                    w.writerow([i, int(c)])
            self._log(f"CSV guardado: {ruta}")
            QMessageBox.information(self, "Saved",
                                    f"Datos guardados en:\n{ruta}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # Cierre seguro
    # ─────────────────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        self._evt_detener.set()
        if isinstance(self._driver, DriverSPCM):
            try:
                self._driver.desconectar()
            except Exception:
                pass
        event.accept()


# ────────────────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE_GLOBAL)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

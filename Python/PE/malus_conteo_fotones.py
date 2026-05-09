"""
Experimento de Ley de Malus por conteo de fotones.

GUI PyQt6 que controla SIMULTÁNEAMENTE el láser TOPTICA iBeam Smart
y el contador Thorlabs SPCM50A/M para realizar un barrido completo de
0 a 360° del segundo polarizador.

Para cada ángulo:
  • Se mide N bins de conteo de fotones durante T_integración segundos.
  • En paralelo se muestrea la potencia óptica del láser para
    compensar deriva y ruido (~1 Hz).
  • Se calcula la intensidad normalizada I_norm = <CPS> / <P>.
  • Se propagan las incertidumbres:
        σ_θ      = ±0.5°    (tornillo del polarizador)
        σ_CPS    = std(CPS_bins) / sqrt(N_bins)
        σ_P      = std(P_samples) / sqrt(N_P)
        σ_I/I    = sqrt((σ_CPS/CPS)^2 + (σ_P/P)^2)
  • Cada punto se añade a una curva Malus con barras de error y se
    compara con cos²(θ-θ₀) puntuado de fondo, donde θ₀ es el ángulo
    del máximo observado.

Al alcanzar 360° aparece el botón "Cerrar y guardar"; los datos crudos
y las gráficas se exportan a una carpeta timestamped.

Tema oscuro Catppuccin Mocha — idéntico a las apps base.

Uso:
    cd Python/PE
    python malus_conteo_fotones.py
"""

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QInputDialog,
    QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
    QSizePolicy, QSpinBox, QSplitter, QStatusBar, QVBoxLayout, QWidget,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ── Importar drivers de las apps existentes ──────────────────────────────
_TOPAS = Path(__file__).resolve().parent.parent / "TopasIbeamSmart"
sys.path.insert(0, str(_TOPAS))
import ibeam_gui as iblg          # IBeamDriver, detectar_puerto
import spcm_gui  as spcmm         # DriverSPCM, detectar_spcm


# ── Constantes ───────────────────────────────────────────────────────────
APP_NAME       = "Ley de Malus — Conteo de fotones"
SIGMA_ANGULO_DEG = 0.5             # incertidumbre del ajuste del polarizador
DEFAULT_POTENCIA_MW   = 5.0
# Réplica fiel de los defaults del software Thorlabs SPCM50A/M:
#   Bin Length 1.000 ms · Time between Bins 0.001 ms · Pulse Blind 0 ns ·
#   Bins per Array 10 000 · Operating Mode "Free Running Timed Counter".
DEFAULT_BIN_LEN_MS    = 1.0
DEFAULT_TIME_BETWEEN_MS = 0.001
DEFAULT_PULSE_BLIND_NS  = 0.0
DEFAULT_BINS_POR_ARRAY = 10000
DEFAULT_PASO_ANGULO   = 10.0       # sólo sugerencia para el dialog
PERIODO_POT_S         = 0.4        # cada cuánto preguntar potencia al láser
T_ESTABILIZACION_S    = 4.0        # espera tras encender láser para calibrar P
N_MUESTRAS_CAL_POT    = 6          # nº de lecturas PIC para promediar al calibrar

# Paleta dark Catppuccin Mocha
COL_BG     = "#1e1e2e"
COL_BG2    = "#181825"
COL_PLOT   = "#181825"
COL_GRID   = "#45475a"
COL_TXT    = "#cdd6f4"
COL_TXT_DIM= "#a6adc8"
COL_BORDE  = "#45475a"
COL_AZUL   = "#89b4fa"
COL_VERDE  = "#a6e3a1"
COL_ROJO   = "#f38ba8"
COL_LILA   = "#cba6f7"
COL_AMBAR  = "#fab387"
COL_BARRA  = "#74c7ec"

ESTILO_LOGO = (
    "color:#f38ba8;font-family:'Helvetica',sans-serif;font-size:18px;"
    "font-weight:bold;letter-spacing:2px;padding-right:12px;"
)

STYLE_GLOBAL = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-size: 11px;
}
QGroupBox {
    background-color: #181825;
    border: 1px solid #45475a;
    border-radius: 4px;
    margin-top: 12px;
    padding: 6px 4px 4px 4px;
    font-weight: bold;
    color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 6px;
    background-color: #1e1e2e;
    color: #89b4fa;
}
QLabel { background-color: transparent; color: #cdd6f4; }
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 3px;
    padding: 5px 10px;
}
QPushButton:hover { background-color: #45475a; }
QPushButton:pressed { background-color: #585b70; }
QPushButton:disabled { color: #6c7086; background-color: #181825; }
QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox {
    background-color: #11111b;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 2px;
    padding: 2px 4px;
    selection-background-color: #585b70;
}
QStatusBar {
    background-color: #11111b;
    color: #cdd6f4;
    border-top: 1px solid #45475a;
}
QSplitter::handle { background-color: #45475a; }
QInputDialog, QMessageBox { background-color: #1e1e2e; color: #cdd6f4; }
"""


def _make_fig(w: float = 5.0, h: float = 3.0):
    fig = Figure(figsize=(w, h), tight_layout=True, facecolor=COL_BG)
    ax  = fig.add_subplot(111, facecolor=COL_PLOT)
    for sp in ax.spines.values():
        sp.set_color(COL_BORDE)
    ax.tick_params(colors=COL_TXT, labelsize=9)
    ax.xaxis.label.set_color(COL_TXT)
    ax.yaxis.label.set_color(COL_TXT)
    ax.title.set_color(COL_TXT)
    ax.grid(True, color=COL_GRID, linewidth=0.5, alpha=0.6)
    return fig, ax


# ────────────────────────────────────────────────────────────────────────
# Diálogo de configuración de polarizadores elípticos
# ────────────────────────────────────────────────────────────────────────
class PolarizadoresDialog(QDialog):
    """
    Segunda ventana donde se introducen los ángulos (θ₁, θ₂) de cada
    polarizador elíptico. Cada polarizador se construye como
        P(θ₁, θ₂) = Q(θ₁ + 90°) · PL(θ₂) · Q(θ₁)
    siguiendo la deducción de Malus_Generalizada.ipynb. Las relaciones
    con los parámetros de la elipse son
        α = θ₁,    χ = θ₁ − θ₂.

    El primer polarizador define el haz incidente (α_in, χ_in); el
    segundo polarizador es el que se rota rígidamente de 0° a 360°
    durante el experimento, con (α_P, χ_P) en su posición inicial.
    Por defecto los cuatro ángulos son 0° → ley de Malus tradicional.
    """

    def __init__(self, parent=None,
                 t1_p1: float = 0.0, t2_p1: float = 0.0,
                 t1_p2: float = 0.0, t2_p2: float = 0.0):
        super().__init__(parent)
        self.setWindowTitle("Polarizadores elípticos — Malus generalizada")
        self.setMinimumWidth(460)
        self.setStyleSheet(STYLE_GLOBAL)

        lay = QVBoxLayout(self)
        lay.setSpacing(8)

        info = QLabel(
            "Cada polarizador elíptico se construye como\n"
            "    P(θ₁, θ₂) = Q(θ₁+90°) · PL(θ₂) · Q(θ₁)\n"
            "donde θ₁ es el eje rápido de los cuartos de onda y θ₂ es\n"
            "el eje del polarizador lineal. Relación con la elipse:\n"
            "    α = θ₁ ,   χ = θ₁ − θ₂.\n"
            "Por defecto (todo a 0°) se reproduce la ley de Malus "
            "tradicional cos²(θ)."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color:{COL_TXT_DIM}; padding: 4px;")
        lay.addWidget(info)

        # 1.er polarizador → haz incidente
        gb1 = QGroupBox("1.er polarizador elíptico — define el haz incidente")
        f1 = QFormLayout(gb1); f1.setSpacing(4)
        self.spn_t1_p1 = self._mk_spin(t1_p1)
        self.spn_t2_p1 = self._mk_spin(t2_p1)
        f1.addRow("θ₁  cuartos de onda", self.spn_t1_p1)
        f1.addRow("θ₂  polarizador lineal", self.spn_t2_p1)
        lay.addWidget(gb1)

        # 2.º polarizador → analizador rotado
        gb2 = QGroupBox("2.º polarizador elíptico — se rota rígidamente 0°→360°")
        f2 = QFormLayout(gb2); f2.setSpacing(4)
        self.spn_t1_p2 = self._mk_spin(t1_p2)
        self.spn_t2_p2 = self._mk_spin(t2_p2)
        f2.addRow("θ₁  cuartos de onda", self.spn_t1_p2)
        f2.addRow("θ₂  polarizador lineal", self.spn_t2_p2)
        lay.addWidget(gb2)

        # Resumen calculado en vivo
        self.lbl_resumen = QLabel("")
        self.lbl_resumen.setStyleSheet(
            f"background:{COL_BG2}; color:{COL_AZUL};"
            f"border:1px solid {COL_BORDE}; padding:6px;"
            "font-family:Menlo,Consolas,monospace; font-weight:bold;")
        lay.addWidget(self.lbl_resumen)
        for s in (self.spn_t1_p1, self.spn_t2_p1, self.spn_t1_p2, self.spn_t2_p2):
            s.valueChanged.connect(self._actualizar_resumen)
        self._actualizar_resumen()

        # Botones
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self._restaurar_defaults)
        lay.addWidget(btns)

    @staticmethod
    def _mk_spin(value: float) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(-360.0, 360.0)
        s.setDecimals(2)
        s.setSingleStep(1.0)
        s.setSuffix(" °")
        s.setValue(value)
        return s

    def _restaurar_defaults(self):
        for s in (self.spn_t1_p1, self.spn_t2_p1, self.spn_t1_p2, self.spn_t2_p2):
            s.setValue(0.0)

    def _actualizar_resumen(self, *_):
        t1_p1 = self.spn_t1_p1.value()
        t2_p1 = self.spn_t2_p1.value()
        t1_p2 = self.spn_t1_p2.value()
        t2_p2 = self.spn_t2_p2.value()
        a_in   = t1_p1
        chi_in = t1_p1 - t2_p1
        a_P    = t1_p2
        chi_P  = t1_p2 - t2_p2
        es_tradicional = (
            abs(t1_p1) < 1e-6 and abs(t2_p1) < 1e-6
            and abs(t1_p2) < 1e-6 and abs(t2_p2) < 1e-6
        )
        modo = ("Malus tradicional  cos²(θ)"
                if es_tradicional else "Malus generalizada")
        self.lbl_resumen.setText(
            f" α_in = {a_in:7.2f}°    χ_in = {chi_in:7.2f}°\n"
            f" α_P  = {a_P:7.2f}°    χ_P  = {chi_P:7.2f}°\n"
            f" → {modo}"
        )

    def values(self) -> tuple[float, float, float, float]:
        return (self.spn_t1_p1.value(), self.spn_t2_p1.value(),
                self.spn_t1_p2.value(), self.spn_t2_p2.value())


# ────────────────────────────────────────────────────────────────────────
# Ventana principal
# ────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):

    sig_log         = pyqtSignal(str)
    sig_conexion    = pyqtSignal(bool, bool, str, str)        # (las_ok, spcm_ok, l_msg, s_msg)
    sig_pt_progreso = pyqtSignal(object, object)              # (idx_arr, counts_arr)
    sig_pt_pot      = pyqtSignal(float, float)                # (t, P_uW_calibrada)
    sig_pt_listo    = pyqtSignal(float)                       # angulo
    sig_error       = pyqtSignal(str)
    sig_cal_pot     = pyqtSignal(float, float)                # (factor, P_uW_pic)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(1280, 820)
        self.resize(1380, 880)

        # Drivers
        self._laser: iblg.IBeamDriver | None = None
        self._spcm:  spcmm.DriverSPCM | None = None

        # Estado experimental
        self._iniciado: bool = False           # láser encendido
        self._punto_en_curso: bool = False
        self._punto_arr: np.ndarray | None = None
        self._punto_bin_s: float = 0.0
        self._punto_potencias: list[tuple[float, float]] = []  # (t, P_uW calibrada)
        self._punto_t0: float = 0.0
        self._punto_evt_detener: threading.Event = threading.Event()
        self._puntos: list[dict] = []

        # Calibración de potencia: el PIC interno del iBeam Smart suele estar
        # desfasado respecto a la potencia óptica realmente emitida (típicamente
        # un factor ×2). Se calibra automáticamente tras encender el láser:
        #     factor = setpoint_uW / <PIC_uW>
        # y se aplica a TODO valor mostrado / almacenado.
        self._factor_pot: float = 1.0
        self._calibrando_pot: bool = False

        # Polarizadores elípticos (Malus generalizada).
        # Defaults = 0 → ley de Malus tradicional cos²(θ).
        #   Polarizador 1 → prepara el haz incidente (α_in, χ_in)
        #   Polarizador 2 → analizador rotado rígidamente 0°→360°
        # con (α_P, χ_P) en su posición inicial.
        self._theta1_p1: float = 0.0
        self._theta2_p1: float = 0.0
        self._theta1_p2: float = 0.0
        self._theta2_p2: float = 0.0

        self._construir_ui()
        self._conectar_senales()

        # Dibujar predicción inicial (Malus tradicional por defecto)
        self._refrescar_malus()

        # Auto-detectar al arrancar
        QTimer.singleShot(200, self._conectar_dispositivos)

    # ─── UI ──────────────────────────────────────────────────────────────
    def _construir_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        ext = QVBoxLayout(central); ext.setContentsMargins(6, 6, 6, 6); ext.setSpacing(6)

        # ── Banner superior con estado de ambos dispositivos + desconectar ──
        fila_top = QHBoxLayout(); fila_top.setSpacing(6)
        self.banner = QLabel("Buscando dispositivos …")
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.setStyleSheet(
            f"background:{COL_BG2};color:{COL_LILA};"
            f"border:1px solid {COL_BORDE};border-radius:3px;"
            "padding:6px;font-weight:bold;font-size:12px;letter-spacing:1px;")
        fila_top.addWidget(self.banner, 1)

        self.btn_desconectar = QPushButton("⏏  Desconectar")
        self.btn_desconectar.setStyleSheet(
            f"background-color:{COL_ROJO};color:#1e1e2e;font-weight:bold;"
            "padding:6px 14px;min-width:120px;border-radius:3px;")
        self.btn_desconectar.clicked.connect(self._desconectar)
        self.btn_desconectar.setEnabled(False)
        fila_top.addWidget(self.btn_desconectar)
        ext.addLayout(fila_top)

        # ── Splitter principal ──
        splitter = QSplitter(Qt.Orientation.Horizontal); splitter.setHandleWidth(4)

        # ─── Panel izquierdo ───
        izq = QWidget(); izq.setMaximumWidth(330); izq.setMinimumWidth(300)
        lay_izq = QVBoxLayout(izq); lay_izq.setSpacing(6); lay_izq.setContentsMargins(2, 2, 2, 2)

        gb_cfg = QGroupBox("Configuración del experimento")
        gc = QFormLayout(gb_cfg); gc.setSpacing(4)
        self.spn_potencia = QDoubleSpinBox()
        # Rango ampliado al máximo del iBeam Smart 488 nm (≈ 100 mW)
        self.spn_potencia.setRange(0.1, 100.0); self.spn_potencia.setDecimals(2)
        self.spn_potencia.setSingleStep(1.0)
        self.spn_potencia.setValue(DEFAULT_POTENCIA_MW); self.spn_potencia.setSuffix(" mW")
        # Parámetros SPCM50A/M idénticos al software de Thorlabs.
        self.spn_bin_ms = QDoubleSpinBox()
        self.spn_bin_ms.setRange(0.001, 60_000.0); self.spn_bin_ms.setDecimals(3)
        self.spn_bin_ms.setValue(DEFAULT_BIN_LEN_MS); self.spn_bin_ms.setSuffix(" ms")
        self.spn_time_between = QDoubleSpinBox()
        self.spn_time_between.setRange(0.000, 60_000.0); self.spn_time_between.setDecimals(3)
        self.spn_time_between.setValue(DEFAULT_TIME_BETWEEN_MS); self.spn_time_between.setSuffix(" ms")
        self.spn_pulse_blind = QDoubleSpinBox()
        self.spn_pulse_blind.setRange(0.0, 1_000.0); self.spn_pulse_blind.setDecimals(3)
        self.spn_pulse_blind.setValue(DEFAULT_PULSE_BLIND_NS); self.spn_pulse_blind.setSuffix(" ns")
        self.spn_bins = QSpinBox()
        self.spn_bins.setRange(1, 10_000_000)
        self.spn_bins.setValue(DEFAULT_BINS_POR_ARRAY)
        self.spn_bins.setGroupSeparatorShown(True)
        self.spn_paso = QDoubleSpinBox()
        self.spn_paso.setRange(1.0, 90.0); self.spn_paso.setDecimals(1)
        self.spn_paso.setValue(DEFAULT_PASO_ANGULO); self.spn_paso.setSuffix(" °")

        gc.addRow("Potencia láser CH1",       self.spn_potencia)
        gc.addRow("Bin Length [ms]",          self.spn_bin_ms)
        gc.addRow("Time between Bins [ms]",   self.spn_time_between)
        gc.addRow("Pulse Blind Time [ns]",    self.spn_pulse_blind)
        gc.addRow("Bins per Array",           self.spn_bins)
        gc.addRow("Paso angular sugerido",    self.spn_paso)

        # Etiqueta informativa: tiempo total estimado del array
        self.lbl_t_total = QLabel("—")
        self.lbl_t_total.setStyleSheet(f"color:{COL_TXT_DIM};font-style:italic;")
        gc.addRow("→ Tiempo estimado",        self.lbl_t_total)
        for w in (self.spn_bin_ms, self.spn_time_between, self.spn_bins):
            w.valueChanged.connect(self._actualizar_t_total)
        self._actualizar_t_total()
        lay_izq.addWidget(gb_cfg)

        # Botones de control
        gb_ctrl = QGroupBox("Control")
        gctrl = QVBoxLayout(gb_ctrl); gctrl.setSpacing(4)
        self.btn_iniciar = QPushButton("▶  Iniciar medición")
        self.btn_iniciar.setStyleSheet(
            "background-color:#3a8c3a;color:white;font-weight:bold;"
            "font-size:13px;padding:8px;")
        self.btn_iniciar.clicked.connect(self._iniciar_medicion)
        gctrl.addWidget(self.btn_iniciar)

        self.btn_tomar = QPushButton("●  Tomar punto")
        self.btn_tomar.setStyleSheet(
            "background-color:#3a5a8c;color:white;font-weight:bold;"
            "font-size:12px;padding:8px;")
        self.btn_tomar.clicked.connect(self._tomar_punto)
        self.btn_tomar.setEnabled(False)
        gctrl.addWidget(self.btn_tomar)

        # Repetir el último ángulo medido (descarta el punto previo y vuelve
        # a medir en el mismo θ — útil cuando una toma sale ruidosa).
        self.btn_repetir = QPushButton("↻  Repetir punto anterior")
        self.btn_repetir.setStyleSheet(
            "background-color:#3a8c8c;color:white;font-weight:bold;"
            "font-size:12px;padding:8px;")
        self.btn_repetir.setToolTip(
            "Descarta el último punto tomado y vuelve a medir en el mismo "
            "ángulo (útil si la lectura quedó ruidosa o si se reposicionó "
            "el polarizador con más precisión).")
        self.btn_repetir.clicked.connect(self._repetir_punto_anterior)
        self.btn_repetir.setEnabled(False)
        gctrl.addWidget(self.btn_repetir)

        self.btn_guardar = QPushButton("💾  Cerrar y guardar")
        self.btn_guardar.setStyleSheet(
            "background-color:#8c5a3a;color:white;font-weight:bold;"
            "font-size:12px;padding:8px;")
        self.btn_guardar.clicked.connect(self._cerrar_y_guardar)
        self.btn_guardar.setEnabled(False)
        gctrl.addWidget(self.btn_guardar)

        # Configurar polarizadores elípticos (Malus generalizada)
        self.btn_polarizadores = QPushButton("⚙  Polarizadores elípticos…")
        self.btn_polarizadores.setStyleSheet(
            "background-color:#5a3a8c;color:white;font-weight:bold;"
            "padding:6px;")
        self.btn_polarizadores.setToolTip(
            "Define θ₁ (cuartos de onda) y θ₂ (polarizador lineal) para los "
            "dos polarizadores elípticos. Por defecto (todo a 0°) la curva "
            "predicha es la ley de Malus tradicional cos²(θ).")
        self.btn_polarizadores.clicked.connect(self._abrir_dialog_polarizadores)
        gctrl.addWidget(self.btn_polarizadores)

        self.lbl_polarizadores = QLabel("Malus tradicional  cos²(θ)")
        self.lbl_polarizadores.setStyleSheet(
            f"background:{COL_BG2};color:{COL_LILA};"
            f"border:1px solid {COL_BORDE};padding:3px 4px;"
            "font-family:Menlo,Consolas,monospace;font-size:10px;")
        self.lbl_polarizadores.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gctrl.addWidget(self.lbl_polarizadores)

        # Reintentar conexión SPCM (cuando el contador no fue detectado)
        self.btn_reintentar_spcm = QPushButton("🔄  Reintentar SPCM")
        self.btn_reintentar_spcm.setStyleSheet(
            "background-color:#3a3a5a;color:#cdd6f4;font-weight:bold;"
            "padding:6px;")
        self.btn_reintentar_spcm.setToolTip(
            "Vuelve a buscar el contador de fotones SPCM50A/M (libera la "
            "interfaz USB cerrando otras instancias antes de pulsar).")
        self.btn_reintentar_spcm.clicked.connect(self._reintentar_spcm)
        gctrl.addWidget(self.btn_reintentar_spcm)

        # Repetir experimento desde cero
        self.btn_reset = QPushButton("🗑  Repetir desde cero")
        self.btn_reset.setStyleSheet(
            "background-color:#5a3a3a;color:#cdd6f4;font-weight:bold;"
            "padding:6px;")
        self.btn_reset.setToolTip(
            "Borra TODOS los puntos tomados, limpia las gráficas y permite "
            "iniciar una nueva medición desde cero (no afecta a los datos "
            "ya guardados en disco).")
        self.btn_reset.clicked.connect(self._repetir_desde_cero)
        gctrl.addWidget(self.btn_reset)
        lay_izq.addWidget(gb_ctrl)

        # Resumen del progreso
        gb_resumen = QGroupBox("Progreso")
        grow = QFormLayout(gb_resumen); grow.setSpacing(2)
        self.lbl_npuntos    = QLabel("0")
        self.lbl_ult_angulo = QLabel("—")
        self.lbl_ult_inten  = QLabel("—")
        self.lbl_max_angulo = QLabel("—")
        for lbl in (self.lbl_npuntos, self.lbl_ult_angulo, self.lbl_ult_inten,
                    self.lbl_max_angulo):
            lbl.setStyleSheet(
                f"background:{COL_BG2};color:{COL_AZUL};"
                f"border:1px solid {COL_BORDE};padding:1px 4px;"
                "font-family:Menlo,Consolas,monospace;font-weight:bold;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grow.addRow("Puntos tomados",        self.lbl_npuntos)
        grow.addRow("Último ángulo",          self.lbl_ult_angulo)
        grow.addRow("Última I_norm",          self.lbl_ult_inten)
        grow.addRow("Ángulo del máximo (θ₀)", self.lbl_max_angulo)
        lay_izq.addWidget(gb_resumen)

        # Log
        gb_log = QGroupBox("Log")
        gl = QVBoxLayout(gb_log); gl.setContentsMargins(4, 4, 4, 4)
        self.log_widget = QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumBlockCount(800)
        self.log_widget.setStyleSheet(
            "background:#11111b;color:#a6e3a1;"
            "font-family:Menlo,Consolas,monospace;font-size:10px;"
            f"border:1px solid {COL_BORDE};")
        gl.addWidget(self.log_widget)
        lay_izq.addWidget(gb_log, 1)

        splitter.addWidget(izq)

        # ─── Panel derecho con 3 gráficas ───
        der = QWidget(); lay_der = QVBoxLayout(der); lay_der.setSpacing(6); lay_der.setContentsMargins(2, 2, 2, 2)

        # En vivo: P(t) y CPS(t) lado a lado
        gb_live = QGroupBox("Medición en vivo (durante la captura del punto actual)")
        gv = QHBoxLayout(gb_live); gv.setSpacing(6)

        # Sub-panel potencia (lectura calibrada al setpoint del CH1)
        sub_p = QWidget(); slp = QVBoxLayout(sub_p); slp.setContentsMargins(0, 0, 0, 0)
        slp.addWidget(QLabel("Potencia del láser P(t)  ·  PIC × factor de calibración"))
        fig_p, self.ax_pot_live = _make_fig(5.0, 2.5)
        self.ax_pot_live.set_xlabel("t [s]")
        self.ax_pot_live.set_ylabel("P [µW]  (calibrada)")
        self.canvas_pot = FigureCanvas(fig_p)
        self.line_pot_live, = self.ax_pot_live.plot([], [], color=COL_AMBAR, marker=".",
                                                    ms=4, lw=1.2, label="P_láser")
        slp.addWidget(self.canvas_pot)
        gv.addWidget(sub_p, 1)

        # Sub-panel conteo de fotones — réplica fiel del software SPCM50A/M
        sub_c = QWidget(); slc = QVBoxLayout(sub_c); slc.setContentsMargins(0, 0, 0, 0)
        slc.addWidget(QLabel("Conteo de fotones — Counts per Bin"))
        fig_c, self.ax_cps_live = _make_fig(5.0, 2.5)
        self.ax_cps_live.set_xlabel("Bin Number")
        self.ax_cps_live.set_ylabel("Counts per Bin")
        self.canvas_cps = FigureCanvas(fig_c)
        self.line_cps_live, = self.ax_cps_live.plot([], [], color=COL_VERDE, lw=1.0,
                                                    label="Counts per Bin")
        slc.addWidget(self.canvas_cps)
        gv.addWidget(sub_c, 1)

        lay_der.addWidget(gb_live, 1)

        # Curva Malus acumulada
        gb_malus = QGroupBox("Ley de Malus — intensidad normalizada vs ángulo")
        gm = QVBoxLayout(gb_malus); gm.setContentsMargins(4, 6, 4, 4)
        fig_m, self.ax_malus = _make_fig(10.0, 4.0)
        self.ax_malus.set_xlabel("Ángulo del polarizador θ [°]")
        self.ax_malus.set_ylabel("I / I_max")
        self.ax_malus.set_xlim(0, 360)
        self.ax_malus.set_ylim(-0.05, 1.15)
        self.canvas_malus = FigureCanvas(fig_m)
        gm.addWidget(self.canvas_malus)
        lay_der.addWidget(gb_malus, 2)

        splitter.addWidget(der)
        splitter.setSizes([320, 1060])
        splitter.setCollapsible(0, False); splitter.setCollapsible(1, False)
        ext.addWidget(splitter, 1)

        # Status bar
        sb = QStatusBar(); self.setStatusBar(sb)
        self.lbl_estado = QLabel("Buscando dispositivos …")
        self.lbl_estado.setStyleSheet("color:#666;")
        sb.addWidget(self.lbl_estado, 1)
        logo = QLabel("MALUS · PHOTON COUNT")
        logo.setStyleSheet(ESTILO_LOGO)
        sb.addPermanentWidget(logo)

    def _conectar_senales(self):
        self.sig_log.connect(self._log)
        self.sig_conexion.connect(self._on_conexion)
        self.sig_pt_progreso.connect(self._on_pt_progreso)
        self.sig_pt_pot.connect(self._on_pt_pot)
        self.sig_pt_listo.connect(self._on_pt_listo)
        self.sig_error.connect(self._on_error)
        self.sig_cal_pot.connect(self._on_cal_pot)

    def _actualizar_t_total(self, *_):
        try:
            t = (self.spn_bin_ms.value() + self.spn_time_between.value()) \
                * self.spn_bins.value() / 1000.0
            self.lbl_t_total.setText(f"{t:.3f} s por punto")
        except Exception:
            self.lbl_t_total.setText("—")

    # ─── Conexión ───────────────────────────────────────────────────────
    def _conectar_laser(self) -> tuple[bool, str]:
        try:
            puerto = iblg.detectar_puerto()
            if puerto is None:
                raise RuntimeError("no se halló iBeam Smart en ningún puerto")
            drv_l = iblg.IBeamDriver()
            drv_l.conectar(puerto)
            niveles = drv_l.leer_niveles()
            if niveles.get(2, 0.0) > 0.0:
                self.sig_log.emit(
                    f"  CH2 = {niveles[2]:.3f} mW de sesión previa → forzando 0")
                drv_l.set_potencia(2, 0.0)
            try: drv_l.apagar()
            except Exception: pass
            self._laser = drv_l
            return True, f"iBeam Smart en {puerto}"
        except Exception as e:
            self._laser = None
            return False, f"láser: {e}"

    def _conectar_spcm(self) -> tuple[bool, str]:
        try:
            info, diag = spcmm.detectar_spcm()
            if info is None:
                raise RuntimeError("no se halló SPCM50A")
            drv_s = spcmm.DriverSPCM()
            drv_s.conectar(info)
            self._spcm = drv_s
            return True, f"SPCM50A S/N {info.get('serial_number','?')}"
        except Exception as e:
            self._spcm = None
            return False, f"SPCM: {e}"

    def _conectar_dispositivos(self):
        self._log("Detectando láser y SPCM…")

        def _t():
            laser_ok, laser_msg = self._conectar_laser()
            spcm_ok,  spcm_msg  = self._conectar_spcm()
            self.sig_conexion.emit(laser_ok, spcm_ok, laser_msg, spcm_msg)

        threading.Thread(target=_t, daemon=True).start()

    def _reintentar_spcm(self):
        """Reintenta SOLO la conexión al contador de fotones."""
        if self._punto_en_curso:
            QMessageBox.warning(
                self, "Medición en curso",
                "Espera a que termine el punto actual antes de reconectar.")
            return
        self._log("Reintentando conexión con SPCM50A/M …")
        # Cerrar conexión previa si quedó colgada
        try:
            if self._spcm is not None:
                self._spcm.desconectar()
        except Exception:
            pass
        self._spcm = None
        self.btn_reintentar_spcm.setEnabled(False)

        def _t():
            spcm_ok, spcm_msg = self._conectar_spcm()
            laser_ok = self._laser is not None and self._laser.conectado()
            laser_msg = "iBeam Smart conectado" if laser_ok else "láser: no conectado"
            self.sig_conexion.emit(laser_ok, spcm_ok, laser_msg, spcm_msg)

        threading.Thread(target=_t, daemon=True).start()

    def _repetir_desde_cero(self):
        """Borra todos los puntos y restablece la GUI a un estado inicial."""
        if self._punto_en_curso:
            QMessageBox.warning(
                self, "Medición en curso",
                "Espera a que termine el punto actual antes de reiniciar.")
            return
        if not self._puntos and not self._iniciado:
            self._log("Nada que reiniciar — no hay puntos ni láser activo.")
            return
        resp = QMessageBox.question(
            self, "Repetir desde cero",
            f"Se borrarán {len(self._puntos)} puntos en memoria y se "
            "apagará el láser.\n\n¿Continuar?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if resp != QMessageBox.StandardButton.Yes:
            return

        # Apagar láser si estaba encendido
        try:
            if self._laser is not None and self._laser.conectado() and self._iniciado:
                self._laser.apagar()
        except Exception:
            pass
        self._iniciado = False
        self._factor_pot = 1.0
        self._puntos.clear()
        self._punto_arr = None
        self._punto_potencias = []

        # Limpiar gráficas
        self.line_pot_live.set_data([], [])
        self.line_cps_live.set_data([], [])
        self.ax_pot_live.set_xlim(0, 1); self.ax_pot_live.set_ylim(0, 1)
        self.ax_cps_live.set_xlim(0, 1); self.ax_cps_live.set_ylim(0, 1)
        self.canvas_pot.draw_idle(); self.canvas_cps.draw_idle()
        # Redibujar la predicción (sin datos) en lugar de dejar la gráfica vacía
        self._refrescar_malus()

        # Restablecer etiquetas de progreso
        self.lbl_npuntos.setText("0")
        self.lbl_ult_angulo.setText("—")
        self.lbl_ult_inten.setText("—")
        self.lbl_max_angulo.setText("—")

        # Botones
        ambos_ok = self._laser is not None and self._spcm is not None
        self.btn_iniciar.setEnabled(ambos_ok)
        self.btn_tomar.setEnabled(False)
        self.btn_repetir.setEnabled(False)
        self.btn_guardar.setEnabled(False)

        # Banner
        if ambos_ok:
            self.banner.setText("●  CONECTADO — listo para iniciar nueva medición")
            self.banner.setStyleSheet(
                f"background:{COL_BG2};color:{COL_VERDE};"
                f"border:2px solid {COL_VERDE};border-radius:3px;"
                "padding:6px;font-weight:bold;font-size:12px;letter-spacing:1px;")

        self._log("✓ Estado reiniciado: 0 puntos, láser apagado, gráficas limpias.")

    def _on_conexion(self, laser_ok: bool, spcm_ok: bool, l_msg: str, s_msg: str):
        if laser_ok and spcm_ok:
            self.banner.setText(f"●  CONECTADO — {l_msg}    ·    {s_msg}")
            self.banner.setStyleSheet(
                f"background:{COL_BG2};color:{COL_VERDE};"
                f"border:2px solid {COL_VERDE};border-radius:3px;"
                "padding:6px;font-weight:bold;font-size:12px;letter-spacing:1px;")
            self.lbl_estado.setText("● Láser y SPCM conectados")
            self.lbl_estado.setStyleSheet(f"color:{COL_VERDE};font-weight:bold;")
            self.btn_desconectar.setEnabled(True)
            self.btn_iniciar.setEnabled(True)
            self._log(f"OK — {l_msg}  |  {s_msg}")
        else:
            estado_l = "✓" if laser_ok else "✗"
            estado_s = "✓" if spcm_ok else "✗"
            self.banner.setText(f"⚠  Conexión incompleta:  Láser {estado_l}  ·  SPCM {estado_s}")
            self.banner.setStyleSheet(
                f"background:{COL_BG2};color:{COL_AMBAR};"
                f"border:2px solid {COL_AMBAR};border-radius:3px;"
                "padding:6px;font-weight:bold;font-size:12px;letter-spacing:1px;")
            self.lbl_estado.setText("⚠ Faltan dispositivos")
            self.lbl_estado.setStyleSheet(f"color:{COL_AMBAR};font-weight:bold;")
            self.btn_desconectar.setEnabled(laser_ok or spcm_ok)
            self.btn_iniciar.setEnabled(False)
            self._log(f"  {l_msg}")
            self._log(f"  {s_msg}")
            self._log("Cierra cualquier instancia abierta de iBeamSmart.app o "
                      "SPCM50AM.app para liberar las interfaces y vuelve a abrir.")

    def _desconectar(self):
        if self._punto_en_curso:
            self._punto_evt_detener.set()
            time.sleep(0.5)
        try:
            if self._laser is not None and self._laser.conectado():
                self._laser.apagar()
                self._laser.desconectar()
        except Exception: pass
        try:
            if self._spcm is not None:
                self._spcm.desconectar()
        except Exception: pass
        self._laser = None; self._spcm = None
        self._iniciado = False
        self.btn_desconectar.setEnabled(False)
        self.btn_iniciar.setEnabled(False)
        self.btn_tomar.setEnabled(False)
        self.btn_repetir.setEnabled(False)
        self.banner.setText("○  DESCONECTADO")
        self.banner.setStyleSheet(
            f"background:{COL_BG2};color:{COL_TXT_DIM};"
            f"border:1px solid {COL_BORDE};border-radius:3px;"
            "padding:6px;font-weight:bold;font-size:12px;")
        self.lbl_estado.setText("○ Desconectado")
        self.lbl_estado.setStyleSheet(f"color:{COL_TXT_DIM};")
        self._log("Desconectado.")

    # ─── Iniciar medición ───────────────────────────────────────────────
    def _iniciar_medicion(self):
        if self._laser is None or self._spcm is None:
            QMessageBox.warning(self, "Sin dispositivos",
                                "Conecta láser y SPCM antes de iniciar.")
            return
        if self._iniciado:
            return
        pot_mW = self.spn_potencia.value()
        try:
            self._laser.set_potencia(2, 0.0)
            self._laser.set_potencia(1, pot_mW)
            self._laser.encender()
        except Exception as e:
            self._on_error(f"al encender láser: {e}")
            return
        self._iniciado = True
        self._factor_pot = 1.0
        self.btn_iniciar.setEnabled(False)
        self.btn_tomar.setEnabled(False)         # se habilita tras calibración
        self._log(f"Láser ON con {pot_mW:.2f} mW en CH1. "
                  f"Estabilizando ~{T_ESTABILIZACION_S:.0f} s y calibrando lectura PIC…")
        self.banner.setText(
            f"●  ESTABILIZANDO — láser ON @ {pot_mW:.2f} mW (configurado)  ·  "
            "calibrando lectura de potencia…")
        self.banner.setStyleSheet(
            f"background:{COL_BG2};color:{COL_AMBAR};"
            f"border:2px solid {COL_AMBAR};border-radius:3px;"
            "padding:6px;font-weight:bold;font-size:12px;letter-spacing:1px;")
        threading.Thread(target=self._calibrar_potencia,
                         args=(pot_mW,), daemon=True).start()

    def _calibrar_potencia(self, setpoint_mW: float):
        """
        El PIC interno del iBeam Smart entrega lecturas en µW que NO siempre
        coinciden con la potencia óptica realmente emitida por la salida
        (típicamente difieren en un factor ×2). Tras la estabilización del
        láser se promedian varias lecturas y se calcula el factor que las
        normaliza al setpoint configurado.
        """
        self._calibrando_pot = True
        try:
            time.sleep(T_ESTABILIZACION_S)
            lecturas = []
            for _ in range(N_MUESTRAS_CAL_POT):
                try:
                    p = self._laser.leer_potencia_uW()
                    if p > 0:
                        lecturas.append(p)
                except Exception:
                    pass
                time.sleep(0.25)
            if not lecturas:
                self.sig_log.emit("⚠ Calibración de potencia: PIC no devolvió "
                                  "lecturas válidas. Factor = 1.0")
                self.sig_cal_pot.emit(1.0, 0.0)
                return
            pic_mean = float(np.mean(lecturas))
            target_uW = setpoint_mW * 1000.0
            factor = target_uW / pic_mean if pic_mean > 0 else 1.0
            self.sig_cal_pot.emit(factor, pic_mean)
        finally:
            self._calibrando_pot = False

    def _on_cal_pot(self, factor: float, pic_uW: float):
        self._factor_pot = factor
        pot_mW = self.spn_potencia.value()
        self._log(
            f"  Calibración PIC → factor = {factor:.4f}  "
            f"(PIC bruto = {pic_uW:.1f} µW, setpoint = {pot_mW*1000:.0f} µW)")
        self.btn_tomar.setEnabled(True)
        self.btn_repetir.setEnabled(len(self._puntos) > 0)
        self.banner.setText(
            f"●  MIDIENDO — láser ON @ {pot_mW:.2f} mW  "
            f"(factor PIC ×{factor:.3f})  ·  Pulsa “Tomar punto” para cada ángulo")
        self.banner.setStyleSheet(
            f"background:{COL_BG2};color:{COL_AZUL};"
            f"border:2px solid {COL_AZUL};border-radius:3px;"
            "padding:6px;font-weight:bold;font-size:12px;letter-spacing:1px;")

    # ─── Tomar punto ─────────────────────────────────────────────────────
    def _sugerir_proximo_angulo(self) -> float:
        if not self._puntos:
            return 0.0
        return min(360.0, self._puntos[-1]["angulo"] + self.spn_paso.value())

    def _tomar_punto(self):
        if not self._iniciado:
            QMessageBox.warning(self, "No iniciado",
                                "Pulsa primero “Iniciar medición”.")
            return
        if self._punto_en_curso:
            return
        sug = self._sugerir_proximo_angulo()
        ang, ok = QInputDialog.getDouble(
            self, "Ángulo del segundo polarizador",
            f"Ingrese θ en grados (sugerido {sug:.1f}°)\n"
            "(σ_θ = ±0.5° por construcción del soporte)",
            value=sug, min=0.0, max=360.0, decimals=1)
        if not ok:
            return
        self._iniciar_punto(ang)

    def _repetir_punto_anterior(self):
        """
        Descarta el último punto guardado y vuelve a medir en el mismo
        ángulo. Útil cuando la última toma quedó ruidosa, hubo deriva del
        láser, o se reposicionó el polarizador con más precisión.
        """
        if not self._iniciado:
            QMessageBox.warning(self, "No iniciado",
                                "Pulsa primero “Iniciar medición”.")
            return
        if self._punto_en_curso:
            return
        if not self._puntos:
            QMessageBox.information(
                self, "Sin puntos previos",
                "Aún no se ha tomado ningún punto que repetir.")
            return

        ult = self._puntos[-1]
        ang_prev = float(ult["angulo"])
        I_prev   = float(ult["I_norm"])

        resp = QMessageBox.question(
            self, "Repetir punto anterior",
            f"Se descartará la última medición:\n\n"
            f"    θ = {ang_prev:.1f}°    I_norm = {I_prev:.4g}\n\n"
            f"¿Volver a medir en θ = {ang_prev:.1f}°?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes)
        if resp != QMessageBox.StandardButton.Yes:
            return

        self._puntos.pop()
        self._refrescar_resumen()
        self._refrescar_malus()
        self._log(f"Repitiendo punto en θ = {ang_prev:.1f}° "
                  f"(descartado I_norm previo = {I_prev:.4g})")
        self._iniciar_punto(ang_prev)

    def _iniciar_punto(self, angulo: float):
        bin_ms       = self.spn_bin_ms.value()
        time_btw_ms  = self.spn_time_between.value()
        pulse_blind  = self.spn_pulse_blind.value()
        n_bins       = int(self.spn_bins.value())
        t_total      = (bin_ms + time_btw_ms) * n_bins / 1000.0

        self._punto_en_curso     = True
        self._punto_evt_detener  = threading.Event()
        self._punto_potencias    = []
        self._punto_t0           = time.time()
        self._punto_angulo_curso = angulo
        self._punto_n_bins       = n_bins

        # Limpiar gráficas en vivo (CPS lo presentamos como en el software de
        # Thorlabs: Counts per Bin vs Bin Number)
        self.line_pot_live.set_data([], [])
        self.line_cps_live.set_data([], [])
        self.ax_pot_live.set_xlim(0, max(t_total, 1.0)); self.ax_pot_live.set_ylim(0, 1)
        self.ax_cps_live.set_xlim(0, n_bins); self.ax_cps_live.set_ylim(0, 1)
        self.canvas_pot.draw_idle(); self.canvas_cps.draw_idle()

        self.btn_tomar.setEnabled(False)
        self.btn_repetir.setEnabled(False)
        self._log(
            f"θ={angulo:6.1f}°  →  {n_bins} bins × {bin_ms:.3f} ms "
            f"(gap {time_btw_ms:.3f} ms, blind {pulse_blind:.3f} ns) "
            f"≈ {t_total:.2f} s")

        # Hilo SPCM (acumula bins, devuelve array final)
        def _t_spcm():
            try:
                bin_s = bin_ms / 1000.0
                arr = self._spcm.leer_array(
                    bin_length_ms=bin_ms,
                    n_bins=n_bins,
                    time_between_ms=time_btw_ms,
                    pulse_blind_ns=pulse_blind,
                    detener_event=self._punto_evt_detener,
                    callback_progreso=lambda frac, partial:
                        self.sig_pt_progreso.emit(
                            np.arange(1, len(partial) + 1),
                            partial.astype(float),
                        ),
                )
                self._punto_arr   = arr
                self._punto_bin_s = bin_s
                # Detener hilo de potencia
                self._punto_evt_detener.set()
                self.sig_pt_listo.emit(angulo)
            except Exception as e:
                self._punto_evt_detener.set()
                self.sig_error.emit(f"SPCM: {e}")

        # Hilo de muestreo de potencia del láser (~2.5 Hz). Aplicamos factor
        # de calibración para mostrar la potencia óptica realmente emitida,
        # no la lectura cruda del fotodiodo PIC.
        def _t_pot():
            while not self._punto_evt_detener.is_set():
                try:
                    p_raw = self._laser.leer_potencia_uW()
                    p = p_raw * self._factor_pot
                    t = time.time() - self._punto_t0
                    self._punto_potencias.append((t, p))
                    self.sig_pt_pot.emit(t, p)
                except Exception:
                    pass
                time.sleep(PERIODO_POT_S)

        threading.Thread(target=_t_spcm, daemon=True).start()
        threading.Thread(target=_t_pot,  daemon=True).start()

    def _on_pt_progreso(self, idx_arr, counts_arr):
        if idx_arr is None or len(idx_arr) == 0:
            return
        self.line_cps_live.set_data(idx_arr, counts_arr)
        n_total = getattr(self, "_punto_n_bins", int(idx_arr[-1]))
        self.ax_cps_live.set_xlim(0, max(n_total, 1))
        ymax = float(counts_arr.max()) if len(counts_arr) else 1.0
        self.ax_cps_live.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)
        self.canvas_cps.draw_idle()

    def _on_pt_pot(self, t: float, P_uW: float):
        xs, ys = self.line_pot_live.get_data()
        xs = list(xs); ys = list(ys)
        xs.append(t); ys.append(P_uW)
        self.line_pot_live.set_data(xs, ys)
        if xs:
            self.ax_pot_live.set_xlim(0, max(xs[-1], 1e-3))
        if ys:
            ymin, ymax = min(ys), max(ys)
            margen = max(0.02 * abs(ymax), 1.0)
            self.ax_pot_live.set_ylim(ymin - margen, ymax + margen)
        self.canvas_pot.draw_idle()

    def _on_pt_listo(self, angulo: float):
        if self._punto_arr is None or len(self._punto_arr) == 0:
            self._on_error("punto sin bins recibidos del SPCM")
            return

        bin_s = self._punto_bin_s
        cps_bins = self._punto_arr.astype(float) / bin_s
        cps_mean = float(cps_bins.mean())
        cps_sem  = float(cps_bins.std(ddof=1) / np.sqrt(len(cps_bins))) \
                   if len(cps_bins) > 1 else 0.0

        pot_arr = np.array(self._punto_potencias) if self._punto_potencias \
                  else np.zeros((0, 2))
        if len(pot_arr) > 1:
            pot_mean = float(pot_arr[:, 1].mean())
            pot_sem  = float(pot_arr[:, 1].std(ddof=1) / np.sqrt(len(pot_arr)))
        elif len(pot_arr) == 1:
            pot_mean = float(pot_arr[0, 1])
            pot_sem  = 0.005 * pot_mean        # 0.5 % típico iBeam Smart
        else:
            pot_mean = 0.0; pot_sem = 0.0

        if pot_mean > 0 and cps_mean > 0:
            I_norm = cps_mean / pot_mean
            rel = np.sqrt((cps_sem / cps_mean) ** 2 +
                          (pot_sem / pot_mean) ** 2)
            sigma_I = I_norm * rel
        else:
            I_norm = 0.0; sigma_I = 0.0

        punto = {
            "angulo":        angulo,
            "sigma_angulo":  SIGMA_ANGULO_DEG,
            "cps_mean":      cps_mean,
            "cps_sem":       cps_sem,
            "pot_mean_uW":   pot_mean,
            "pot_sem_uW":    pot_sem,
            "I_norm":        I_norm,
            "sigma_I_norm":  sigma_I,
            "bins":          self._punto_arr.copy(),
            "potencias":     pot_arr.copy(),
            "bin_length_s":  bin_s,
            "t_iso":         datetime.now().isoformat(timespec="seconds"),
        }
        self._puntos.append(punto)

        self._log(
            f"  → CPS = {cps_mean:>9.1f} ± {cps_sem:>5.1f}   "
            f"P = {pot_mean:>7.2f} ± {pot_sem:>5.2f} µW   "
            f"I_norm = {I_norm:.4g} ± {sigma_I:.2g}")

        self._refrescar_resumen()
        self._refrescar_malus()

        self._punto_en_curso = False
        self._punto_arr = None
        self._punto_potencias = []
        self.btn_tomar.setEnabled(True)
        self.btn_repetir.setEnabled(len(self._puntos) > 0)

        if angulo >= 360.0 - 1e-6:
            self.btn_guardar.setEnabled(True)
            QMessageBox.information(
                self, "Barrido completo",
                f"Se alcanzó θ = {angulo:.1f}°.\n"
                "Pulsa “Cerrar y guardar” para exportar los datos.")

    # ─── Resumen / Malus plot ──────────────────────────────────────────
    def _refrescar_resumen(self):
        n = len(self._puntos)
        self.lbl_npuntos.setText(f"{n}")
        if n:
            ult = self._puntos[-1]
            self.lbl_ult_angulo.setText(f"{ult['angulo']:.1f}°")
            self.lbl_ult_inten.setText(f"{ult['I_norm']:.4g}")
            Is = np.array([p["I_norm"] for p in self._puntos])
            ang0 = self._puntos[int(Is.argmax())]["angulo"]
            self.lbl_max_angulo.setText(f"{ang0:.1f}°")

    def _es_malus_tradicional(self) -> bool:
        return (abs(self._theta1_p1) < 1e-6 and abs(self._theta2_p1) < 1e-6
                and abs(self._theta1_p2) < 1e-6 and abs(self._theta2_p2) < 1e-6)

    def _actualizar_lbl_polarizadores(self):
        if self._es_malus_tradicional():
            self.lbl_polarizadores.setText("Malus tradicional  cos²(θ)")
        else:
            chi_in = self._theta1_p1 - self._theta2_p1
            chi_P  = self._theta1_p2 - self._theta2_p2
            self.lbl_polarizadores.setText(
                f"χ_in = {chi_in:.1f}°   χ_P = {chi_P:.1f}°  (generalizada)")

    def _abrir_dialog_polarizadores(self):
        dlg = PolarizadoresDialog(
            self,
            self._theta1_p1, self._theta2_p1,
            self._theta1_p2, self._theta2_p2)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        (self._theta1_p1, self._theta2_p1,
         self._theta1_p2, self._theta2_p2) = dlg.values()
        self._actualizar_lbl_polarizadores()
        self._refrescar_malus()
        chi_in = self._theta1_p1 - self._theta2_p1
        chi_P  = self._theta1_p2 - self._theta2_p2
        self._log(
            f"Polarizadores → P1(θ₁={self._theta1_p1:.2f}°, "
            f"θ₂={self._theta2_p1:.2f}°)  ·  "
            f"P2(θ₁={self._theta1_p2:.2f}°, θ₂={self._theta2_p2:.2f}°)   "
            f"⇒  α_in={self._theta1_p1:.2f}°, χ_in={chi_in:.2f}°,  "
            f"α_P={self._theta1_p2:.2f}°, χ_P={chi_P:.2f}°  "
            + ("(Malus tradicional)" if self._es_malus_tradicional()
               else "(Malus generalizada)"))

    def _refrescar_malus(self):
        # Datos experimentales (si los hay)
        if self._puntos:
            angs = np.array([p["angulo"]       for p in self._puntos])
            Is   = np.array([p["I_norm"]        for p in self._puntos])
            sIs  = np.array([p["sigma_I_norm"] for p in self._puntos])
            I_max = float(Is.max()) if Is.max() > 0 else 1.0
            Is_n   = Is  / I_max
            sIs_n  = sIs / I_max
            idx_max = int(Is.argmax())
            theta_0 = float(angs[idx_max])
        else:
            angs = Is = sIs = Is_n = sIs_n = None
            # Sin datos: el pico natural del modelo está en α_in − α_P (mod 180°)
            theta_0 = (self._theta1_p1 - self._theta1_p2) % 180.0

        # Predicción de la ley de Malus generalizada:
        #   I(θ) = ½ [1 + cos2χ_in cos2χ_P cos2(θ_0 − θ) + sin2χ_in sin2χ_P]
        # El pico se alinea con θ_0 (máximo de los datos o predicción natural)
        # para que el desfase del montaje no oculte la forma. La forma de la
        # curva (amplitud y "piso") sí depende de χ_in, χ_P.
        chi_in_rad = np.deg2rad(self._theta1_p1 - self._theta2_p1)
        chi_P_rad  = np.deg2rad(self._theta1_p2 - self._theta2_p2)
        theta_grid = np.linspace(0, 360, 721)
        phi = np.deg2rad(theta_0 - theta_grid)
        I_pred = 0.5 * (
            1.0
            + np.cos(2*chi_in_rad) * np.cos(2*chi_P_rad) * np.cos(2*phi)
            + np.sin(2*chi_in_rad) * np.sin(2*chi_P_rad)
        )
        I_pred_max = float(I_pred.max()) if I_pred.max() > 0 else 1.0
        I_pred_n = I_pred / I_pred_max
        I_pred_min_n = float(I_pred.min()) / I_pred_max

        self.ax_malus.cla()
        self.ax_malus.set_facecolor(COL_PLOT)
        for sp in self.ax_malus.spines.values():
            sp.set_color(COL_BORDE)
        self.ax_malus.tick_params(colors=COL_TXT, labelsize=9)
        self.ax_malus.grid(True, color=COL_GRID, linewidth=0.5, alpha=0.6)
        self.ax_malus.set_xlabel("Ángulo del polarizador θ [°]", color=COL_TXT)
        self.ax_malus.set_ylabel("I / I_max",                     color=COL_TXT)

        if self._es_malus_tradicional():
            label_pred = f"cos²(θ − {theta_0:.1f}°)  (Malus tradicional)"
        else:
            chi_in_deg = self._theta1_p1 - self._theta2_p1
            chi_P_deg  = self._theta1_p2 - self._theta2_p2
            label_pred = (
                f"Malus generalizada  ·  χ_in={chi_in_deg:.1f}°, "
                f"χ_P={chi_P_deg:.1f}°  ·  I_min/I_max={I_pred_min_n:.3f}"
            )

        self.ax_malus.plot(theta_grid, I_pred_n, color=COL_LILA, ls=":", lw=1.6,
                           label=label_pred)

        if angs is not None:
            self.ax_malus.errorbar(
                angs, Is_n, xerr=SIGMA_ANGULO_DEG, yerr=sIs_n,
                fmt="o", color=COL_AZUL, ecolor=COL_AMBAR,
                markersize=5, lw=1.0, capsize=3, capthick=1,
                label=f"datos (N = {len(angs)})")

        self.ax_malus.legend(facecolor=COL_BG2, edgecolor=COL_BORDE,
                             labelcolor=COL_TXT, loc="upper right",
                             fontsize=9)
        self.ax_malus.set_xlim(0, 360)
        self.ax_malus.set_ylim(-0.08, 1.18)
        self.canvas_malus.draw_idle()

    # ─── Cerrar y guardar ──────────────────────────────────────────────
    def _cerrar_y_guardar(self):
        if not self._puntos:
            QMessageBox.information(self, "Sin datos",
                                    "Aún no hay puntos para guardar.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ofrecer al usuario elegir carpeta padre — por defecto Python/PE/Malus
        carpeta_padre = QFileDialog.getExistingDirectory(
            self, "Carpeta donde guardar la medición",
            str(Path(__file__).resolve().parent))
        if not carpeta_padre:
            return
        base = Path(carpeta_padre) / f"malus_conteo_{ts}"
        base.mkdir(parents=True, exist_ok=True)

        try:
            self._exportar_datos(base)
            self._exportar_graficas(base)
        except Exception as e:
            self._on_error(f"al guardar: {e}")
            return

        # Apagar láser
        try:
            if self._laser and self._laser.conectado():
                self._laser.apagar()
        except Exception: pass

        self._log(f"Guardado en: {base.resolve()}")
        QMessageBox.information(self, "Guardado",
                                f"Datos y gráficas guardados en:\n{base.resolve()}")

    def _exportar_datos(self, base: Path):
        bin_ms = self.spn_bin_ms.value()
        t_btw  = self.spn_time_between.value()
        n_bins = int(self.spn_bins.value())
        t_int  = (bin_ms + t_btw) * n_bins / 1000.0
        # Resumen tabulado
        with open(base / "datos.txt", "w", encoding="utf-8") as f:
            f.write("# Ley de Malus por conteo de fotones\n")
            f.write(f"# fecha: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"# potencia_CH1_mW:    {self.spn_potencia.value():.3f}\n")
            f.write(f"# factor_calib_PIC:   {self._factor_pot:.5f}\n")
            f.write(f"# bin_length_ms:      {bin_ms:.3f}\n")
            f.write(f"# time_between_ms:    {t_btw:.3f}\n")
            f.write(f"# pulse_blind_ns:     {self.spn_pulse_blind.value():.3f}\n")
            f.write(f"# bins_per_array:     {n_bins}\n")
            f.write(f"# t_integracion_s:    {t_int:.3f}\n")
            f.write(f"# sigma_angulo_deg:   {SIGMA_ANGULO_DEG}\n")
            f.write("# columnas: angulo[deg]\tsigma_angulo[deg]\t"
                    "CPS\tsigma_CPS\tP[uW]\tsigma_P[uW]\t"
                    "I_norm[CPS/uW]\tsigma_I_norm\n")
            for p in self._puntos:
                f.write(
                    f"{p['angulo']:.2f}\t{p['sigma_angulo']:.3f}\t"
                    f"{p['cps_mean']:.4f}\t{p['cps_sem']:.4f}\t"
                    f"{p['pot_mean_uW']:.5f}\t{p['pot_sem_uW']:.5f}\t"
                    f"{p['I_norm']:.6e}\t{p['sigma_I_norm']:.6e}\n")

        # Detalle por punto: bins y ruido del láser
        sub_b = base / "bins";  sub_b.mkdir(exist_ok=True)
        sub_r = base / "ruido"; sub_r.mkdir(exist_ok=True)
        for p in self._puntos:
            tag = f"theta_{p['angulo']:06.2f}".replace(".", "p")
            with open(sub_b / f"{tag}.txt", "w", encoding="utf-8") as f:
                f.write(f"# bins en theta = {p['angulo']:.2f} deg\n")
                f.write(f"# bin_length_s = {p['bin_length_s']:.6f}\n")
                f.write("# bin\tcounts\tCPS\n")
                for j, c in enumerate(p["bins"], start=1):
                    f.write(f"{j}\t{int(c)}\t{c / p['bin_length_s']:.3f}\n")
            with open(sub_r / f"{tag}.txt", "w", encoding="utf-8") as f:
                f.write(f"# potencia laser durante la medida en theta = {p['angulo']:.2f} deg\n")
                f.write("# t[s]\tP[uW]\n")
                for t, P in p["potencias"]:
                    f.write(f"{t:.4f}\t{P:.5f}\n")

    def _exportar_graficas(self, base: Path):
        # 1) Curva de Malus principal
        self.canvas_malus.figure.savefig(
            base / "malus_curva.png", dpi=150, facecolor=COL_BG)

        # 2) Ruido del láser global (todos los puntos)
        fig, ax = _make_fig(10.0, 3.5)
        ax.set_xlabel("ángulo medido θ [°]")
        ax.set_ylabel("P láser [µW]")
        ax.set_title("Ruido y deriva del láser durante el barrido",
                     color=COL_TXT, fontsize=10)
        for p in self._puntos:
            if len(p["potencias"]):
                ts = p["potencias"][:, 0]
                Ps = p["potencias"][:, 1]
                # Mapear t local a un offset alrededor del ángulo
                xs = p["angulo"] + (ts - ts.mean()) * 0.3
                ax.plot(xs, Ps, ".", color=COL_AMBAR, ms=2.5, alpha=0.7)
        ax.set_xlim(0, 360)
        fig.savefig(base / "ruido_laser.png", dpi=150, facecolor=COL_BG)

        # 3) Conteo crudo por punto
        fig, ax = _make_fig(10.0, 3.5)
        ax.set_xlabel("ángulo medido θ [°]")
        ax.set_ylabel("CPS")
        ax.set_yscale("log")
        ax.set_title("Tasa de conteo por bin (todos los puntos)",
                     color=COL_TXT, fontsize=10)
        for p in self._puntos:
            cps = p["bins"].astype(float) / p["bin_length_s"]
            xs = p["angulo"] + np.linspace(-1.5, 1.5, len(cps))
            ax.plot(xs, np.maximum(cps, 1.0), ".", color=COL_VERDE,
                    ms=1.5, alpha=0.5)
        ax.set_xlim(0, 360)
        fig.savefig(base / "conteo_fotones.png", dpi=150, facecolor=COL_BG)

    # ─── Utilidades ─────────────────────────────────────────────────────
    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_widget.appendPlainText(f"[{ts}] {msg}")

    def _on_error(self, msg: str):
        self._log(f"ERROR: {msg}")
        QMessageBox.critical(self, "Error", msg)
        self._punto_en_curso = False
        self.btn_tomar.setEnabled(self._iniciado)
        self.btn_repetir.setEnabled(self._iniciado and len(self._puntos) > 0)

    # ─── Cierre seguro ─────────────────────────────────────────────────
    def closeEvent(self, event):
        self._punto_evt_detener.set()
        try:
            if self._laser is not None and self._laser.conectado():
                self._laser.apagar()
                self._laser.desconectar()
        except Exception: pass
        try:
            if self._spcm is not None:
                self._spcm.desconectar()
        except Exception: pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE_GLOBAL)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

"""
Experimento de Ley de Malus por conteo de fotones.

GUI PyQt6 que controla SIMULTÁNEAMENTE el láser TOPTICA iBeam Smart
y el contador Thorlabs SPCM50A/M para realizar un barrido completo de
0 a 360° del segundo polarizador.

Antes del barrido se calibra el RUIDO DE FONDO (medición de "blanco",
práctica estándar de metrología en conteo de fotones — Currie 1968,
ISO 11843): con el láser ENCENDIDO y dos polarizadores lineales cruzados
a 90° (configuración de extinción) se mide la tasa de fondo

    R_b = cuentas oscuras del SPCM + luz ambiente + fuga por la
          extinción finita de los polarizadores  [CPS]

durante un tiempo ≥ 3× el de un punto (blanco "bien conocido": reduce
σ_Rb como 1/√T_b). Cada punto del barrido se corrige por sustracción
del blanco y se reportan el nivel crítico L_C y el límite de detección
L_D de Currie para juzgar los puntos cercanos al mínimo.

Para cada ángulo:
  • Se mide N bins de conteo de fotones durante T_integración segundos.
  • En paralelo se muestrea la potencia óptica del láser para
    compensar deriva y ruido (~1 Hz).
  • Se calcula la intensidad normalizada NETA
        I_norm = <(CPS − R_b) / P>.
  • Se propagan las incertidumbres:
        σ_θ      = ±2°      (cada goniómetro; el montaje usa 6:
                            dos cuartos de onda + un polarizador lineal
                            por cada uno de los dos polarizadores elípticos)
        σ_CPS    = std(CPS_bins) / sqrt(N_bins)
        σ_P      = std(P_samples) / sqrt(N_P)
        σ_I/I    = sqrt((σ_CPS/CPS)^2 + (σ_P/P)^2)
        σ_Rb     = incertidumbre del blanco: entra en σ_I como componente
                   SISTEMÁTICA correlacionada (σ_Rb·<1/P>, sin dividir
                   por √N, pues R_b se resta por igual a todos los bins)
  • Cada punto se dibuja con barra de error en x (grados, ±2° del
    goniómetro) y en y (intensidad normalizada, propagación de conteo
    y potencia). La curva teórica lleva además una banda ±1σ propagada
    desde los goniómetros de configuración (χ_in, χ_P y el pico θ₀).

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
    QScrollArea, QSizePolicy, QSpinBox, QSplitter, QStatusBar, QVBoxLayout,
    QWidget,
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
# Incertidumbre angular: el montaje usa 6 goniómetros (dos cuartos de onda +
# un polarizador lineal por cada uno de los dos polarizadores elípticos), cada
# uno con una resolución de lectura de ±2°. Esta σ se usa tanto para la barra
# de error en x (grados) de cada dato como para propagar la banda de
# incertidumbre de la curva teórica (ver _malus_pred_con_banda).
SIGMA_GONIO_DEG  = 2.0             # incertidumbre de cada goniómetro [°]
SIGMA_ANGULO_DEG = SIGMA_GONIO_DEG # alias retro-compatible (barra de error x)
DEFAULT_POTENCIA_MW   = 5.0
# Réplica fiel de los defaults del software Thorlabs SPCM50A/M:
#   Bin Length 1.000 ms · Time between Bins 0.001 ms · Pulse Blind 0 ns ·
#   Bins per Array 10 000 · Operating Mode "Free Running Timed Counter".
DEFAULT_BIN_LEN_MS    = 1.0
DEFAULT_TIME_BETWEEN_MS = 0.001
DEFAULT_PULSE_BLIND_NS  = 0.0
DEFAULT_BINS_POR_ARRAY = 10000
DEFAULT_PASO_ANGULO   = 10.0       # sólo sugerencia para el dialog
# Calibración de fondo: el blanco se mide por defecto con 3× los bins de un
# punto normal, de modo que σ_Rb ≪ σ del punto ("blanco bien conocido" en el
# sentido de Currie 1968) y su contribución sistemática quede subdominante.
FACTOR_BINS_FONDO_DEF = 3
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


def estilo_boton(bg: str, fg: str = "#ffffff", fs: int = 12) -> str:
    """Hoja de estilo consistente para los botones de control.

    Usa el mismo padding, radio y peso en todos los botones para que el
    texto (con emoji incluido) quede centrado y holgado dentro del recuadro,
    e incorpora estados :hover/:pressed/:disabled coherentes con el tema.
    """
    return (
        f"QPushButton{{background-color:{bg};color:{fg};font-weight:bold;"
        f"font-size:{fs}px;padding:6px 12px;border:none;border-radius:5px;"
        f"text-align:center;}}"
        f"QPushButton:hover{{background-color:{bg};border:1px solid {COL_TXT};}}"
        f"QPushButton:pressed{{background-color:#585b70;}}"
        f"QPushButton:disabled{{background-color:#313244;color:#6c7086;}}"
    )


def _malus_pred_con_banda(theta_grid, a, b, c, d, sigma_gonio,
                          theta_0=None, incluir_barrido=False):
    """Fórmula de Malus generalizada y su incertidumbre 1σ propagada.

    Intensidad (sin normalizar aún) según

        I(θ) = ½[1 + cos2χ_in cos2χ_P cos2φ + sin2χ_in sin2χ_P],
        χ_in = a − b,   χ_P = c − d,   φ = θ_0 − θ,

    donde (a, b) = (θ₁, θ₂) del 1.er polarizador y (c, d) las del 2.º.
    El pico teórico estricto es θ_0 = a − c (α_in − α_P); si se pasa
    ``theta_0`` se fija ese pico (modo "alinear a los datos").

    La incertidumbre σ_I es la propagación EXACTA (primer orden) de la
    fórmula por todas sus variables angulares, cada una con desviación
    ``sigma_gonio`` grados, vía derivadas numéricas:

        σ_I² = Σ_x (∂I/∂x)² σ_gonio².

    - ``incluir_barrido=False`` (banda de la curva teórica): se propaga
      x ∈ {a, b, c, d}. El ángulo θ es la variable independiente del eje,
      no se propaga.
    - ``incluir_barrido=True`` (barra de error en y de cada dato): se
      propaga además el ángulo de barrido θ, x ∈ {a, b, c, d, θ}, porque
      la lectura del goniómetro de barrido también tiene σ = sigma_gonio.

    En ambos casos se captura la correlación de ``a`` y ``c`` (que aparecen
    a la vez en χ_in/χ_P y en el pico θ_0 cuando éste es teórico).
    """
    theta_grid = np.asarray(theta_grid, float)

    def I_de(a, b, c, d, theta):
        chi_in = np.deg2rad(a - b)
        chi_P  = np.deg2rad(c - d)
        t0  = (a - c) if theta_0 is None else theta_0
        phi = np.deg2rad(t0 - theta)
        return 0.5 * (
            1.0
            + np.cos(2*chi_in) * np.cos(2*chi_P) * np.cos(2*phi)
            + np.sin(2*chi_in) * np.sin(2*chi_P)
        )

    I0 = I_de(a, b, c, d, theta_grid)
    var = np.zeros_like(I0)
    delta = 1e-3                       # paso [°] para la derivada numérica
    base = [a, b, c, d]
    for i in range(4):
        mas = list(base); menos = list(base)
        mas[i]   += delta
        menos[i] -= delta
        dIdx = (I_de(*mas, theta_grid) - I_de(*menos, theta_grid)) / (2 * delta)
        var += (dIdx * sigma_gonio) ** 2
    if incluir_barrido:
        dIdt = (I_de(a, b, c, d, theta_grid + delta)
                - I_de(a, b, c, d, theta_grid - delta)) / (2 * delta)
        var += (dIdt * sigma_gonio) ** 2
    return I0, np.sqrt(var)


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
        # Abrir tan alta como permita la pantalla (hasta 1040 px): así la
        # columna de control cabe entera y no hay que desplazarla.
        pantalla = QApplication.primaryScreen()
        if pantalla is not None:
            disp = pantalla.availableGeometry()
            self.resize(min(1380, disp.width()), min(1040, disp.height()))
        else:
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

        # Punto en vivo sobre la curva de Malus: marcador que sigue la
        # estimación corriente de I_norm mientras entran los bins. Al terminar
        # la medición lo sustituye el punto definitivo de _refrescar_malus.
        self._pv_marker = None
        self._pv_texto = None
        self._pv_traza = None
        self._pv_barra = None
        self._pv_activo: bool = False
        self._pv_ultimo_draw: float = 0.0
        self._pv_I_ref: float | None = None   # normalización fija del punto
        self._pv_hist: list[float] = []       # rastro de estimaciones
        self._malus_I_max: float = 1.0   # normalización vigente de la gráfica

        # Calibración de ruido de fondo (blanco metrológico): tasa medida con
        # el láser ON y dos polarizadores lineales cruzados (extinción). Si es
        # None, los puntos se guardan sin corrección (I_norm = I_norm_bruto).
        self._fondo: dict | None = None
        self._punto_es_fondo: bool = False
        self._aviso_fondo_mostrado: bool = False

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
        self.btn_desconectar.setStyleSheet(estilo_boton(COL_ROJO, "#1e1e2e", 12))
        self.btn_desconectar.setMinimumWidth(130)
        self.btn_desconectar.clicked.connect(self._desconectar)
        self.btn_desconectar.setEnabled(False)
        fila_top.addWidget(self.btn_desconectar)
        ext.addLayout(fila_top)

        # ── Splitter principal ──
        splitter = QSplitter(Qt.Orientation.Horizontal); splitter.setHandleWidth(4)

        # ─── Panel izquierdo ───
        izq = QWidget()
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
        self.btn_iniciar.setStyleSheet(estilo_boton("#3a8c3a", "#ffffff", 13))
        self.btn_iniciar.clicked.connect(self._iniciar_medicion)
        gctrl.addWidget(self.btn_iniciar)

        # Calibración del ruido de fondo (blanco): láser ON + dos polarizadores
        # lineales cruzados. Debe hacerse ANTES de tomar puntos.
        self.btn_fondo = QPushButton("🌑  Calibrar ruido de fondo")
        self.btn_fondo.setStyleSheet(estilo_boton("#4a4a6a", "#ffffff", 12))
        self.btn_fondo.setToolTip(
            "Medición de blanco (metrología): con el láser encendido, coloca "
            "dos polarizadores lineales cruzados a 90° (extinción) frente al "
            "SPCM y mide la tasa de fondo R_b (cuentas oscuras + luz ambiente "
            "+ fuga por extinción finita). Todos los puntos del barrido se "
            "corrigen restando R_b y σ_Rb entra en la incertidumbre.")
        self.btn_fondo.clicked.connect(self._calibrar_fondo)
        self.btn_fondo.setEnabled(False)
        gctrl.addWidget(self.btn_fondo)

        self.lbl_fondo = QLabel("Fondo: sin calibrar")
        self.lbl_fondo.setStyleSheet(
            f"background:{COL_BG2};color:{COL_AMBAR};"
            f"border:1px solid {COL_BORDE};padding:3px 4px;"
            "font-family:Menlo,Consolas,monospace;font-size:10px;")
        self.lbl_fondo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gctrl.addWidget(self.lbl_fondo)

        self.btn_tomar = QPushButton("●  Tomar punto")
        self.btn_tomar.setStyleSheet(estilo_boton("#3a5a8c", "#ffffff", 12))
        self.btn_tomar.clicked.connect(self._tomar_punto)
        self.btn_tomar.setEnabled(False)
        gctrl.addWidget(self.btn_tomar)

        # Repetir/corregir un punto ya tomado (por defecto el último): descarta
        # el dato existente en ese ángulo y vuelve a medirlo.
        self.btn_repetir = QPushButton("↻  Repetir / corregir punto")
        self.btn_repetir.setStyleSheet(estilo_boton("#3a8c8c", "#ffffff", 12))
        self.btn_repetir.setToolTip(
            "Vuelve a medir un punto ya tomado. Por defecto propone el último "
            "ángulo, pero puedes escribir cualquier otro ángulo previo: se "
            "eliminará el dato que ya existía en ese ángulo y se reemplazará "
            "por la nueva medición.")
        self.btn_repetir.clicked.connect(self._repetir_punto_anterior)
        self.btn_repetir.setEnabled(False)
        gctrl.addWidget(self.btn_repetir)

        self.btn_guardar = QPushButton("💾  Cerrar y guardar")
        self.btn_guardar.setStyleSheet(estilo_boton("#8c5a3a", "#ffffff", 12))
        self.btn_guardar.clicked.connect(self._cerrar_y_guardar)
        self.btn_guardar.setEnabled(False)
        gctrl.addWidget(self.btn_guardar)

        # Configurar polarizadores elípticos (Malus generalizada)
        self.btn_polarizadores = QPushButton("⚙  Polarizadores elípticos…")
        self.btn_polarizadores.setStyleSheet(estilo_boton("#5a3a8c", "#ffffff", 12))
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

        # Alinear el pico de la curva teórica al máximo de los datos.
        # Desactivado por defecto ⇒ se muestra la curva teórica estricta,
        # con el pico en la posición natural del modelo α_in − α_P.
        self.chk_alinear_pico = QCheckBox("Alinear pico de la curva a los datos")
        self.chk_alinear_pico.setChecked(False)
        self.chk_alinear_pico.setStyleSheet(
            f"color:{COL_TXT};font-size:10px;padding:2px;")
        self.chk_alinear_pico.setToolTip(
            "Si está activado, el pico de la curva teórica se desplaza para "
            "coincidir con el ángulo del máximo experimental (compensa el "
            "desfase del montaje). Si está desactivado (por defecto), la curva "
            "es la teórica estricta con el pico en α_in − α_P (mod 180°).")
        self.chk_alinear_pico.toggled.connect(self._refrescar_malus)
        gctrl.addWidget(self.chk_alinear_pico)

        # Seguimiento en vivo del punto que se está midiendo.
        self.chk_punto_vivo = QCheckBox("Ver el punto en vivo sobre la curva")
        self.chk_punto_vivo.setChecked(True)
        self.chk_punto_vivo.setStyleSheet(
            f"color:{COL_TXT};font-size:10px;padding:2px;")
        self.chk_punto_vivo.setToolTip(
            "Mientras se toma un punto, dibuja sobre la curva de Malus un "
            "marcador con la estimación de I/I_max calculada con los bins "
            "acumulados hasta ese instante. Al terminar la medición el "
            "marcador desaparece y queda el punto definitivo (con sus barras "
            "de error). No altera los datos: es solo visualización.")
        self.chk_punto_vivo.toggled.connect(self._on_toggle_punto_vivo)
        gctrl.addWidget(self.chk_punto_vivo)

        # Reintentar conexión SPCM (cuando el contador no fue detectado)
        self.btn_reintentar_spcm = QPushButton("🔄  Reintentar SPCM")
        self.btn_reintentar_spcm.setStyleSheet(estilo_boton("#3a3a5a", COL_TXT, 12))
        self.btn_reintentar_spcm.setToolTip(
            "Vuelve a buscar el contador de fotones SPCM50A/M (libera la "
            "interfaz USB cerrando otras instancias antes de pulsar).")
        self.btn_reintentar_spcm.clicked.connect(self._reintentar_spcm)
        gctrl.addWidget(self.btn_reintentar_spcm)

        # Repetir experimento desde cero
        self.btn_reset = QPushButton("🗑  Repetir desde cero")
        self.btn_reset.setStyleSheet(estilo_boton("#5a3a3a", COL_TXT, 12))
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
        lay_izq.addStretch(1)

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
        self.log_widget.setMinimumHeight(90)
        self.log_widget.setMaximumHeight(128)
        gl.addWidget(self.log_widget)

        # Ningún control del panel izquierdo debe poder comprimirse por debajo
        # de su tamaño natural: si la ventana es más baja que la columna, Qt
        # aplastaba los botones hasta recortarles el texto. Ahora conservan su
        # altura y, si no caben, la zona de controles se desplaza.
        for btn in izq.findChildren(QPushButton):
            btn.setMinimumHeight(btn.sizeHint().height())
        for chk in izq.findChildren(QCheckBox):
            chk.setMinimumHeight(chk.sizeHint().height())

        scroll_izq = QScrollArea()
        scroll_izq.setWidget(izq)
        scroll_izq.setWidgetResizable(True)
        scroll_izq.setFrameShape(QFrame.Shape.NoFrame)
        scroll_izq.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # El log queda fuera del área desplazable para que siempre esté a la
        # vista, por baja que sea la ventana.
        panel_izq = QWidget()
        panel_izq.setMaximumWidth(348); panel_izq.setMinimumWidth(318)
        lay_panel = QVBoxLayout(panel_izq)
        lay_panel.setSpacing(6); lay_panel.setContentsMargins(0, 0, 0, 0)
        lay_panel.addWidget(scroll_izq, 1)
        lay_panel.addWidget(gb_log)
        splitter.addWidget(panel_izq)

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
        # El blanco también se descarta: si el usuario cambia la potencia del
        # láser, la componente de fuga por extinción finita (∝ P) cambia y el
        # fondo debe volver a medirse en las nuevas condiciones.
        self._fondo = None
        self._punto_es_fondo = False
        self._aviso_fondo_mostrado = False
        self._actualizar_lbl_fondo()
        self.btn_fondo.setEnabled(False)

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
        self.btn_fondo.setEnabled(False)
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
        self.btn_fondo.setEnabled(False)         # ídem
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
        self.btn_fondo.setEnabled(True)
        self.btn_repetir.setEnabled(len(self._puntos) > 0)
        if self._fondo is None:
            self._log(
                "→ Recomendado: calibra el RUIDO DE FONDO antes del primer "
                "punto (láser ON + dos polarizadores lineales cruzados a 90°).")
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
        # Aviso metrológico (una sola vez): sin blanco calibrado, I_norm
        # incluye cuentas oscuras + luz ambiente y sesga el mínimo de la curva.
        if self._fondo is None and not self._puntos \
                and not self._aviso_fondo_mostrado:
            self._aviso_fondo_mostrado = True
            resp = QMessageBox.question(
                self, "Fondo sin calibrar",
                "No se ha calibrado el ruido de fondo (blanco).\n\n"
                "Sin esta corrección, I_norm incluirá las cuentas oscuras del "
                "SPCM y la luz ambiente, sesgando sobre todo los puntos "
                "cercanos al mínimo de la curva.\n\n"
                "¿Tomar puntos SIN corrección de fondo?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if resp != QMessageBox.StandardButton.Yes:
                return
            self._log("⚠ Midiendo SIN corrección de fondo (blanco no calibrado).")

        sug = self._sugerir_proximo_angulo()
        ang, ok = QInputDialog.getDouble(
            self, "Ángulo del segundo polarizador",
            f"Ingrese θ en grados (sugerido {sug:.1f}°)\n"
            f"(σ_θ = ±{SIGMA_GONIO_DEG:.0f}° por goniómetro)",
            value=sug, min=0.0, max=360.0, decimals=1)
        if not ok:
            return
        self._iniciar_punto(ang)

    def _repetir_punto_anterior(self):
        """
        Vuelve a medir un punto ya tomado, reemplazándolo. Por defecto
        propone el último ángulo medido, pero el usuario puede escribir
        cualquier otro ángulo previo: el dato que ya existía en ese ángulo
        se elimina y se sustituye por la nueva medición. Útil cuando una
        toma quedó ruidosa, hubo deriva del láser, o se reposicionó el
        polarizador con más precisión.
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

        # Ángulo propuesto por defecto: el del último punto tomado.
        ang_default = float(self._puntos[-1]["angulo"])
        ang, ok = QInputDialog.getDouble(
            self, "Repetir / corregir punto",
            "Ángulo θ del punto a corregir (por defecto, el último).\n"
            "Se eliminará el dato que ya existía en ese ángulo y se\n"
            "reemplazará por la nueva medición.\n"
            f"(σ_θ = ±{SIGMA_GONIO_DEG:.0f}° por goniómetro)",
            value=ang_default, min=0.0, max=360.0, decimals=1)
        if not ok:
            return

        # Buscar el punto existente más cercano al ángulo pedido.
        angs = [float(p["angulo"]) for p in self._puntos]
        idx  = min(range(len(angs)), key=lambda i: abs(angs[i] - ang))
        TOL  = max(0.5, SIGMA_GONIO_DEG)   # ° — margen para identificar el punto

        if abs(angs[idx] - ang) <= TOL:
            p_viejo  = self._puntos.pop(idx)
            ang_real = float(p_viejo["angulo"])
            self._log(
                f"Corrigiendo punto en θ = {ang_real:.1f}° "
                f"(descartado I_norm previo = {float(p_viejo['I_norm']):.4g})")
            ang = ang_real   # medir exactamente en el ángulo del dato eliminado
        else:
            # No existe un punto en ese ángulo: se mide uno nuevo, no se borra.
            resp = QMessageBox.question(
                self, "Sin dato en ese ángulo",
                f"No hay ningún punto registrado en θ = {ang:.1f}° "
                f"(el más cercano está en {angs[idx]:.1f}°).\n\n"
                f"¿Medir un punto nuevo en θ = {ang:.1f}°?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes)
            if resp != QMessageBox.StandardButton.Yes:
                return
            self._log(f"Nuevo punto en θ = {ang:.1f}° "
                      f"(no existía dato previo en ese ángulo)")

        self._refrescar_resumen()
        self._refrescar_malus()
        self._iniciar_punto(ang)

    # ─── Calibración del ruido de fondo (blanco metrológico) ────────────
    def _calibrar_fondo(self):
        """
        Medición de blanco previa al barrido, según la práctica estándar de
        metrología en conteo (Currie 1968; ISO 11843): con el láser ENCENDIDO
        y dos polarizadores lineales cruzados a 90° (extinción) se mide la
        tasa de fondo

            R_b = cuentas oscuras del SPCM + luz ambiente + fuga por la
                  extinción finita de los polarizadores  [CPS].

        Medir con el láser ON (y no apagado) hace que el blanco capture
        también la luz esparcida del propio láser en el montaje, que estará
        presente durante el barrido real. El blanco se mide por defecto con
        FACTOR_BINS_FONDO_DEF× los bins de un punto para que sea un "blanco
        bien conocido" (σ_Rb ∝ 1/√T_b, subdominante frente al punto).
        """
        if not self._iniciado:
            QMessageBox.warning(self, "No iniciado",
                                "Pulsa primero “Iniciar medición”.")
            return
        if self._punto_en_curso:
            return

        extra = ("\n\n⚠ Ya existe un fondo calibrado "
                 f"({self._fondo['tasa_cps']:.1f} CPS): será REEMPLAZADO y la "
                 "corrección se recalculará para todos los puntos ya tomados."
                 if self._fondo is not None else "")
        resp = QMessageBox.question(
            self, "Calibrar ruido de fondo (blanco)",
            "Con el LÁSER ENCENDIDO, coloca DOS polarizadores lineales "
            "CRUZADOS a 90° (configuración de extinción) delante del SPCM.\n\n"
            "Se medirá la tasa de fondo R_b:\n"
            "  • cuentas oscuras del detector\n"
            "  • luz ambiente de la sala\n"
            "  • fuga por la extinción finita de los polarizadores\n\n"
            "Todos los puntos del barrido se corregirán como\n"
            "    I_norm = ⟨(CPS − R_b)/P⟩\n"
            "y σ_Rb entrará en la incertidumbre como componente sistemática."
            + extra + "\n\n¿Polarizadores cruzados y listo para medir?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes)
        if resp != QMessageBox.StandardButton.Yes:
            return

        factor, ok = QInputDialog.getInt(
            self, "Duración del blanco",
            "Duración del fondo en múltiplos de un punto normal\n"
            "(≥ 3× recomendado: blanco “bien conocido”, σ_Rb ∝ 1/√T):",
            value=FACTOR_BINS_FONDO_DEF, min=1, max=50)
        if not ok:
            return

        n_bins_fondo = int(self.spn_bins.value()) * factor
        self._log(f"FONDO → midiendo blanco con {n_bins_fondo} bins "
                  f"({factor}× un punto normal)…")
        self._iniciar_punto(0.0, es_fondo=True, n_bins_override=n_bins_fondo)

    def _procesar_fondo(self):
        """Reduce la medición de blanco a (R_b, σ_Rb) y límites de Currie."""
        bin_s    = self._punto_bin_s
        arr      = self._punto_arr
        cps_bins = arr.astype(float) / bin_s
        n        = len(cps_bins)
        R_b      = float(cps_bins.mean())
        total    = float(arr.sum())
        T_b      = n * bin_s                      # tiempo vivo del blanco [s]

        # SEM por chunks: robusta frente a fondo NO estacionario o
        # super-poissoniano (parpadeo de luminarias a 50/60 Hz, tránsito de
        # personas). Igual esquema que en la reducción de los puntos.
        M = max(10, min(100, n // 50))
        chunk = n // M
        if chunk >= 1 and M >= 2:
            medias = cps_bins[: M * chunk].reshape(M, chunk).mean(axis=1)
            sem = float(medias.std(ddof=1) / np.sqrt(M))
        else:
            sem = (float(cps_bins.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)
        # Piso de Poisson: σ(R) = √N_total / T_b para un proceso de conteo puro
        sigma_poisson = (np.sqrt(total) / T_b) if total > 0 else 0.0
        sigma_Rb = max(sem, sigma_poisson)

        # Límites de Currie (1968) para UN punto del barrido con el tiempo
        # vivo actual T_s: cuentas de fondo esperadas B = R_b·T_s y
        #     σ₀ = √(B(1 + T_s/T_b)),  L_C = 1.645·σ₀,  L_D = 2.71 + 3.29·σ₀.
        # Un punto con cuentas netas < L_C es indistinguible del fondo.
        T_s   = int(self.spn_bins.value()) * bin_s
        B_esp = R_b * T_s
        sigma0 = np.sqrt(max(B_esp, 0.0) * (1.0 + T_s / T_b))
        L_C = 1.645 * sigma0
        L_D = 2.71 + 3.29 * sigma0

        self._fondo = {
            "tasa_cps":    R_b,
            "sigma_cps":   sigma_Rb,
            "n_bins":      n,
            "bin_length_s": bin_s,
            "t_total_s":   T_b,
            "total_cuentas": total,
            "L_C_cuentas": float(L_C),
            "L_D_cuentas": float(L_D),
            "bins":        arr.copy(),
            "t_iso":       datetime.now().isoformat(timespec="seconds"),
        }

        self._log(f"✓ FONDO calibrado: R_b = {R_b:.2f} ± {sigma_Rb:.2f} CPS  "
                  f"({total:.0f} cuentas en {T_b:.1f} s)")
        self._log(f"  Currie por punto (T_s = {T_s:.1f} s): "
                  f"B_esp ≈ {B_esp:.0f} cuentas, "
                  f"L_C ≈ {L_C:.0f}, L_D ≈ {L_D:.0f} cuentas netas")
        self._actualizar_lbl_fondo()

        # Recalcular la corrección de los puntos ya tomados (exacto: restar la
        # constante R_b bin a bin equivale a I_bruto − R_b·⟨1/P⟩, y no cambia
        # la varianza muestral, por lo que basta con los estadísticos guardados).
        if self._puntos:
            for p in self._puntos:
                self._aplicar_fondo_a_punto(p)
            self._log(f"  Corrección aplicada retroactivamente a "
                      f"{len(self._puntos)} punto(s).")
            self._refrescar_resumen()
        self._refrescar_malus()

    def _aplicar_fondo_a_punto(self, p: dict):
        """
        Sustracción del blanco: I_norm = I_bruto − R_b·⟨1/P⟩.

        σ_Rb entra como componente SISTEMÁTICA correlacionada (la misma R_b
        se resta a todos los bins del punto, así que NO se reduce con √N):
            σ_I² = σ_I,bruto² + (σ_Rb·⟨1/P⟩)².
        """
        if self._fondo is None:
            return
        Rb   = self._fondo["tasa_cps"]
        sRb  = self._fondo["sigma_cps"]
        invP = p.get("inv_P_mean", 0.0)
        p["fondo_cps"]     = Rb
        p["I_norm"]        = p["I_norm_bruto"] - Rb * invP
        p["sigma_I_norm"]  = float(np.hypot(p["sigma_I_bruto"], sRb * invP))

    def _actualizar_lbl_fondo(self):
        if self._fondo is None:
            self.lbl_fondo.setText("Fondo: sin calibrar")
            self.lbl_fondo.setStyleSheet(
                f"background:{COL_BG2};color:{COL_AMBAR};"
                f"border:1px solid {COL_BORDE};padding:3px 4px;"
                "font-family:Menlo,Consolas,monospace;font-size:10px;")
        else:
            f = self._fondo
            self.lbl_fondo.setText(
                f"Fondo: {f['tasa_cps']:.1f} ± {f['sigma_cps']:.1f} CPS  "
                f"(L_C≈{f['L_C_cuentas']:.0f})")
            self.lbl_fondo.setStyleSheet(
                f"background:{COL_BG2};color:{COL_VERDE};"
                f"border:1px solid {COL_VERDE};padding:3px 4px;"
                "font-family:Menlo,Consolas,monospace;font-size:10px;")

    def _iniciar_punto(self, angulo: float, es_fondo: bool = False,
                       n_bins_override: int | None = None):
        bin_ms       = self.spn_bin_ms.value()
        time_btw_ms  = self.spn_time_between.value()
        pulse_blind  = self.spn_pulse_blind.value()
        n_bins       = int(n_bins_override if n_bins_override
                           else self.spn_bins.value())
        t_total      = (bin_ms + time_btw_ms) * n_bins / 1000.0

        self._punto_es_fondo     = es_fondo
        self._punto_en_curso     = True
        self._punto_evt_detener  = threading.Event()
        self._punto_potencias    = []
        self._punto_t0           = time.time()
        self._punto_angulo_curso = angulo
        self._punto_n_bins       = n_bins
        self._punto_bin_s        = bin_ms / 1000.0

        # Punto en vivo: solo para puntos de la curva (el blanco no es un punto)
        self._limpiar_punto_vivo()
        self._pv_activo = (self.chk_punto_vivo.isChecked() and not es_fondo)
        self._pv_ultimo_draw = 0.0

        # Limpiar gráficas en vivo (CPS lo presentamos como en el software de
        # Thorlabs: Counts per Bin vs Bin Number)
        self.line_pot_live.set_data([], [])
        self.line_cps_live.set_data([], [])
        self.ax_pot_live.set_xlim(0, max(t_total, 1.0)); self.ax_pot_live.set_ylim(0, 1)
        self.ax_cps_live.set_xlim(0, n_bins); self.ax_cps_live.set_ylim(0, 1)
        self.canvas_pot.draw_idle(); self.canvas_cps.draw_idle()

        self.btn_tomar.setEnabled(False)
        self.btn_repetir.setEnabled(False)
        self.btn_fondo.setEnabled(False)
        etiqueta = "FONDO (blanco)" if es_fondo else f"θ={angulo:6.1f}°"
        self._log(
            f"{etiqueta}  →  {n_bins} bins × {bin_ms:.3f} ms "
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

        # Marcador en vivo sobre la curva de Malus. Se limita a ~8 Hz porque
        # el callback puede llegar a cada bin (kHz) y redibujar la curva es
        # mucho más caro que actualizar la gráfica de bins.
        if self._pv_activo and self.chk_punto_vivo.isChecked():
            ahora = time.time()
            if ahora - self._pv_ultimo_draw >= 0.12:
                self._pv_ultimo_draw = ahora
                try:
                    self._actualizar_punto_vivo(counts_arr)
                except Exception:
                    # Nunca dejar que un fallo de dibujo aborte la medición.
                    self._pv_activo = False
                    self._limpiar_punto_vivo()

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

    def _reducir_punto(self, arr, bin_s: float, pot_arr) -> dict:
        """
        Reducción de un punto (parcial o completo) a intensidad normalizada.

        1) Counts → CPS por bin
        2) Potencia P(t) → interpolada en el tiempo de cada bin
        3) Cociente bin-a-bin   I_bin = CPS_bin / P(t_bin)
           Esto CANCELA el ruido común del láser (deriva, modos
           relajación) que afecta por igual a CPS y a P.
        4) La media de los bins se agrupa en M chunks (≈ 50) y la
           incertidumbre de I_norm se calcula como la SEM de las
           medias de chunk. Esto incluye la varianza residual
           (no-cancelada bin-a-bin) y los efectos correlacionados
           a tiempos largos (deriva térmica, vibraciones).

        `I_norm` es BRUTA (sin restar fondo); la sustracción del blanco se
        aplica aparte en _aplicar_fondo_a_punto.

        Se usa tanto para el punto definitivo como para la traza en vivo
        (bin a bin), de modo que el punto en vivo converge exactamente al
        valor final.
        """
        cps_bins = arr.astype(float) / bin_s
        n_bins   = len(cps_bins)
        cps_mean = float(cps_bins.mean()) if n_bins else 0.0
        cps_sem  = (float(cps_bins.std(ddof=1) / np.sqrt(n_bins))
                    if n_bins > 1 else 0.0)

        # Período por bin (Bin Length + Time between Bins)
        t_step_s = (self.spn_bin_ms.value() + self.spn_time_between.value()) / 1000.0
        t_bins = (np.arange(n_bins) + 0.5) * t_step_s     # tiempo del centro

        if len(pot_arr) >= 2:
            # Interpolación lineal de P(t) en el tiempo de cada bin.
            # np.interp extrapola con los extremos (clip), válido aquí
            # porque el muestreo de potencia cubre todo el punto.
            P_at_bin = np.interp(t_bins, pot_arr[:, 0], pot_arr[:, 1])
            pot_mean = float(pot_arr[:, 1].mean())
            pot_sem  = float(pot_arr[:, 1].std(ddof=1) / np.sqrt(len(pot_arr)))
        elif len(pot_arr) == 1:
            P_at_bin = np.full(n_bins, float(pot_arr[0, 1]))
            pot_mean = float(pot_arr[0, 1])
            pot_sem  = 0.005 * pot_mean        # 0.5 % típico iBeam Smart
        else:
            P_at_bin = np.zeros(n_bins)
            pot_mean = 0.0; pot_sem = 0.0

        valid = (P_at_bin > 0)
        if valid.sum() > 1 and cps_mean > 0:
            inv_P_mean = float(np.mean(1.0 / P_at_bin[valid]))
            I_per_bin = cps_bins[valid] / P_at_bin[valid]
            # Estadística por chunks: M ∈ [10, 100], cada chunk con
            # ≥ 50 bins para promediar el ruido de Poisson antes de
            # calcular la SEM. Captura mejor las correlaciones a tiempos
            # largos que la SEM directa de los bins individuales.
            n_v = int(valid.sum())
            M = max(10, min(100, n_v // 50))
            chunk_size = n_v // M
            if chunk_size >= 1 and M >= 2:
                I_chunked = (I_per_bin[: M * chunk_size]
                             .reshape(M, chunk_size).mean(axis=1))
                I_norm  = float(I_chunked.mean())
                sigma_I = float(I_chunked.std(ddof=1) / np.sqrt(M))
            else:
                I_norm  = float(I_per_bin.mean())
                sigma_I = (float(I_per_bin.std(ddof=1) / np.sqrt(len(I_per_bin)))
                           if len(I_per_bin) > 1 else 0.0)
            # Cota inferior por Poisson: si cae por debajo del piso
            # de Poisson (laser perfectamente estable), usamos ese piso.
            #   σ_Poisson(I) = sqrt(N_total) / (P · T_total)
            #               = I_norm / sqrt(N_total)
            N_total = float(arr.sum())
            if N_total > 0:
                sigma_poisson = I_norm / np.sqrt(N_total)
                sigma_I = max(sigma_I, sigma_poisson)
        elif pot_mean > 0 and cps_mean > 0:
            # Fallback: sin lecturas de potencia válidas, propagación clásica
            inv_P_mean = 1.0 / pot_mean
            I_norm = cps_mean / pot_mean
            rel = np.sqrt((cps_sem / cps_mean) ** 2
                          + (pot_sem / pot_mean) ** 2)
            sigma_I = I_norm * rel
        else:
            inv_P_mean = 0.0
            I_norm = 0.0; sigma_I = 0.0

        return {
            "n_bins":     n_bins,
            "cps_mean":   cps_mean,   "cps_sem": cps_sem,
            "pot_mean":   pot_mean,   "pot_sem": pot_sem,
            "inv_P_mean": inv_P_mean,
            "I_norm":     I_norm,     "sigma_I": sigma_I,
        }

    # ─── Punto en vivo sobre la curva de Malus ─────────────────────────
    def _on_toggle_punto_vivo(self, activado: bool):
        if activado:
            # Se reanuda solo cuando llegue el siguiente bin.
            self._pv_activo = self._punto_en_curso and not self._punto_es_fondo
        else:
            self._limpiar_punto_vivo()
            self.canvas_malus.draw_idle()

    def _limpiar_punto_vivo(self):
        for attr in ("_pv_marker", "_pv_traza", "_pv_barra", "_pv_texto"):
            art = getattr(self, attr, None)
            if art is not None:
                try:
                    art.remove()
                except (ValueError, NotImplementedError):
                    pass
                setattr(self, attr, None)
        self._pv_activo = False
        self._pv_I_ref = None
        self._pv_hist = []

    def _asegurar_artistas_vivo(self):
        """Crea los artistas del punto en vivo si no existen o si un
        redibujado de la curva (cla()) los eliminó del eje.

        Todos van con clip_on=False: en θ = 0° o 360° el marcador cae justo
        sobre el borde del eje y, recortado, se veía como una flecha en vez
        de como un punto.
        """
        if getattr(self, "_pv_traza", None) is None or self._pv_traza.axes is None:
            # Rastro de las últimas estimaciones: deja ver la oscilación
            # y cómo se va estrechando al converger.
            (self._pv_traza,) = self.ax_malus.plot(
                [], [], marker="o", markersize=3, linestyle="none",
                color=COL_AMBAR, alpha=0.30, zorder=11, clip_on=False)
        if getattr(self, "_pv_barra", None) is None or self._pv_barra.axes is None:
            # Barra ±1σ de la estimación corriente: encoge como 1/√N y hace
            # visible la convergencia aunque la media ya casi no se mueva.
            (self._pv_barra,) = self.ax_malus.plot(
                [], [], color=COL_AMBAR, lw=1.6, alpha=0.75, zorder=12,
                solid_capstyle="butt", clip_on=False)
        if getattr(self, "_pv_marker", None) is None or self._pv_marker.axes is None:
            (self._pv_marker,) = self.ax_malus.plot(
                [], [], marker="o", markersize=10, linestyle="none",
                markerfacecolor=COL_AMBAR, markeredgecolor="#ffffff",
                markeredgewidth=1.4, zorder=13, clip_on=False)
        if getattr(self, "_pv_texto", None) is None or self._pv_texto.axes is None:
            self._pv_texto = self.ax_malus.text(
                0, 0, "", color=COL_AMBAR, fontsize=8, zorder=13,
                ha="left", va="center", clip_on=False)

    def _actualizar_punto_vivo(self, counts_arr):
        """Dibuja la estimación corriente del punto en curso sobre la curva
        de Malus, usando los bins acumulados hasta ahora."""
        pot_arr = np.array(list(self._punto_potencias)) \
            if self._punto_potencias else np.zeros((0, 2))
        red = self._reducir_punto(np.asarray(counts_arr), self._punto_bin_s,
                                  pot_arr)
        if red["I_norm"] <= 0.0:
            # Aún no hay lecturas de potencia del láser (llegan cada
            # PERIODO_POT_S): sin ellas la estimación sería 0 y el marcador
            # daría un salto falso hasta el suelo de la gráfica.
            return
        I_vivo = red["I_norm"]
        if self._fondo is not None:
            I_vivo -= self._fondo["tasa_cps"] * red["inv_P_mean"]

        # Normalización I / I_max. La referencia se CONGELA en la primera
        # estimación válida del punto y no se toca hasta que termine: si se
        # recalculase con el propio valor en vivo, el cociente daría 1 siempre
        # y el marcador quedaría clavado sin mostrar la convergencia.
        if self._pv_I_ref is None:
            self._pv_I_ref = self._malus_I_max if self._puntos else I_vivo
        y = I_vivo / self._pv_I_ref if self._pv_I_ref > 0 else 0.0

        self._asegurar_artistas_vivo()
        x = self._punto_angulo_curso
        self._pv_marker.set_data([x], [y])
        sigma_n = red["sigma_I"] / self._pv_I_ref
        self._pv_barra.set_data([x, x], [y - sigma_n, y + sigma_n])
        # Rastro de las últimas estimaciones (se ve oscilar y estrecharse).
        self._pv_hist.append(y)
        del self._pv_hist[:-60]
        self._pv_traza.set_data([x] * len(self._pv_hist), self._pv_hist)
        frac = red["n_bins"] / max(self._punto_n_bins, 1)
        # Cerca del extremo derecho la etiqueta se pasa al otro lado del
        # marcador para no salirse del eje (0–360°).
        if x > 260.0:
            self._pv_texto.set_ha("right"); self._pv_texto.set_position((x - 12, y))
        else:
            self._pv_texto.set_ha("left");  self._pv_texto.set_position((x + 12, y))
        self._pv_texto.set_text(
            f"en vivo · {frac*100:.0f} %\nI/I_max = {y:.3f} ± {sigma_n:.3f}")
        self.canvas_malus.draw_idle()

    def _on_pt_listo(self, angulo: float):
        # El punto definitivo (con barras de error) sustituye al marcador vivo.
        self._pv_activo = False
        self._limpiar_punto_vivo()

        if self._punto_arr is None or len(self._punto_arr) == 0:
            self._on_error("punto sin bins recibidos del SPCM")
            return

        # ¿Era la medición de blanco? Se reduce aparte y no genera punto.
        if self._punto_es_fondo:
            self._punto_es_fondo = False
            self._procesar_fondo()
            self._punto_en_curso  = False
            self._punto_arr       = None
            self._punto_potencias = []
            self.btn_tomar.setEnabled(True)
            self.btn_fondo.setEnabled(True)
            self.btn_repetir.setEnabled(len(self._puntos) > 0)
            return

        bin_s = self._punto_bin_s
        pot_arr = (np.array(self._punto_potencias)
                   if self._punto_potencias else np.zeros((0, 2)))
        red = self._reducir_punto(self._punto_arr, bin_s, pot_arr)
        n_bins     = red["n_bins"]
        cps_mean   = red["cps_mean"];   cps_sem = red["cps_sem"]
        pot_mean   = red["pot_mean"];   pot_sem = red["pot_sem"]
        inv_P_mean = red["inv_P_mean"]
        I_norm     = red["I_norm"];     sigma_I = red["sigma_I"]

        punto = {
            "angulo":        angulo,
            "sigma_angulo":  SIGMA_ANGULO_DEG,
            "cps_mean":      cps_mean,
            "cps_sem":       cps_sem,
            "pot_mean_uW":   pot_mean,
            "pot_sem_uW":    pot_sem,
            # Estadísticos brutos (sin blanco); I_norm/sigma_I_norm se
            # sobreescriben en _aplicar_fondo_a_punto si hay fondo calibrado.
            "I_norm":        I_norm,
            "sigma_I_norm":  sigma_I,
            "I_norm_bruto":  I_norm,
            "sigma_I_bruto": sigma_I,
            "inv_P_mean":    inv_P_mean,
            "fondo_cps":     0.0,
            "bins":          self._punto_arr.copy(),
            "potencias":     pot_arr.copy(),
            "bin_length_s":  bin_s,
            "t_iso":         datetime.now().isoformat(timespec="seconds"),
        }
        self._aplicar_fondo_a_punto(punto)
        self._puntos.append(punto)

        self._log(
            f"  → CPS = {cps_mean:>9.1f} ± {cps_sem:>5.1f}   "
            f"P = {pot_mean:>7.2f} ± {pot_sem:>5.2f} µW   "
            f"I_norm = {punto['I_norm']:.4g} ± {punto['sigma_I_norm']:.2g}"
            + (f"  (bruto {I_norm:.4g}; fondo "
               f"{self._fondo['tasa_cps']:.1f} CPS restado)"
               if self._fondo is not None else "  (SIN corrección de fondo)"))

        # Criterio de detección de Currie: ¿las cuentas netas del punto son
        # distinguibles del fondo puro?
        if self._fondo is not None:
            T_s = n_bins * bin_s
            netas = (cps_mean - self._fondo["tasa_cps"]) * T_s
            if netas < self._fondo["L_C_cuentas"]:
                self._log(
                    f"  ⚠ Cuentas netas ({netas:.0f}) < L_C de Currie "
                    f"({self._fondo['L_C_cuentas']:.0f}): este punto es "
                    "estadísticamente compatible con fondo puro (mínimo real).")

        self._refrescar_resumen()
        self._refrescar_malus()

        self._punto_en_curso = False
        self._punto_arr = None
        self._punto_potencias = []
        self.btn_tomar.setEnabled(True)
        self.btn_fondo.setEnabled(True)
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
        # Pico natural del modelo (curva teórica estricta): α_in − α_P (mod 180°)
        theta_0_teorico = (self._theta1_p1 - self._theta1_p2) % 180.0
        alinear = self.chk_alinear_pico.isChecked()

        # Datos experimentales (si los hay)
        if self._puntos:
            angs = np.array([p["angulo"]       for p in self._puntos])
            Is   = np.array([p["I_norm"]        for p in self._puntos])
            sIs  = np.array([p["sigma_I_norm"] for p in self._puntos])
            I_max = float(Is.max()) if Is.max() > 0 else 1.0
            self._malus_I_max = I_max
            Is_n   = Is  / I_max
            sIs_n  = sIs / I_max
            idx_max = int(Is.argmax())
            # Solo se alinea el pico al máximo de los datos si el usuario lo
            # activa; en caso contrario se respeta el pico teórico estricto.
            theta_0 = float(angs[idx_max]) if alinear else theta_0_teorico
        else:
            angs = Is = sIs = Is_n = sIs_n = None
            theta_0 = theta_0_teorico

        # Predicción de la ley de Malus generalizada:
        #   I(θ) = ½ [1 + cos2χ_in cos2χ_P cos2(θ_0 − θ) + sin2χ_in sin2χ_P]
        # Por defecto θ_0 es el pico teórico estricto α_in − α_P (mod 180°).
        # Si se activa "Alinear pico…", θ_0 pasa a ser el máximo de los datos
        # para compensar el desfase del montaje. La forma de la curva (amplitud
        # y "piso") siempre depende solo de χ_in, χ_P.
        #
        # La banda sombreada es la incertidumbre 1σ propagada desde los
        # goniómetros de configuración (θ₁_p1, θ₂_p1, θ₁_p2, θ₂_p2), cada uno
        # con σ = SIGMA_GONIO_DEG (= 2°). En modo "alinear", θ_0 lo fijan los
        # datos y la banda solo recoge la incertidumbre de χ_in y χ_P.
        theta_grid = np.linspace(0, 360, 721)
        theta_0_band = theta_0 if alinear else None
        I_pred, sigma_pred = _malus_pred_con_banda(
            theta_grid,
            self._theta1_p1, self._theta2_p1,
            self._theta1_p2, self._theta2_p2,
            SIGMA_GONIO_DEG, theta_0=theta_0_band)
        I_pred_max = float(I_pred.max()) if I_pred.max() > 0 else 1.0
        I_pred_n    = I_pred / I_pred_max
        sigma_pred_n = sigma_pred / I_pred_max
        I_pred_min_n = float(I_pred.min()) / I_pred_max

        self.ax_malus.cla()
        # cla() destruye los artistas del punto en vivo; se recrean solos en
        # la siguiente actualización si la medición sigue en curso.
        self._pv_marker = None
        self._pv_traza = None
        self._pv_barra = None
        self._pv_texto = None
        self.ax_malus.set_facecolor(COL_PLOT)
        for sp in self.ax_malus.spines.values():
            sp.set_color(COL_BORDE)
        self.ax_malus.tick_params(colors=COL_TXT, labelsize=9)
        self.ax_malus.grid(True, color=COL_GRID, linewidth=0.5, alpha=0.6)
        self.ax_malus.set_xlabel("Ángulo del polarizador θ [°]", color=COL_TXT)
        self.ax_malus.set_ylabel("Intensidad normalizada  I / I_max",
                                 color=COL_TXT)

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
        # Banda de incertidumbre 1σ propagada desde los 6 goniómetros (±2°)
        if np.any(sigma_pred_n > 0):
            self.ax_malus.fill_between(
                theta_grid, I_pred_n - sigma_pred_n, I_pred_n + sigma_pred_n,
                color=COL_LILA, alpha=0.18, linewidth=0,
                label=f"±1σ teórica (goniómetros ±{SIGMA_GONIO_DEG:.0f}°)")

        if angs is not None:
            # Barras de error de cada dato:
            #   x = grados → lectura del goniómetro de barrido, σ_θ = ±2°.
            #   y = intensidad normalizada. La incertidumbre de conteo
            #       (σ_I_norm) es minúscula (Poisson con millones de cuentas);
            #       la barra vertical la DOMINA la propagación completa de la
            #       fórmula de Malus generalizada por TODAS sus variables
            #       angulares (χ_in, χ_P, θ₀ y el ángulo de barrido θ), cada
            #       una ±2°:
            #           σ_I² = Σ_x (∂I/∂x)² σ_gonio²,  x ∈ {θ₁_p1, θ₂_p1,
            #                                              θ₁_p2, θ₂_p2, θ}.
            #       Se suma en cuadratura con σ_I_norm.
            _, sig_malus = _malus_pred_con_banda(
                angs, self._theta1_p1, self._theta2_p1,
                self._theta1_p2, self._theta2_p2,
                SIGMA_GONIO_DEG, theta_0=theta_0_band, incluir_barrido=True)
            sig_malus_n = sig_malus / I_pred_max
            yerr_total = np.sqrt(sIs_n**2 + sig_malus_n**2)
            #   La σ de conteo (sIs_n) ya incluye, si el blanco está
            #   calibrado, la componente sistemática del fondo σ_Rb·⟨1/P⟩.
            etiq_fondo = (", fondo restado" if self._fondo is not None
                          else ", SIN fondo")
            self.ax_malus.errorbar(
                angs, Is_n, xerr=SIGMA_GONIO_DEG, yerr=yerr_total,
                fmt="o", color=COL_AZUL, ecolor=COL_AMBAR,
                markersize=5, lw=1.0, capsize=3, capthick=1,
                label=f"datos (N = {len(angs)}{etiq_fondo})")

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
            if self._fondo is not None:
                fo = self._fondo
                f.write("# --- calibracion de fondo (blanco: laser ON + "
                        "polarizadores lineales cruzados 90 deg) ---\n")
                f.write(f"# fondo_cps:          {fo['tasa_cps']:.4f}\n")
                f.write(f"# sigma_fondo_cps:    {fo['sigma_cps']:.4f}\n")
                f.write(f"# fondo_n_bins:       {fo['n_bins']}\n")
                f.write(f"# fondo_t_total_s:    {fo['t_total_s']:.3f}\n")
                f.write(f"# fondo_cuentas_tot:  {fo['total_cuentas']:.0f}\n")
                f.write(f"# fondo_t_iso:        {fo['t_iso']}\n")
                f.write(f"# currie_L_C_cuentas: {fo['L_C_cuentas']:.1f}\n")
                f.write(f"# currie_L_D_cuentas: {fo['L_D_cuentas']:.1f}\n")
                f.write("# correccion: I_norm = <(CPS - fondo_cps)/P>; "
                        "sigma_I_norm incluye sigma_fondo*<1/P> "
                        "(sistematica correlacionada)\n")
            else:
                f.write("# fondo_cps:          NO CALIBRADO "
                        "(I_norm sin correccion de fondo)\n")
            f.write("# columnas: angulo[deg]\tsigma_angulo[deg]\t"
                    "CPS\tsigma_CPS\tP[uW]\tsigma_P[uW]\t"
                    "I_norm[CPS/uW]\tsigma_I_norm\t"
                    "I_norm_bruto[CPS/uW]\tfondo_cps_restado\n")
            for p in self._puntos:
                f.write(
                    f"{p['angulo']:.2f}\t{p['sigma_angulo']:.3f}\t"
                    f"{p['cps_mean']:.4f}\t{p['cps_sem']:.4f}\t"
                    f"{p['pot_mean_uW']:.5f}\t{p['pot_sem_uW']:.5f}\t"
                    f"{p['I_norm']:.6e}\t{p['sigma_I_norm']:.6e}\t"
                    f"{p.get('I_norm_bruto', p['I_norm']):.6e}\t"
                    f"{p.get('fondo_cps', 0.0):.4f}\n")

        # Bins crudos del blanco (trazabilidad de la calibración de fondo)
        if self._fondo is not None:
            fo = self._fondo
            with open(base / "fondo_bins.txt", "w", encoding="utf-8") as f:
                f.write("# blanco: laser ON + polarizadores lineales "
                        "cruzados 90 deg\n")
                f.write(f"# fecha: {fo['t_iso']}\n")
                f.write(f"# bin_length_s = {fo['bin_length_s']:.6f}\n")
                f.write(f"# R_b = {fo['tasa_cps']:.4f} +- "
                        f"{fo['sigma_cps']:.4f} CPS\n")
                f.write("# bin\tcounts\tCPS\n")
                for j, c in enumerate(fo["bins"], start=1):
                    f.write(f"{j}\t{int(c)}\t{c / fo['bin_length_s']:.3f}\n")

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
        self._punto_es_fondo = False
        self._pv_activo = False
        self._limpiar_punto_vivo()
        self.canvas_malus.draw_idle()
        self.btn_tomar.setEnabled(self._iniciado)
        self.btn_fondo.setEnabled(self._iniciado)
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

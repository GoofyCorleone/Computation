"""
Interfaz gráfica (PyQt6) para el láser TOPTICA iBeam Smart.

Pestaña 1 — Control:
  encender/apagar, canales, temperatura, estabilidad, gráficas P(t) y T(t).

Pestaña 2 — FINE / SKILL:
  FINE (Feedback Induced Noise Eraser): off / modo A / modo B.
  SKILL (Speckle Killer): off / modo 1 / modo 2.
  Gráfica de potencia a largo plazo (minutos) y ruido de intensidad.

Log compartido, siempre visible bajo las dos pestañas.

Notas sobre el firmware:
  - Potencia de salida = SUMA de canales; dejar CH2 en 0 mW para control simple.
  - Setpoint TEC de fábrica 25 °C; cambio requiere contraseña de mantenimiento.
  - FINE: 'fine on/off', 'fine a' (modo A), 'fine b' (modo B). Estado: 'sta fine'.
  - SKILL: 'skill on/off', 'skill 1/2'. Estado no disponible en este firmware;
    se rastrea localmente en la app.
"""

import math
import sys
import threading
import time
from collections import deque

import serial
from serial.tools import list_ports
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QDoubleSpinBox, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSlider, QTabWidget, QVBoxLayout, QWidget,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ── Comunicación ────────────────────────────────────────────────────────────
BAUD              = 115200
TIMEOUT           = 1.5
PROMPT            = b"CMD> "
INTERVALO_POLL_MS = 700

# ── Estabilidad de potencia ─────────────────────────────────────────────────
VENTANA_ESTAB_S    = 8.0
UMBRAL_REL_ESTABLE = 0.005
MIN_MUESTRAS_EST   = 6
TAU_DIODO_S        = 25.0
TOL_TEMP_C         = 0.30

# ── Gráficas ────────────────────────────────────────────────────────────────
VENTANA_PLOT_S      = 60.0      # Tab 1: ventana deslizante (segundos)
VENTANA_LARGO_MIN   = 10.0      # Tab 2: historial de potencia (minutos)
VENTANA_RUIDO_S     = 30.0      # ventana para calcular ruido relativo

# ── Paleta dark (Catppuccin Mocha) ──────────────────────────────────────────
BG       = "#1e1e2e"
AX_BG    = "#181825"
FG       = "#cdd6f4"
GRID_COL = "#45475a"
C_POW    = "#89b4fa"   # azul
C_TEMP   = "#f38ba8"   # rosa
C_NOISE  = "#a6e3a1"   # verde
C_SET    = "#6c7086"   # gris (setpoints)


# ── Helpers de figura dark ───────────────────────────────────────────────────
def _make_fig(w=4.2, h=2.6):
    fig = Figure(figsize=(w, h), tight_layout=True, facecolor=BG)
    ax  = fig.add_subplot(111, facecolor=AX_BG)
    for sp in ax.spines.values():
        sp.set_color(GRID_COL)
    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.6)
    return fig, ax


# ────────────────────────────────────────────────────────────────────────────
# Detección automática de puerto
# ────────────────────────────────────────────────────────────────────────────
def candidatos_puerto() -> list[str]:
    candidatos = []
    for p in list_ports.comports():
        nombre = p.device
        desc   = (p.description  or "").lower()
        manuf  = (p.manufacturer or "").lower()
        if (
            "usbserial" in nombre.lower()
            or "usbmodem" in nombre.lower()
            or nombre.lower().startswith(("/dev/ttyusb", "/dev/ttyacm"))
            or nombre.upper().startswith("COM")
            or "ch340" in desc or "ftdi" in desc or "toptica" in manuf
            or ("usb" in desc and "serial" in desc)
        ):
            candidatos.append(nombre)
    return candidatos


def puerto_responde_ibeam(puerto: str) -> bool:
    try:
        with serial.Serial(puerto, BAUD, timeout=1.0) as s:
            time.sleep(0.8)
            s.reset_input_buffer()
            s.write(b"\r\n")
            return PROMPT in s.read_until(PROMPT, size=200)
    except Exception:
        return False


def detectar_puerto() -> str | None:
    for p in candidatos_puerto():
        if puerto_responde_ibeam(p):
            return p
    return None


# ────────────────────────────────────────────────────────────────────────────
# Estabilidad de potencia
# ────────────────────────────────────────────────────────────────────────────
class EstabilidadPotencia:
    def __init__(self, ventana_s: float = VENTANA_ESTAB_S):
        self.ventana_s = ventana_s
        self.muestras: deque[tuple[float, float]] = deque()

    def reset(self):
        self.muestras.clear()

    def agregar(self, t: float, p_mW: float):
        self.muestras.append((t, p_mW))
        t_min = t - self.ventana_s
        while self.muestras and self.muestras[0][0] < t_min:
            self.muestras.popleft()

    def _stats(self):
        n = len(self.muestras)
        if n < MIN_MUESTRAS_EST:
            return None
        ts = [t for t, _ in self.muestras]
        ps = [p for _, p in self.muestras]
        media = sum(ps) / n
        var   = sum((p - media) ** 2 for p in ps) / n
        std   = math.sqrt(var)
        t_med = sum(ts) / n
        num   = sum((ts[i] - t_med) * (ps[i] - media) for i in range(n))
        den   = sum((ts[i] - t_med) ** 2 for i in range(n))
        pend  = num / den if den > 1e-9 else 0.0
        return media, std, pend, n

    def estado(self) -> tuple[str, float | None, float | None]:
        s = self._stats()
        if s is None:
            return ("Sin datos", None, None)
        media, std, pend, _ = s
        if abs(media) < 1e-6:
            return ("Sin emisión", 0.0, None)
        rel_std   = std / abs(media)
        rel_drift = abs(pend * self.ventana_s) / abs(media)
        rel = max(rel_std, rel_drift)
        if rel < UMBRAL_REL_ESTABLE:
            return ("Estabilizado", 1.0, 0.0)
        eta  = min(600.0, max(0.0, TAU_DIODO_S * math.log(rel / UMBRAL_REL_ESTABLE)))
        frac = max(0.0, min(1.0, 1.0 - math.log10(rel / UMBRAL_REL_ESTABLE) / 2.0))
        txt  = "Estabilizando" if rel < 5 * UMBRAL_REL_ESTABLE else "Calentando"
        return (txt, frac, eta)

    def ruido_relativo_pct(self) -> float | None:
        """Desviación estándar relativa en %, proxy de ruido de intensidad."""
        s = self._stats()
        if s is None:
            return None
        media, std, _, _ = s
        if abs(media) < 1e-6:
            return None
        return 100.0 * std / abs(media)


# ────────────────────────────────────────────────────────────────────────────
# Driver serie
# ────────────────────────────────────────────────────────────────────────────
class IBeamDriver:
    def __init__(self):
        self.ser: serial.Serial | None = None
        self._lock = threading.Lock()

    def conectar(self, puerto: str):
        self.ser = serial.Serial(puerto, BAUD, timeout=TIMEOUT)
        time.sleep(0.8)
        self.ser.reset_input_buffer()
        self.ser.write(b"\r\n")
        self.ser.read_until(PROMPT)

    def desconectar(self):
        if self.ser and self.ser.is_open:
            try:
                self.enviar("la off")
            except Exception:
                pass
            self.ser.close()
        self.ser = None

    def conectado(self) -> bool:
        return self.ser is not None and self.ser.is_open

    def enviar(self, comando: str) -> str:
        if not self.conectado():
            raise RuntimeError("No conectado")
        with self._lock:
            self.ser.reset_input_buffer()
            self.ser.write((comando + "\r\n").encode())
            crudo = self.ser.read_until(PROMPT).decode(errors="replace")
        lineas = [l.strip() for l in crudo.splitlines()
                  if l.strip() and l.strip() != "CMD>"]
        if lineas and lineas[0].lower() == comando.lower():
            lineas = lineas[1:]
        return "\n".join(lineas)

    # ── emisión ──────────────────────────────────────────────────────────────
    def encender(self):  self.enviar("la on")
    def apagar(self):    self.enviar("la off")
    def estado(self) -> str: return self.enviar("sta la")

    def set_potencia(self, canal: int, mW: float):
        self.enviar(f"ch {canal} pow {mW:.3f}")

    def leer_niveles(self) -> dict[int, float]:
        resp = self.enviar("sh level pow")
        niveles = {}
        for linea in resp.splitlines():
            s = linea.strip()
            if s.upper().startswith("CH") and "PWR" in s.upper():
                try:
                    canal = int(s[2])
                    trozo = s.split(":", 1)[1].replace("mW", "").strip()
                    niveles[canal] = float(trozo)
                except (ValueError, IndexError):
                    pass
        return niveles

    def leer_potencia_uW(self) -> float:
        resp = self.enviar("sh pow")
        for linea in resp.splitlines():
            if "PIC" in linea and "uW" in linea:
                for tok in linea.replace("=", " ").split():
                    if tok.isdigit():
                        return float(tok)
        return 0.0

    # ── temperatura ──────────────────────────────────────────────────────────
    def leer_temperatura_C(self) -> float | None:
        resp = self.enviar("sh temp")
        for linea in resp.splitlines():
            s = linea.strip().upper()
            if s.startswith("TEMP") and "=" in s:
                try:
                    return float(s.split("=", 1)[1].replace("C", "").strip())
                except ValueError:
                    pass
        return None

    def leer_setpoint_temp_C(self) -> float | None:
        resp = self.enviar("sh syst data")
        for linea in resp.splitlines():
            s = linea.strip()
            if s.lower().startswith("tec setpoint") and "->" in s:
                try:
                    return float(s.split("->", 1)[1].replace("C", "").strip())
                except ValueError:
                    pass
        return None

    def leer_estado_tec(self) -> str:
        return self.enviar("sta tec").strip().upper()

    def set_temperatura_C(self, temp_C: float) -> str:
        return self.enviar(f"set temp {temp_C:.2f}")

    # ── FINE ─────────────────────────────────────────────────────────────────
    def leer_estado_fine(self) -> str:
        """Devuelve 'ON' u 'OFF'."""
        return self.enviar("sta fine").strip().upper()

    def set_fine(self, modo: str):
        """modo: 'off', 'a', 'b'."""
        if modo == "off":
            self.enviar("fine off")
        else:
            self.enviar("fine on")
            self.enviar(f"fine {modo}")

    # ── SKILL ────────────────────────────────────────────────────────────────
    def set_skill(self, modo: str):
        """modo: 'off', '1', '2'."""
        if modo == "off":
            self.enviar("skill off")
        else:
            self.enviar("skill on")
            self.enviar(f"skill {modo}")


# ────────────────────────────────────────────────────────────────────────────
# GUI principal
# ────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    # ── señales cross-thread ─────────────────────────────────────────────────
    sig_log           = pyqtSignal(str)
    sig_estado        = pyqtSignal(str)
    sig_potencia      = pyqtSignal(float)
    sig_niveles       = pyqtSignal(dict)
    sig_niveles_spin  = pyqtSignal(dict)
    sig_error_poll    = pyqtSignal(str)
    sig_detect_done   = pyqtSignal(str)
    sig_temperatura   = pyqtSignal(object)   # (temp, setpoint, tec)
    sig_estabilidad   = pyqtSignal(object)   # (texto, frac, eta)
    sig_setp_resp     = pyqtSignal(str, str)
    sig_fine_estado   = pyqtSignal(str)      # 'ON' / 'OFF'

    def __init__(self):
        super().__init__()
        self.setWindowTitle("iBeam Smart — Control")
        self.setMinimumSize(1020, 960)
        self.resize(1020, 1000)

        self.driver      = IBeamDriver()
        self.estabilidad = EstabilidadPotencia()
        self.estab_ruido = EstabilidadPotencia(ventana_s=VENTANA_RUIDO_S)

        # Historial gráficas
        self._t0:           float | None = None
        self._hist_pot:     deque[tuple[float, float]] = deque()  # (t_s, mW)
        self._hist_temp:    deque[tuple[float, float]] = deque()
        self._hist_pot_min: deque[tuple[float, float]] = deque()  # (t_min, mW)
        self._hist_noise:   deque[tuple[float, float]] = deque()  # (t_min, %)

        # Estado FINE/SKILL rastreado localmente
        self._fine_modo  = "off"   # 'off', 'a', 'b'
        self._skill_modo = "off"   # 'off', '1', '2'

        self._construir_ui()
        self._conectar_senales()

        self.timer_poll = QTimer(self)
        self.timer_poll.timeout.connect(self._poll_async)

    # ──────────────────────────────────────────────────────────────────────────
    # Construcción de la UI
    # ──────────────────────────────────────────────────────────────────────────
    def _construir_ui(self):
        raiz = QWidget()
        self.setCentralWidget(raiz)
        lay_raiz = QVBoxLayout(raiz)
        lay_raiz.setSpacing(6)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab { padding: 8px 20px; font-size: 13px; font-weight: bold; }
            QTabBar::tab:selected { color: #89b4fa; border-bottom: 2px solid #89b4fa; }
        """)
        lay_raiz.addWidget(self.tabs, 1)

        # Pestaña 1 ─ Control
        tab1 = QWidget()
        self.tabs.addTab(tab1, "  Control  ")
        self._construir_tab1(tab1)

        # Pestaña 2 ─ FINE / SKILL
        tab2 = QWidget()
        self.tabs.addTab(tab2, "  FINE / SKILL  ")
        self._construir_tab2(tab2)

        # Log compartido (siempre visible)
        gb_log = QGroupBox("Log")
        lay_log = QVBoxLayout(gb_log)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(600)
        self.log.setMaximumHeight(130)
        self.log.setStyleSheet(
            "background-color:#111;color:#0f0;font-family:Menlo;font-size:11px;")
        lay_log.addWidget(self.log)
        lay_raiz.addWidget(gb_log)

        # Detectar puerto al arrancar
        QTimer.singleShot(100, self._auto_detectar)

    # ── Pestaña 1: Control ────────────────────────────────────────────────────
    def _construir_tab1(self, parent: QWidget):
        lay = QVBoxLayout(parent)
        lay.setSpacing(6)

        # Conexión
        gb_con = QGroupBox("Conexión")
        g = QGridLayout(gb_con)
        g.addWidget(QLabel("Puerto:"), 0, 0)
        self.txt_puerto = QLineEdit("(detectando...)")
        g.addWidget(self.txt_puerto, 0, 1)
        self.btn_detectar = QPushButton("Auto-detectar")
        self.btn_detectar.clicked.connect(self._auto_detectar)
        g.addWidget(self.btn_detectar, 0, 2)
        self.btn_conectar = QPushButton("Conectar")
        self.btn_conectar.clicked.connect(self._toggle_conexion)
        g.addWidget(self.btn_conectar, 0, 3)
        lay.addWidget(gb_con)

        # Canales
        gb_ch = QGroupBox("Canales  (salida = SUMA de canales)")
        grid = QGridLayout(gb_ch)
        grid.addWidget(QLabel("<b>Canal</b>"),         0, 0)
        grid.addWidget(QLabel("<b>Setpoint [mW]</b>"), 0, 1)
        grid.addWidget(QLabel(""),                      0, 2)
        grid.addWidget(QLabel("<b>Nivel actual</b>"),   0, 3)
        self.spn_pow   = {}
        self.lbl_nivel = {}
        for i, canal in enumerate([1, 2], 1):
            grid.addWidget(QLabel(f"CH{canal}"), i, 0)
            spn = QDoubleSpinBox()
            spn.setRange(0.0, 200.0); spn.setDecimals(3)
            spn.setSingleStep(0.1); spn.setSuffix(" mW")
            spn.setValue(0.0 if canal == 2 else 1.0)
            grid.addWidget(spn, i, 1)
            btn = QPushButton("Aplicar")
            btn.clicked.connect(lambda _, c=canal, s=spn: self._aplicar_potencia(c, s.value()))
            grid.addWidget(btn, i, 2)
            lbl = QLabel("—"); lbl.setStyleSheet("font-family:Menlo;color:#444;")
            grid.addWidget(lbl, i, 3)
            self.spn_pow[canal] = spn; self.lbl_nivel[canal] = lbl
        lay.addWidget(gb_ch)

        # Temperatura
        gb_temp = QGroupBox("Temperatura del diodo (TEC)")
        gt = QGridLayout(gb_temp)
        gt.addWidget(QLabel("Setpoint:"), 0, 0)
        self.spn_temp = QDoubleSpinBox()
        self.spn_temp.setRange(15.0, 40.0); self.spn_temp.setDecimals(2)
        self.spn_temp.setSingleStep(0.5); self.spn_temp.setSuffix(" °C")
        self.spn_temp.setValue(25.0)
        gt.addWidget(self.spn_temp, 0, 1)
        btn_t = QPushButton("Aplicar")
        btn_t.clicked.connect(self._aplicar_temperatura)
        gt.addWidget(btn_t, 0, 2)
        self.lbl_temp_actual = QLabel("Actual: —")
        self.lbl_temp_actual.setStyleSheet("font-weight:bold;font-size:13px;")
        gt.addWidget(self.lbl_temp_actual, 1, 0, 1, 2)
        self.lbl_tec = QLabel("TEC: —")
        self.lbl_tec.setStyleSheet("font-weight:bold;font-size:13px;")
        gt.addWidget(self.lbl_tec, 1, 2)
        self.lbl_estab_temp = QLabel("Estabilidad térmica: —")
        self.lbl_estab_temp.setTextFormat(Qt.TextFormat.RichText)
        gt.addWidget(self.lbl_estab_temp, 2, 0, 1, 3)
        lay.addWidget(gb_temp)

        # Emisión
        gb_em = QGroupBox("Emisión")
        le = QVBoxLayout(gb_em)
        fila_btns = QHBoxLayout()
        self.btn_on  = QPushButton("Encender (LA ON)")
        self.btn_off = QPushButton("Apagar (LA OFF)")
        self.btn_on.setStyleSheet(
            "background-color:#3a7d3a;color:white;font-weight:bold;padding:8px;")
        self.btn_off.setStyleSheet(
            "background-color:#7d3a3a;color:white;font-weight:bold;padding:8px;")
        self.btn_on.clicked.connect(self._encender)
        self.btn_off.clicked.connect(self._apagar)
        self.btn_on.setEnabled(False); self.btn_off.setEnabled(False)
        fila_btns.addWidget(self.btn_on); fila_btns.addWidget(self.btn_off)
        le.addLayout(fila_btns)

        fila_est = QHBoxLayout()
        self.lbl_estado   = QLabel("Estado: —")
        self.lbl_potencia = QLabel("Potencia medida: —")
        for lbl in (self.lbl_estado, self.lbl_potencia):
            lbl.setStyleSheet("font-weight:bold;font-size:13px;")
        fila_est.addWidget(self.lbl_estado); fila_est.addWidget(self.lbl_potencia)
        le.addLayout(fila_est)

        fila_stab = QHBoxLayout()
        self.lbl_estab_pow = QLabel("Estabilidad: —")
        self.lbl_estab_pow.setStyleSheet("font-weight:bold;font-size:13px;")
        self.lbl_estab_pow.setTextFormat(Qt.TextFormat.RichText)
        fila_stab.addWidget(self.lbl_estab_pow)
        self.lbl_eta = QLabel("ETA: —")
        fila_stab.addWidget(self.lbl_eta)
        le.addLayout(fila_stab)

        self.bar_estab = QProgressBar()
        self.bar_estab.setRange(0, 100); self.bar_estab.setValue(0)
        self.bar_estab.setFormat("%p% estabilizado")
        le.addWidget(self.bar_estab)
        lay.addWidget(gb_em)

        # Gráficas en tiempo real ─ Tab 1 (60 s)
        fila_plots = QHBoxLayout(); fila_plots.setSpacing(8)

        gb_pp = QGroupBox("Potencia en tiempo real")
        lpp = QVBoxLayout(gb_pp)
        fig_p, self.ax_pot = _make_fig(4.8, 3.2)
        self.ax_pot.set_xlabel("t [s]"); self.ax_pot.set_ylabel("P [mW]")
        self.canvas_pot = FigureCanvas(fig_p)
        self.line_pot,     = self.ax_pot.plot([], [], color=C_POW,  lw=1.2,
                                               marker=".", ms=3)
        self.line_pot_set  = self.ax_pot.axhline(0, color=C_SET, ls="--",
                                                  lw=0.8, visible=False)
        lpp.addWidget(self.canvas_pot); fila_plots.addWidget(gb_pp, 1)

        gb_pt = QGroupBox("Temperatura en tiempo real")
        lpt = QVBoxLayout(gb_pt)
        fig_t, self.ax_temp = _make_fig(4.8, 3.2)
        self.ax_temp.set_xlabel("t [s]"); self.ax_temp.set_ylabel("T [°C]")
        self.canvas_temp = FigureCanvas(fig_t)
        self.line_temp,    = self.ax_temp.plot([], [], color=C_TEMP, lw=1.2,
                                                marker=".", ms=3)
        self.line_temp_set = self.ax_temp.axhline(25.0, color=C_SET, ls="--", lw=0.8)
        lpt.addWidget(self.canvas_temp); fila_plots.addWidget(gb_pt, 1)
        lay.addLayout(fila_plots)

    # ── Pestaña 2: FINE / SKILL ───────────────────────────────────────────────
    def _construir_tab2(self, parent: QWidget):
        lay = QVBoxLayout(parent)
        lay.setSpacing(8)

        fila_ctrl = QHBoxLayout()
        fila_ctrl.setSpacing(10)

        # ── FINE ──────────────────────────────────────────────────────────────
        gb_fine = QGroupBox("FINE — Feedback Induced Noise Eraser")
        lf = QVBoxLayout(gb_fine)
        lf.setSpacing(4)

        self.sld_fine = QSlider(Qt.Orientation.Horizontal)
        self.sld_fine.setRange(0, 2)
        self.sld_fine.setValue(0)
        self.sld_fine.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sld_fine.setTickInterval(1)
        self.sld_fine.setSingleStep(1)
        self.sld_fine.setPageStep(1)
        self.sld_fine.valueChanged.connect(self._on_fine_slider)
        lf.addWidget(self.sld_fine)

        fila_lf = QHBoxLayout()
        fila_lf.addWidget(QLabel("Apagado"), 0, Qt.AlignmentFlag.AlignLeft)
        fila_lf.addWidget(QLabel("Modo A"), 0, Qt.AlignmentFlag.AlignCenter)
        fila_lf.addWidget(QLabel("Modo B"), 0, Qt.AlignmentFlag.AlignRight)
        lf.addLayout(fila_lf)

        self.lbl_fine_estado = QLabel("Estado FINE: —")
        self.lbl_fine_estado.setStyleSheet(
            "color:#89b4fa;font-weight:bold;margin-top:4px;")
        lf.addWidget(self.lbl_fine_estado)

        frm_help_f = QFrame()
        frm_help_f.setFrameShape(QFrame.Shape.StyledPanel)
        frm_help_f.setStyleSheet(
            "background-color:#2a2a3e;border:1px solid #45475a;border-radius:4px;")
        lf_hf = QVBoxLayout(frm_help_f)
        lf_hf.setContentsMargins(8, 6, 8, 6)
        lbl_hf = QLabel(
            "<b style='color:#89b4fa;'>HELP — FINE</b><br>"
            "<b>Modo A:</b> lazo de retroalimentación de baja frecuencia (≲ 100 Hz).<br>"
            "&nbsp;&nbsp;Corrige ruido mecánico y de la corriente de bombeo.<br>"
            "<b>Modo B:</b> lazo extendido hasta ≈ 10 MHz.<br>"
            "&nbsp;&nbsp;Reduce fluctuaciones rápidas de amplitud; ideal cuando "
            "la coherencia temporal es crítica."
        )
        lbl_hf.setWordWrap(True)
        lbl_hf.setStyleSheet("font-size:11px;color:#cdd6f4;")
        lf_hf.addWidget(lbl_hf)
        lf.addWidget(frm_help_f)

        fila_ctrl.addWidget(gb_fine, 1)

        # ── SKILL ─────────────────────────────────────────────────────────────
        gb_skill = QGroupBox("SKILL — Speckle Killer")
        ls = QVBoxLayout(gb_skill)
        ls.setSpacing(4)

        self.sld_skill = QSlider(Qt.Orientation.Horizontal)
        self.sld_skill.setRange(0, 2)
        self.sld_skill.setValue(0)
        self.sld_skill.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sld_skill.setTickInterval(1)
        self.sld_skill.setSingleStep(1)
        self.sld_skill.setPageStep(1)
        self.sld_skill.valueChanged.connect(self._on_skill_slider)
        ls.addWidget(self.sld_skill)

        fila_ls = QHBoxLayout()
        fila_ls.addWidget(QLabel("Apagado"), 0, Qt.AlignmentFlag.AlignLeft)
        fila_ls.addWidget(QLabel("Modo 1"), 0, Qt.AlignmentFlag.AlignCenter)
        fila_ls.addWidget(QLabel("Modo 2"), 0, Qt.AlignmentFlag.AlignRight)
        ls.addLayout(fila_ls)

        self.lbl_skill_estado = QLabel("Estado SKILL: Apagado")
        self.lbl_skill_estado.setStyleSheet(
            "color:#a6e3a1;font-weight:bold;margin-top:4px;")
        ls.addWidget(self.lbl_skill_estado)

        frm_help_s = QFrame()
        frm_help_s.setFrameShape(QFrame.Shape.StyledPanel)
        frm_help_s.setStyleSheet(
            "background-color:#2a2a3e;border:1px solid #45475a;border-radius:4px;")
        ls_hs = QVBoxLayout(frm_help_s)
        ls_hs.setContentsMargins(8, 6, 8, 6)
        lbl_hs = QLabel(
            "<b style='color:#a6e3a1;'>HELP — SKILL</b><br>"
            "<b>Modo 1:</b> modulación de fase de baja amplitud (~π/4).<br>"
            "&nbsp;&nbsp;Reducción leve de speckle; mínimo impacto en coherencia temporal.<br>"
            "<b>Modo 2:</b> modulación de fase de mayor amplitud (~π).<br>"
            "&nbsp;&nbsp;Máxima supresión de speckle al acoplar a fibra multimodo."
        )
        lbl_hs.setWordWrap(True)
        lbl_hs.setStyleSheet("font-size:11px;color:#cdd6f4;")
        ls_hs.addWidget(lbl_hs)
        ls.addWidget(frm_help_s)

        fila_ctrl.addWidget(gb_skill, 1)

        lay.addLayout(fila_ctrl)

        # Gráficas dark ─ Tab 2
        fila_gr = QHBoxLayout(); fila_gr.setSpacing(8)

        gb_long = QGroupBox(f"Potencia — últimos {int(VENTANA_LARGO_MIN)} min")
        ll = QVBoxLayout(gb_long)
        fig_l, self.ax_long = _make_fig(4.4, 3.2)
        self.ax_long.set_xlabel("t [min]")
        self.ax_long.set_ylabel("P [mW]")
        self.canvas_long = FigureCanvas(fig_l)
        fig_l.set_facecolor(BG)
        self.line_long,    = self.ax_long.plot([], [], color=C_POW, lw=1.4,
                                                marker=".", ms=3)
        self.line_long_set = self.ax_long.axhline(0, color=C_SET, ls="--",
                                                   lw=0.8, visible=False)
        ll.addWidget(self.canvas_long)
        fila_gr.addWidget(gb_long, 1)

        gb_noise = QGroupBox("Ruido de intensidad — proxy de coherencia de amplitud")
        ln = QVBoxLayout(gb_noise)
        fig_n, self.ax_noise = _make_fig(4.4, 3.2)
        self.ax_noise.set_xlabel("t [min]")
        self.ax_noise.set_ylabel("Ruido relativo σ/μ [%]")
        self.canvas_noise = FigureCanvas(fig_n)
        fig_n.set_facecolor(BG)
        self.line_noise, = self.ax_noise.plot([], [], color=C_NOISE, lw=1.4,
                                               marker=".", ms=3)
        self.ax_noise.axhline(0.5, color=C_SET, ls="--", lw=0.8,
                              label="Umbral estabilidad (0.5 %)")
        self.ax_noise.legend(loc="upper right",
                             labelcolor=FG, facecolor=AX_BG,
                             edgecolor=GRID_COL, fontsize=7)
        ln.addWidget(self.canvas_noise)
        fila_gr.addWidget(gb_noise, 1)

        lay.addLayout(fila_gr)

    # ──────────────────────────────────────────────────────────────────────────
    # Conexión de señales
    # ──────────────────────────────────────────────────────────────────────────
    def _conectar_senales(self):
        self.sig_log.connect(self._log)
        self.sig_estado.connect(lambda s: self.lbl_estado.setText(f"Estado: {s}"))
        self.sig_potencia.connect(self._on_potencia)
        self.sig_niveles.connect(self._actualizar_niveles)
        self.sig_niveles_spin.connect(self._actualizar_spinboxes)
        self.sig_error_poll.connect(self._manejar_error_poll)
        self.sig_detect_done.connect(self._on_detect_done)
        self.sig_temperatura.connect(self._on_temperatura)
        self.sig_estabilidad.connect(self._on_estabilidad)
        self.sig_setp_resp.connect(self._on_setp_resp)
        self.sig_fine_estado.connect(self._on_fine_estado)

    # ──────────────────────────────────────────────────────────────────────────
    # Handlers de señales
    # ──────────────────────────────────────────────────────────────────────────
    def _log(self, msg: str):
        self.log.appendPlainText(msg)

    def _on_potencia(self, p_uW: float):
        p_mW = p_uW / 1000.0
        self.lbl_potencia.setText(f"Potencia medida: {p_mW:.3f} mW")

        ahora = time.time()
        if self._t0 is None:
            self._t0 = ahora
        t_s   = ahora - self._t0
        t_min = t_s / 60.0

        # Estabilidad
        self.estabilidad.agregar(ahora, p_mW)
        self.estab_ruido.agregar(ahora, p_mW)
        self.sig_estabilidad.emit(self.estabilidad.estado())

        # Historial Tab 1 (60 s)
        self._hist_pot.append((t_s, p_mW))
        self._recortar(self._hist_pot, t_s, VENTANA_PLOT_S)
        self._refrescar_pot()

        # Historial Tab 2 (minutos)
        self._hist_pot_min.append((t_min, p_mW))
        self._recortar(self._hist_pot_min, t_min, VENTANA_LARGO_MIN)

        ruido = self.estab_ruido.ruido_relativo_pct()
        if ruido is not None:
            self._hist_noise.append((t_min, ruido))
            self._recortar(self._hist_noise, t_min, VENTANA_LARGO_MIN)

        self._refrescar_largo()
        self._refrescar_noise()

    def _on_estabilidad(self, datos):
        texto, frac, eta = datos
        color = {"Estabilizado": "#3a7d3a", "Estabilizando": "#b8860b",
                 "Calentando": "#a04020"}.get(texto, "#555")
        self.lbl_estab_pow.setText(
            f"Estabilidad: <span style='color:{color}'>{texto}</span>")
        self.bar_estab.setValue(int(round(frac * 100)) if frac is not None else 0)
        if eta is None or texto in ("Sin datos", "Sin emisión"):
            self.lbl_eta.setText("ETA: —")
        elif eta < 1:
            self.lbl_eta.setText("ETA: < 1 s")
        elif eta < 60:
            self.lbl_eta.setText(f"ETA: ~ {eta:.0f} s")
        else:
            self.lbl_eta.setText(f"ETA: ~ {eta/60:.1f} min")

    def _on_temperatura(self, datos):
        temp, setpoint, tec = datos
        self.lbl_temp_actual.setText(f"Actual: {temp:.2f} °C" if temp is not None else "Actual: —")
        if setpoint is not None:
            self.spn_temp.blockSignals(True)
            self.spn_temp.setValue(setpoint)
            self.spn_temp.blockSignals(False)
            self.line_temp_set.set_ydata([setpoint, setpoint])
        self.lbl_tec.setText(f"TEC: {tec or '—'}")

        if temp is None or setpoint is None:
            self.lbl_estab_temp.setText("Estabilidad térmica: —")
        else:
            delta = temp - setpoint
            ok = abs(delta) < TOL_TEMP_C
            color = "#3a7d3a" if ok else "#a04020"
            etiq  = "Térmica estable" if ok else "Térmica fuera de tolerancia"
            self.lbl_estab_temp.setText(
                f"<span style='color:{color};font-weight:bold;'>{etiq}</span> "
                f"(Δ = {delta:+.2f} °C)")

        if temp is not None:
            ahora = time.time()
            if self._t0 is None:
                self._t0 = ahora
            t_s = ahora - self._t0
            self._hist_temp.append((t_s, temp))
            self._recortar(self._hist_temp, t_s, VENTANA_PLOT_S)
            self._refrescar_temp(setpoint)

    def _on_setp_resp(self, pedido: str, respuesta: str):
        if "access restricted" in respuesta.lower():
            QMessageBox.warning(
                self, "Setpoint restringido",
                f"El firmware restringe el acceso al setpoint del TEC sin contraseña.\n"
                f"Setpoint de fábrica: 25.0 °C.\n\nRespuesta: {respuesta}")
        elif respuesta:
            self._log(f"set temp {pedido} → {respuesta}")
        else:
            self._log(f"set temp {pedido} → OK")

    def _on_fine_estado(self, estado: str):
        self.lbl_fine_estado.setText(f"Estado FINE: {estado}")
        if estado == "OFF" and self._fine_modo != "off":
            self._fine_modo = "off"
            self.sld_fine.blockSignals(True)
            self.sld_fine.setValue(0)
            self.sld_fine.blockSignals(False)

    def _on_fine_slider(self, valor: int):
        modo = {0: "off", 1: "a", 2: "b"}[valor]
        if modo == self._fine_modo or not self.driver.conectado():
            return
        self._fine_modo = modo
        def _t():
            try:
                self.driver.set_fine(modo)
                etiqueta = {"off": "OFF", "a": "Modo A", "b": "Modo B"}[modo]
                self.sig_log.emit(f">>> fine {modo}  OK  ({etiqueta})")
            except Exception as e:
                self.sig_log.emit(f"!!! fine {modo}  ERROR: {e}")
        threading.Thread(target=_t, daemon=True).start()

    def _on_skill_slider(self, valor: int):
        modo = {0: "off", 1: "1", 2: "2"}[valor]
        if modo == self._skill_modo or not self.driver.conectado():
            return
        self._skill_modo = modo
        etiqueta = {"off": "Apagado", "1": "Modo 1", "2": "Modo 2"}[modo]
        self.lbl_skill_estado.setText(f"Estado SKILL: {etiqueta}")
        def _t():
            try:
                self.driver.set_skill(modo)
                self.sig_log.emit(f">>> skill {modo}  OK  ({etiqueta})")
            except Exception as e:
                self.sig_log.emit(f"!!! skill {modo}  ERROR: {e}")
        threading.Thread(target=_t, daemon=True).start()

    def _actualizar_niveles(self, niveles: dict):
        for canal, mW in niveles.items():
            if canal in self.lbl_nivel:
                self.lbl_nivel[canal].setText(f"{mW:.3f} mW")
        if niveles:
            total = sum(niveles.values())
            self.line_pot_set.set_ydata([total, total])
            self.line_pot_set.set_visible(total > 0.0)
            self.line_long_set.set_ydata([total, total])
            self.line_long_set.set_visible(total > 0.0)

    def _actualizar_spinboxes(self, niveles: dict):
        for canal, mW in niveles.items():
            if canal in self.spn_pow:
                self.spn_pow[canal].blockSignals(True)
                self.spn_pow[canal].setValue(mW)
                self.spn_pow[canal].blockSignals(False)

    # ──────────────────────────────────────────────────────────────────────────
    # Gráficas helpers
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _recortar(hist: deque, t_actual: float, ventana: float):
        t_min = t_actual - ventana
        while hist and hist[0][0] < t_min:
            hist.popleft()

    def _refrescar_pot(self):
        if not self._hist_pot:
            return
        xs = [t for t, _ in self._hist_pot]
        ys = [p for _, p in self._hist_pot]
        self.line_pot.set_data(xs, ys)
        t_max = xs[-1]
        self.ax_pot.set_xlim(max(0.0, t_max - VENTANA_PLOT_S), max(t_max, 1.0))
        y_lo, y_hi = min(ys), max(ys)
        if self.line_pot_set.get_visible():
            sp = float(self.line_pot_set.get_ydata()[0])
            y_lo, y_hi = min(y_lo, sp), max(y_hi, sp)
        mg = max(0.05 * max(abs(y_hi), 1e-3), 0.02)
        self.ax_pot.set_ylim(y_lo - mg, y_hi + mg)
        self.canvas_pot.draw_idle()

    def _refrescar_temp(self, setpoint):
        if not self._hist_temp:
            return
        xs = [t for t, _ in self._hist_temp]
        ys = [T for _, T in self._hist_temp]
        self.line_temp.set_data(xs, ys)
        t_max = xs[-1]
        self.ax_temp.set_xlim(max(0.0, t_max - VENTANA_PLOT_S), max(t_max, 1.0))
        centro = setpoint if setpoint is not None else (min(ys) + max(ys)) / 2
        amp    = max(1.0, max(ys) - min(ys))
        self.ax_temp.set_ylim(centro - amp, centro + amp)
        self.canvas_temp.draw_idle()

    def _refrescar_largo(self):
        if not self._hist_pot_min:
            return
        xs = [t for t, _ in self._hist_pot_min]
        ys = [p for _, p in self._hist_pot_min]
        self.line_long.set_data(xs, ys)
        t_max = xs[-1]
        self.ax_long.set_xlim(max(0.0, t_max - VENTANA_LARGO_MIN), max(t_max, 0.1))
        y_lo, y_hi = min(ys), max(ys)
        if self.line_long_set.get_visible():
            sp = float(self.line_long_set.get_ydata()[0])
            y_lo, y_hi = min(y_lo, sp), max(y_hi, sp)
        mg = max(0.05 * max(abs(y_hi), 1e-3), 0.02)
        self.ax_long.set_ylim(y_lo - mg, y_hi + mg)
        self.canvas_long.draw_idle()

    def _refrescar_noise(self):
        if not self._hist_noise:
            return
        xs = [t for t, _ in self._hist_noise]
        ys = [n for _, n in self._hist_noise]
        self.line_noise.set_data(xs, ys)
        t_max = xs[-1]
        self.ax_noise.set_xlim(max(0.0, t_max - VENTANA_LARGO_MIN), max(t_max, 0.1))
        y_hi = max(max(ys), 0.5) * 1.2
        self.ax_noise.set_ylim(0, max(y_hi, 1.0))
        self.canvas_noise.draw_idle()

    def _reset_graficos(self):
        self._t0 = None
        for h in (self._hist_pot, self._hist_temp,
                  self._hist_pot_min, self._hist_noise):
            h.clear()
        self.line_pot.set_data([], [])
        self.line_temp.set_data([], [])
        self.line_long.set_data([], [])
        self.line_noise.set_data([], [])
        self.line_pot_set.set_visible(False)
        self.line_long_set.set_visible(False)
        self.ax_pot.set_xlim(0, VENTANA_PLOT_S)
        self.ax_pot.set_ylim(0, 1)
        self.ax_temp.set_xlim(0, VENTANA_PLOT_S)
        self.ax_temp.set_ylim(24, 26)
        self.ax_long.set_xlim(0, VENTANA_LARGO_MIN)
        self.ax_long.set_ylim(0, 1)
        self.ax_noise.set_xlim(0, VENTANA_LARGO_MIN)
        self.ax_noise.set_ylim(0, 2)
        for c in (self.canvas_pot, self.canvas_temp,
                  self.canvas_long, self.canvas_noise):
            c.draw_idle()

    # ──────────────────────────────────────────────────────────────────────────
    # Conexión / detección
    # ──────────────────────────────────────────────────────────────────────────
    def _auto_detectar(self, conectar_auto: bool = True):
        self._auto_connect_pendiente = conectar_auto
        self.btn_detectar.setEnabled(False)
        self.btn_detectar.setText("Buscando...")
        self._log("Buscando puerto del iBeam Smart ...")
        def _t():
            self.sig_detect_done.emit(detectar_puerto() or "")
        threading.Thread(target=_t, daemon=True).start()

    def _on_detect_done(self, puerto: str):
        self.btn_detectar.setEnabled(True)
        self.btn_detectar.setText("Auto-detectar")
        if puerto:
            self.txt_puerto.setText(puerto)
            self._log(f"Puerto detectado: {puerto}")
            if getattr(self, "_auto_connect_pendiente", False) and not self.driver.conectado():
                self._conectar()
        else:
            self.txt_puerto.setText("")
            self._log("No se encontró iBeam Smart.")

    def _toggle_conexion(self):
        if self.driver.conectado():
            self._desconectar()
        else:
            self._conectar()

    def _conectar(self):
        puerto = self.txt_puerto.text().strip()
        if not puerto:
            QMessageBox.warning(self, "Sin puerto",
                                "No hay puerto. Usa 'Auto-detectar'.")
            return
        try:
            self.driver.conectar(puerto)
            self._log(f"Conectado a {puerto} @ {BAUD}")
            self.btn_conectar.setText("Desconectar")
            self.btn_on.setEnabled(True)
            self.btn_off.setEnabled(True)
            self.timer_poll.start(INTERVALO_POLL_MS)
            self.estabilidad.reset()
            self.estab_ruido.reset()
            self._reset_graficos()
            threading.Thread(target=self._sincronizar, daemon=True).start()
        except Exception as e:
            QMessageBox.critical(self, "Error de conexión", str(e))

    def _sincronizar(self):
        niveles  = self.driver.leer_niveles()
        # La potencia óptica de salida del iBeam Smart = SUMA de los dos
        # canales (CH1 + CH2). Si CH2 quedó con un valor alto de una sesión
        # anterior, los cambios en CH1 parecen no surtir efecto (la salida
        # queda 'pegada' al valor de CH2). Para evitar esa confusión,
        # forzamos CH2 = 0 al conectar y dejamos a CH1 como único control.
        if niveles.get(2, 0.0) > 0.0:
            self.sig_log.emit(
                f"[AVISO] CH2 = {niveles[2]:.3f} mW (de una sesión previa). "
                "Forzando CH2 = 0 para que CH1 controle la potencia.")
            try:
                self.driver.set_potencia(2, 0.0)
                niveles_recheq = self.driver.leer_niveles()
                if niveles_recheq:
                    niveles = niveles_recheq
            except Exception as e:
                self.sig_log.emit(f"!!! No se pudo poner CH2 = 0: {e}")
        setpoint = self.driver.leer_setpoint_temp_C()
        temp     = self.driver.leer_temperatura_C()
        tec      = self.driver.leer_estado_tec()
        fine     = self.driver.leer_estado_fine()
        self.sig_niveles.emit(niveles)
        self.sig_niveles_spin.emit(niveles)
        self.sig_temperatura.emit((temp, setpoint, tec))
        self.sig_fine_estado.emit(fine)
        self.sig_log.emit(f"Niveles: {niveles}  TEC: {tec} {setpoint}°C  FINE: {fine}")

    def _desconectar(self):
        self.timer_poll.stop()
        try:
            self.driver.desconectar()
        except Exception as e:
            self._log(f"Error al desconectar: {e}")
        self.btn_conectar.setText("Conectar")
        self.btn_on.setEnabled(False)
        self.btn_off.setEnabled(False)
        for lbl in (self.lbl_estado, self.lbl_potencia, self.lbl_temp_actual,
                    self.lbl_tec):
            lbl.setText(lbl.text().split(":")[0] + ": —")
        self.lbl_estab_temp.setText("Estabilidad térmica: —")
        self.lbl_estab_pow.setText("Estabilidad: —")
        self.lbl_eta.setText("ETA: —")
        self.bar_estab.setValue(0)
        for lbl in self.lbl_nivel.values():
            lbl.setText("—")
        self.lbl_fine_estado.setText("Estado FINE: —")
        self.lbl_skill_estado.setText("Estado SKILL: Apagado")
        self.sld_fine.blockSignals(True); self.sld_fine.setValue(0); self.sld_fine.blockSignals(False)
        self.sld_skill.blockSignals(True); self.sld_skill.setValue(0); self.sld_skill.blockSignals(False)
        self._fine_modo = "off"; self._skill_modo = "off"
        self.estabilidad.reset()
        self.estab_ruido.reset()
        self._reset_graficos()
        self._log("Desconectado")

    # ──────────────────────────────────────────────────────────────────────────
    # Acciones
    # ──────────────────────────────────────────────────────────────────────────
    def _encender(self):
        self.estabilidad.reset()
        self.estab_ruido.reset()
        self._ejecutar(self.driver.encender, "la on")

    def _apagar(self):
        self._ejecutar(self.driver.apagar, "la off")

    def _aplicar_potencia(self, canal: int, mW: float):
        self.estabilidad.reset()
        self.estab_ruido.reset()
        self._ejecutar(lambda: self.driver.set_potencia(canal, mW),
                       f"ch {canal} pow {mW:.3f}")

    def _aplicar_temperatura(self):
        if not self.driver.conectado():
            return
        v = self.spn_temp.value()
        def _t():
            try:
                resp = self.driver.set_temperatura_C(v)
            except Exception as e:
                resp = f"ERROR: {e}"
            self.sig_setp_resp.emit(f"{v:.2f}", resp)
        threading.Thread(target=_t, daemon=True).start()

    def _ejecutar(self, accion, etiqueta: str):
        if not self.driver.conectado():
            self._log("!!! No conectado")
            return
        def _t():
            try:
                accion()
                self.sig_log.emit(f">>> {etiqueta}  OK")
            except Exception as e:
                self.sig_log.emit(f"!!! {etiqueta}  ERROR: {e}")
        threading.Thread(target=_t, daemon=True).start()

    # ──────────────────────────────────────────────────────────────────────────
    # Polling
    # ──────────────────────────────────────────────────────────────────────────
    def _poll_async(self):
        if not self.driver.conectado():
            return
        def _t():
            try:
                estado   = self.driver.estado()
                potencia = self.driver.leer_potencia_uW()
                niveles  = self.driver.leer_niveles()
                temp     = self.driver.leer_temperatura_C()
                tec      = self.driver.leer_estado_tec()
                fine     = self.driver.leer_estado_fine()
                self.sig_estado.emit(estado)
                self.sig_potencia.emit(potencia)
                self.sig_niveles.emit(niveles)
                self.sig_temperatura.emit((temp, self.spn_temp.value(), tec))
                self.sig_fine_estado.emit(fine)
            except Exception as e:
                self.sig_error_poll.emit(str(e))
        threading.Thread(target=_t, daemon=True).start()

    def _manejar_error_poll(self, msg: str):
        self._log(f"Poll error: {msg}")
        self.timer_poll.stop()

    def closeEvent(self, event):
        self.timer_poll.stop()
        try:
            self.driver.desconectar()
        except Exception:
            pass
        event.accept()


# ────────────────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

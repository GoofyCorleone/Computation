"""
Interfaz gráfica (PyQt6) para el contador de fotones Thorlabs SPCM50A/M.

Pestaña 1 — Medición:
  Display digital de tasa de conteo (cps / kcps / Mcps), tiempo de
  integración configurable, modo continuo / única adquisición, umbral
  de alerta, estadísticas en vivo y gráfica P(t) deslizante (60 s).

Pestaña 2 — Análisis:
  Histograma de tasas de conteo, traza de largo plazo (10 min),
  estadísticas globales de sesión y exportación CSV.

Log compartido siempre visible bajo las pestañas.

Comunicación:
  - Modo real: USB HID via biblioteca `hid` (pip install hid).
    El SPCM50A/M se identifica con VID=0x1313 (Thorlabs).
    Para encontrar el PID exacto ejecutar:
      python -c "import hid; [print(hex(d['vendor_id']),
        hex(d['product_id']), d['product_string'])
        for d in hid.enumerate(0x1313,0)]"
  - Modo simulación: activo automáticamente si no se detecta el
    dispositivo. Genera fotocuentas Poisson con deriva lenta y
    cuentas oscuras realistas.

Protocolo HID — PENDIENTE:
  Ver DriverSPCM.leer_conteos(). Capturar tráfico USB con Wireshark
  (USBPcap en Windows) para determinar el formato exacto de los
  mensajes HID.

Notas de firmware SPCM50A/M:
  - Máx. tasa de conteo: ~50 Mcps.
  - Dead time: ≈ 22 ns.
  - Cuentas oscuras típicas: < 50 cps (Si APD refrigerado).
  - Rango espectral: 400 – 1000 nm.
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
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QComboBox, QDoubleSpinBox,
    QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QRadioButton, QSizePolicy, QTabWidget,
    QVBoxLayout, QWidget,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    import hid as _hid
    HID_DISPONIBLE = True
except ImportError:
    HID_DISPONIBLE = False

# ── Identificadores USB Thorlabs ─────────────────────────────────────────────
THORLABS_VID = 0x1313
SPCM_PID     = 0x804A   # actualizar tras ejecutar el snippet del docstring

# ── Tiempos de integración disponibles ──────────────────────────────────────
TIEMPOS_INTEG_MS = [1, 2, 5, 10, 20, 50, 100, 200, 500,
                    1000, 2000, 5000, 10000]
INTEG_DEFAULT_MS = 100

# ── Ventanas de gráficas ─────────────────────────────────────────────────────
VENTANA_PLOT_S  = 60.0    # Tab 1: ventana deslizante (s)
VENTANA_LARGO_S = 600.0   # Tab 2: traza de largo plazo (10 min)

# ── Paleta dark (Catppuccin Mocha) ───────────────────────────────────────────
BG       = "#1e1e2e"
AX_BG    = "#181825"
FG       = "#cdd6f4"
GRID_COL = "#45475a"
C_MAIN   = "#89b4fa"   # azul   — curva principal
C_HIST   = "#cba6f7"   # lila   — histograma
C_ALERT  = "#f38ba8"   # rosa   — umbral / alerta
C_STAT   = "#a6e3a1"   # verde  — estadísticas
C_SET    = "#6c7086"   # gris   — referencias

# Estilo del display digital (LCD style)
_DISPLAY_BASE = (
    "background-color:#0d0d1a;"
    "color:#89b4fa;"
    "font-family:'Courier New',monospace;"
    "font-size:42px;"
    "font-weight:bold;"
    "border:2px solid #45475a;"
    "border-radius:6px;"
    "padding:10px 24px;"
    "letter-spacing:4px;"
)
_DISPLAY_ALERTA = (
    "background-color:#0d0d1a;"
    "color:#f38ba8;"
    "font-family:'Courier New',monospace;"
    "font-size:42px;"
    "font-weight:bold;"
    "border:2px solid #f38ba8;"
    "border-radius:6px;"
    "padding:10px 24px;"
    "letter-spacing:4px;"
)


# ────────────────────────────────────────────────────────────────────────────
# Simulador (modo demo)
# ────────────────────────────────────────────────────────────────────────────
class SimuladorSPCM:
    """
    Genera fotocuentas sintéticas: distribución de Poisson sobre una
    tasa base con deriva sinusoidal lenta y cuentas oscuras.
    """
    def __init__(self):
        self._activo   = False
        self._t_inicio: float | None = None
        self.tasa_base_cps   = 48_500.0
        self.cuentas_oscuras = 28
        self._rng = np.random.default_rng()

    def iniciar(self):
        self._activo   = True
        self._t_inicio = time.time()

    def detener(self):
        self._activo = False

    def leer_conteos(self, integ_ms: int) -> int:
        if not self._activo:
            return 0
        t = time.time() - (self._t_inicio or time.time())
        # Deriva lenta ~10 % con período 60 s + ruido rápido ~1 %
        deriva = 1.0 + 0.10 * math.sin(2 * math.pi * t / 60.0)
        tasa   = self.tasa_base_cps * deriva + self.cuentas_oscuras
        return int(self._rng.poisson(tasa * integ_ms / 1000.0))

    @property
    def info_dispositivo(self) -> str:
        return "SIMULACIÓN — Thorlabs SPCM50A/M (demo, sin hardware)"


# ────────────────────────────────────────────────────────────────────────────
# Driver real (USB HID)
# ────────────────────────────────────────────────────────────────────────────
class DriverSPCM:
    """
    Driver USB HID para Thorlabs SPCM50A/M.

    PROTOCOLO PENDIENTE DE IMPLEMENTACIÓN.
    Pasos:
      1. Conectar el SPCM al Mac.
      2. Ejecutar el snippet del docstring principal para hallar el PID.
      3. Actualizar SPCM_PID en este archivo.
      4. Capturar tráfico HID con Wireshark / USBPcap para determinar
         el formato real del mensaje de solicitud y respuesta.
      5. Implementar leer_conteos() con el protocolo observado.
    """
    def __init__(self):
        self._dev  = None
        self._lock = threading.Lock()
        self._info: dict = {}

    @staticmethod
    def enumerar_thorlabs() -> list[dict]:
        if not HID_DISPONIBLE:
            return []
        return list(_hid.enumerate(THORLABS_VID, 0))

    def detectar(self) -> dict | None:
        for d in self.enumerar_thorlabs():
            if d.get("product_id") == SPCM_PID:
                return d
        return None

    def conectar(self, vid: int = THORLABS_VID, pid: int = SPCM_PID):
        if not HID_DISPONIBLE:
            raise RuntimeError("Instalar biblioteca HID: pip install hid")
        self._dev = _hid.device()
        self._dev.open(vid, pid)
        self._dev.set_nonblocking(False)
        for d in self.enumerar_thorlabs():
            if d.get("product_id") == pid:
                self._info = d
                break

    def desconectar(self):
        if self._dev:
            try:
                self._dev.close()
            except Exception:
                pass
            self._dev = None

    def conectado(self) -> bool:
        return self._dev is not None

    def leer_conteos(self, integ_ms: int) -> int:
        """
        ⚠ IMPLEMENTAR: protocolo HID real del SPCM50A/M.

        Esquema tentativo (ajustar según captura real):
          Enviar  → [0x00, 0x01, integ_ms & 0xFF, integ_ms >> 8, 0...0]
          Esperar → integ_ms ms
          Recibir ← [status, cnt0, cnt1, cnt2, cnt3, ...]
          Conteos  = cnt0 | (cnt1<<8) | (cnt2<<16) | (cnt3<<24)
        """
        with self._lock:
            raise NotImplementedError(
                "Protocolo HID no implementado — ver docstring de "
                "DriverSPCM.leer_conteos(). La app usa simulación."
            )

    @property
    def info_dispositivo(self) -> str:
        sn = self._info.get("serial_number", "?")
        ps = self._info.get("product_string", "SPCM50A/M")
        return f"Thorlabs {ps}  S/N: {sn}"


# ────────────────────────────────────────────────────────────────────────────
# Helper figuras dark
# ────────────────────────────────────────────────────────────────────────────
def _make_fig(w: float = 4.5, h: float = 3.0):
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
# Ventana principal
# ────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):

    # Señales cross-thread
    sig_log      = pyqtSignal(str)
    sig_conteos  = pyqtSignal(int, float)   # (N_raw, t_s)
    sig_error    = pyqtSignal(str)
    sig_conexion = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thorlabs SPCM50A/M — Single Photon Counter")
        self.setMinimumSize(1020, 920)
        self.resize(1060, 980)

        # Driver
        self._driver: SimuladorSPCM | DriverSPCM = SimuladorSPCM()
        self._simulacion = True

        # Estado de medición
        self._midiendo         = False
        self._modo_unico       = False
        self._seguir_midiendo  = threading.Event()
        self._hilo_med:  threading.Thread | None = None
        self._integ_ms   = INTEG_DEFAULT_MS
        self._umbral_cps: float | None = None
        self._t0:         float | None = None

        # Historial
        self._hist_tasa:  deque[tuple[float, float]] = deque()  # (t_s, cps)
        self._hist_largo: deque[tuple[float, float]] = deque()  # (t_s, cps)
        self._hist_bins:  list[float] = []
        self._timestamps: list[str]   = []

        # Acumuladores
        self._total_conteos = 0
        self._mediciones_n  = 0
        self._ventana_stats: deque[float] = deque(maxlen=200)

        self._construir_ui()
        self._conectar_senales()

        QTimer.singleShot(200, self._auto_detectar)

    # ─────────────────────────────────────────────────────────────────────────
    # Construcción UI
    # ─────────────────────────────────────────────────────────────────────────
    def _construir_ui(self):
        raiz = QWidget()
        self.setCentralWidget(raiz)
        lay  = QVBoxLayout(raiz)
        lay.setSpacing(6)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabBar::tab { padding: 8px 22px; font-size: 13px; font-weight: bold; }"
            "QTabBar::tab:selected { color: #89b4fa; border-bottom: 2px solid #89b4fa; }"
        )
        lay.addWidget(self.tabs, 1)

        tab1 = QWidget(); self.tabs.addTab(tab1, "  Medición  ")
        tab2 = QWidget(); self.tabs.addTab(tab2, "  Análisis  ")
        self._construir_tab_medicion(tab1)
        self._construir_tab_analisis(tab2)

        # Log compartido
        gb_log = QGroupBox("Log")
        ll = QVBoxLayout(gb_log)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        self.log.setMaximumHeight(118)
        self.log.setStyleSheet(
            "background:#111;color:#0f0;font-family:Menlo;font-size:11px;")
        ll.addWidget(self.log)
        lay.addWidget(gb_log)

    # ── Tab 1 — Medición ─────────────────────────────────────────────────────
    def _construir_tab_medicion(self, parent: QWidget):
        lay = QVBoxLayout(parent)
        lay.setSpacing(6)

        # Fila superior: Dispositivo + Configuración
        fila_top = QHBoxLayout()

        # Dispositivo
        gb_dev = QGroupBox("Dispositivo")
        gd = QVBoxLayout(gb_dev)
        self.lbl_dispositivo = QLabel("Buscando …")
        self.lbl_dispositivo.setStyleSheet("font-size:11px;color:#cba6f7;")
        self.lbl_dispositivo.setWordWrap(True)
        gd.addWidget(self.lbl_dispositivo)
        fila_dev = QHBoxLayout()
        self.btn_conectar = QPushButton("Conectar")
        self.btn_conectar.clicked.connect(self._toggle_conexion)
        self.btn_sim = QPushButton("Modo demo")
        self.btn_sim.setStyleSheet("color:#cba6f7;")
        self.btn_sim.clicked.connect(self._activar_simulacion_manual)
        fila_dev.addWidget(self.btn_conectar)
        fila_dev.addWidget(self.btn_sim)
        gd.addLayout(fila_dev)
        fila_top.addWidget(gb_dev, 2)

        # Configuración
        gb_cfg = QGroupBox("Configuración")
        gcf = QGridLayout(gb_cfg)

        gcf.addWidget(QLabel("T integración:"), 0, 0)
        self.cmb_integ = QComboBox()
        for t in TIEMPOS_INTEG_MS:
            lbl = f"{t} ms" if t < 1000 else f"{t//1000} s"
            self.cmb_integ.addItem(lbl, t)
        self.cmb_integ.setCurrentIndex(TIEMPOS_INTEG_MS.index(INTEG_DEFAULT_MS))
        self.cmb_integ.currentIndexChanged.connect(self._cambiar_integracion)
        gcf.addWidget(self.cmb_integ, 0, 1)

        gcf.addWidget(QLabel("Umbral alerta:"), 1, 0)
        fila_u = QHBoxLayout()
        self.spn_umbral = QDoubleSpinBox()
        self.spn_umbral.setRange(0, 1e8)
        self.spn_umbral.setDecimals(0)
        self.spn_umbral.setSuffix(" cps")
        self.spn_umbral.setValue(0)
        self.spn_umbral.setSpecialValueText("— sin umbral")
        fila_u.addWidget(self.spn_umbral)
        btn_u = QPushButton("Aplicar")
        btn_u.clicked.connect(self._aplicar_umbral)
        fila_u.addWidget(btn_u)
        gcf.addLayout(fila_u, 1, 1)

        gcf.addWidget(QLabel("Modo:"), 2, 0)
        fila_modo = QHBoxLayout()
        self.rb_continuo = QRadioButton("Continuo")
        self.rb_unico    = QRadioButton("Única adquisición")
        self.rb_continuo.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self.rb_continuo)
        bg.addButton(self.rb_unico)
        fila_modo.addWidget(self.rb_continuo)
        fila_modo.addWidget(self.rb_unico)
        gcf.addLayout(fila_modo, 2, 1)
        fila_top.addWidget(gb_cfg, 3)

        lay.addLayout(fila_top)

        # Display digital grande
        gb_disp = QGroupBox("Tasa de conteo")
        ld = QVBoxLayout(gb_disp)
        self.lbl_tasa = QLabel("0  cps")
        self.lbl_tasa.setStyleSheet(_DISPLAY_BASE)
        self.lbl_tasa.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ld.addWidget(self.lbl_tasa)

        fila_info = QHBoxLayout()
        self.lbl_total   = QLabel("Total: 0 cuentas")
        self.lbl_elapsed = QLabel("Tiempo: 0.0 s")
        self.lbl_n_med   = QLabel("N: 0 mediciones")
        for lbl in (self.lbl_total, self.lbl_elapsed, self.lbl_n_med):
            lbl.setStyleSheet("font-size:12px;color:#a6adc8;")
            fila_info.addWidget(lbl)
        ld.addLayout(fila_info)
        lay.addWidget(gb_disp)

        # Botones de control
        gb_ctrl = QGroupBox()
        lc = QHBoxLayout(gb_ctrl)
        self.btn_iniciar = QPushButton("▶  Iniciar")
        self.btn_detener = QPushButton("■  Detener")
        self.btn_reset   = QPushButton("↺  Reiniciar")
        self.btn_iniciar.setStyleSheet(
            "background:#3a7d3a;color:white;font-weight:bold;"
            "padding:10px 24px;font-size:14px;")
        self.btn_detener.setStyleSheet(
            "background:#7d3a3a;color:white;font-weight:bold;"
            "padding:10px 24px;font-size:14px;")
        self.btn_reset.setStyleSheet(
            "background:#45475a;color:white;font-weight:bold;"
            "padding:10px 24px;font-size:14px;")
        self.btn_iniciar.clicked.connect(self._iniciar_medicion)
        self.btn_detener.clicked.connect(self._detener_medicion)
        self.btn_reset.clicked.connect(self._reiniciar)
        self.btn_detener.setEnabled(False)
        for b in (self.btn_iniciar, self.btn_detener, self.btn_reset):
            lc.addWidget(b)
        lay.addWidget(gb_ctrl)

        # Estadísticas en vivo
        gb_est = QGroupBox("Estadísticas en vivo (ventana 200 muestras)")
        ge = QGridLayout(gb_est)
        self.lbl_media  = QLabel("Media: —")
        self.lbl_sigma  = QLabel("σ: —")
        self.lbl_min    = QLabel("Mín: —")
        self.lbl_max    = QLabel("Máx: —")
        self.lbl_snr    = QLabel("SNR: —")
        self.lbl_sigma_p = QLabel("σ/μ: —")
        stats_lbls = [self.lbl_media, self.lbl_sigma, self.lbl_min,
                      self.lbl_max, self.lbl_snr, self.lbl_sigma_p]
        for i, lbl in enumerate(stats_lbls):
            lbl.setStyleSheet("font-family:Menlo;font-size:12px;")
            ge.addWidget(lbl, i // 3, i % 3)
        lay.addWidget(gb_est)

        # Gráfica tasa vs tiempo (60 s)
        gb_plot = QGroupBox(f"Tasa de conteo — últimos {int(VENTANA_PLOT_S)} s")
        lp = QVBoxLayout(gb_plot)
        fig_m, self.ax_med = _make_fig(9.4, 3.0)
        self.ax_med.set_xlabel("t [s]")
        self.ax_med.set_ylabel("Tasa [cps]")
        self.canvas_med  = FigureCanvas(fig_m)
        self.line_med,   = self.ax_med.plot([], [], color=C_MAIN, lw=1.2,
                                            marker=".", ms=3)
        self.line_umbral = self.ax_med.axhline(
            0, color=C_ALERT, ls="--", lw=0.9,
            visible=False, label="Umbral")
        lp.addWidget(self.canvas_med)
        lay.addWidget(gb_plot)

    # ── Tab 2 — Análisis ──────────────────────────────────────────────────────
    def _construir_tab_analisis(self, parent: QWidget):
        lay = QVBoxLayout(parent)
        lay.setSpacing(6)

        fila_gr = QHBoxLayout(); fila_gr.setSpacing(8)

        # Traza largo plazo
        gb_largo = QGroupBox(
            f"Tasa de conteo — últimos {int(VENTANA_LARGO_S // 60)} min")
        ll = QVBoxLayout(gb_largo)
        fig_l, self.ax_largo = _make_fig(4.8, 3.5)
        self.ax_largo.set_xlabel("t [s]")
        self.ax_largo.set_ylabel("Tasa [cps]")
        self.canvas_largo = FigureCanvas(fig_l)
        self.line_largo,  = self.ax_largo.plot([], [], color=C_MAIN, lw=1.0,
                                               marker=".", ms=2)
        ll.addWidget(self.canvas_largo)
        fila_gr.addWidget(gb_largo, 1)

        # Histograma
        gb_hist = QGroupBox("Histograma de tasa de conteo")
        lh = QVBoxLayout(gb_hist)
        fig_h, self.ax_hist = _make_fig(4.8, 3.5)
        self.ax_hist.set_xlabel("Tasa [cps]")
        self.ax_hist.set_ylabel("Frecuencia")
        self.canvas_hist = FigureCanvas(fig_h)
        lh.addWidget(self.canvas_hist)
        fila_gr.addWidget(gb_hist, 1)

        lay.addLayout(fila_gr)

        # Panel inferior: estadísticas globales + exportación
        fila_bot = QHBoxLayout(); fila_bot.setSpacing(8)

        gb_g = QGroupBox("Estadísticas globales (sesión)")
        gg = QGridLayout(gb_g)
        _filas = [("Media:", "media"), ("Mediana:", "mediana"),
                  ("σ:", "sigma"), ("Mín:", "min"),
                  ("Máx:", "max"), ("N muestras:", "n")]
        self._g: dict[str, QLabel] = {}
        for i, (etiq, key) in enumerate(_filas):
            gg.addWidget(QLabel(etiq), i // 3, (i % 3) * 2)
            v = QLabel("—")
            v.setStyleSheet(
                "font-family:Menlo;color:#89b4fa;font-weight:bold;")
            gg.addWidget(v, i // 3, (i % 3) * 2 + 1)
            self._g[key] = v
        fila_bot.addWidget(gb_g, 2)

        gb_exp = QGroupBox("Exportar")
        le = QVBoxLayout(gb_exp)
        self.btn_csv = QPushButton("Guardar CSV …")
        self.btn_csv.setStyleSheet(
            "background:#313244;color:#cdd6f4;padding:8px;font-weight:bold;")
        self.btn_csv.clicked.connect(self._guardar_csv)
        self.lbl_export = QLabel("(sin exportar)")
        self.lbl_export.setStyleSheet("font-size:11px;color:#a6adc8;")
        le.addWidget(self.btn_csv)
        le.addWidget(self.lbl_export)
        btn_limpiar = QPushButton("Limpiar historial")
        btn_limpiar.clicked.connect(self._reiniciar)
        le.addWidget(btn_limpiar)
        fila_bot.addWidget(gb_exp, 1)

        lay.addLayout(fila_bot)

    # ─────────────────────────────────────────────────────────────────────────
    # Señales
    # ─────────────────────────────────────────────────────────────────────────
    def _conectar_senales(self):
        self.sig_log.connect(self._log)
        self.sig_conteos.connect(self._on_conteos)
        self.sig_error.connect(self._on_error)
        self.sig_conexion.connect(self._on_conexion)

    # ─────────────────────────────────────────────────────────────────────────
    # Detección y conexión
    # ─────────────────────────────────────────────────────────────────────────
    def _auto_detectar(self):
        self._log("Buscando Thorlabs SPCM50A/M (VID=0x1313) …")
        def _t():
            drv = DriverSPCM()
            info = drv.detectar()
            if info:
                try:
                    drv.conectar(info["vendor_id"], info["product_id"])
                    self._driver     = drv
                    self._simulacion = False
                    self.sig_conexion.emit(drv.info_dispositivo)
                    self.sig_log.emit(f"Conectado: {drv.info_dispositivo}")
                    return
                except Exception as e:
                    self.sig_log.emit(
                        f"Dispositivo encontrado pero error al conectar: {e}")
            else:
                if HID_DISPONIBLE:
                    lista = DriverSPCM.enumerar_thorlabs()
                    if lista:
                        self.sig_log.emit(
                            "Dispositivos Thorlabs detectados (PID incorrecto):")
                        for d in lista:
                            self.sig_log.emit(
                                f"  VID={hex(d['vendor_id'])} "
                                f"PID={hex(d['product_id'])} "
                                f"'{d.get('product_string','')}'")
                self.sig_log.emit(
                    "SPCM50A/M no encontrado — activando modo simulación.")
            self._usar_simulacion()
        threading.Thread(target=_t, daemon=True).start()

    def _usar_simulacion(self):
        sim = SimuladorSPCM()
        self._driver     = sim
        self._simulacion = True
        self.sig_conexion.emit(sim.info_dispositivo)

    def _activar_simulacion_manual(self):
        if self._midiendo:
            self._detener_medicion()
        self._usar_simulacion()
        self._log("Modo simulación activado manualmente.")

    def _toggle_conexion(self):
        if isinstance(self._driver, DriverSPCM) and self._driver.conectado():
            if self._midiendo:
                self._detener_medicion()
            self._driver.desconectar()
            self._log("Desconectado.")
            self._usar_simulacion()
        else:
            self._auto_detectar()

    def _on_conexion(self, info: str):
        self.lbl_dispositivo.setText(info)
        color = "#cba6f7" if self._simulacion else "#a6e3a1"
        self.lbl_dispositivo.setStyleSheet(f"font-size:11px;color:{color};")
        self.btn_conectar.setText(
            "Desconectar"
            if (not self._simulacion
                and isinstance(self._driver, DriverSPCM)
                and self._driver.conectado())
            else "Conectar")

    # ─────────────────────────────────────────────────────────────────────────
    # Control de medición
    # ─────────────────────────────────────────────────────────────────────────
    def _iniciar_medicion(self):
        if self._midiendo:
            return
        self._modo_unico = self.rb_unico.isChecked()
        self._midiendo   = True
        self._seguir_midiendo.set()
        self.btn_iniciar.setEnabled(False)
        self.btn_detener.setEnabled(True)
        if isinstance(self._driver, SimuladorSPCM):
            self._driver.iniciar()
        self._hilo_med = threading.Thread(
            target=self._bucle_medicion, daemon=True)
        self._hilo_med.start()
        modo = "única adquisición" if self._modo_unico else "continuo"
        self._log(f"Medición iniciada — T_int={self._integ_ms} ms  modo={modo}")

    def _detener_medicion(self):
        self._seguir_midiendo.clear()
        self._midiendo = False
        if isinstance(self._driver, SimuladorSPCM):
            self._driver.detener()
        self.btn_iniciar.setEnabled(True)
        self.btn_detener.setEnabled(False)
        self._log("Medición detenida.")

    def _reiniciar(self):
        estaba_midiendo = self._midiendo
        if estaba_midiendo:
            self._detener_medicion()
        self._t0            = None
        self._total_conteos = 0
        self._mediciones_n  = 0
        self._hist_tasa.clear()
        self._hist_largo.clear()
        self._hist_bins.clear()
        self._timestamps.clear()
        self._ventana_stats.clear()
        self.lbl_tasa.setText("0  cps")
        self.lbl_tasa.setStyleSheet(_DISPLAY_BASE)
        self.lbl_total.setText("Total: 0 cuentas")
        self.lbl_elapsed.setText("Tiempo: 0.0 s")
        self.lbl_n_med.setText("N: 0 mediciones")
        for lbl in (self.lbl_media, self.lbl_sigma, self.lbl_min,
                    self.lbl_max, self.lbl_snr, self.lbl_sigma_p):
            lbl.setText(lbl.text().split(":")[0] + ": —")
        for v in self._g.values():
            v.setText("—")
        self._limpiar_graficas()
        self._log("Reiniciado.")
        if estaba_midiendo:
            self._iniciar_medicion()

    def _cambiar_integracion(self, idx: int):
        self._integ_ms = self.cmb_integ.itemData(idx)
        self._log(f"T integración → {self._integ_ms} ms")

    def _aplicar_umbral(self):
        v = self.spn_umbral.value()
        if v <= 0:
            self._umbral_cps = None
            self.line_umbral.set_visible(False)
            self._log("Umbral desactivado.")
        else:
            self._umbral_cps = v
            self.line_umbral.set_ydata([v, v])
            self.line_umbral.set_visible(True)
            self._log(f"Umbral → {v:,.0f} cps")
        self.canvas_med.draw_idle()

    # ─────────────────────────────────────────────────────────────────────────
    # Bucle de medición (hilo)
    # ─────────────────────────────────────────────────────────────────────────
    def _bucle_medicion(self):
        unico = self._modo_unico
        while self._seguir_midiendo.is_set():
            t_ini = time.time()
            try:
                conteos = self._driver.leer_conteos(self._integ_ms)
            except NotImplementedError:
                self.sig_error.emit(
                    "Protocolo USB no implementado — activando simulación.")
                self._usar_simulacion()
                if isinstance(self._driver, SimuladorSPCM):
                    self._driver.iniciar()
                conteos = self._driver.leer_conteos(self._integ_ms)
            except Exception as e:
                self.sig_error.emit(str(e))
                break
            ahora = time.time()
            if self._t0 is None:
                self._t0 = ahora
            t_s = ahora - self._t0
            self.sig_conteos.emit(conteos, t_s)
            if unico:
                break
            # Mantener la cadencia descontando el tiempo de ejecución
            pausa = self._integ_ms / 1000.0 - (time.time() - t_ini)
            if pausa > 0:
                self._seguir_midiendo.wait(timeout=pausa)
        self.sig_log.emit("Bucle de medición finalizado.")

    # ─────────────────────────────────────────────────────────────────────────
    # Handlers de señales
    # ─────────────────────────────────────────────────────────────────────────
    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log.appendPlainText(f"[{ts}] {msg}")

    def _on_error(self, msg: str):
        self._log(f"ERROR: {msg}")
        if self._midiendo:
            self._detener_medicion()

    def _on_conteos(self, conteos: int, t_s: float):
        cps = conteos / (self._integ_ms / 1000.0)
        self._total_conteos += conteos
        self._mediciones_n  += 1
        self._ventana_stats.append(cps)
        self._hist_bins.append(cps)
        self._timestamps.append(datetime.now().isoformat(timespec="milliseconds"))

        # Display
        self.lbl_tasa.setText(self._fmt(cps))
        if self._umbral_cps and cps > self._umbral_cps:
            self.lbl_tasa.setStyleSheet(_DISPLAY_ALERTA)
        else:
            self.lbl_tasa.setStyleSheet(_DISPLAY_BASE)

        elapsed = self._mediciones_n * self._integ_ms / 1000.0
        self.lbl_total.setText(f"Total: {self._total_conteos:,} cuentas")
        self.lbl_elapsed.setText(f"Tiempo: {elapsed:.1f} s")
        self.lbl_n_med.setText(f"N: {self._mediciones_n:,} mediciones")

        # Historial gráficas
        self._hist_tasa.append((t_s, cps))
        self._recortar(self._hist_tasa, t_s, VENTANA_PLOT_S)
        self._hist_largo.append((t_s, cps))
        self._recortar(self._hist_largo, t_s, VENTANA_LARGO_S)

        # Estadísticas
        self._actualizar_stats()

        # Gráficas (no bloquean: draw_idle)
        self._refrescar_medicion()
        self._refrescar_largo()
        if self._mediciones_n % 5 == 0:  # histograma cada 5 puntos
            self._refrescar_histograma()

    # ─────────────────────────────────────────────────────────────────────────
    # Estadísticas
    # ─────────────────────────────────────────────────────────────────────────
    def _actualizar_stats(self):
        arr = np.array(self._ventana_stats)
        if arr.size == 0:
            return
        mu, sg = arr.mean(), arr.std()
        mn, mx = arr.min(), arr.max()
        snr     = mu / sg if sg > 0 else float("inf")
        sigma_p = 100.0 * sg / mu if mu > 0 else float("inf")
        self.lbl_media.setText(  f"Media: {self._fmt(mu).strip()}")
        self.lbl_sigma.setText(  f"σ: {self._fmt(sg).strip()}")
        self.lbl_min.setText(    f"Mín: {self._fmt(mn).strip()}")
        self.lbl_max.setText(    f"Máx: {self._fmt(mx).strip()}")
        self.lbl_snr.setText(    f"SNR: {snr:.1f}")
        self.lbl_sigma_p.setText(f"σ/μ: {sigma_p:.2f} %")

        # Estadísticas globales (Tab 2)
        g = np.array(self._hist_bins)
        self._g["media"].setText(self._fmt(g.mean()).strip())
        self._g["mediana"].setText(self._fmt(float(np.median(g))).strip())
        self._g["sigma"].setText(self._fmt(g.std()).strip())
        self._g["min"].setText(self._fmt(g.min()).strip())
        self._g["max"].setText(self._fmt(g.max()).strip())
        self._g["n"].setText(str(len(g)))

    @staticmethod
    def _fmt(cps: float) -> str:
        """Autoescalado: cps / kcps / Mcps."""
        if cps >= 1e6:
            return f"{cps/1e6:9.4f}  Mcps"
        if cps >= 1e3:
            return f"{cps/1e3:9.3f}  kcps"
        return f"{cps:9.1f}  cps"

    # ─────────────────────────────────────────────────────────────────────────
    # Gráficas
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _recortar(hist: deque, t_actual: float, ventana: float):
        lim = t_actual - ventana
        while hist and hist[0][0] < lim:
            hist.popleft()

    def _refrescar_medicion(self):
        if not self._hist_tasa:
            return
        xs = [t for t, _ in self._hist_tasa]
        ys = [c for _, c in self._hist_tasa]
        self.line_med.set_data(xs, ys)
        t_max = xs[-1]
        self.ax_med.set_xlim(max(0.0, t_max - VENTANA_PLOT_S), max(t_max, 1.0))
        y_lo, y_hi = min(ys), max(ys)
        if self._umbral_cps:
            y_lo = min(y_lo, self._umbral_cps)
            y_hi = max(y_hi, self._umbral_cps)
        mg = max(0.05 * max(abs(y_hi), 1.0), 5.0)
        self.ax_med.set_ylim(max(0.0, y_lo - mg), y_hi + mg)
        self.canvas_med.draw_idle()

    def _refrescar_largo(self):
        if not self._hist_largo:
            return
        xs = [t for t, _ in self._hist_largo]
        ys = [c for _, c in self._hist_largo]
        self.line_largo.set_data(xs, ys)
        t_max = xs[-1]
        self.ax_largo.set_xlim(
            max(0.0, t_max - VENTANA_LARGO_S), max(t_max, 1.0))
        mg = max(0.05 * max(abs(max(ys)), 1.0), 5.0)
        self.ax_largo.set_ylim(max(0.0, min(ys) - mg), max(ys) + mg)
        self.canvas_largo.draw_idle()

    def _refrescar_histograma(self):
        if len(self._hist_bins) < 5:
            return
        arr    = np.array(self._hist_bins)
        n_bins = min(60, max(10, len(arr) // 8))
        # Redibujar el eje completo (cla mantiene el facecolor del ax)
        self.ax_hist.cla()
        self.ax_hist.set_facecolor(AX_BG)
        for sp in self.ax_hist.spines.values():
            sp.set_color(GRID_COL)
        self.ax_hist.tick_params(colors=FG, labelsize=8)
        self.ax_hist.xaxis.label.set_color(FG)
        self.ax_hist.yaxis.label.set_color(FG)
        self.ax_hist.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.6)
        self.ax_hist.hist(arr, bins=n_bins, color=C_HIST, alpha=0.85,
                          edgecolor=AX_BG, linewidth=0.3)
        mu, sg = arr.mean(), arr.std()
        self.ax_hist.axvline(mu, color=C_MAIN, ls="--", lw=1.1,
                             label=f"μ = {self._fmt(mu).strip()}")
        self.ax_hist.axvline(mu - sg, color=C_SET, ls=":", lw=0.8)
        self.ax_hist.axvline(mu + sg, color=C_SET, ls=":", lw=0.8,
                             label=f"σ = {self._fmt(sg).strip()}")
        self.ax_hist.set_xlabel("Tasa [cps]")
        self.ax_hist.set_ylabel("Frecuencia")
        self.ax_hist.legend(labelcolor=FG, facecolor=AX_BG,
                            edgecolor=GRID_COL, fontsize=7,
                            loc="upper right")
        self.canvas_hist.draw_idle()

    def _limpiar_graficas(self):
        self.line_med.set_data([], [])
        self.line_largo.set_data([], [])
        self.ax_med.set_xlim(0, VENTANA_PLOT_S)
        self.ax_med.set_ylim(0, 100_000)
        self.ax_largo.set_xlim(0, VENTANA_LARGO_S)
        self.ax_largo.set_ylim(0, 100_000)
        self.ax_hist.cla()
        self.ax_hist.set_facecolor(AX_BG)
        for c in (self.canvas_med, self.canvas_largo, self.canvas_hist):
            c.draw_idle()

    # ─────────────────────────────────────────────────────────────────────────
    # Exportación CSV
    # ─────────────────────────────────────────────────────────────────────────
    def _guardar_csv(self):
        if not self._hist_bins:
            QMessageBox.information(self, "Sin datos",
                                    "No hay datos para exportar.")
            return
        ruta, _ = QFileDialog.getSaveFileName(
            self, "Guardar datos SPCM",
            f"spcm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV (*.csv)")
        if not ruta:
            return
        try:
            rows = list(self._hist_largo)
            ts_list = self._timestamps[-len(rows):]
            with open(ruta, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "t_s", "tasa_cps",
                             "integ_ms", "dispositivo"])
                dispositivo = self._driver.info_dispositivo
                for i, (t_s, cps) in enumerate(rows):
                    ts = ts_list[i] if i < len(ts_list) else ""
                    w.writerow([ts, f"{t_s:.4f}", f"{cps:.2f}",
                                self._integ_ms, dispositivo])
            self.lbl_export.setText(f"Guardado: {Path(ruta).name}")
            self._log(f"CSV guardado ({len(rows)} filas): {ruta}")
        except Exception as e:
            QMessageBox.critical(self, "Error al guardar", str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # Cierre seguro
    # ─────────────────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        self._seguir_midiendo.clear()
        if isinstance(self._driver, DriverSPCM):
            try:
                self._driver.desconectar()
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

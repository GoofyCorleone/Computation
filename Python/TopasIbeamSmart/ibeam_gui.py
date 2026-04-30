"""
Interfaz gráfica (PyQt6) para el láser TOPTICA iBeam Smart.
Encender/apagar emisión, modificar la potencia de cada canal, monitorear y
controlar la temperatura del diodo, e indicar la estabilidad de la potencia.

Detecta automáticamente el puerto del adaptador USB-Serial al que está
conectado el iBeam Smart (sondea cada puerto hasta encontrar el prompt
'CMD> ' a 115200 baud).

Notas:
  - La potencia de salida del iBeam Smart es la SUMA de los canales activos.
    Para que el control de un canal se corresponda con la salida, el otro
    canal debe estar en 0 mW. Los comandos 'en N' / 'di N' no modifican la
    contribución del canal en este firmware — la única forma confiable es
    fijar el nivel con 'ch N pow 0'.
  - El setpoint de temperatura (TEC) viene fijado de fábrica (típicamente
    25.0 °C). El comando 'set temp X' está protegido por contraseña en el
    firmware estándar y devolverá '%SYS-W-047, access restricted'. La app
    intenta el cambio y avisa si fue rechazado.
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
    QApplication, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QVBoxLayout, QWidget,
)

# Backend Qt de matplotlib para embeber gráficos en la GUI
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


BAUD               = 115200
TIMEOUT            = 1.5
PROMPT             = b"CMD> "
INTERVALO_POLL_MS  = 700

# Estabilidad de potencia
VENTANA_ESTAB_S    = 8.0     # ventana móvil para evaluar estabilidad
UMBRAL_REL_ESTABLE = 0.005   # < 0.5 % de variación relativa
MIN_MUESTRAS_EST   = 6
TAU_DIODO_S        = 25.0    # constante térmica típica del diodo (warmup)

# Estabilidad térmica
TOL_TEMP_C         = 0.30    # |T - setpoint| < 0.3 °C => térmica estable

# Gráficos en tiempo real
VENTANA_PLOT_S     = 60.0    # ventana visible en los gráficos en vivo


# --------------------------------------------------------------------------
# Detección automática de puerto
# --------------------------------------------------------------------------
def candidatos_puerto() -> list[str]:
    """Lista de puertos serie candidatos ordenados por plausibilidad."""
    candidatos = []
    for p in list_ports.comports():
        nombre = p.device
        desc = (p.description or "").lower()
        manuf = (p.manufacturer or "").lower()
        if (
            "usbserial" in nombre.lower()
            or "usbmodem" in nombre.lower()
            or nombre.lower().startswith(("/dev/ttyusb", "/dev/ttyacm"))
            or nombre.upper().startswith("COM")
            or "ch340" in desc or "ftdi" in desc or "toptica" in manuf
            or "usb" in desc and "serial" in desc
        ):
            candidatos.append(nombre)
    return candidatos


def puerto_responde_ibeam(puerto: str) -> bool:
    """Abre el puerto, envía CR y comprueba si llega el prompt 'CMD> '."""
    try:
        with serial.Serial(puerto, BAUD, timeout=1.0) as s:
            time.sleep(0.8)
            s.reset_input_buffer()
            s.write(b"\r\n")
            raw = s.read_until(PROMPT, size=200)
            return PROMPT in raw
    except Exception:
        return False


def detectar_puerto() -> str | None:
    """Devuelve el primer puerto que responde al prompt del iBeam Smart."""
    for puerto in candidatos_puerto():
        if puerto_responde_ibeam(puerto):
            return puerto
    return None


# --------------------------------------------------------------------------
# Estabilidad de potencia
# --------------------------------------------------------------------------
class EstabilidadPotencia:
    """Mantiene un historial de (t, potencia) y calcula estabilidad y ETA."""

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

    def _stats(self) -> tuple[float, float, float, int] | None:
        n = len(self.muestras)
        if n < MIN_MUESTRAS_EST:
            return None
        ts = [t for t, _ in self.muestras]
        ps = [p for _, p in self.muestras]
        media = sum(ps) / n
        var = sum((p - media) ** 2 for p in ps) / n
        std = math.sqrt(var)
        t_med = sum(ts) / n
        num = sum((ts[i] - t_med) * (ps[i] - media) for i in range(n))
        den = sum((ts[i] - t_med) ** 2 for i in range(n))
        pendiente = num / den if den > 1e-9 else 0.0
        return media, std, pendiente, n

    def estado(self) -> tuple[str, float | None, float | None]:
        """
        Devuelve (texto, fraccion_progreso, eta_segundos).
        - texto: 'Estabilizado', 'Estabilizando', 'Calentando', 'Sin datos'
        - fraccion_progreso: 0..1 (None si sin datos)
        - eta_segundos: estimación de tiempo restante (None si no aplica)
        """
        s = self._stats()
        if s is None:
            return ("Sin datos", None, None)
        media, std, pendiente, _ = s
        if abs(media) < 1e-6:
            return ("Sin emisión", 0.0, None)

        rel_std   = std / abs(media)
        rel_drift = abs(pendiente * self.ventana_s) / abs(media)
        rel = max(rel_std, rel_drift)

        if rel < UMBRAL_REL_ESTABLE:
            return ("Estabilizado", 1.0, 0.0)

        # Estimar ETA suponiendo decaimiento exponencial:
        # rel(t) = rel_actual * exp(-t/τ) → t = τ · ln(rel/UMBRAL)
        eta = TAU_DIODO_S * math.log(rel / UMBRAL_REL_ESTABLE)
        eta = max(0.0, min(eta, 600.0))  # acotar a [0, 10 min]

        # Mapear a fracción 0..1 (logarítmica entre rel y UMBRAL)
        # Cuando rel/UMBRAL = 10  → fracción ≈ 0.0 (lejos)
        # Cuando rel/UMBRAL = 1   → fracción = 1.0 (estable)
        frac = max(0.0, min(1.0, 1.0 - math.log10(rel / UMBRAL_REL_ESTABLE) / 2.0))

        texto = "Estabilizando" if rel < 5 * UMBRAL_REL_ESTABLE else "Calentando"
        return (texto, frac, eta)


# --------------------------------------------------------------------------
# Driver serie
# --------------------------------------------------------------------------
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
        lineas = [l.strip() for l in crudo.splitlines() if l.strip() and l.strip() != "CMD>"]
        if lineas and lineas[0].lower() == comando.lower():
            lineas = lineas[1:]
        return "\n".join(lineas)

    def encender(self): self.enviar("la on")
    def apagar(self):   self.enviar("la off")
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

    def leer_temperatura_C(self) -> float | None:
        """Lee la temperatura actual del diodo en °C ('sh temp' → 'TEMP = 025.0 C')."""
        resp = self.enviar("sh temp")
        for linea in resp.splitlines():
            s = linea.strip().upper()
            if s.startswith("TEMP") and "=" in s:
                try:
                    valor = s.split("=", 1)[1].replace("C", "").replace("°", "").strip()
                    return float(valor)
                except ValueError:
                    pass
        return None

    def leer_setpoint_temp_C(self) -> float | None:
        """Extrae el setpoint del TEC desde 'sh syst data' (línea 'TEC setpoint: ... -> X C')."""
        resp = self.enviar("sh syst data")
        for linea in resp.splitlines():
            s = linea.strip()
            if s.lower().startswith("tec setpoint") and "->" in s:
                try:
                    cola = s.split("->", 1)[1].replace("C", "").replace("°", "").strip()
                    return float(cola)
                except ValueError:
                    pass
        return None

    def leer_estado_tec(self) -> str:
        """Devuelve 'ON' / 'OFF' / texto-de-error según 'sta tec'."""
        return self.enviar("sta tec").strip().upper()

    def set_temperatura_C(self, temp_C: float) -> str:
        """Intenta fijar el setpoint del TEC. Suele estar restringido en
        firmware estándar y devolver '%SYS-W-047, access restricted'."""
        return self.enviar(f"set temp {temp_C:.2f}")


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
class MainWindow(QMainWindow):
    sig_log           = pyqtSignal(str)
    sig_estado        = pyqtSignal(str)
    sig_potencia      = pyqtSignal(float)
    sig_niveles       = pyqtSignal(dict)
    sig_niveles_spin  = pyqtSignal(dict)
    sig_error_poll    = pyqtSignal(str)
    sig_detect_done   = pyqtSignal(str)
    sig_temperatura   = pyqtSignal(object)  # (temp_actual, setpoint, tec_estado)
    sig_estabilidad   = pyqtSignal(object)  # (texto, frac, eta)
    sig_setp_resp     = pyqtSignal(str, str) # (setpoint_pedido, respuesta)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("iBeam Smart - Control")
        self.setMinimumSize(960, 820)
        self.resize(960, 880)
        self.driver = IBeamDriver()
        self.estabilidad = EstabilidadPotencia()

        # Historial para gráficos en tiempo real
        self._t0_grafica: float | None = None
        self._hist_pot:  deque[tuple[float, float]] = deque()
        self._hist_temp: deque[tuple[float, float]] = deque()

        self._construir_ui()
        self._conectar_senales()

        self.timer_poll = QTimer(self)
        self.timer_poll.timeout.connect(self._poll_async)

    # ------- UI -------
    def _construir_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Conexión
        gb_con = QGroupBox("Conexión")
        lay_con = QGridLayout(gb_con)
        lay_con.addWidget(QLabel("Puerto:"), 0, 0)
        self.txt_puerto = QLineEdit("(detectando...)")
        lay_con.addWidget(self.txt_puerto, 0, 1)
        self.btn_detectar = QPushButton("Auto-detectar")
        self.btn_detectar.clicked.connect(self._auto_detectar)
        lay_con.addWidget(self.btn_detectar, 0, 2)
        self.btn_conectar = QPushButton("Conectar")
        self.btn_conectar.clicked.connect(self._toggle_conexion)
        lay_con.addWidget(self.btn_conectar, 0, 3)
        layout.addWidget(gb_con)

        # Canales
        gb_ch = QGroupBox("Canales (potencia en mW — la salida es la SUMA)")
        grid = QGridLayout(gb_ch)
        grid.addWidget(QLabel("<b>Canal</b>"),          0, 0)
        grid.addWidget(QLabel("<b>Setpoint [mW]</b>"),  0, 1)
        grid.addWidget(QLabel(""),                       0, 2)
        grid.addWidget(QLabel("<b>Nivel actual</b>"),    0, 3)

        self.spn_pow   = {}
        self.lbl_nivel = {}
        for i, canal in enumerate([1, 2], start=1):
            grid.addWidget(QLabel(f"CH{canal}"), i, 0)
            spn = QDoubleSpinBox()
            spn.setRange(0.0, 200.0)
            spn.setDecimals(3)
            spn.setSingleStep(0.1)
            spn.setValue(0.0 if canal == 2 else 1.0)
            spn.setSuffix(" mW")
            grid.addWidget(spn, i, 1)

            btn = QPushButton("Aplicar")
            btn.clicked.connect(lambda _, c=canal, s=spn: self._aplicar_potencia(c, s.value()))
            grid.addWidget(btn, i, 2)

            lbl = QLabel("—")
            lbl.setStyleSheet("font-family: Menlo; color: #444;")
            grid.addWidget(lbl, i, 3)

            self.spn_pow[canal]   = spn
            self.lbl_nivel[canal] = lbl
        layout.addWidget(gb_ch)

        # Temperatura
        gb_temp = QGroupBox("Temperatura del diodo (TEC)")
        lay_t = QGridLayout(gb_temp)
        lay_t.addWidget(QLabel("Setpoint:"), 0, 0)
        self.spn_temp = QDoubleSpinBox()
        self.spn_temp.setRange(15.0, 40.0)
        self.spn_temp.setDecimals(2)
        self.spn_temp.setSingleStep(0.5)
        self.spn_temp.setValue(25.0)
        self.spn_temp.setSuffix(" °C")
        lay_t.addWidget(self.spn_temp, 0, 1)
        self.btn_temp = QPushButton("Aplicar")
        self.btn_temp.clicked.connect(self._aplicar_temperatura)
        lay_t.addWidget(self.btn_temp, 0, 2)

        self.lbl_temp_actual = QLabel("Actual: —")
        self.lbl_temp_actual.setStyleSheet("font-weight: bold; font-size: 13px;")
        lay_t.addWidget(self.lbl_temp_actual, 1, 0, 1, 2)
        self.lbl_tec = QLabel("TEC: —")
        self.lbl_tec.setStyleSheet("font-weight: bold; font-size: 13px;")
        lay_t.addWidget(self.lbl_tec, 1, 2)

        self.lbl_estab_temp = QLabel("Estabilidad térmica: —")
        self.lbl_estab_temp.setTextFormat(Qt.TextFormat.RichText)
        lay_t.addWidget(self.lbl_estab_temp, 2, 0, 1, 3)
        layout.addWidget(gb_temp)

        # Emisión
        gb_em = QGroupBox("Emisión")
        lay_em = QVBoxLayout(gb_em)
        fila_btns = QHBoxLayout()
        self.btn_on  = QPushButton("Encender (LA ON)")
        self.btn_off = QPushButton("Apagar (LA OFF)")
        self.btn_on.setStyleSheet("background-color: #3a7d3a; color: white; font-weight: bold; padding: 8px;")
        self.btn_off.setStyleSheet("background-color: #7d3a3a; color: white; font-weight: bold; padding: 8px;")
        self.btn_on.clicked.connect(self._encender)
        self.btn_off.clicked.connect(self._apagar)
        self.btn_on.setEnabled(False)
        self.btn_off.setEnabled(False)
        fila_btns.addWidget(self.btn_on)
        fila_btns.addWidget(self.btn_off)
        lay_em.addLayout(fila_btns)

        fila_est = QHBoxLayout()
        self.lbl_estado   = QLabel("Estado: —")
        self.lbl_potencia = QLabel("Potencia medida: —")
        for lbl in (self.lbl_estado, self.lbl_potencia):
            lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
        fila_est.addWidget(self.lbl_estado)
        fila_est.addWidget(self.lbl_potencia)
        lay_em.addLayout(fila_est)

        # Estabilidad de potencia
        fila_stab = QHBoxLayout()
        self.lbl_estab_pow = QLabel("Estabilidad: —")
        self.lbl_estab_pow.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.lbl_estab_pow.setTextFormat(Qt.TextFormat.RichText)
        fila_stab.addWidget(self.lbl_estab_pow)
        self.lbl_eta = QLabel("ETA: —")
        fila_stab.addWidget(self.lbl_eta)
        lay_em.addLayout(fila_stab)

        self.bar_estab = QProgressBar()
        self.bar_estab.setRange(0, 100)
        self.bar_estab.setValue(0)
        self.bar_estab.setTextVisible(True)
        self.bar_estab.setFormat("%p% estabilizado")
        lay_em.addWidget(self.bar_estab)
        layout.addWidget(gb_em)

        # Gráficos en tiempo real (potencia y temperatura, en paralelo)
        fila_plots = QHBoxLayout()
        fila_plots.setSpacing(8)

        gb_plot_p = QGroupBox("Potencia en tiempo real")
        lay_plot_p = QVBoxLayout(gb_plot_p)
        fig_p = Figure(figsize=(4.2, 2.6), tight_layout=True)
        self.canvas_pot = FigureCanvas(fig_p)
        self.ax_pot = fig_p.add_subplot(111)
        self.ax_pot.set_xlabel("t [s]")
        self.ax_pot.set_ylabel("P [mW]")
        self.ax_pot.grid(True, alpha=0.3)
        self.line_pot, = self.ax_pot.plot([], [], "b.-", markersize=3)
        self.line_pot_set = self.ax_pot.axhline(0, color="gray", linestyle="--",
                                                 linewidth=0.8, visible=False)
        lay_plot_p.addWidget(self.canvas_pot)
        fila_plots.addWidget(gb_plot_p, 1)

        gb_plot_t = QGroupBox("Temperatura en tiempo real")
        lay_plot_t = QVBoxLayout(gb_plot_t)
        fig_t = Figure(figsize=(4.2, 2.6), tight_layout=True)
        self.canvas_temp = FigureCanvas(fig_t)
        self.ax_temp = fig_t.add_subplot(111)
        self.ax_temp.set_xlabel("t [s]")
        self.ax_temp.set_ylabel("T [°C]")
        self.ax_temp.grid(True, alpha=0.3)
        self.line_temp, = self.ax_temp.plot([], [], "r.-", markersize=3)
        self.line_temp_set = self.ax_temp.axhline(25.0, color="gray", linestyle="--",
                                                   linewidth=0.8, label="Setpoint")
        lay_plot_t.addWidget(self.canvas_temp)
        fila_plots.addWidget(gb_plot_t, 1)
        layout.addLayout(fila_plots)

        # Log
        gb_log = QGroupBox("Log")
        lay_log = QVBoxLayout(gb_log)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMaximumHeight(140)
        self.log.setStyleSheet("background-color: #111; color: #0f0; font-family: Menlo; font-size: 11px;")
        lay_log.addWidget(self.log)
        layout.addWidget(gb_log)

        QTimer.singleShot(100, self._auto_detectar)

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

    # ------- handlers -------
    def _log(self, msg: str):
        self.log.appendPlainText(msg)

    def _on_potencia(self, p_uW: float):
        p_mW = p_uW / 1000.0
        self.lbl_potencia.setText(f"Potencia medida: {p_mW:.3f} mW")
        ahora = time.time()
        # actualizar historial de estabilidad
        self.estabilidad.agregar(ahora, p_mW)
        texto, frac, eta = self.estabilidad.estado()
        self.sig_estabilidad.emit((texto, frac, eta))
        # actualizar gráfico de potencia
        if self._t0_grafica is None:
            self._t0_grafica = ahora
        t_rel = ahora - self._t0_grafica
        self._hist_pot.append((t_rel, p_mW))
        self._recortar_hist(self._hist_pot, t_rel)
        self._refrescar_plot_pot()

    def _on_estabilidad(self, datos):
        texto, frac, eta = datos
        color = {
            "Estabilizado": "#3a7d3a",
            "Estabilizando": "#b8860b",
            "Calentando":   "#a04020",
            "Sin emisión":  "#555",
            "Sin datos":    "#555",
        }.get(texto, "#444")
        self.lbl_estab_pow.setText(f"Estabilidad: <span style='color:{color}'>{texto}</span>")
        if frac is not None:
            self.bar_estab.setValue(int(round(frac * 100)))
        else:
            self.bar_estab.setValue(0)
        if eta is None or texto in ("Sin datos", "Sin emisión"):
            self.lbl_eta.setText("ETA: —")
        elif eta < 1.0:
            self.lbl_eta.setText("ETA: < 1 s")
        elif eta < 60:
            self.lbl_eta.setText(f"ETA: ~ {eta:.0f} s")
        else:
            self.lbl_eta.setText(f"ETA: ~ {eta/60:.1f} min")

    def _on_temperatura(self, datos):
        temp, setpoint, tec_estado = datos
        if temp is not None:
            self.lbl_temp_actual.setText(f"Actual: {temp:.2f} °C")
        else:
            self.lbl_temp_actual.setText("Actual: —")
        if setpoint is not None:
            self.spn_temp.blockSignals(True)
            self.spn_temp.setValue(setpoint)
            self.spn_temp.blockSignals(False)
            self.line_temp_set.set_ydata([setpoint, setpoint])
        self.lbl_tec.setText(f"TEC: {tec_estado or '—'}")

        # Estabilidad térmica
        if temp is None or setpoint is None:
            self.lbl_estab_temp.setText("Estabilidad térmica: —")
        else:
            delta = temp - setpoint
            if abs(delta) < TOL_TEMP_C:
                self.lbl_estab_temp.setText(
                    f"<span style='color:#3a7d3a; font-weight:bold;'>Térmica estable</span> "
                    f"(Δ = {delta:+.2f} °C)"
                )
            else:
                self.lbl_estab_temp.setText(
                    f"<span style='color:#a04020; font-weight:bold;'>Térmica fuera de tolerancia</span> "
                    f"(Δ = {delta:+.2f} °C)"
                )

        # Actualizar gráfico de temperatura
        if temp is not None:
            ahora = time.time()
            if self._t0_grafica is None:
                self._t0_grafica = ahora
            t_rel = ahora - self._t0_grafica
            self._hist_temp.append((t_rel, temp))
            self._recortar_hist(self._hist_temp, t_rel)
            self._refrescar_plot_temp(setpoint)

    def _on_setp_resp(self, pedido: str, respuesta: str):
        if "access restricted" in respuesta.lower():
            QMessageBox.warning(
                self, "Setpoint restringido",
                f"El firmware del iBeam Smart no permite cambiar el setpoint del TEC "
                f"sin contraseña de mantenimiento (acceso restringido).\n\n"
                f"El setpoint de fábrica suele ser 25.0 °C, valor estándar para esta "
                f"familia de láseres y al que están calibrados los parámetros de potencia.\n\n"
                f"Comando enviado: set temp {pedido}\nRespuesta: {respuesta}"
            )
        elif respuesta:
            self._log(f"set temp {pedido} → {respuesta}")
        else:
            self._log(f"set temp {pedido} → OK")

    def _actualizar_niveles(self, niveles: dict):
        for canal, mW in niveles.items():
            if canal in self.lbl_nivel:
                self.lbl_nivel[canal].setText(f"{mW:.3f} mW")
        # Actualizar la línea de setpoint en el gráfico de potencia
        if niveles:
            total = sum(niveles.values())
            self.line_pot_set.set_ydata([total, total])
            self.line_pot_set.set_visible(total > 0.0)

    @staticmethod
    def _recortar_hist(hist: deque, t_actual: float):
        t_min = t_actual - VENTANA_PLOT_S
        while hist and hist[0][0] < t_min:
            hist.popleft()

    def _refrescar_plot_pot(self):
        if not self._hist_pot:
            return
        xs = [t for t, _ in self._hist_pot]
        ys = [p for _, p in self._hist_pot]
        self.line_pot.set_data(xs, ys)
        t_max = xs[-1]
        t_min = max(0.0, t_max - VENTANA_PLOT_S)
        self.ax_pot.set_xlim(t_min, max(t_max, t_min + 1.0))
        # Y limits con margen
        y_lo, y_hi = min(ys), max(ys)
        if self.line_pot_set.get_visible():
            sp = float(self.line_pot_set.get_ydata()[0])
            y_lo, y_hi = min(y_lo, sp), max(y_hi, sp)
        margen = max(0.05 * max(abs(y_hi), 1e-3), 0.02)
        self.ax_pot.set_ylim(y_lo - margen, y_hi + margen)
        self.canvas_pot.draw_idle()

    def _refrescar_plot_temp(self, setpoint: float | None):
        if not self._hist_temp:
            return
        xs = [t for t, _ in self._hist_temp]
        ys = [T for _, T in self._hist_temp]
        self.line_temp.set_data(xs, ys)
        t_max = xs[-1]
        t_min = max(0.0, t_max - VENTANA_PLOT_S)
        self.ax_temp.set_xlim(t_min, max(t_max, t_min + 1.0))
        y_lo, y_hi = min(ys), max(ys)
        if setpoint is not None:
            y_lo, y_hi = min(y_lo, setpoint), max(y_hi, setpoint)
        # Mantener al menos ±1 °C alrededor del setpoint o el rango medido
        centro = setpoint if setpoint is not None else (y_lo + y_hi) / 2.0
        amplitud = max(1.0, y_hi - y_lo)
        self.ax_temp.set_ylim(centro - amplitud, centro + amplitud)
        self.canvas_temp.draw_idle()

    def _actualizar_spinboxes(self, niveles: dict):
        for canal, mW in niveles.items():
            if canal in self.spn_pow:
                self.spn_pow[canal].blockSignals(True)
                self.spn_pow[canal].setValue(mW)
                self.spn_pow[canal].blockSignals(False)

    # ------- conexión / detección -------
    def _auto_detectar(self, conectar_automaticamente: bool = True):
        self._auto_connect_pendiente = conectar_automaticamente
        self.btn_detectar.setEnabled(False)
        self.btn_detectar.setText("Buscando...")
        self._log("Buscando puerto del iBeam Smart ...")

        def _tarea():
            puerto = detectar_puerto() or ""
            self.sig_detect_done.emit(puerto)
        threading.Thread(target=_tarea, daemon=True).start()

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
            self._log("No se encontró un iBeam Smart conectado.")

    def _toggle_conexion(self):
        if self.driver.conectado():
            self._desconectar()
        else:
            self._conectar()

    def _conectar(self):
        puerto = self.txt_puerto.text().strip()
        if not puerto:
            QMessageBox.warning(self, "Sin puerto", "No hay puerto definido. Usa 'Auto-detectar' o escríbelo.")
            return
        try:
            self.driver.conectar(puerto)
            self._log(f"Conectado a {puerto} @ {BAUD}")
            self.btn_conectar.setText("Desconectar")
            self.btn_on.setEnabled(True)
            self.btn_off.setEnabled(True)
            self.timer_poll.start(INTERVALO_POLL_MS)
            self.estabilidad.reset()
            self._reset_graficos()
            self._ejecutar_async(self._sincronizar_desde_dispositivo)
        except Exception as e:
            QMessageBox.critical(self, "Error de conexión", str(e))

    def _sincronizar_desde_dispositivo(self):
        niveles  = self.driver.leer_niveles()
        setpoint = self.driver.leer_setpoint_temp_C()
        temp     = self.driver.leer_temperatura_C()
        tec      = self.driver.leer_estado_tec()
        self.sig_niveles.emit(niveles)
        self.sig_niveles_spin.emit(niveles)
        self.sig_temperatura.emit((temp, setpoint, tec))
        self.sig_log.emit(f"Niveles leídos: {niveles}")
        self.sig_log.emit(f"TEC: {tec}, setpoint: {setpoint} °C, actual: {temp} °C")

    def _desconectar(self):
        self.timer_poll.stop()
        try:
            self.driver.desconectar()
        except Exception as e:
            self._log(f"Error al desconectar: {e}")
        self.btn_conectar.setText("Conectar")
        self.btn_on.setEnabled(False)
        self.btn_off.setEnabled(False)
        self.lbl_estado.setText("Estado: —")
        self.lbl_potencia.setText("Potencia medida: —")
        self.lbl_temp_actual.setText("Actual: —")
        self.lbl_tec.setText("TEC: —")
        self.lbl_estab_temp.setText("Estabilidad térmica: —")
        self.lbl_estab_pow.setText("Estabilidad: —")
        self.lbl_eta.setText("ETA: —")
        self.bar_estab.setValue(0)
        for lbl in self.lbl_nivel.values():
            lbl.setText("—")
        self.estabilidad.reset()
        self._reset_graficos()
        self._log("Desconectado")

    def _reset_graficos(self):
        """Vacía los buffers de los gráficos en tiempo real y los redibuja."""
        self._t0_grafica = None
        self._hist_pot.clear()
        self._hist_temp.clear()
        self.line_pot.set_data([], [])
        self.line_temp.set_data([], [])
        self.line_pot_set.set_visible(False)
        self.ax_pot.set_xlim(0, VENTANA_PLOT_S)
        self.ax_pot.set_ylim(0, 1)
        self.ax_temp.set_xlim(0, VENTANA_PLOT_S)
        self.ax_temp.set_ylim(24, 26)
        self.canvas_pot.draw_idle()
        self.canvas_temp.draw_idle()

    # ------- acciones -------
    def _encender(self):
        self.estabilidad.reset()
        self._ejecutar(lambda: self.driver.encender(), "la on")

    def _apagar(self):
        self._ejecutar(lambda: self.driver.apagar(), "la off")

    def _aplicar_potencia(self, canal: int, mW: float):
        # un cambio de setpoint relanza la fase de estabilización
        self.estabilidad.reset()
        self._ejecutar(lambda: self.driver.set_potencia(canal, mW), f"ch {canal} pow {mW:.3f}")

    def _aplicar_temperatura(self):
        if not self.driver.conectado():
            self._log("!!! No conectado")
            return
        valor = self.spn_temp.value()
        def _tarea():
            try:
                resp = self.driver.set_temperatura_C(valor)
            except Exception as e:
                resp = f"ERROR: {e}"
            self.sig_setp_resp.emit(f"{valor:.2f}", resp)
        threading.Thread(target=_tarea, daemon=True).start()

    def _ejecutar(self, accion, etiqueta: str):
        if not self.driver.conectado():
            self._log("!!! No conectado")
            return
        def _tarea():
            try:
                accion()
                self.sig_log.emit(f">>> {etiqueta}  OK")
            except Exception as e:
                self.sig_log.emit(f"!!! {etiqueta}  ERROR: {e}")
        threading.Thread(target=_tarea, daemon=True).start()

    def _ejecutar_async(self, accion):
        threading.Thread(target=accion, daemon=True).start()

    # ------- polling -------
    def _poll_async(self):
        if not self.driver.conectado():
            return
        def _tarea():
            try:
                estado   = self.driver.estado()
                potencia = self.driver.leer_potencia_uW()
                niveles  = self.driver.leer_niveles()
                temp     = self.driver.leer_temperatura_C()
                tec      = self.driver.leer_estado_tec()
                # Setpoint cambia rara vez: solo lo refrescamos en sincronización inicial
                setpoint = None
                self.sig_estado.emit(estado)
                self.sig_potencia.emit(potencia)
                self.sig_niveles.emit(niveles)
                self.sig_temperatura.emit((temp, setpoint if setpoint is not None
                                            else self.spn_temp.value(), tec))
            except Exception as e:
                self.sig_error_poll.emit(str(e))
        threading.Thread(target=_tarea, daemon=True).start()

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


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

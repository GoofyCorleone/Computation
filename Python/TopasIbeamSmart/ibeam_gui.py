"""
Interfaz gráfica (PyQt6) para el láser TOPTICA iBeam Smart.
Encender/apagar emisión y modificar la potencia de cada canal.

Detecta automáticamente el puerto del adaptador USB-Serial al que está
conectado el iBeam Smart (sondea cada puerto hasta encontrar el prompt
'CMD> ' a 115200 baud).

Nota sobre canales:
  La potencia de salida del iBeam Smart es la SUMA de los canales activos.
  Para que el control de un canal se corresponda con la salida, el otro
  canal debe estar en 0 mW. Los comandos 'en N' / 'di N' no modifican la
  contribución del canal en este firmware — la única forma confiable es
  fijar el nivel con 'ch N pow 0'.
"""

import sys
import threading
import time

import serial
from serial.tools import list_ports
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
    QVBoxLayout, QWidget,
)


BAUD           = 115200
TIMEOUT        = 1.5
PROMPT         = b"CMD> "
INTERVALO_POLL_MS = 700


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
        # macOS: /dev/cu.usbserial-*, /dev/cu.usbmodem*
        # Windows: COMn con descripción CH340/FTDI
        # Linux: /dev/ttyUSB*, /dev/ttyACM*
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
        # Fijar el nivel del canal. Para "apagar" un canal, enviar 0 mW.
        self.enviar(f"ch {canal} pow {mW:.3f}")

    def leer_niveles(self) -> dict[int, float]:
        """Devuelve {1: mW_CH1, 2: mW_CH2} a partir de 'sh level pow'."""
        resp = self.enviar("sh level pow")
        niveles = {}
        for linea in resp.splitlines():
            # Formato: 'CH1, PWR:  0.500 mW'
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
    sig_detect_done   = pyqtSignal(str)  # puerto o "" si no encontrado

    def __init__(self):
        super().__init__()
        self.setWindowTitle("iBeam Smart - Control")
        self.setFixedSize(540, 540)
        self.driver = IBeamDriver()

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
        grid.addWidget(QLabel("<b>Canal</b>"),         0, 0)
        grid.addWidget(QLabel("<b>Setpoint [mW]</b>"), 0, 1)
        grid.addWidget(QLabel(""),                      0, 2)
        grid.addWidget(QLabel("<b>Nivel actual</b>"),   0, 3)

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
        layout.addWidget(gb_em)

        # Log
        gb_log = QGroupBox("Log")
        lay_log = QVBoxLayout(gb_log)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setStyleSheet("background-color: #111; color: #0f0; font-family: Menlo; font-size: 11px;")
        lay_log.addWidget(self.log)
        layout.addWidget(gb_log)

        # Detectar puerto al arrancar (en segundo plano)
        QTimer.singleShot(100, self._auto_detectar)

    def _conectar_senales(self):
        self.sig_log.connect(self._log)
        self.sig_estado.connect(lambda s: self.lbl_estado.setText(f"Estado: {s}"))
        self.sig_potencia.connect(lambda p: self.lbl_potencia.setText(f"Potencia medida: {p/1000:.3f} mW"))
        self.sig_niveles.connect(self._actualizar_niveles)
        self.sig_niveles_spin.connect(self._actualizar_spinboxes)
        self.sig_error_poll.connect(self._manejar_error_poll)
        self.sig_detect_done.connect(self._on_detect_done)

    # ------- acciones -------
    def _log(self, msg: str):
        self.log.appendPlainText(msg)

    def _actualizar_niveles(self, niveles: dict):
        for canal, mW in niveles.items():
            if canal in self.lbl_nivel:
                self.lbl_nivel[canal].setText(f"{mW:.3f} mW")

    def _auto_detectar(self, conectar_automaticamente: bool = True):
        self._auto_connect_pendiente = conectar_automaticamente
        self.btn_detectar.setEnabled(False)
        self.btn_detectar.setText("Buscando...")
        self._log("Buscando puerto del iBeam Smart ...")

        def _tarea():
            puerto = detectar_puerto() or ""
            # Emitir señal: cross-thread signal dispatch correcto
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
            # Sincronizar GUI con el estado actual del dispositivo
            self._ejecutar_async(self._sincronizar_desde_dispositivo)
        except Exception as e:
            QMessageBox.critical(self, "Error de conexión", str(e))

    def _sincronizar_desde_dispositivo(self):
        niveles = self.driver.leer_niveles()
        self.sig_niveles.emit(niveles)
        self.sig_niveles_spin.emit(niveles)
        self.sig_log.emit(f"Niveles leídos del dispositivo: {niveles}")

    def _actualizar_spinboxes(self, niveles: dict):
        for canal, mW in niveles.items():
            if canal in self.spn_pow:
                self.spn_pow[canal].blockSignals(True)
                self.spn_pow[canal].setValue(mW)
                self.spn_pow[canal].blockSignals(False)

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
        for lbl in self.lbl_nivel.values():
            lbl.setText("—")
        self._log("Desconectado")

    def _encender(self):
        self._ejecutar(lambda: self.driver.encender(), "la on")

    def _apagar(self):
        self._ejecutar(lambda: self.driver.apagar(), "la off")

    def _aplicar_potencia(self, canal: int, mW: float):
        self._ejecutar(lambda: self.driver.set_potencia(canal, mW), f"ch {canal} pow {mW:.3f}")

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
                self.sig_estado.emit(estado)
                self.sig_potencia.emit(potencia)
                self.sig_niveles.emit(niveles)
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

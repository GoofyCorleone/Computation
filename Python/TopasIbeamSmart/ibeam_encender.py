"""
Control del láser diodo TOPTICA iBeam Smart vía puerto serial (RS-232).

Funciones:
  - Enciende el láser, mide potencia y temperatura del diodo.
  - Detecta estabilidad de potencia y muestra ETA de estabilización.
  - Genera dos gráficos: potencia(t) y temperatura(t).

Configuración verificada:
  - Baud:    115200
  - Prompt:  'CMD> '
  - Comandos: en/di <ch>, la on/off, sta la, sh pow, ch <n> pow <mW>,
              sh temp, sh syst data, sta tec
"""

import math
import time
from collections import deque

import pylab as pl
import serial


PUERTO  = "/dev/cu.usbserial-1140"
BAUD    = 115200
TIMEOUT = 2.0
PROMPT  = b"CMD> "

# Estabilidad de potencia
VENTANA_ESTAB_S    = 8.0
UMBRAL_REL_ESTABLE = 0.005
MIN_MUESTRAS_EST   = 6
TAU_DIODO_S        = 25.0


def enviar(ser: serial.Serial, comando: str) -> str:
    """Envía un comando y devuelve la respuesta sin prompt ni eco."""
    ser.reset_input_buffer()
    ser.write((comando + "\r\n").encode())
    crudo = ser.read_until(PROMPT).decode(errors="replace")
    lineas = [l.strip() for l in crudo.splitlines() if l.strip() and l.strip() != "CMD>"]
    if lineas and lineas[0].lower() == comando.lower():
        lineas = lineas[1:]
    return "\n".join(lineas)


def leer_potencia_uW(ser: serial.Serial) -> float:
    resp = enviar(ser, "sh pow")
    for linea in resp.splitlines():
        if "PIC" in linea and "uW" in linea:
            for tok in linea.replace("=", " ").split():
                if tok.isdigit():
                    return float(tok)
    return 0.0


def leer_temperatura_C(ser: serial.Serial) -> float | None:
    """Lee la temperatura del diodo ('TEMP = 025.0 C')."""
    resp = enviar(ser, "sh temp")
    for linea in resp.splitlines():
        s = linea.strip().upper()
        if s.startswith("TEMP") and "=" in s:
            try:
                return float(s.split("=", 1)[1].replace("C", "").strip())
            except ValueError:
                pass
    return None


def leer_setpoint_temp_C(ser: serial.Serial) -> float | None:
    """Lee el setpoint del TEC (línea 'TEC setpoint: ... -> 025.0 C')."""
    resp = enviar(ser, "sh syst data")
    for linea in resp.splitlines():
        s = linea.strip()
        if s.lower().startswith("tec setpoint") and "->" in s:
            try:
                return float(s.split("->", 1)[1].replace("C", "").strip())
            except ValueError:
                pass
    return None


def estado_estabilidad(historial: deque) -> tuple[str, float | None]:
    """Devuelve (texto, eta_segundos) según el historial reciente."""
    n = len(historial)
    if n < MIN_MUESTRAS_EST:
        return ("sin datos", None)
    ts = [t for t, _ in historial]
    ps = [p for _, p in historial]
    media = sum(ps) / n
    if abs(media) < 1e-6:
        return ("sin emisión", None)
    var = sum((p - media) ** 2 for p in ps) / n
    std = math.sqrt(var)
    t_med = sum(ts) / n
    num = sum((ts[i] - t_med) * (ps[i] - media) for i in range(n))
    den = sum((ts[i] - t_med) ** 2 for i in range(n))
    pendiente = num / den if den > 1e-9 else 0.0
    rel_std   = std / abs(media)
    rel_drift = abs(pendiente * VENTANA_ESTAB_S) / abs(media)
    rel = max(rel_std, rel_drift)
    if rel < UMBRAL_REL_ESTABLE:
        return ("ESTABLE", 0.0)
    eta = TAU_DIODO_S * math.log(rel / UMBRAL_REL_ESTABLE)
    eta = max(0.0, min(eta, 600.0))
    texto = "estabilizando" if rel < 5 * UMBRAL_REL_ESTABLE else "calentando"
    return (texto, eta)


def main():
    print(f"Conectando a iBeam Smart en {PUERTO} a {BAUD} baud ...")
    with serial.Serial(PUERTO, BAUD, timeout=TIMEOUT) as laser:
        time.sleep(1.0)
        laser.read_until(PROMPT)

        enviar(laser, "la off")

        estado    = enviar(laser, "sta la")
        niveles   = enviar(laser, "sh level pow")
        setpoint  = leer_setpoint_temp_C(laser)
        tec       = enviar(laser, "sta tec").strip()
        temp_ini  = leer_temperatura_C(laser)
        print(f"Estado inicial    : {estado}")
        print(f"TEC               : {tec}, setpoint = {setpoint} °C, actual = {temp_ini} °C")
        print(f"Niveles           :\n{niveles}")

        # Configurar CH1 = 5 mW, CH2 = 0 mW
        potencia_mW = 5.0
        print(f"\nConfigurando CH1 a {potencia_mW} mW (CH2 = 0 mW) ...")
        enviar(laser, "ch 2 pow 0")
        enviar(laser, f"ch 1 pow {potencia_mW}")
        enviar(laser, "en 1")

        print("Encendiendo láser ...")
        enviar(laser, "la on")
        time.sleep(0.3)

        estado   = enviar(laser, "sta la")
        potencia = leer_potencia_uW(laser)
        print(f"Estado            : {estado}")
        print(f"Potencia inicial  : {potencia/1000:.3f} mW")

        print("\nAdquiriendo potencia y temperatura durante 12 s ...")
        tiempos, potencias, temperaturas = [], [], []
        historial = deque()
        t0 = time.time()
        ultima_temp = temp_ini if temp_ini is not None else 25.0
        while time.time() - t0 < 12.0:
            t = time.time() - t0
            p_mW = leer_potencia_uW(laser) / 1000.0
            temp = leer_temperatura_C(laser)
            if temp is not None:
                ultima_temp = temp
            tiempos.append(t)
            potencias.append(p_mW)
            temperaturas.append(ultima_temp)
            historial.append((t, p_mW))
            t_min = t - VENTANA_ESTAB_S
            while historial and historial[0][0] < t_min:
                historial.popleft()
            est, eta = estado_estabilidad(historial)
            eta_str = f"{eta:.0f} s" if eta is not None else "—"
            print(
                f"  t={t:5.1f} s  P={p_mW:.3f} mW  T={ultima_temp:.2f} °C  "
                f"[{est:14s}]  ETA: {eta_str}",
                end="\r",
            )
            time.sleep(0.25)
        print()

        print("\nApagando láser ...")
        enviar(laser, "la off")
        estado_final = enviar(laser, "sta la")
        print(f"Estado final      : {estado_final}")

    # Gráfico potencia + temperatura
    fig, (ax1, ax2) = pl.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(tiempos, potencias, "b.-", markersize=4)
    ax1.set_ylabel("Potencia [mW]")
    ax1.set_title("iBeam Smart — potencia y temperatura durante emisión")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(potencia_mW, color="gray", linestyle="--", linewidth=0.8,
                label=f"Setpoint {potencia_mW} mW")
    ax1.legend(loc="lower right")

    ax2.plot(tiempos, temperaturas, "r.-", markersize=4)
    ax2.set_ylabel("Temperatura [°C]")
    ax2.set_xlabel("Tiempo [s]")
    if setpoint is not None:
        ax2.axhline(setpoint, color="gray", linestyle="--", linewidth=0.8,
                    label=f"Setpoint TEC {setpoint:.1f} °C")
        ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    pl.tight_layout()
    pl.savefig("potencia_laser.png", dpi=120)
    print("\nGráfico guardado: potencia_laser.png")
    pl.show()


if __name__ == "__main__":
    main()

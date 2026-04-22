"""
Control del láser diodo TOPTICA iBeam Smart via puerto serial (RS-232).

Configuración verificada:
  - Puerto:     /dev/cu.usbserial-1130
  - Baud rate:  115200
  - Prompt:     'CMD> '
  - Comandos:   en <ch>, di <ch>, la on, la off, sta la, sh pow, ch <n> pow <mW>
"""

import time
import pylab as pl
import serial


PUERTO  = "/dev/cu.usbserial-1130"
BAUD    = 115200
TIMEOUT = 2.0
PROMPT  = b"CMD> "


def enviar(ser: serial.Serial, comando: str) -> str:
    """Envía un comando y devuelve la respuesta sin prompt ni eco."""
    ser.reset_input_buffer()
    ser.write((comando + "\r\n").encode())
    crudo = ser.read_until(PROMPT).decode(errors="replace")
    # Quitar prompt y líneas vacías; quitar eco si aparece
    lineas = [l.strip() for l in crudo.splitlines() if l.strip() and l.strip() != "CMD>"]
    if lineas and lineas[0].lower() == comando.lower():
        lineas = lineas[1:]
    return "\n".join(lineas)


def leer_potencia_uW(ser: serial.Serial) -> float:
    """Lee la potencia actual en microwatts desde 'sh pow'."""
    resp = enviar(ser, "sh pow")
    # Busca patrón 'PIC  = NNNNNN uW'
    for linea in resp.splitlines():
        if "PIC" in linea and "uW" in linea:
            partes = linea.replace("=", " ").split()
            for tok in partes:
                if tok.isdigit():
                    return float(tok)
    return 0.0


def main():
    print(f"Conectando a iBeam Smart en {PUERTO} a {BAUD} baud ...")
    with serial.Serial(PUERTO, BAUD, timeout=TIMEOUT) as laser:
        time.sleep(1.0)
        laser.read_until(PROMPT)  # consumir prompt inicial

        # Apagar por seguridad al arrancar
        enviar(laser, "la off")

        estado = enviar(laser, "sta la")
        print(f"Estado inicial : {estado}")

        niveles = enviar(laser, "sh level pow")
        print(f"Niveles        :\n{niveles}")

        # Configurar canal 1 a 5 mW y habilitarlo
        potencia_mW = 5.0
        print(f"\nConfigurando CH1 a {potencia_mW} mW ...")
        enviar(laser, f"ch 1 pow {potencia_mW}")
        enviar(laser, "en 1")

        # Encender emisión
        print("Encendiendo láser ...")
        enviar(laser, "la on")
        time.sleep(0.3)

        estado = enviar(laser, "sta la")
        potencia = leer_potencia_uW(laser)
        print(f"Estado         : {estado}")
        print(f"Potencia       : {potencia:.1f} uW ({potencia/1000:.3f} mW)")

        # Medir potencia durante unos segundos y graficar con pylab
        print("\nMidiendo potencia durante 5 s ...")
        tiempos, potencias = [], []
        t0 = time.time()
        while time.time() - t0 < 5.0:
            tiempos.append(time.time() - t0)
            potencias.append(leer_potencia_uW(laser) / 1000.0)  # mW
            time.sleep(0.1)

        # Apagar
        print("Apagando láser ...")
        enviar(laser, "la off")
        estado = enviar(laser, "sta la")
        print(f"Estado final   : {estado}")

    # Graficar evolución temporal
    pl.figure(figsize=(8, 4))
    pl.plot(tiempos, potencias, "b.-", markersize=4)
    pl.xlabel("Tiempo [s]")
    pl.ylabel("Potencia [mW]")
    pl.title("iBeam Smart - potencia durante emisión")
    pl.grid(True, alpha=0.3)
    pl.tight_layout()
    pl.savefig("potencia_laser.png", dpi=120)
    print("\nGráfico guardado: potencia_laser.png")
    pl.show()


if __name__ == "__main__":
    main()

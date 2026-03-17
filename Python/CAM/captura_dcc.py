"""
Captura de imágenes RAW desde cámara Thorlabs DCC (IDS uEye) en macOS.

Estrategia:
  1. Intenta captura con pyueye (SDK nativo, datos raw a profundidad completa).
  2. Si el SDK no está instalado, cae a OpenCV como UVC fallback.

Requiere (sistema):
  - IDS uEye SDK para macOS → https://en.ids-imaging.com/downloads.html
    Busca "uEye SDK" → macOS → instala el .dmg.

Requiere (Python, ya instalados en este venv):
  - pyueye
  - opencv-python
  - numpy
  - Pillow
"""

import numpy as np
import cv2
from PIL import Image
import os
from datetime import datetime

OUTPUT_DIR = "capturas_raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# CAPTURA CON PYUEYE (nativa, raw completo)
# ==============================================================================

def capturar_pyueye(n_imagenes=4, exposicion_ms=10.0):
    """
    Captura n_imagenes usando el SDK uEye.
    Guarda como TIFF de 16 bits (o 8 bits si la cámara no soporta 16).
    Retorna lista de rutas guardadas.
    """
    from pyueye import ueye

    hcam = ueye.HIDS(0)  # 0 = primera cámara disponible

    # Inicializar
    ret = ueye.is_InitCamera(hcam, None)
    if ret != ueye.IS_SUCCESS:
        raise RuntimeError(f"No se pudo inicializar la cámara uEye (código {ret}). "
                           "¿Está instalado el SDK y conectada la cámara?")

    try:
        # Modo de color: raw 8-bit mono (cambiar a IS_CM_MONO12 si la cámara lo soporta)
        ueye.is_SetColorMode(hcam, ueye.IS_CM_MONO8)

        # Obtener tamaño del sensor
        sensor_info = ueye.SENSORINFO()
        ueye.is_GetSensorInfo(hcam, sensor_info)
        width  = int(sensor_info.nMaxWidth)
        height = int(sensor_info.nMaxHeight)
        print(f"  Sensor: {width}×{height} px — {sensor_info.strSensorName.decode()}")

        # Configurar exposición
        exp = ueye.double(exposicion_ms)
        ueye.is_Exposure(hcam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exp, 8)
        print(f"  Exposición: {exposicion_ms} ms")

        # Asignar memoria de imagen
        mem_ptr  = ueye.c_mem_p()
        mem_id   = ueye.int()
        bitdepth = 8  # bits por píxel para MONO8
        ueye.is_AllocImageMem(hcam, width, height, bitdepth, mem_ptr, mem_id)
        ueye.is_SetImageMem(hcam, mem_ptr, mem_id)

        # Modo de captura: single shot
        ueye.is_SetDisplayMode(hcam, ueye.IS_SET_DM_DIB)

        rutas = []
        for i in range(n_imagenes):
            ueye.is_FreezeVideo(hcam, ueye.IS_WAIT)

            # Copiar datos a numpy
            datos = ueye.get_data(mem_ptr, width, height, bitdepth, width, copy=True)
            frame = np.reshape(datos, (height, width))

            # Normalizar a float64 [0, 1] (consistente con CamPol_5)
            frame_float = frame.astype(np.float64) / np.iinfo(frame.dtype).max

            # Guardar como TIFF 16-bit
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            ruta = os.path.join(OUTPUT_DIR, f"captura_{i:02d}_{ts}.tif")
            img_16 = (frame_float * 65535).astype(np.uint16)
            Image.fromarray(img_16).save(ruta)
            rutas.append(ruta)
            print(f"  [{i+1}/{n_imagenes}] Guardada: {ruta}  "
                  f"(min={frame.min()}, max={frame.max()})")

        return rutas

    finally:
        ueye.is_FreeImageMem(hcam, mem_ptr, mem_id)
        ueye.is_ExitCamera(hcam)


# ==============================================================================
# CAPTURA CON OPENCV (fallback UVC — sin control de bit-depth)
# ==============================================================================

def capturar_opencv(n_imagenes=4, indice_camara=0, exposicion=-6.0):
    """
    Captura usando OpenCV (backend AVFoundation en macOS).
    Útil si la cámara expone interfaz UVC. Los datos serán BGR 8-bit.

    NOTA: Para que funcione, debes dar permiso de cámara a Terminal en:
      Configuración del Sistema → Privacidad y Seguridad → Cámara → Terminal ✓
    """
    cap = cv2.VideoCapture(indice_camara, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV no pudo abrir la cámara (índice {indice_camara}).\n"
            "Verifica:\n"
            "  1. Permisos de cámara: Configuración del Sistema → Privacidad → Cámara → Terminal ✓\n"
            "  2. Que la cámara esté conectada y encendida.\n"
            "  3. Prueba otros índices (0, 1, 2...)."
        )

    # Configurar resolución máxima
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  4096)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 4096)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposicion)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Resolución obtenida: {w}×{h} px")

    # Descartar primeros frames (la cámara necesita estabilizarse)
    for _ in range(5):
        cap.read()

    rutas = []
    for i in range(n_imagenes):
        ret, frame = cap.read()
        if not ret:
            print(f"  [!] No se pudo leer el frame {i+1}")
            continue

        # Convertir a escala de grises y normalizar a [0,1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ruta = os.path.join(OUTPUT_DIR, f"captura_{i:02d}_{ts}.tif")
        Image.fromarray(gray).save(ruta)
        rutas.append(ruta)
        print(f"  [{i+1}/{n_imagenes}] Guardada: {ruta}  "
              f"(min={gray.min()}, max={gray.max()})")

    cap.release()
    return rutas


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=== Captura Thorlabs DCC ===\n")

    usar_opencv = False

    # Intentar primero con el SDK nativo (pyueye)
    try:
        print("Intentando captura con SDK uEye (pyueye)...")
        rutas = capturar_pyueye(n_imagenes=4, exposicion_ms=10.0)
        print(f"\n✓ Captura completada con pyueye. {len(rutas)} imágenes en '{OUTPUT_DIR}/'")

    except (ImportError, RuntimeError) as e:
        print(f"[!] pyueye no disponible: {e}")
        usar_opencv = True

    if usar_opencv:
        print("\nIntentando captura con OpenCV (modo UVC)...")
        try:
            rutas = capturar_opencv(n_imagenes=4, indice_camara=0)
            print(f"\n✓ Captura completada con OpenCV. {len(rutas)} imágenes en '{OUTPUT_DIR}/'")
        except RuntimeError as e2:
            print(f"\n[!] OpenCV también falló: {e2}")
            print("\n── Diagnóstico ────────────────────────────────────────────────")
            print("Para cámaras Thorlabs DCC en macOS necesitas UNA de estas opciones:")
            print()
            print("  Opción A (recomendada — raw completo):")
            print("    1. Descarga IDS uEye SDK para macOS:")
            print("       https://en.ids-imaging.com/downloads.html")
            print("    2. Instala el .dmg")
            print("    3. Ejecuta este script de nuevo")
            print()
            print("  Opción B (OpenCV UVC — solo 8-bit):")
            print("    1. Abre: Configuración del Sistema → Privacidad y Seguridad → Cámara")
            print("    2. Activa el permiso para Terminal (o iTerm2)")
            print("    3. Ejecuta este script de nuevo")
            print("────────────────────────────────────────────────────────────────")

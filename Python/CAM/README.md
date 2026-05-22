# CAM — Análisis Polarimétrico

Scripts de análisis de imágenes para cámara polarimétrica. A partir de cuatro fotografías capturadas con distintas orientaciones de polarizador y retardador de cuarto de onda, se reconstruyen los **parámetros de Stokes** completos y se generan mapas 2D del estado de polarización píxel a píxel.

![Panel de resultados](panel_polarimetria.png)

---

## Cómo funciona el código

### 1. Adquisición del montaje óptico

El sistema usa una cámara CCD (Thorlabs DCC) precedida por un polarizador lineal y un retardador de cuarto de onda (λ/4). Al girar ambos elementos a cuatro ángulos distintos se obtienen cuatro medidas de intensidad independientes que permiten determinar el vector de Stokes completo:

```
Medida        Polarizador   Retardador   Variable
─────────────────────────────────────────────────
I_0_0.bmp         0°            0°         I0
I_45_0.bmp        45°           0°         I45
I_90_0.bmp        90°           0°         I90
I_45_90.bmp       45°           90°        I4590
```

### 2. Normalización a punto flotante

Cada imagen BMP (uint8, 8 bits por píxel) se convierte a `float64` en el rango `[0, 1]` dividiendo por el valor máximo del tipo de dato:

```python
image_float = img.astype(np.float64) / np.iinfo(img.dtype).max
```

Esto preserva la precisión numérica completa durante todas las operaciones algebraicas posteriores, evitando errores de redondeo o desbordamiento que ocurrirían si se trabajara en enteros.

### 3. Cálculo de los parámetros de Stokes

Con las cuatro intensidades se construye el vector de Stokes mediante combinaciones lineales:

```
S0 = I0 + I90              → intensidad total
S1 = I0 − I90              → polarización lineal a 0°/90°
S2 = 2·I45 − I0 − I90      → polarización lineal a ±45°
S3 = 2·I4590 − I0 − I90    → polarización circular
```

Normalizados respecto a S0 (para eliminar la dependencia de la intensidad de la fuente):

```
s1 = S1/S0,  s2 = S2/S0,  s3 = S3/S0   ∈ [−1, 1]
```

### 4. Propiedades de polarización derivadas

```
DoP  = √(s1² + s2² + s3²)     Grado de polarización total   ∈ [0, 1]
AoLP = ½·arctan2(s2, s1)       Ángulo de polarización lineal ∈ [0°, 180°)
DoCP = |s3|                     Grado de polarización circular ∈ [0, 1]
```

### 5. Enmascarado

Se aplica un umbral sobre S0 (`umbral = 0.05`) para ignorar píxeles sin señal (fondo oscuro, bordes de la apertura). Los píxeles enmascarados se ponen a cero en todas las salidas.

### 6. Visualización

Se genera un panel de seis mapas (`panel_polarimetria.png`):

| Panel | Descripción |
|---|---|
| S0 | Intensidad total (escala de grises) |
| s1 | Stokes lineal horizontal/vertical (mapa rojo–azul) |
| s2 | Stokes lineal diagonal (mapa rojo–azul) |
| DoP | Grado de polarización (mapa térmico) |
| AoLP | Ángulo de polarización en grados (mapa cíclico HSV) |
| Falso color | Imagen RGB con R=s1, G=s2, B=s3 |

---

## Instalación

```bash
python -m venv venv
source venv/bin/activate
pip install numpy matplotlib opencv-python Pillow
```

## Uso

1. Coloca las cuatro imágenes BMP en el directorio del proyecto con los nombres exactos:
   `I_0_0.bmp`, `I_45_0.bmp`, `I_90_0.bmp`, `I_45_90.bmp`

2. Ejecuta:
   ```bash
   source venv/bin/activate
   python CamPol_3.py
   ```

3. El script imprime en consola las estadísticas promedio de la región iluminada y abre el panel de resultados. Todos los mapas se guardan como PNG en el mismo directorio.

---

## Archivos de salida

| Archivo | Contenido |
|---|---|
| `panel_polarimetria.png` | Panel resumen con los seis mapas |
| `DoLP_map.png` | Grado de polarización lineal [0, 1] |
| `AoLP_map.png` | Ángulo de polarización lineal [0°, 180°) |
| `DoCP_map.png` | Grado de polarización circular [0, 1] |
| `RGB_Stokes.png` | Falso color (R=s1, G=s2, B=s3) |
| `Mask.png` | Máscara binaria de píxeles válidos |

---

## Estructura del proyecto

| Archivo | Descripción |
|---|---|
| `CamPol_3.py` | Script principal — BMP entrada, Stokes en float64 |
| `CamPol_4.py` | Variante con enmascarado adicional basado en S0 |
| `CamPol_5.py` | Pipeline completo con salida estadística y 4 PNGs |
| `CamPolCalib.py` | Análisis de calibración |
| `captura_dcc.py` | Captura de imágenes desde cámara Thorlabs DCC |

---

## Aplicación ThorCam (GUI nativa macOS)

Aplicación gráfica para captura interactiva con vista en vivo y controles de cámara.

### Características

- **Vista en vivo** con histograma en tiempo real
- **Estadísticas** del frame: min/max/media/desviación estándar/% saturación
- **Controles de cámara:** exposición, ganancia
- **Procesamiento software:** brillo, contraste, falso color (mapa térmico), resaltado de saturación
- **Captura rápida** con tecla `Espacio` o botón
- **5 formatos de guardado:**
  - `PNG` — 8-bit sin pérdida
  - `JPG` — 8-bit comprimido
  - `TIFF 8-bit`
  - `TIFF float32` — **valores decimales en [0,1]**, para polarimetría
  - `NPY float64` — formato NumPy nativo, máxima precisión
- **Prefijos polarimétricos preconfigurados:** `I_0_0`, `I_45_0`, `I_90_0`, `I_45_90` (compatibles con `CamPol_3.py`)

### Ejecutar desde código fuente

```bash
source venv/bin/activate
pip install PySide6 tifffile
python ThorCam_App.py
```

### Generar aplicación nativa macOS

```bash
./build_app.sh
```

Esto produce `dist/ThorCam.app`. Para instalarla:

```bash
mv dist/ThorCam.app /Applications/
open /Applications/ThorCam.app
```

La primera vez macOS pedirá permiso de cámara (`Configuración → Privacidad y Seguridad → Cámara`).

### Flujo polarimétrico con ThorCam

1. Conecta la CCD y abre ThorCam
2. Ajusta exposición/ganancia hasta evitar saturación (panel de estadísticas)
3. Selecciona formato **TIFF float32** o **NPY**
4. Para cada combinación polarizador/retardador:
   - Selecciona el prefijo polarimétrico correspondiente (`I_0_0`, etc.)
   - Presiona `Espacio` para capturar
5. Mueve las cuatro imágenes a la carpeta del proyecto y ejecuta `python CamPol_3.py`

> **Nota sobre macOS y cámaras DCC:** IDS Imaging descontinuó el SDK uEye para macOS. La aplicación intenta primero el backend nativo (`pyueye`) si está disponible; si no, usa OpenCV (UVC). Para acceso completo a la DCC en Mac, usar **IDS Peak** (`https://en.ids-imaging.com/ids-peak.html`) o ejecutar la app desde Windows/Linux donde el SDK funciona.

## Captura por línea de comandos (`captura_dcc.py`)

```bash
python captura_dcc.py
```

Versión CLI: captura 4 imágenes secuencialmente y las guarda como TIFF en `capturas_raw/`.

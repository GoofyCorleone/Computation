# CAM — Análisis Polarimétrico

Scripts de análisis de imágenes para cámara polarimétrica. Calculan los parámetros de Stokes y las propiedades de polarización derivadas (DoP, AoLP, DoCP) a partir de cuatro imágenes capturadas con distintas combinaciones de polarizador y retardador.

## Estructura

| Archivo | Descripción |
|---|---|
| `CamPol_3.py` | Script principal — BMP entrada, Stokes en float64, panel de salida |
| `CamPol_4.py` | Variante con enmascarado adicional basado en S0 |
| `CamPol_5.py` | Pipeline completo con salida estadística y 4 PNGs |
| `CamPolCalib.py` | Análisis de calibración |
| `captura_dcc.py` | Captura de imágenes desde cámara Thorlabs DCC |

## Instalación

```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
pip install numpy Pillow matplotlib opencv-python
```

## Datos de entrada

Cuatro imágenes BMP en el directorio raíz del proyecto:

| Archivo | Polarizador | Retardador |
|---|---|---|
| `I_0_0.bmp` | 0° | 0° |
| `I_45_0.bmp` | 45° | 0° |
| `I_90_0.bmp` | 90° | 0° |
| `I_45_90.bmp` | 45° | 90° |

Las imágenes deben ser en escala de grises o BGRA de 8 bits (1024×1280 px).

## Uso

```bash
source venv/bin/activate
python CamPol_3.py
```

El script imprime las estadísticas de polarización en consola y guarda las imágenes de salida.

## Salidas

| Archivo | Contenido |
|---|---|
| `DoLP_map.png` | Mapa del grado de polarización lineal [0, 1] |
| `AoLP_map.png` | Mapa del ángulo de polarización lineal [0°, 180°) |
| `DoCP_map.png` | Mapa del grado de polarización circular [0, 1] |
| `RGB_Stokes.png` | Imagen de falso color (R=s1, G=s2, B=s3) |
| `Mask.png` | Máscara de píxeles con señal válida |
| `panel_polarimetria.png` | Panel resumen con todos los mapas |

## Convención de Stokes

```
S0  = I0 + I90                    intensidad total
S1  = I0 - I90                    polarización lineal horizontal/vertical
S2  = 2·I45 - I0 - I90            polarización lineal diagonal
S3  = 2·I45_90 - I0 - I90         polarización circular

Normalizados (respecto a S0):
  s1 = S1/S0,  s2 = S2/S0,  s3 = S3/S0   ∈ [-1, 1]

DoP  = sqrt(s1² + s2² + s3²)      ∈ [0, 1]
AoLP = 0.5·arctan2(s2, s1)        ∈ [0°, 180°)
DoCP = |s3|                        ∈ [0, 1]
```

Las imágenes se normalizan a `float64` en `[0, 1]` antes de calcular los parámetros (`image_float = img / 255`), lo que garantiza precisión numérica completa.

## Captura con cámara Thorlabs DCC

```bash
python captura_dcc.py
```

Intenta captura con el SDK IDS uEye (pyueye). Si no está instalado, usa OpenCV como fallback UVC. Las imágenes se guardan en `capturas_raw/` como TIFF compatibles con `CamPol_5.py`.

Para habilitar el SDK nativo descarga IDS Peak desde `https://en.ids-imaging.com/ids-peak.html`.

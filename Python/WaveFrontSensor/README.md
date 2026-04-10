# WaveFrontSensor

Módulo de análisis de aberraciones ópticas para el sensor de frente de onda Shack-Hartmann (Thorlabs WFS). Procesa los archivos CSV que exporta el sensor, reconstruye el frente de onda a partir de los coeficientes de Zernike y caracteriza las aberraciones ópticas en el formalismo clásico de Seidel.

---

## Contenido

```
WaveFrontSensor/
├── SensorWavefront.py       ← módulo principal (este archivo)
├── ModuloAberraciones.py    ← versión anterior (referencia)
├── lenteThorlabs.csv        ← datos de lente Thorlabs (ejemplo)
├── lenteImpresa.csv         ← datos de lente impresa en 3D
├── plano.csv                ← referencia de plano óptico
└── Datos_3D/
    └── 25_10_25/            ← mediciones multi-ángulo
        ├── Analisis.py      ← script de análisis en lote
        ├── Horizontal0_difpupila/  {−20°…+20°}.csv
        ├── Horizontal1/
        ├── Horizontal2/
        └── vertical/
```

---

## Dependencias

```bash
pip install numpy matplotlib scipy chardet
```

---

## Uso rápido

```python
from SensorWavefront import SensorShackHartmann

sensor = SensorShackHartmann('lenteThorlabs.csv', guardar=True)
resultados = sensor.analizar()
```

Genera tres figuras y, con `guardar=True`, exporta un PDF multipágina y un reporte `.txt`.

También desde terminal:

```bash
python SensorWavefront.py lenteThorlabs.csv
```

---

## API

### `SensorShackHartmann(archivo_csv, guardar, carpeta_salida, angulo)`

| Parámetro | Tipo | Descripción |
|---|---|---|
| `archivo_csv` | `str` | Ruta al CSV exportado por el software Thorlabs WFS |
| `guardar` | `bool` | Si `True`, exporta PDF y TXT al terminar (defecto: `False`) |
| `carpeta_salida` | `str` | Directorio de salida; si `None`, escribe junto al CSV |
| `angulo` | `float` | Ángulo de inclinación en grados para mediciones multi-ángulo |

> **Alias de compatibilidad:** también acepta `guardar_archivos` y `nombre_carpeta`
> (parámetros usados por `Datos_3D/25_10_25/Analisis.py`).

### Métodos principales

| Método | Descripción |
|---|---|
| `.analizar(mostrar=True)` | Ejecuta el pipeline completo; devuelve dict con todos los resultados |
| `.reconstruir_wavefront()` | Calcula W_rec = Σ cⱼ · Zₙᵐ(ρ,θ) sobre la malla del CSV |
| `.calcular_aberraciones()` | Mapas por grupo de aberración + coeficientes de Seidel |
| `.figura_wavefront()` | Figura 1: mapas 2D/3D y residuo |
| `.figura_zernike()` | Figura 2: espectro de coeficientes Zernike |
| `.figura_aberraciones()` | Figura 3: mapas de aberraciones individuales + tabla de Seidel |
| `.exportar_pdf(figs)` | Guarda lista de figuras como PDF multipágina |
| `.exportar_txt(aber)` | Escribe reporte numérico completo en `.txt` |

### Dict de retorno de `.analizar()`

```python
{
  'terminos':     [{'n': int, 'm': int, 'c': float}, ...],
  'W_medido':     ndarray (21×21) [µm],
  'W_rec':        ndarray (21×21) [µm],
  'X':            ndarray [mm],
  'Y':            ndarray [mm],
  'aberraciones': {
      'grupos':       {'Pistón': {...}, 'Desenfoque': {...}, ...},
      'W_ho':         ndarray,   # frente sin pistón ni inclinación
      'metricas':     {'rms_total', 'pv_total', 'strehl', 'diametro_pupila'},
      'tabla_seidel': [(nombre, c_Z, W_Seidel, forma), ...]
  },
  'figuras': [fig1, fig2, fig3]
}
```

---

## Tutorial paso a paso

### 1. Análisis de un solo archivo

```python
from SensorWavefront import SensorShackHartmann

# Análisis con exportación
s = SensorShackHartmann('lenteThorlabs.csv', guardar=True)
r = s.analizar()

# Acceder a los coeficientes de Zernike
for t in r['terminos']:
    print(f"n={t['n']:>2}  m={t['m']:>+3}  c = {t['c']:>8.4f} µm")
```

Salida (lenteThorlabs.csv):
```
n= 0  m= +0  c =  -23.5050 µm   Pistón
n= 1  m= -1  c =   11.5780 µm   Inclinación Y (Tilt)
n= 1  m= +1  c =   15.2260 µm   Inclinación X (Tip)
n= 2  m= -2  c =   -0.1730 µm   Astigmatismo 45°
n= 2  m= +0  c =   -3.3700 µm   Desenfoque
...
```

### 2. Acceder a métricas de calidad

```python
met = r['aberraciones']['metricas']
print(f"PV     = {met['pv_total']:.4f} µm")
print(f"RMS    = {met['rms_total']:.4f} µm")
print(f"Strehl ≈ {met['strehl']:.4f}")
print(f"Ø pup. = {met['diametro_pupila']:.3f} mm")
```

### 3. Coeficientes de Seidel

```python
for nombre, c_Z, W_Seidel, forma in r['aberraciones']['tabla_seidel']:
    print(f"{nombre:<22}  c_Z = {c_Z:+.4f} µm   W_S = {W_Seidel:+.4f} µm   {forma}")
```

Los coeficientes de Seidel se derivan de los coeficientes de Zernike normalizados
según las relaciones:

| Aberración | Relación |
|---|---|
| Desenfoque | W₀₂₀ = 2√3 · c(2,0) |
| Astigmatismo | W₂₂₂ = 2√6 · √(c(2,−2)² + c(2,2)²) |
| Coma | W₁₃₁ = 3√8 · √(c(3,−1)² + c(3,1)²) |
| Esférica primaria | W₀₄₀ = 6√5 · c(4,0) |

### 4. Comparar dos lentes

```python
from SensorWavefront import SensorShackHartmann

for archivo in ['lenteThorlabs.csv', 'lenteImpresa.csv']:
    s = SensorShackHartmann(archivo)
    s._leer_csv()
    s._parsear_coeficientes()
    s._parsear_wavefront()
    s.reconstruir_wavefront()
    aber = s.calcular_aberraciones()
    met  = aber['metricas']
    print(f"{archivo:<25}  RMS={met['rms_total']:.4f} µm  Strehl={met['strehl']:.4f}")
```

### 5. Análisis en lote (multi-ángulo)

El script `Datos_3D/25_10_25/Analisis.py` analiza series de mediciones a
distintos ángulos de inclinación (−20° a +20°, paso 2°):

```python
import SensorWavefront as sw
import numpy as np

rutas = ['Horizontal1/{}.csv'.format(i) for i in np.arange(-20, 22, 2)]

for angulo, ruta in zip(np.arange(-20, 22, 2), rutas):
    sensor = sw.SensorShackHartmann(
        archivo_csv=ruta,
        guardar=True,
        carpeta_salida='AnalisisHorizontal1',
        angulo=angulo
    )
    sensor.main()
```

Cada iteración produce `AnalisisHorizontal1/<ruta>_<angulo>deg_analisis.pdf`
y el reporte `.txt` correspondiente.

---

## Fundamento físico

### Polinomios de Zernike

El frente de onda se expande en la base de polinomios de Zernike normalizados
(convención OSA/ANSI) sobre la pupila circular de radio R:

```
W(x,y) = Σⱼ  cⱼ · Zₙⱼᵐʲ(ρ, θ)      ρ = √(x²+y²)/R,  θ = arctan2(y,x)
```

Los polinomios se definen como:

```
Zₙᵐ(ρ,θ) = Nₙᵐ · Rₙ|ᵐ|(ρ) · { cos(mθ)  si m > 0
                                { sin(|m|θ) si m < 0
                                { 1         si m = 0

Nₙᵐ = √(2(n+1))  (m≠0),  √(n+1)  (m=0)
```

La normalización garantiza que el coeficiente `cⱼ` es directamente el RMS
de la contribución de ese término sobre la pupila (en µm).

El CSV de Thorlabs WFS exporta los coeficientes en orden OSA/ANSI estricto,
con las columnas `Order = n` y `Frequency = m` para identificar cada término
de forma inequívoca.

### Aberraciones de Seidel

Las aberraciones primarias (Seidel) se obtienen de los coeficientes de Zernike
de 2.° y 3.° orden. En notación de Hopkins `W(H, ρ, θ)` para un punto en el eje
(H = 0), los términos supervivientes son:

| Nombre | Forma | Índices Zernike |
|---|---|---|
| Desenfoque | W₀₂₀ ρ² | (n=2, m=0) |
| Astigmatismo | W₂₂₂ ρ² cos²θ | (n=2, m=±2) |
| Coma | W₁₃₁ ρ³ cosθ | (n=3, m=±1) |
| Esférica primaria | W₀₄₀ ρ⁴ | (n=4, m=0) |

Los coeficientes W_klm (en µm sobre pupila unitaria) se calculan extrayendo el
término de mayor potencia en ρ de cada polinomio de Zernike balanceado.

### Estimación de Strehl

Usando la aproximación de Maréchal (válida exactamente para RMS ≪ λ):

```
SR ≈ exp[−(2π · RMS / λ)²]
```

con λ = 0.6328 µm (He-Ne). Un sistema limitado por difracción tiene SR > 0.8,
equivalente a RMS < λ/14 ≈ 0.045 µm.

---

## Ejemplo de resultados: lenteThorlabs.csv

```
Coeficientes dominantes
  n=0  m= 0   c = −23.505 µm   Pistón          (offset de referencia)
  n=1  m=−1   c = +11.578 µm   Inclinación Y   (desalineación mecánica)
  n=1  m=+1   c = +15.226 µm   Inclinación X   (desalineación mecánica)
  n=2  m= 0   c =  −3.370 µm   Desenfoque      (foco fuera del plano imagen)

Métricas (sin pistón ni inclinación)
  PV     = 12.21 µm
  RMS    =  3.42 µm
  Strehl ≈  0.00   (dominado por desenfoque)

Aberraciones de alto orden (≤ 0.03 µm)
  Astigmatismo, Coma, Esférica — todos < λ/20
  → La lente tiene calidad óptica excelente; el error medido
    es instrumental (desenfoque + desalineación del montaje).
```

---

## Historial de versiones

| Archivo | Versión | Descripción |
|---|---|---|
| `Analisis.py` | 1.0 | Función procedural simple |
| `Anallisis2.py` / `Analisis3.py` | 1.x | Añade detección de codificación |
| `Analisis4.py` | 2.0 | Refactorizado en funciones separadas |
| `AberrationsModule.py` | 3.0 | Clase OO sin exportación |
| `ModuloAberraciones.py` | 3.1 | Clase OO con PDF/TXT — **índices Zernike incorrectos** |
| **`SensorWavefront.py`** | **4.0** | **Lee (n,m) del CSV · reconstrucción correcta · Seidel** |

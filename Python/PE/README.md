# Experimentos de Polarización (PE)

Conjunto de scripts y cuadernos Jupyter para estudiar la **ley de Malus generalizada** con polarización elíptica, tanto en simulación como con hardware real de conteo de fotones.

---

## Tabla de contenidos

1. [Fundamentos teóricos](#fundamentos-teóricos)
   - [Vector de Jones y polarización elíptica](#vector-de-jones-y-polarización-elíptica)
   - [El polarizador elíptico](#el-polarizador-elíptico)
   - [Relación (θ₁, θ₂) ↔ (α, χ)](#relación-θ₁-θ₂--α-χ)
   - [Ley de Malus generalizada](#ley-de-malus-generalizada)
   - [Casos límite](#casos-límite)
2. [Estructura del proyecto](#estructura-del-proyecto)
3. [Dependencias](#dependencias)
4. [Tutorial: GUI de conteo de fotones](#tutorial-gui-de-conteo-de-fotones)
   - [Requisitos de hardware](#requisitos-de-hardware)
   - [Arrancar la aplicación](#arrancar-la-aplicación)
   - [Conexión de dispositivos](#conexión-de-dispositivos)
   - [Configurar el experimento](#configurar-el-experimento)
   - [Configurar los polarizadores elípticos](#configurar-los-polarizadores-elípticos)
   - [Iniciar y tomar puntos](#iniciar-y-tomar-puntos)
   - [Interpretar las gráficas](#interpretar-las-gráficas)
   - [Guardar los datos](#guardar-los-datos)
5. [Scripts y cuadernos](#scripts-y-cuadernos)
6. [Formato de datos Thorlabs](#formato-de-datos-thorlabs)

---

## Fundamentos teóricos

### Vector de Jones y polarización elíptica

Un estado de polarización elíptica con ángulo de orientación α (del eje mayor de la elipse respecto al eje x) y ángulo de elipticidad χ (relacionado con la razón de los semiejes, χ ∈ [−π/4, π/4]) se representa con el vector de Jones en la convención de Azzam:

```
|ψ(α, χ)⟩ = R(α) · (cos χ, i sin χ)ᵀ
```

donde R(α) es la matriz de rotación bidimensional. Los casos especiales son:

| χ | Tipo de polarización |
|---|---|
| 0 | Lineal (eje mayor a α) |
| +π/4 | Circular izquierda |
| −π/4 | Circular derecha |
| otro | Elíptica general |

### El polarizador elíptico

El esquema óptico que produce (o filtra) un estado elíptico arbitrario consiste en tres elementos en serie:

```
P(θ₁, θ₂) = Q(θ₁ + 90°) · PL(θ₂) · Q(θ₁)
```

donde:

- **Q(θ)** — Lámina de cuarto de onda (QWP) con eje rápido a θ respecto al eje x. Matriz de Jones:

```
Q(θ) = Rᵀ(θ) · diag(1, i) · e^{−iπ/4} · R(θ)
```

- **PL(θ₂)** — Polarizador lineal ideal con eje de transmisión a θ₂:

```
PL(θ₂) = |e(θ₂)⟩⟨e(θ₂)|   con   e(θ₂) = (cos θ₂, sin θ₂)ᵀ
```

La deducción algebraica completa (verificada con SymPy) se encuentra en [`Malus_Generalizada.ipynb`](Malus_Generalizada.ipynb).

### Relación (θ₁, θ₂) ↔ (α, χ)

Del formalismo algebraico del artículo base (Ecs. 31-32):

```
α = θ₁           (orientación de la elipse)
χ = θ₁ − θ₂      (elipticidad)
```

Inversamente:
```
θ₁ = α
θ₂ = α − χ
```

Esto significa que para producir un estado con elipse orientada a α con elipticidad χ, basta ajustar los ángulos de montaje a `θ₁ = α` y `θ₂ = α − χ`.

**Ejemplo — polarizador circular izquierdo:** α = 0°, χ = 45° → θ₁ = 0°, θ₂ = −45°.

**Ejemplo — polarizador lineal a 30°:** α = 30°, χ = 0° → θ₁ = 30°, θ₂ = 30°.

### Ley de Malus generalizada

Cuando un haz con polarización elíptica (α_in, χ_in) incide sobre un polarizador elíptico (α_P, χ_P), la intensidad transmitida es:

**Forma de Poincaré:**

```
I(φ) = (I₀/2) [1 + cos2χ_in · cos2χ_P · cos2φ + sin2χ_in · sin2χ_P]
```

**Forma del haversine (equivalente):**

```
I(φ) = I₀ [1 − sin²(χ_in − χ_P) − cos2χ_in · cos2χ_P · sin²φ]
```

donde **φ = α_in − α_P** es el ángulo azimutal relativo entre el haz y el eje del polarizador.

Cuando el segundo polarizador se rota **rígidamente** un ángulo θ desde su posición inicial, ambos cuartos de onda y el polarizador lineal giran en conjunto, con lo cual χ_P permanece constante y α_P varía como `α_P(θ) = α_P₀ + θ`. En consecuencia:

```
φ(θ) = α_in − α_P₀ − θ
```

y la curva medida es exactamente I(θ) según la fórmula anterior.

### Casos límite

| χ_in | χ_P | Resultado |
|---|---|---|
| 0 | 0 | I = I₀ cos²φ (Malus clásica) |
| π/4 | 0 | I = I₀/2 (constante, circular sobre lineal) |
| χ | χ, φ=0 | I = I₀ (transmisión máxima) |
| χ | χ, φ=π/2 | I = I₀ sin²(2χ) |
| 0 | χ_P | I = (I₀/2)[1 + cos2χ_P · cos2φ] |

**Intensidad mínima y máxima:**

```
I_max = I₀ cos²(χ_in − χ_P)
I_min = I₀ sin²(χ_in − χ_P)
```

Cuando χ_in ≠ χ_P la curva **no llega a cero** ni a I₀, lo cual es la firma de la ley generalizada y no puede explicarse con la ley de Malus clásica.

---

## Estructura del proyecto

```
PE/
├── malus_conteo_fotones.py      # GUI principal (PyQt6) — conteo de fotones
├── SoloMalus.py                 # Gráfica rápida desde archivos .txt de Thorlabs
├── Distributions.py             # Biblioteca: esfera de Poincaré + ajuste geométrico
│
├── Malus_Generalizada.ipynb     # Deducción algebraica completa con SymPy
├── MalusLaw_Simulation.ipynb    # Simulación clásica y cuántica de la ley de Malus
├── DataAnalysis.ipynb           # Ajuste de datos reales del SPCM a la fórmula generalizada
├── Propuesta.ipynb              # Derivación Jones de polarizador elíptico sintonizable
│
├── DatosPE/                     # Datos SPCM (5° pasos) + normalización
├── DatosPE2/, DatosPE3/, DatosPE4/   # Corridas adicionales
├── DATICOS/PElip/               # Barridos con polarización elíptica (χ fijo)
├── Malus/                       # Archivos .txt Thorlabs (10° pasos, SoloMalus.py)
└── Malus_Tradicional*/          # Carpetas de salida de malus_conteo_fotones.py
```

---

## Dependencias

```
numpy, matplotlib, pandas, scipy   # todos los scripts y notebooks
plotly                             # Distributions.py, Propuesta.ipynb
sympy                              # Malus_Generalizada.ipynb, Propuesta.ipynb
PyQt6                              # malus_conteo_fotones.py
```

El entorno virtual compartido está en `../venv/`:

```bash
source ../venv/bin/activate
```

---

## Tutorial: GUI de conteo de fotones

`malus_conteo_fotones.py` es la aplicación principal para realizar un barrido completo 0°→360° del segundo polarizador elíptico midiendo fotones en coincidencia con el láser.

### Requisitos de hardware

| Componente | Descripción |
|---|---|
| **TOPTICA iBeam Smart** | Láser sintonizable, conectado por USB/serial |
| **Thorlabs SPCM50A/M** | Contador de fotones de avalancha (APD), conectado por USB |
| **Montura rotativa** | Para el segundo polarizador elíptico (Q·PL·Q) |

Los drivers se importan automáticamente desde `../TopasIbeamSmart/`:
- `ibeam_gui.py` — `IBeamDriver`, `detectar_puerto`
- `spcm_gui.py` — `DriverSPCM`, `detectar_spcm`

### Arrancar la aplicación

```bash
cd Python/PE
source ../venv/bin/activate
python malus_conteo_fotones.py
```

La ventana principal tiene tres zonas:

```
┌─────────────────────────────────────────────────────────────┐
│                   Banner de estado                          │
├──────────────────────┬──────────────────────────────────────┤
│  Panel izquierdo     │  Gráficas en tiempo real             │
│  ─────────────────   │  ┌─────────────┬───────────────────┐ │
│  Configuración       │  │  P(t) láser │  Conteo por bin   │ │
│  Control             │  └─────────────┴───────────────────┘ │
│  Progreso            │  ┌─────────────────────────────────┐ │
│  Log                 │  │   Curva de Malus acumulada      │ │
│                      │  └─────────────────────────────────┘ │
└──────────────────────┴──────────────────────────────────────┘
```

### Conexión de dispositivos

Al iniciar, la aplicación detecta automáticamente el láser y el SPCM. El banner superior indica el estado:

- **Verde** — ambos dispositivos conectados, listo para medir
- **Ámbar** — conexión incompleta (uno o ningún dispositivo detectado)

Si el SPCM no aparece, cierra cualquier instancia abierta del software Thorlabs (que bloquea el USB) y pulsa **"Reintentar SPCM"**.

Para desconectar de forma segura (apaga el láser y libera USB) usa el botón **"Desconectar"**.

### Configurar el experimento

En el panel **"Configuración del experimento"** ajusta los parámetros antes de comenzar:

| Campo | Descripción | Valor típico |
|---|---|---|
| Potencia láser CH1 | Potencia de salida del iBeam Smart | 5 mW |
| Bin Length | Duración de cada ventana temporal de conteo | 1 ms |
| Time between Bins | Pausa entre bins | 0.001 ms |
| Pulse Blind Time | Tiempo de ceguera del APD tras detección | 0 ns |
| Bins per Array | Número de bins por punto angular | 10 000 |
| Paso angular sugerido | Incremento sugerido al pedir el siguiente ángulo | 10° |

El **tiempo estimado por punto** se calcula automáticamente:
```
t = (Bin Length + Time between Bins) × Bins per Array
```
Con los valores por defecto: `(1 + 0.001) × 10 000 / 1000 ≈ 10 s` por ángulo.

### Configurar los polarizadores elípticos

Pulsa **"⚙ Polarizadores elípticos…"** para abrir la ventana de configuración:

```
┌─────────────────────────────────────────────────────────────┐
│  Polarizadores elípticos — Malus generalizada               │
├─────────────────────────────────────────────────────────────┤
│  1.er polarizador elíptico — define el haz incidente        │
│    θ₁  cuartos de onda:    [   0.00 °]                      │
│    θ₂  polarizador lineal: [   0.00 °]                      │
├─────────────────────────────────────────────────────────────┤
│  2.º polarizador elíptico — se rota rígidamente 0°→360°     │
│    θ₁  cuartos de onda:    [   0.00 °]                      │
│    θ₂  polarizador lineal: [   0.00 °]                      │
├─────────────────────────────────────────────────────────────┤
│   α_in =   0.00°    χ_in =   0.00°                         │
│   α_P  =   0.00°    χ_P  =   0.00°                         │
│   → Malus tradicional  cos²(θ)                              │
└─────────────────────────────────────────────────────────────┘
```

**Cómo usar el diálogo:**

1. Introduce los ángulos de los elementos ópticos del **primer polarizador** (el que prepara el haz). Los ángulos en laboratorio del montaje Q·PL·Q se mapean a la elipse como `α_in = θ₁` y `χ_in = θ₁ − θ₂`.

2. Introduce los ángulos **iniciales** del **segundo polarizador** (el que se rota). La elipticidad `χ_P = θ₁ − θ₂` permanece constante durante la rotación; solo varía la orientación `α_P`.

3. El cuadro inferior muestra en tiempo real los parámetros de la elipse resultantes y si la configuración produce la ley de Malus **tradicional** o **generalizada**.

4. Pulsa **OK** — la gráfica de Malus se actualiza al instante con la curva predicha.

5. **Restaurar defaults** (botón "Restore Defaults") lleva todos los ángulos a 0° → Malus tradicional cos²(θ).

**Ejemplos de configuración:**

| Experimento | θ₁¹ | θ₂¹ | θ₁² | θ₂² | Resultado esperado |
|---|---|---|---|---|---|
| Malus tradicional | 0° | 0° | 0° | 0° | I = cos²(θ) |
| Incidencia circular | 0° | −45° | 0° | 0° | I = I₀/2 (plana) |
| Elípticos iguales (χ=20°) | 0° | −20° | 0° | −20° | I varía entre 0 y 1 |
| Elípticos cruzados | 0° | −20° | 0° | 20° | I con piso elevado |

La curva predicha se dibuja automáticamente en la gráfica de Malus **antes de tomar cualquier punto**, para que puedas anticipar la forma del experimento.

### Iniciar y tomar puntos

#### 1. Iniciar medición

Pulsa **"▶ Iniciar medición"**. El láser se enciende a la potencia configurada y la aplicación espera ~4 s para que estabilice. Durante ese tiempo calibra automáticamente el fotodiodo PIC interno del láser:

```
factor_calib = Potencia_setpoint / Lectura_PIC_promedio
```

Este factor corrige la deriva típica (×2) entre la lectura interna y la potencia óptica real emitida.

#### 2. Tomar un punto

Posiciona el segundo polarizador en el ángulo deseado y pulsa **"● Tomar punto"**. Un diálogo te pide confirmar el ángulo (pre-relleno con el siguiente paso sugerido):

```
Ingrese θ en grados (sugerido 10.0°)
(σ_θ = ±0.5° por construcción del soporte)
```

Durante la medición verás en tiempo real:
- **Izquierda:** Potencia calibrada del láser P(t) en µW
- **Derecha:** Histograma de conteos por bin (réplica del software Thorlabs)

Al terminar se calcula automáticamente:
```
CPS_medio = ⟨Counts per Bin⟩ / Bin_Length_s
I_norm = CPS_medio / P_medio
σ_I/I = √[(σ_CPS/CPS)² + (σ_P/P)²]
```

#### 3. Barrido completo

Repite "Tomar punto" para cada ángulo de 0° a 360°. La curva acumulada se actualiza con cada punto, mostrando:

- **Línea de puntos (lila)** — predicción teórica con los ángulos configurados
- **Puntos azules con barras de error** — datos experimentales normalizados

Cuando llegues a 360° el botón **"💾 Cerrar y guardar"** se habilita.

**Opciones adicionales:**

- **"🔄 Reintentar SPCM"** — reconecta el contador sin reiniciar el experimento
- **"🗑 Repetir desde cero"** — borra todos los puntos y apaga el láser (los archivos ya guardados no se borran)

### Interpretar las gráficas

#### Gráfica de Malus

El eje x es el ángulo θ [°], el eje y es I/I_max normalizado.

La **curva teórica** muestra en la leyenda los valores de χ_in y χ_P usados. Si la curva no alcanza cero, es porque `χ_in ≠ χ_P` y la intensidad mínima teórica es `sin²(χ_in − χ_P) > 0`.

Un **buen ajuste** entre la curva teórica y los datos indica que los ángulos de los polarizadores están correctamente caracterizados. Una discrepancia en la **forma** (no solo en la posición del pico) sugiere que los ángulos χ son incorrectos.

#### Gráfica de potencia P(t)

Muestra la deriva del láser durante cada punto. Una variación grande (>2%) indica inestabilidad; considera aumentar el tiempo de integración o revisar la temperatura del láser.

#### Gráfica de conteos (Counts per Bin)

Réplica fiel del histograma del software SPCM50A/M. El número de cuentas por bin debe ser estable en tiempo (sin tendencia), lo que confirma que el láser y el APD funcionan correctamente.

### Guardar los datos

Pulsa **"💾 Cerrar y guardar"** y elige una carpeta. Se crea un subdirectorio con marca de tiempo (`malus_conteo_YYYYMMDD_HHMMSS/`) que contiene:

```
malus_conteo_YYYYMMDD_HHMMSS/
├── datos.txt            # Resumen tabulado: θ, σ_θ, CPS, σ_CPS, P, σ_P, I_norm, σ_I
├── malus_curva.png      # Gráfica de Malus exportada en alta resolución
├── ruido_laser.png      # Deriva y ruido del láser durante todo el barrido
├── conteo_fotones.png   # Tasa de conteo cruda (log scale) por punto
├── bins/                # Un archivo .txt por ángulo con todos los bins
│   └── theta_XXXXX.txt
└── ruido/               # Lecturas de potencia durante cada punto
    └── theta_XXXXX.txt
```

El archivo `datos.txt` tiene cabecera con todos los parámetros del experimento y es directamente legible con `pandas`:

```python
import pandas as pd
df = pd.read_csv('datos.txt', comment='#', sep='\t',
                 names=['theta', 'sigma_theta', 'CPS', 'sigma_CPS',
                        'P_uW', 'sigma_P_uW', 'I_norm', 'sigma_I'])
```

---

## Scripts y cuadernos

### `SoloMalus.py`

Gráfica rápida de archivos `.txt` exportados por el software Thorlabs. Lee todos los archivos en la carpeta `Malus/`, normaliza los conteos y los superpone con cos²(θ).

```bash
python SoloMalus.py
```

### `Distributions.py`

Biblioteca para:
- Leer CSVs con parámetros de Stokes
- Renderizar la esfera de Poincaré interactiva con Plotly
- Ajustar la intersección cono-esfera (ley de los birrefringentes elípticos)

### `Malus_Generalizada.ipynb`

Deducción paso a paso con SymPy:
1. Vector de Jones del estado elíptico general
2. Matriz del polarizador elíptico P(θ₁, θ₂)
3. Verificación de idempotencia y traza
4. Relación (θ₁, θ₂) ↔ (α_P, χ_P)
5. Intensidad general y simplificación a la ley de Malus generalizada
6. Verificación numérica contra el cálculo matricial de Jones

### `MalusLaw_Simulation.ipynb`

Simulación Monte Carlo de la ley de Malus en dos regímenes:
- **Clásico** (`ClassicalMalus`): intensidad continua con ruido gaussiano
- **Cuántico** (`QuantumMalus`): distribución de Poisson sobre la probabilidad de absorción del fotón

Ilustra cómo la estadística de Poisson del conteo de fotones introduce fluctuaciones que siguen la ley de Malus generalizada en promedio.

### `DataAnalysis.ipynb`

Ajuste de datos reales del SPCM (carpetas `DatosPE*`) a la fórmula generalizada:

```python
C12(alpha, chi, alphap, chip) = (
    cos(chi)**2 * cos(chip)**2 * cos(alpha - alphap)**2
    + sin(chi)**2 * sin(chip)**2
    + 0.5 * sin(2*chi) * sin(2*chip) * cos(2*(alpha - alphap))
)
```

Extrae los parámetros óptimos (α_in, χ_in) del haz incidente por mínimos cuadrados.

### `Propuesta.ipynb`

Derivación vía matrices de Jones de un polarizador elíptico sintonizable en tiempo real. Usa `Distributions.py` para la visualización de la trayectoria en la esfera de Poincaré.

---

## Formato de datos Thorlabs

Los archivos `.txt` exportados por el software SPCM50A/M tienen una **cabecera de 19 líneas** antes de los datos tabulados:

```python
import pandas as pd
df = pd.read_csv(ruta, header=19, delimiter='\t')
# Columnas: 'Bin Number', 'Counts per Bin'
```

Esto aplica a todos los archivos en `Malus/`, `DatosPE/`, `DatosPE2/`, etc.

---

## Referencia rápida de la fórmula

```
I(θ) = (I₀/2) · [1 + cos(2χ_in)·cos(2χ_P)·cos(2φ) + sin(2χ_in)·sin(2χ_P)]

φ = α_in − α_P − θ        (ángulo relativo durante la rotación)
α = θ₁                     (orientación de la elipse)
χ = θ₁ − θ₂               (elipticidad)

Casos límite:
  χ_in = χ_P = 0  →  I = I₀ cos²(θ)     [Malus clásica]
  χ_in = π/4, χ_P = 0  →  I = I₀/2      [circular → lineal]
  χ_in = χ_P, φ = 0     →  I = I₀        [transmisión total]
```

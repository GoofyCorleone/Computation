# Prácticas Manim — Esfera de Poincaré

Librería para animar transformaciones de polarización óptica en la **esfera de Poincaré**, implementada con [Manim Community](https://www.manim.community/).

---

## Instalación

```bash
# Desde el directorio Practicas Manim/
python3 -m venv .venv
source .venv/bin/activate
pip install manim
```

---

## Estructura del proyecto

```
Practicas Manim/
├── esfera_poincare/          # Librería principal
│   ├── __init__.py           # Exports públicos
│   ├── fisica.py             # Funciones de óptica (Jones, Stokes)
│   ├── base.py               # Clase base EsferaPoincare + Parametros
│   └── escenas.py            # Escenas reutilizables
├── ejemplos.py               # Cuatro animaciones de demostración
├── Prueba1.py                # Primer ejemplo Manim: círculo animado
└── PuntoEnEsfera.py          # Punto en espiral sobre esfera 3D
```

---

## Uso de la librería

### 1. Importar y subclasificar

```python
from esfera_poincare import Parametros
from esfera_poincare.escenas import RetardadorFijo

class MiRetardador(RetardadorFijo):
    params = Parametros(
        alpha_deg=45,           # orientación de la elipse (°)
        chi_deg=0,              # elipticidad (°)
        retardancia_fija_deg=180,  # media onda
    )
```

```bash
manim -qh mi_script.py MiRetardador
```

### 2. Escenas disponibles

| Escena | Descripción |
|---|---|
| `RetardadorFijo` | Retardador con eje rápido fijo en S₁. El estado gira alrededor de S₁ un ángulo Γ. |
| `PolarizadorParcial` | Polarizador parcial con eje fijo en S₁. El estado se desplaza hacia el polo H al aumentar la diatenuación. |
| `RetardadorGiratorio` | Retardador con Γ fija y eje girando θ: 0° → 360°. El estado traza una curva de Lissajous esférica. |
| `PolarizadorGiratorio` | Polarizador parcial con eje girando θ: 0° → 360°. |

### 3. Parámetros (`Parametros`)

| Atributo | Significado | Rango |
|---|---|---|
| `alpha_deg` | Orientación de la elipse de polarización (°) | [0, 180] |
| `chi_deg` | Elipticidad (°) | [−45, 45] |
| `retardancia_fija_deg` | Retardancia del retardador fijo (°) | — |
| `p1`, `p2` | Transmisiones de amplitud del polarizador parcial | [0, 1], p2 ≤ p1 |
| `retardancia_gir_deg` | Retardancia del retardador giratorio (°) | — |

### 4. API física

```python
from esfera_poincare import (
    jones_desde_angulos,   # (alpha_rad, chi_rad) → vector Jones
    jones_a_stokes,        # vector Jones → (S₁, S₂, S₃)
    jones_retardador,      # (Gamma_rad, theta_rad) → matriz Jones
    jones_pol_parcial,     # (p1, p2, theta_rad) → matriz Jones
)
```

---

## Ejemplos incluidos (`ejemplos.py`)

| Escena | Elemento óptico | Configuración |
|---|---|---|
| `CuartoDeOnda` | Retardador λ/4 fijo | Γ = 90°, θ = 0° |
| `PolParcialFijo` | Polarizador parcial fijo | p₁ = 1, p₂ = 0.3, θ = 0° |
| `CuartoDeOndaGiratorio` | Retardador λ/4 giratorio | Γ = 90°, θ: 0° → 360° |
| `PolParcialGiratorio` | Polarizador parcial giratorio | p₁ = 1, p₂ = 0.3, θ: 0° → 360° |

```bash
source .venv/bin/activate

# Una escena en alta calidad (1080p 60fps):
manim -qh --fps 60 ejemplos.py CuartoDeOnda

# Preview rápido:
manim -ql ejemplos.py CuartoDeOnda

# Todas las escenas:
manim -qh --fps 60 ejemplos.py CuartoDeOnda PolParcialFijo CuartoDeOndaGiratorio PolParcialGiratorio
```

Los videos se guardan en `media/videos/ejemplos/1080p60/`.

---

## Animaciones generadas

### Retardador λ/4 fijo — `CuartoDeOnda`

El estado (α=30°, χ=20°) gira 90° alrededor del eje S₁.

https://github.com/user-attachments/assets/cuarto-de-onda

### Polarizador parcial fijo — `PolParcialFijo`

Con p₂ variando 1 → 0.3, el estado se desplaza hacia el polo H.

### Retardador λ/4 giratorio — `CuartoDeOndaGiratorio`

El eje rápido gira de 0° a 360°. El estado traza una curva de Lissajous esférica (periodo π en θ).

### Polarizador parcial giratorio — `PolParcialGiratorio`

El eje de diatenuación gira de 0° a 360°.

---

## Descripción física

### La esfera de Poincaré

Representación geométrica de los estados de polarización completamente polarizados. Cada punto en la superficie de la esfera unitaria corresponde a un estado único descrito por (S₁, S₂, S₃).

| Punto | (S₁, S₂, S₃) | Polarización |
|---|---|---|
| H | (+1, 0, 0) | Lineal horizontal |
| V | (−1, 0, 0) | Lineal vertical |
| D | (0, +1, 0) | Lineal +45° |
| A | (0, −1, 0) | Lineal −45° |
| R | (0, 0, +1) | Circular derecha |
| L | (0, 0, −1) | Circular izquierda |

### Vector Jones desde (α, χ)

```
Eₓ = cos α · cos χ − i · sin α · sin χ
Eᵧ = sin α · cos χ + i · cos α · sin χ

S₁ = cos(2χ) · cos(2α)
S₂ = cos(2χ) · sin(2α)
S₃ = sin(2χ)
```

### Retardador

Introduce una diferencia de fase Γ entre dos componentes ortogonales del campo. Con el eje en θ, la acción sobre la esfera es una rotación rígida alrededor del eje (cos 2θ, sin 2θ, 0) de ángulo Γ.

### Polarizador parcial

Atenúa las dos componentes con transmisiones p₁ ≥ p₂. Al aumentar la diatenuación (p₂ → 0), el estado se desplaza hacia el polo correspondiente al eje de mayor transmisión.

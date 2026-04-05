# Prácticas Manim — Esfera de Poincaré

Animaciones de transformaciones de polarización en la **esfera de Poincaré**, implementadas con [Manim Community](https://www.manim.community/).

---

## Contenido

| Archivo | Descripción |
|---|---|
| `Prueba1.py` | Primer ejemplo: círculo animado con `Write` |
| `PuntoEnEsfera.py` | Punto moviéndose en espiral sobre una esfera 3D |
| `EsferaPoincare.py` | Simulación completa de transformaciones de polarización en la esfera de Poincaré |

---

## Instalación

Se requiere Python 3.10+ y el entorno virtual compartido del repositorio:

```bash
# Desde la raíz del repositorio Python/
python3 -m venv .venv
source .venv/bin/activate
pip install manim
```

---

## Uso — `EsferaPoincare.py`

### Parámetros de usuario

Todos los parámetros físicos se editan al inicio del archivo:

```python
ALPHA_DEG            = 30    # Orientación de la elipse (°) ∈ [0, 180]
CHI_DEG              = 20    # Elipticidad (°)              ∈ [-45, 45]
RETARDANCIA_FIJA_DEG = 120   # Retardancia del retardador fijo (°)
P1                   = 1.0   # Transmisión eje rápido del polarizador parcial
P2                   = 0.3   # Transmisión eje lento  del polarizador parcial
RETARDANCIA_GIR_DEG  = 90    # Retardancia del retardador giratorio (°)
```

### Renderizar escenas

```bash
source /ruta/a/.venv/bin/activate

# Escena 1 — Punto en la esfera para el estado (alpha, chi)
manim -ql EsferaPoincare.py MostrarPunto

# Escena 2 — Retardador fijo (eje rápido ∥ S₁)
manim -ql EsferaPoincare.py RetardadorFijo

# Escena 3 — Polarizador parcial (eje ∥ S₁)
manim -ql EsferaPoincare.py PolarizadorParcial

# Escena 4 — Retardador giratorio (θ: 0° → 360°)
manim -ql EsferaPoincare.py RetardadorGiratorio

# Alta calidad
manim -qh EsferaPoincare.py <NombreEscena>
```

Los videos se guardan en `media/videos/EsferaPoincare/`.

---

## Descripción física

### La esfera de Poincaré

La esfera de Poincaré es una representación geométrica del espacio de todos los estados de polarización de la luz completamente polarizada. Cada punto en la superficie de la esfera unitaria corresponde a un estado de polarización único, descrito por los parámetros de Stokes normalizados (S₁, S₂, S₃).

Los polos y el ecuador tienen interpretaciones físicas directas:

| Punto | Coordenadas (S₁, S₂, S₃) | Polarización |
|---|---|---|
| H | (+1, 0, 0) | Lineal horizontal |
| V | (−1, 0, 0) | Lineal vertical |
| D | (0, +1, 0) | Lineal +45° |
| A | (0, −1, 0) | Lineal −45° |
| R | (0, 0, +1) | Circular derecha |
| L | (0, 0, −1) | Circular izquierda |

### Vector Jones y ángulos de la elipse

Un estado de polarización elíptica queda caracterizado por:
- **α** (orientación): ángulo de la elipse respecto al eje x, ∈ [0°, 180°]
- **χ** (elipticidad): ángulo de elipticidad, ∈ [−45°, 45°]

El vector Jones correspondiente es:

```
Eₓ = cos α · cos χ − i · sin α · sin χ
Eᵧ = sin α · cos χ + i · cos α · sin χ
```

Y su representación en la esfera de Poincaré:

```
S₁ = cos(2χ) · cos(2α)
S₂ = cos(2χ) · sin(2α)
S₃ = sin(2χ)
```

### Escena 1 — Mostrar punto (α, χ)

Calcula el vector Jones a partir de (α, χ), obtiene (S₁, S₂, S₃) y muestra el punto en la esfera con su radio desde el origen. La cámara gira lentamente para ofrecer perspectiva 3D.

### Escena 2 — Retardador fijo

Un retardador es un elemento óptico birrefringente que introduce una diferencia de fase **Γ** (retardancia) entre dos componentes ortogonales del campo. Con el eje rápido a lo largo de S₁, la acción sobre la esfera de Poincaré es una **rotación rígida alrededor del eje S₁** de ángulo Γ.

Matriz de Jones del retardador con eje rápido en θ:

```
J = ⎡ cos²θ · e^(iΓ/2) + sin²θ · e^(−iΓ/2)    cos θ sin θ · (e^(iΓ/2) − e^(−iΓ/2)) ⎤
    ⎣ cos θ sin θ · (e^(iΓ/2) − e^(−iΓ/2))    sin²θ · e^(iΓ/2) + cos²θ · e^(−iΓ/2) ⎦
```

Para θ = 0 la trayectoria del estado es un arco circular alrededor de S₁.

### Escena 3 — Polarizador parcial

Un polarizador parcial atenúa las dos componentes ortogonales con transmisiones de amplitud p₁ y p₂ (p₁ ≥ p₂). Su matriz de Jones con eje en θ = 0 es:

```
J = ⎡ p₁   0  ⎤
    ⎣  0   p₂ ⎦
```

La animación muestra p₂ variando de 1 (elemento neutro) hasta el valor P2. Conforme p₂ → 0, el estado se desplaza en la esfera hacia el polo H (S₁ = +1), pues la componente Eₓ domina progresivamente. A diferencia del retardador, el polarizador parcial no conserva la intensidad total.

### Escena 4 — Retardador giratorio

Se mantiene la retardancia Γ constante y se rota el eje rápido de θ = 0° a θ = 360°. En la esfera de Poincaré, el **eje de rotación** del retardador es el punto (cos 2θ, sin 2θ, 0), que recorre el ecuador conforme θ varía.

La trayectoria resultante del estado de polarización es una **curva de Lissajous esférica** de periodo π en θ (el estado regresa a su posición inicial tras θ = 180°, trazando el lazo dos veces durante la rotación completa de 360°).

Un indicador teal muestra la posición del eje del retardador en el ecuador en tiempo real (coordenada 2θ).

---

## Estructura del código

```
EsferaPoincare
├── Funciones físicas
│   ├── jones_desde_angulos(alpha, chi)      → vector Jones desde (α, χ)
│   ├── jones_a_stokes(jones)                → (S₁, S₂, S₃) en la esfera
│   ├── jones_retardador(Gamma, theta)       → matriz Jones del retardador
│   └── jones_pol_parcial(p1, p2, theta)     → matriz Jones del polarizador parcial
│
├── EsferaPoincare(ThreeDScene)              — clase base
│   ├── s2m(S)                               → Stokes → coordenada Manim
│   ├── iniciar_escena()                     → esfera + ejes + etiquetas
│   ├── overlay(lineas, colores, sizes)      → texto fijo en pantalla
│   └── mk_tray(jones_0, J_func)             → ParametricFunction de la trayectoria
│
├── MostrarPunto(EsferaPoincare)             — Escena 1
├── RetardadorFijo(EsferaPoincare)           — Escena 2
├── PolarizadorParcial(EsferaPoincare)       — Escena 3
└── RetardadorGiratorio(EsferaPoincare)      — Escena 4
```

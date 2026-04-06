# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Entorno virtual

```bash
# Activar (creado en Practicas Manim/.venv)
source .venv/bin/activate
```

## Renderizar escenas

```bash
# Alta calidad 1080p 60fps (recomendado)
manim -qh --fps 60 ejemplos.py CuartoDeOnda

# Preview rápido
manim -ql ejemplos.py CuartoDeOnda
```

Escenas en `ejemplos.py`: `CuartoDeOnda`, `PolParcialFijo`, `CuartoDeOndaGiratorio`, `PolParcialGiratorio`.

Los videos se guardan en `media/videos/ejemplos/1080p60/`.

## Arquitectura de la librería

El código está dividido en tres capas:

**`esfera_poincare/fisica.py`** — Funciones puras sin dependencias de Manim:
- `jones_desde_angulos(alpha, chi)` — vector Jones desde ángulos de la elipse
- `jones_a_stokes(j)` — Jones normalizado → (S₁, S₂, S₃)
- `jones_retardador(Gamma, theta)` — matriz Jones de un retardador
- `jones_pol_parcial(p1, p2, theta)` — matriz Jones de un polarizador parcial

**`esfera_poincare/base.py`** — Clase base y parámetros:
- `Parametros` — dataclass con todos los parámetros físicos; expone propiedades `alpha`, `chi`, `retardancia_fija`, etc. ya en radianes, y `jones_inicial`/`stokes_inicial`.
- `EsferaPoincare(ThreeDScene)` — helpers reutilizables:
  - `s2m(S)` — Stokes → coordenadas Manim (escala por `self.R = 2.0`)
  - `iniciar_escena()` — añade esfera (resolución 48×96), círculos meridianos, ejes S₁S₂S₃ y puntos H/V/D/A/R/L
  - `overlay(lineas, colores, sizes)` — texto fijo en pantalla (`add_fixed_in_frame_mobjects`)
  - `mk_tray(jones_0, J_func, n=600)` — `ParametricFunction` de la trayectoria del estado bajo `J_func(t)`, t ∈ [0,1]

**`esfera_poincare/escenas.py`** — Cuatro escenas reutilizables:
- `RetardadorFijo`, `PolarizadorParcial` — eje fijo en S₁
- `RetardadorGiratorio`, `PolarizadorGiratorio` — eje girando θ: 0 → 2π (incluye indicador teal del eje en el ecuador)

## Cómo crear una nueva escena

Subclasifica la escena base y sobreescribe `params`:

```python
from esfera_poincare import Parametros
from esfera_poincare.escenas import RetardadorFijo

class MediaOnda(RetardadorFijo):
    params = Parametros(alpha_deg=45, retardancia_fija_deg=180)
```

## Convención de ejes y colores

- Sistema de coordenadas: x ↔ S₁, y ↔ S₂, z ↔ S₃. Radio visual `_R = 2.0`.
- Paleta en `base.COLORES`: `"ret"` (retardador fijo), `"parp"` (polarizador fijo), `"gir"` (retardador giratorio), `"pgir"` (polarizador giratorio), `"eje"` (indicador de eje, teal).
- Muestras de trayectoria: `n=600` para escenas fijas, `n=600` también para giratorias (índice del eje usa `1/600`).

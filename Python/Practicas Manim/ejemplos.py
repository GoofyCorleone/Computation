"""
ejemplos.py — Cuatro animaciones de demostración de la librería esfera_poincare.

Escenas
-------
CuartoDeOnda         : retardador λ/4 (Γ = 90°) con eje fijo en S₁.
PolParcialFijo        : polarizador parcial (p₁=1, p₂=0.3) con eje fijo en S₁.
CuartoDeOndaGiratorio: retardador λ/4 con eje girando θ: 0° → 360°.
PolParcialGiratorio   : polarizador parcial (p₁=1, p₂=0.3) con eje girando θ: 0° → 360°.

Uso
---
    # Baja calidad (preview rápido):
    manim -ql ejemplos.py CuartoDeOnda

    # Alta calidad (1080p 60 fps):
    manim -qh ejemplos.py CuartoDeOnda

    # Todas las escenas en alta calidad:
    manim -qh ejemplos.py CuartoDeOnda PolParcialFijo CuartoDeOndaGiratorio PolParcialGiratorio
"""

from esfera_poincare import Parametros
from esfera_poincare.escenas import (
    RetardadorFijo,
    PolarizadorParcial,
    RetardadorGiratorio,
    PolarizadorGiratorio,
)

# ── Estado inicial compartido ──────────────────────────────────────────────────
# α = 30°, χ = 20°  →  polarización elíptica general
_ESTADO = dict(alpha_deg=30, chi_deg=20)


# ── Ejemplo 1: Retardador cuarto de onda (Γ = 90°), eje fijo ──────────────────
class CuartoDeOnda(RetardadorFijo):
    """
    Retardador λ/4 con eje rápido a lo largo de S₁.
    El estado gira 90° alrededor del eje S₁.
    """
    params = Parametros(**_ESTADO, retardancia_fija_deg=90)


# ── Ejemplo 2: Polarizador parcial, eje fijo ───────────────────────────────────
class PolParcialFijo(PolarizadorParcial):
    """
    Polarizador parcial (p₁=1, p₂=0.3) con eje a lo largo de S₁.
    El estado se desplaza hacia el polo H conforme p₂ → 0.3.
    """
    params = Parametros(**_ESTADO, p1=1.0, p2=0.3)


# ── Ejemplo 3: Retardador cuarto de onda, eje giratorio ───────────────────────
class CuartoDeOndaGiratorio(RetardadorGiratorio):
    """
    Retardador λ/4 con eje rápido girando θ: 0° → 360°.
    El estado traza una curva de Lissajous esférica.
    """
    params = Parametros(**_ESTADO, retardancia_gir_deg=90)


# ── Ejemplo 4: Polarizador parcial, eje giratorio ─────────────────────────────
class PolParcialGiratorio(PolarizadorGiratorio):
    """
    Polarizador parcial (p₁=1, p₂=0.3) con eje girando θ: 0° → 360°.
    """
    params = Parametros(**_ESTADO, p1=1.0, p2=0.3)

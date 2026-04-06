"""
escenas.py — Escenas de Manim reutilizables para transformaciones ópticas
en la esfera de Poincaré.

Escenas disponibles
-------------------
RetardadorFijo      : retardador con eje fijo (θ = 0).
PolarizadorParcial  : polarizador parcial con eje fijo (θ = 0).
RetardadorGiratorio : retardador con eje girando θ: 0 → 2π.
PolarizadorGiratorio: polarizador parcial con eje girando θ: 0 → 2π.

Uso
---
Subclasifica e instancia `self.params = Parametros(...)` antes de llamar
a los métodos de animación, o usa directamente las subclases de ejemplo
en `ejemplos.py`.
"""

from __future__ import annotations
import numpy as np
from manim import *

from .base import EsferaPoincare, Parametros, COLORES
from .fisica import (
    jones_retardador,
    jones_pol_parcial,
    jones_a_stokes,
    _norm,
)


class RetardadorFijo(EsferaPoincare):
    """
    Retardador con eje rápido fijo a lo largo de S₁ (θ = 0).

    El estado de polarización gira en la esfera alrededor del eje S₁
    un ángulo igual a la retardancia Γ.

    Personalización
    ---------------
    Sobreescribe `params` en la subclase o modifícalo antes de llamar
    a `construct()`:

        class MiEscena(RetardadorFijo):
            params = Parametros(alpha_deg=45, retardancia_fija_deg=180)
    """
    params: Parametros = Parametros()

    def construct(self):
        p = self.params
        self.iniciar_escena()

        path = self.mk_tray(
            p.jones_inicial,
            lambda t: jones_retardador(t * p.retardancia_fija, theta=0.0),
            color=COLORES["ret"],
        )
        S_0 = p.stokes_inicial
        dot     = Dot3D(self.s2m(S_0),  radius=0.13, color=COLORES["ret"])
        dot_fin = Dot3D(path.get_end(), radius=0.09, color=WHITE)

        self.overlay(
            lineas  = [
                "Retardador fijo  (eje rápido ∥ S₁)",
                f"Retardancia  Γ = {p.retardancia_fija_deg:.1f}°",
                "Rotación alrededor del eje S₁",
                f"Estado inicial:  α = {p.alpha_deg}°   χ = {p.chi_deg}°",
            ],
            colores = [COLORES["ret"], WHITE, GRAY_A, GRAY_A],
            sizes   = [25, 22, 20, 20],
        )

        self.add(dot)
        self.wait(0.5)
        self.begin_ambient_camera_rotation(rate=0.10)
        self.play(
            Create(path),
            MoveAlongPath(dot, path),
            run_time=7.0, rate_func=linear,
        )
        self.play(FadeIn(dot_fin), run_time=0.4)
        self.wait(3)
        self.stop_ambient_camera_rotation()


class PolarizadorParcial(EsferaPoincare):
    """
    Polarizador parcial con eje fijo a lo largo de S₁ (θ = 0).

    Anima la diatenuación: p₂ varía de 1 (elemento neutro) → params.p2.
    El estado se desplaza hacia el polo H (S₁ = +1).

    Personalización
    ---------------
    Sobreescribe `params` en la subclase:

        class MiEscena(PolarizadorParcial):
            params = Parametros(alpha_deg=45, p2=0.1)
    """
    params: Parametros = Parametros()

    def construct(self):
        p = self.params
        self.iniciar_escena()

        path = self.mk_tray(
            p.jones_inicial,
            lambda t: jones_pol_parcial(p.p1, 1.0 + t * (p.p2 - 1.0), theta=0.0),
            color=COLORES["parp"],
        )
        S_0 = p.stokes_inicial
        dot     = Dot3D(self.s2m(S_0),  radius=0.13, color=COLORES["parp"])
        dot_fin = Dot3D(path.get_end(), radius=0.09, color=WHITE)

        self.overlay(
            lineas  = [
                "Polarizador Parcial  (eje ∥ S₁)",
                f"p₁ = {p.p1:.2f}     p₂ : 1.00 → {p.p2:.2f}",
                "El estado se desplaza hacia H",
                f"Estado inicial:  α = {p.alpha_deg}°   χ = {p.chi_deg}°",
            ],
            colores = [COLORES["parp"], WHITE, GRAY_A, GRAY_A],
            sizes   = [25, 22, 20, 20],
        )

        self.add(dot)
        self.wait(0.5)
        self.begin_ambient_camera_rotation(rate=0.10)
        self.play(
            Create(path),
            MoveAlongPath(dot, path),
            run_time=7.0, rate_func=linear,
        )
        self.play(FadeIn(dot_fin), run_time=0.4)
        self.wait(3)
        self.stop_ambient_camera_rotation()


class RetardadorGiratorio(EsferaPoincare):
    """
    Retardador con retardancia Γ fija y eje rápido girando θ: 0 → 2π.

    En la esfera de Poincaré el estado traza una curva de Lissajous
    esférica (periodo π en θ; el lazo se recorre dos veces en 2π).
    Un indicador teal muestra la posición del eje del retardador en el
    ecuador (coordenada 2θ).

    Personalización
    ---------------
        class MiEscena(RetardadorGiratorio):
            params = Parametros(retardancia_gir_deg=180)
    """
    params: Parametros = Parametros()

    def construct(self):
        p = self.params
        R = self.R
        self.iniciar_escena()

        # Trayectoria del estado de polarización
        path_est = self.mk_tray(
            p.jones_inicial,
            lambda t: jones_retardador(p.retardancia_gir, theta=t * TAU),
            color=COLORES["gir"],
            grosor=4,
        )
        S_0 = p.stokes_inicial
        dot_est = Dot3D(self.s2m(S_0), radius=0.13, color=COLORES["gir"])

        # Indicador del eje rápido en el ecuador (posición 2θ)
        path_eje = ParametricFunction(
            lambda t: R * np.array([np.cos(t * 2 * TAU), np.sin(t * 2 * TAU), 0.]),
            t_range=[0., 1., 1. / 600.],
            color=COLORES["eje"], stroke_width=2, stroke_opacity=0.50,
        )
        dot_eje = Dot3D(R * np.array([1., 0., 0.]), radius=0.09, color=COLORES["eje"])

        self.overlay(
            lineas  = [
                f"Retardador giratorio  Γ = {p.retardancia_gir_deg:.1f}°",
                "θ : 0° → 360°  (eje rápido)",
                "Teal: eje del retardador en la esfera (2θ)",
                f"Estado inicial:  α = {p.alpha_deg}°   χ = {p.chi_deg}°",
            ],
            colores = [COLORES["gir"], WHITE, COLORES["eje"], GRAY_A],
            sizes   = [25, 22, 20, 20],
        )

        self.add(dot_est, dot_eje)
        self.wait(0.5)
        self.begin_ambient_camera_rotation(rate=0.08)
        self.play(
            Create(path_est), MoveAlongPath(dot_est, path_est),
            Create(path_eje), MoveAlongPath(dot_eje, path_eje),
            run_time=10.0, rate_func=linear,
        )
        self.wait(3)
        self.stop_ambient_camera_rotation()


class PolarizadorGiratorio(EsferaPoincare):
    """
    Polarizador parcial con diatenuación fija (p1, p2) y eje girando θ: 0 → 2π.

    El eje de diatenuación recorre el ecuador de la esfera (posición 2θ).
    El estado de polarización traza una curva cerrada al completar π en θ
    (se recorre dos veces en la rotación completa de 2π).

    Personalización
    ---------------
        class MiEscena(PolarizadorGiratorio):
            params = Parametros(p2=0.0)   # polarizador lineal perfecto
    """
    params: Parametros = Parametros()

    def construct(self):
        p = self.params
        R = self.R
        self.iniciar_escena()

        path_est = self.mk_tray(
            p.jones_inicial,
            lambda t: jones_pol_parcial(p.p1, p.p2, theta=t * TAU),
            color=COLORES["pgir"],
            grosor=4,
        )
        S_0 = p.stokes_inicial
        dot_est = Dot3D(self.s2m(S_0), radius=0.13, color=COLORES["pgir"])

        # Indicador del eje rápido en el ecuador
        path_eje = ParametricFunction(
            lambda t: R * np.array([np.cos(t * 2 * TAU), np.sin(t * 2 * TAU), 0.]),
            t_range=[0., 1., 1. / 600.],
            color=COLORES["eje"], stroke_width=2, stroke_opacity=0.50,
        )
        dot_eje = Dot3D(R * np.array([1., 0., 0.]), radius=0.09, color=COLORES["eje"])

        self.overlay(
            lineas  = [
                "Polarizador Parcial Giratorio",
                f"p₁ = {p.p1:.2f}   p₂ = {p.p2:.2f}",
                "θ : 0° → 360°  (eje de diatenuación)",
                f"Estado inicial:  α = {p.alpha_deg}°   χ = {p.chi_deg}°",
            ],
            colores = [COLORES["pgir"], WHITE, GRAY_A, GRAY_A],
            sizes   = [25, 22, 20, 20],
        )

        self.add(dot_est, dot_eje)
        self.wait(0.5)
        self.begin_ambient_camera_rotation(rate=0.08)
        self.play(
            Create(path_est), MoveAlongPath(dot_est, path_est),
            Create(path_eje), MoveAlongPath(dot_eje, path_eje),
            run_time=10.0, rate_func=linear,
        )
        self.wait(3)
        self.stop_ambient_camera_rotation()

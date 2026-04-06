"""
base.py — Clase base EsferaPoincare y dataclass Parametros.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from manim import *

from .fisica import jones_desde_angulos, jones_a_stokes, _norm


# ── Paleta de colores ──────────────────────────────────────────────────────────
COLORES = {
    "S1"   : "#FF7070",
    "S2"   : "#70FFAA",
    "S3"   : "#70AAFF",
    "ret"  : "#FFD700",   # retardador fijo
    "parp" : "#FFAA44",   # polarizador parcial
    "gir"  : "#44FF88",   # retardador giratorio
    "pgir" : "#FF88DD",   # polarizador giratorio
    "eje"  : TEAL,
}

_R_VISUAL = 2.0   # radio de la esfera en la escena


# ── Parámetros de usuario ──────────────────────────────────────────────────────
@dataclass
class Parametros:
    """
    Todos los parámetros físicos de la simulación.

    Atributos
    ---------
    alpha_deg : orientación de la elipse de polarización (°) ∈ [0, 180]
    chi_deg   : elipticidad (°)                              ∈ [-45, 45]
    retardancia_fija_deg   : retardancia del retardador fijo (°)
    p1, p2                 : transmisiones del polarizador parcial ∈ [0,1]
    retardancia_gir_deg    : retardancia del retardador giratorio (°)
    """
    alpha_deg             : float = 30.0
    chi_deg               : float = 20.0
    retardancia_fija_deg  : float = 90.0    # cuarto de onda por defecto
    p1                    : float = 1.0
    p2                    : float = 0.3
    retardancia_gir_deg   : float = 90.0

    @property
    def alpha(self) -> float:
        return np.radians(self.alpha_deg)

    @property
    def chi(self) -> float:
        return np.radians(self.chi_deg)

    @property
    def retardancia_fija(self) -> float:
        return np.radians(self.retardancia_fija_deg)

    @property
    def retardancia_gir(self) -> float:
        return np.radians(self.retardancia_gir_deg)

    @property
    def jones_inicial(self) -> np.ndarray:
        return jones_desde_angulos(self.alpha, self.chi)

    @property
    def stokes_inicial(self) -> np.ndarray:
        return jones_a_stokes(self.jones_inicial)


# ── Clase base ─────────────────────────────────────────────────────────────────
class EsferaPoincare(ThreeDScene):
    """
    Clase base con la esfera de Poincaré.

    Sistema de coordenadas Manim: x ↔ S₁,  y ↔ S₂,  z ↔ S₃.

    Subclases deben implementar `construct()` e instanciar sus propios
    `Parametros`. El helper `iniciar_escena()` añade la esfera y los ejes.
    """

    R = _R_VISUAL

    # ── mapeo Stokes → coordenadas Manim ──────────────────────────────────────
    def s2m(self, S: np.ndarray) -> np.ndarray:
        return self.R * np.array([S[0], S[1], S[2]], dtype=float)

    # ── constructores de objetos 3D ────────────────────────────────────────────
    def _mk_esfera(self) -> Surface:
        sf = Surface(
            lambda u, v: self.R * np.array([
                np.sin(u) * np.cos(v),
                np.sin(u) * np.sin(v),
                np.cos(u),
            ]),
            u_range=[0, PI], v_range=[0, TAU],
            resolution=(48, 96),
            fill_opacity=0.07,
            stroke_opacity=0.0,
        )
        sf.set_color(BLUE_D)
        return sf

    def _mk_circulos(self) -> VGroup:
        kw = dict(stroke_opacity=0.18, color=WHITE)
        r  = self.R
        return VGroup(
            Circle(radius=r, **kw),
            Circle(radius=r, **kw).rotate(PI / 2, np.array([1., 0., 0.])),
            Circle(radius=r, **kw).rotate(PI / 2, np.array([0., 0., 1.])),
        )

    def _mk_ejes(self) -> VGroup:
        L, d = self.R * 1.30, 0.28
        datos = [
            ("S₁", np.array([1., 0., 0.]), COLORES["S1"]),
            ("S₂", np.array([0., 1., 0.]), COLORES["S2"]),
            ("S₃", np.array([0., 0., 1.]), COLORES["S3"]),
        ]
        g = VGroup()
        for lbl, v, c in datos:
            g.add(Arrow3D(ORIGIN, L * v, color=c, resolution=8))
            g.add(Text(lbl, font_size=26, color=c).move_to((L + d) * v))
        return g

    def _mk_puntos_especiales(self) -> VGroup:
        est = [
            ("H", np.array([ 1.,  0.,  0.])),
            ("V", np.array([-1.,  0.,  0.])),
            ("D", np.array([ 0.,  1.,  0.])),
            ("A", np.array([ 0., -1.,  0.])),
            ("R", np.array([ 0.,  0.,  1.])),
            ("L", np.array([ 0.,  0., -1.])),
        ]
        g = VGroup()
        for nombre, S in est:
            pos = self.s2m(S)
            g.add(Dot3D(pos, radius=0.06, color=GRAY_A))
            g.add(Text(nombre, font_size=20, color=GRAY_A).move_to(pos * 1.16))
        return g

    def iniciar_escena(self, phi: float = 70 * DEGREES,
                       theta: float = -45 * DEGREES) -> None:
        """Añade esfera, círculos meridianos, ejes y puntos H/V/D/A/R/L."""
        self.set_camera_orientation(phi=phi, theta=theta)
        self.add(
            self._mk_esfera(),
            self._mk_circulos(),
            self._mk_ejes(),
            self._mk_puntos_especiales(),
        )

    # ── texto fijo en pantalla ─────────────────────────────────────────────────
    def overlay(self, lineas: list, colores: list = None,
                sizes: list = None) -> VGroup:
        """Bloque de texto fijo en la esquina superior izquierda."""
        n = len(lineas)
        if colores is None:
            colores = [WHITE] * n
        if sizes is None:
            sizes = [23] * n
        grupo, prev = VGroup(), None
        for txt, col, sz in zip(lineas, colores, sizes):
            t = Text(txt, font_size=sz, color=col)
            if prev is None:
                t.to_corner(UL).shift(RIGHT * 0.15 + DOWN * 0.05)
            else:
                t.next_to(prev, DOWN, buff=0.12, aligned_edge=LEFT)
            grupo.add(t)
            prev = t
        self.add_fixed_in_frame_mobjects(*grupo)
        return grupo

    # ── generador de trayectorias ──────────────────────────────────────────────
    def mk_tray(self, jones_0: np.ndarray, J_func,
                color=YELLOW, grosor: int = 4,
                n: int = 600) -> ParametricFunction:
        """
        ParametricFunction de la trayectoria del estado jones_0
        bajo la familia de matrices Jones J_func(t), t ∈ [0, 1].

        Parámetros
        ----------
        jones_0 : vector Jones del estado inicial
        J_func  : función t → np.ndarray (2×2), familia de matrices Jones
        color   : color de la curva
        grosor  : stroke_width
        n       : número de muestras (mayor → más suave)
        """
        R = self.R

        def f(t: float) -> np.ndarray:
            j = _norm(J_func(t) @ jones_0)
            S = jones_a_stokes(j)
            return R * np.array([S[0], S[1], S[2]], dtype=float)

        return ParametricFunction(
            f, t_range=[0., 1., 1. / n],
            color=color, stroke_width=grosor,
        )

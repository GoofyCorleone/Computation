from manim import *
import numpy as np

# ╔══════════════════════════════════════════════════════════════╗
# ║                  PARÁMETROS DE USUARIO                       ║
# ║          — editar aquí para cambiar el estado/óptico —       ║
# ╚══════════════════════════════════════════════════════════════╝

ALPHA_DEG            = 30    # Orientación de la elipse (°)  ∈ [0, 180]
CHI_DEG              = 20    # Elipticidad          (°)  ∈ [-45, 45]
RETARDANCIA_FIJA_DEG = 120   # Retardancia del retardador fijo (°)
P1                   = 1.0   # Transmisión eje rápido del polarizador parcial ∈ [0,1]
P2                   = 0.3   # Transmisión eje lento  del polarizador parcial ∈ [0,1]
RETARDANCIA_GIR_DEG  = 90    # Retardancia del retardador giratorio (°)


# ╔══════════════════════════════════════════════════════════════╗
# ║                         FÍSICA                               ║
# ╚══════════════════════════════════════════════════════════════╝

def jones_desde_angulos(alpha: float, chi: float) -> np.ndarray:
    """
    Vector Jones normalizado para polarización elíptica.
      alpha : orientación de la elipse (rad)
      chi   : elipticidad (rad),  ∈ [-π/4, π/4]

    Punto resultante en la esfera de Poincaré:
      S₁ = cos(2χ)·cos(2α),  S₂ = cos(2χ)·sin(2α),  S₃ = sin(2χ)
    """
    ex = np.cos(alpha)*np.cos(chi) - 1j*np.sin(alpha)*np.sin(chi)
    ey = np.sin(alpha)*np.cos(chi) + 1j*np.cos(alpha)*np.sin(chi)
    return np.array([ex, ey], dtype=complex)


def jones_a_stokes(j: np.ndarray) -> np.ndarray:
    """Jones normalizado → (S₁, S₂, S₃) en la esfera unitaria de Poincaré."""
    ex, ey = j[0], j[1]
    S0 = abs(ex)**2 + abs(ey)**2
    if S0 < 1e-14:
        return np.zeros(3)
    return np.array([
        (abs(ex)**2 - abs(ey)**2) / S0,
         2.0 * np.real(ex * np.conj(ey)) / S0,   # S₂
         2.0 * np.imag(ey * np.conj(ex)) / S0,   # S₃
    ])


def _norm(j: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(j)
    return j / n if n > 1e-12 else j


def jones_retardador(Gamma: float, theta: float = 0.0) -> np.ndarray:
    """
    Matriz Jones de un retardador.
      Gamma : retardancia (rad) — rota el vector de Stokes alrededor
              del eje (cos 2θ, sin 2θ, 0) un ángulo Gamma.
      theta : ángulo del eje rápido respecto a x (rad)
    """
    c, s   = np.cos(theta), np.sin(theta)
    ph, pm = np.exp(1j*Gamma/2), np.exp(-1j*Gamma/2)
    return np.array([
        [c**2*ph + s**2*pm,  c*s*(ph - pm)    ],
        [c*s*(ph - pm),      s**2*ph + c**2*pm],
    ], dtype=complex)


def jones_pol_parcial(p1: float, p2: float, theta: float = 0.0) -> np.ndarray:
    """
    Matriz Jones de un polarizador parcial.
      p1, p2 : transmisiones de amplitud (eje rápido / lento) ∈ [0,1]
      theta  : ángulo del eje rápido (rad)
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [p1*c**2 + p2*s**2,  (p1-p2)*c*s      ],
        [(p1-p2)*c*s,         p1*s**2 + p2*c**2],
    ], dtype=complex)


# ╔══════════════════════════════════════════════════════════════╗
# ║              CLASE BASE — ESFERA DE POINCARÉ                 ║
# ╚══════════════════════════════════════════════════════════════╝

_R = 2.0   # radio visual de la esfera en la escena

_C = {
    "S1"  : "#FF7070",
    "S2"  : "#70FFAA",
    "S3"  : "#70AAFF",
    "ret" : YELLOW,
    "parp": "#FFAA44",
    "gir" : "#44FF88",
    "eje" : TEAL,
}


class EsferaPoincare(ThreeDScene):
    """
    Clase base con la esfera de Poincaré.
    Sistema de coordenadas: x ↔ S₁,  y ↔ S₂,  z ↔ S₃.
    """
    R = _R

    # ─── mapeo Stokes → coordenadas Manim ────────────────────
    def s2m(self, S: np.ndarray) -> np.ndarray:
        return self.R * np.array([S[0], S[1], S[2]], dtype=float)

    # ─── constructores de objetos 3D ─────────────────────────
    def _mk_esfera(self) -> Surface:
        sf = Surface(
            lambda u, v: self.R * np.array([
                np.sin(u)*np.cos(v),
                np.sin(u)*np.sin(v),
                np.cos(u),
            ]),
            u_range=[0, PI], v_range=[0, TAU],
            resolution=(30, 60),
            fill_opacity=0.07,
            stroke_opacity=0.0,
        )
        sf.set_color(BLUE_D)
        return sf

    def _mk_circulos(self) -> VGroup:
        kw = dict(stroke_opacity=0.20, color=WHITE)
        r  = self.R
        return VGroup(
            Circle(radius=r, **kw),
            Circle(radius=r, **kw).rotate(PI/2, np.array([1.,0.,0.])),
            Circle(radius=r, **kw).rotate(PI/2, np.array([0.,0.,1.])),
        )

    def _mk_ejes(self) -> VGroup:
        L, d = self.R*1.30, 0.28
        datos = [
            ("S₁", np.array([1.,0.,0.]), _C["S1"]),
            ("S₂", np.array([0.,1.,0.]), _C["S2"]),
            ("S₃", np.array([0.,0.,1.]), _C["S3"]),
        ]
        g = VGroup()
        for lbl, v, c in datos:
            g.add(Arrow3D(ORIGIN, L*v, color=c, resolution=8))
            g.add(Text(lbl, font_size=26, color=c).move_to((L+d)*v))
        return g

    def _mk_puntos_especiales(self) -> VGroup:
        est = [
            ("H", np.array([ 1., 0., 0.])),
            ("V", np.array([-1., 0., 0.])),
            ("D", np.array([ 0., 1., 0.])),
            ("A", np.array([ 0.,-1., 0.])),
            ("R", np.array([ 0., 0., 1.])),
            ("L", np.array([ 0., 0.,-1.])),
        ]
        g = VGroup()
        for nombre, S in est:
            pos = self.s2m(S)
            g.add(Dot3D(pos, radius=0.06, color=GRAY_A))
            g.add(Text(nombre, font_size=20, color=GRAY_A).move_to(pos*1.16))
        return g

    def iniciar_escena(self, phi: float = 70*DEGREES,
                       theta: float = -45*DEGREES) -> None:
        self.set_camera_orientation(phi=phi, theta=theta)
        self.add(
            self._mk_esfera(),
            self._mk_circulos(),
            self._mk_ejes(),
            self._mk_puntos_especiales(),
        )

    # ─── texto fijo en pantalla ───────────────────────────────
    def overlay(self, lineas: list, colores: list = None,
                sizes: list = None) -> VGroup:
        """Coloca un bloque de texto fijo en la esquina superior izquierda."""
        n = len(lineas)
        if colores is None: colores = [WHITE]*n
        if sizes   is None: sizes   = [23]*n
        grupo, prev = VGroup(), None
        for txt, col, sz in zip(lineas, colores, sizes):
            t = Text(txt, font_size=sz, color=col)
            if prev is None:
                t.to_corner(UL).shift(RIGHT*0.15 + DOWN*0.05)
            else:
                t.next_to(prev, DOWN, buff=0.12, aligned_edge=LEFT)
            grupo.add(t)
            prev = t
        self.add_fixed_in_frame_mobjects(*grupo)
        return grupo

    # ─── generador de trayectorias ────────────────────────────
    def mk_tray(self, jones_0: np.ndarray, J_func,
                color=YELLOW, grosor: int = 4,
                n: int = 300) -> ParametricFunction:
        """
        ParametricFunction de la trayectoria que sigue el estado
        jones_0 bajo la familia de matrices Jones J_func(t), t ∈ [0,1].
        """
        R = self.R
        def f(t: float) -> np.ndarray:
            j = _norm(J_func(t) @ jones_0)
            S = jones_a_stokes(j)
            return R * np.array([S[0], S[1], S[2]], dtype=float)
        return ParametricFunction(f, t_range=[0., 1., 1./n],
                                  color=color, stroke_width=grosor)


# ╔══════════════════════════════════════════════════════════════╗
# ║  ESCENA 1 — MostrarPunto                                     ║
# ║                                                              ║
# ║  Dado cualquier vector Jones caracterizado por (alpha, chi), ║
# ║  calcula S₁, S₂, S₃ y muestra el punto en la esfera.        ║
# ╚══════════════════════════════════════════════════════════════╝

class MostrarPunto(EsferaPoincare):
    """Escena 1: estado (ALPHA_DEG, CHI_DEG) → punto en la esfera de Poincaré."""

    def construct(self):
        alpha = np.radians(ALPHA_DEG)
        chi   = np.radians(CHI_DEG)

        self.iniciar_escena()

        jones = jones_desde_angulos(alpha, chi)
        S     = jones_a_stokes(jones)
        pos   = self.s2m(S)

        self.overlay(
            lineas  = [
                "Estado de polarización",
                f"α = {ALPHA_DEG}°     χ = {CHI_DEG}°",
                f"S₁={S[0]:.3f}   S₂={S[1]:.3f}   S₃={S[2]:.3f}",
            ],
            colores = [YELLOW, WHITE, GRAY_A],
            sizes   = [27, 23, 20],
        )

        radio = DashedLine(ORIGIN, pos, color=YELLOW,
                           dash_length=0.13, stroke_opacity=0.60)
        dot   = Dot3D(pos, radius=0.14, color=YELLOW)

        self.begin_ambient_camera_rotation(rate=0.18)
        self.play(Create(radio), FadeIn(dot), run_time=2.0)
        self.wait(5)
        self.stop_ambient_camera_rotation()


# ╔══════════════════════════════════════════════════════════════╗
# ║  ESCENA 2 — RetardadorFijo                                   ║
# ║                                                              ║
# ║  Retardador con eje rápido a lo largo de S₁ (θ = 0).        ║
# ║  El estado gira alrededor del eje S₁ en la esfera de         ║
# ║  Poincaré un ángulo igual a la retardancia Γ.                ║
# ╚══════════════════════════════════════════════════════════════╝

class RetardadorFijo(EsferaPoincare):

    def construct(self):
        alpha     = np.radians(ALPHA_DEG)
        chi       = np.radians(CHI_DEG)
        Gamma_max = np.radians(RETARDANCIA_FIJA_DEG)

        self.iniciar_escena()

        jones_0 = jones_desde_angulos(alpha, chi)
        S_0     = jones_a_stokes(jones_0)

        path    = self.mk_tray(
            jones_0,
            lambda t: jones_retardador(t * Gamma_max, theta=0.),
            color=_C["ret"],
        )
        dot     = Dot3D(self.s2m(S_0),  radius=0.13, color=_C["ret"])
        dot_fin = Dot3D(path.get_end(), radius=0.09, color=WHITE)

        self.overlay(
            lineas  = [
                "Retardador fijo  (eje rápido ∥ S₁)",
                f"Retardancia  Γ = {RETARDANCIA_FIJA_DEG}°",
                "Rotación alrededor del eje S₁",
                f"Estado inicial:  α={ALPHA_DEG}°  χ={CHI_DEG}°",
            ],
            colores = [_C["ret"], WHITE, GRAY_A, GRAY_A],
            sizes   = [25, 22, 20, 20],
        )

        self.add(dot)
        self.wait(0.5)
        self.begin_ambient_camera_rotation(rate=0.10)
        self.play(
            Create(path),
            MoveAlongPath(dot, path),
            run_time=6.0, rate_func=linear,
        )
        self.play(FadeIn(dot_fin), run_time=0.5)
        self.wait(3)
        self.stop_ambient_camera_rotation()


# ╔══════════════════════════════════════════════════════════════╗
# ║  ESCENA 3 — PolarizadorParcial                               ║
# ║                                                              ║
# ║  Polarizador parcial con eje rápido ∥ S₁.                    ║
# ║  Se anima la diatenuación: p₂ varía de 1 → P2.               ║
# ║  El estado se desplaza hacia el polo H (S₁=+1).              ║
# ╚══════════════════════════════════════════════════════════════╝

class PolarizadorParcial(EsferaPoincare):

    def construct(self):
        alpha = np.radians(ALPHA_DEG)
        chi   = np.radians(CHI_DEG)

        self.iniciar_escena()

        jones_0 = jones_desde_angulos(alpha, chi)
        S_0     = jones_a_stokes(jones_0)

        # p₂ varía de 1 (sin efecto) → P2 (máxima diatenuación)
        path    = self.mk_tray(
            jones_0,
            lambda t: jones_pol_parcial(P1, 1.0 + t*(P2 - 1.0), theta=0.),
            color=_C["parp"],
        )
        dot     = Dot3D(self.s2m(S_0),  radius=0.13, color=_C["parp"])
        dot_fin = Dot3D(path.get_end(), radius=0.09, color=WHITE)

        self.overlay(
            lineas  = [
                "Polarizador Parcial  (eje ∥ S₁)",
                f"p₁ = {P1:.2f}     p₂ : 1.00 → {P2:.2f}",
                "El estado se desplaza hacia H",
                f"Estado inicial:  α={ALPHA_DEG}°  χ={CHI_DEG}°",
            ],
            colores = [_C["parp"], WHITE, GRAY_A, GRAY_A],
            sizes   = [25, 22, 20, 20],
        )

        self.add(dot)
        self.wait(0.5)
        self.begin_ambient_camera_rotation(rate=0.10)
        self.play(
            Create(path),
            MoveAlongPath(dot, path),
            run_time=6.0, rate_func=linear,
        )
        self.play(FadeIn(dot_fin), run_time=0.5)
        self.wait(3)
        self.stop_ambient_camera_rotation()


# ╔══════════════════════════════════════════════════════════════╗
# ║  ESCENA 4 — RetardadorGiratorio                              ║
# ║                                                              ║
# ║  Retardador con retardancia Γ fija cuyo eje rápido gira      ║
# ║  de θ=0 a θ=2π.  El estado traza una curva de Lissajous      ║
# ║  esférica (periodo π en θ, doble lazo visible).              ║
# ║  Un punto teal marca el eje del retardador en el ecuador     ║
# ║  de la esfera de Poincaré (posición 2θ).                     ║
# ╚══════════════════════════════════════════════════════════════╝

class RetardadorGiratorio(EsferaPoincare):

    def construct(self):
        alpha = np.radians(ALPHA_DEG)
        chi   = np.radians(CHI_DEG)
        Gamma = np.radians(RETARDANCIA_GIR_DEG)
        R     = self.R

        self.iniciar_escena()

        jones_0 = jones_desde_angulos(alpha, chi)
        S_0     = jones_a_stokes(jones_0)

        # — trayectoria del estado de polarización —————————————
        path_est = self.mk_tray(
            jones_0,
            lambda t: jones_retardador(Gamma, theta=t*TAU),
            color=_C["gir"],
            grosor=4,
        )
        dot_est = Dot3D(self.s2m(S_0), radius=0.13, color=_C["gir"])

        # — indicador del eje rápido en el ecuador —————————————
        # Eje rápido en θ ↔ punto (cos 2θ, sin 2θ, 0) en la esfera de Poincaré.
        # Como θ va de 0 a 2π, el eje recorre el ecuador dos veces (t·2·TAU).
        path_eje = ParametricFunction(
            lambda t: R * np.array([np.cos(t*2*TAU), np.sin(t*2*TAU), 0.]),
            t_range=[0., 1., 1./300.],
            color=_C["eje"], stroke_width=2, stroke_opacity=0.50,
        )
        dot_eje = Dot3D(R*np.array([1.,0.,0.]), radius=0.09, color=_C["eje"])

        self.overlay(
            lineas  = [
                f"Retardador giratorio  Γ = {RETARDANCIA_GIR_DEG}°",
                "θ : 0° → 360°  (eje rápido)",
                "Teal: eje del retardador en la esfera (2θ)",
                f"Estado inicial:  α={ALPHA_DEG}°  χ={CHI_DEG}°",
            ],
            colores = [_C["gir"], WHITE, _C["eje"], GRAY_A],
            sizes   = [25, 22, 20, 20],
        )

        self.add(dot_est, dot_eje)
        self.wait(0.5)
        self.begin_ambient_camera_rotation(rate=0.08)
        self.play(
            Create(path_est), MoveAlongPath(dot_est, path_est),
            Create(path_eje), MoveAlongPath(dot_eje, path_eje),
            run_time=9.0, rate_func=linear,
        )
        self.wait(3)
        self.stop_ambient_camera_rotation()

from manim import *


class PuntoEnEsfera(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)

        esfera = Surface(
            lambda u, v: np.array([
                np.sin(u) * np.cos(v),
                np.sin(u) * np.sin(v),
                np.cos(u),
            ]),
            u_range=[0, PI],
            v_range=[0, TAU],
            resolution=(24, 48),
            fill_opacity=0.3,
            checkerboard_colors=[BLUE_D, BLUE_E],
        )

        punto = Dot3D(point=np.array([1, 0, 0]), radius=0.07, color=YELLOW)

        self.add(esfera, punto)
        self.begin_ambient_camera_rotation(rate=0.2)

        # Trayectoria: punto recorre la esfera en una espiral
        def posicion_esfera(t):
            phi = t * PI          # de polo a polo
            theta = t * TAU * 3   # tres vueltas completas
            return np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            ])

        trayectoria = ParametricFunction(
            posicion_esfera,
            t_range=[0, 1],
            color=YELLOW,
            stroke_width=2,
        )

        self.play(
            MoveAlongPath(punto, trayectoria),
            Create(trayectoria),
            run_time=8,
            rate_func=linear,
        )

        self.wait(2)
        self.stop_ambient_camera_rotation()

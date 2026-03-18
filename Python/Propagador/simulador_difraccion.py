"""
Simulador de difracción por doble rendija
==========================================
Compara la integral de Huyghens-Fresnel (exacta) con la integral de Fresnel
(aproximación paraxial) para dos rendijas rectangulares.

Geometría
---------
  Rendija 1 : centrada en (0, 0, n),   dimensiones p × q  [m]
  Rendija 2 : centrada en (a, b, n+c), dimensiones p2 × q2 [m]
  Plano de observación : z = z0

Método de integración
---------------------
  Cuadratura de Gauss-Legendre (numpy.polynomial.legendre.leggauss).
  Para integrales sobre dominios finitos con integrando analítico y
  moderadamente oscilatorio, GL converge exponencialmente con el número
  de puntos N_quad, siendo la opción más eficiente y confiable.

  - Huyghens-Fresnel : doble integral anidada (S1) dentro de doble integral
    (S2), calculada por filas para mantener uso de memoria bajo.
  - Fresnel : integral separable en x e y → dos integrales dobles 1D
    independientes.

λ = 632.8 nm (He-Ne)

Referencias
-----------
  Ecs. (7), (8) y (9) del documento "Teoría sobre la propagación de fotones
  por una y dos rendijas", febrero 2026.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy.polynomial.legendre import leggauss


# ======================================================================
class DifraccionDosRendijas:
    """
    Clase principal.  Orquesta el cálculo y la visualización.

    Parámetros
    ----------
    p, q     : ancho (x) y alto (y) de la rendija 1  [m]
    p2, q2   : ancho (x) y alto (y) de la rendija 2  [m]
    a, b     : posición (x, y) del centro de la rendija 2  [m]
    c        : separación longitudinal entre las dos rendijas  [m]
    z0       : posición z del plano de observación  [m]
    n        : posición z de la rendija 1  [m]
    N_grid   : número de puntos en cada eje del plano de observación
    N_quad   : puntos de cuadratura Gauss-Legendre (mayor → más preciso)
    x_range  : extensión total en x del plano de observación  [m]
    y_range  : extensión total en y del plano de observación  [m]
    """

    LAMBDA = 632.8e-9  # longitud de onda He-Ne [m]

    def __init__(
        self,
        p=0.10e-3, q=0.10e-3,
        p2=0.10e-3, q2=0.10e-3,
        a=0.30e-3,  b=0.0,
        c=20e-3,
        z0=200e-3,
        n=10e-3,
        N_grid=64,
        N_quad=30,
        x_range=2.5e-3,
        y_range=2.5e-3,
    ):
        self.lam = self.LAMBDA
        self.k   = 2.0 * np.pi / self.lam

        # geometría
        self.p,  self.q  = p,  q
        self.p2, self.q2 = p2, q2
        self.a,  self.b  = a,  b
        self.c   = c
        self.z0  = z0
        self.n   = n
        self.d2  = z0 - n - c   # distancia rendija-2 → plano de observación

        if self.d2 <= 0:
            raise ValueError(
                f"z0 ({z0*1e3:.1f} mm) debe ser mayor que n+c ({(n+c)*1e3:.1f} mm)."
            )

        # grilla de observación
        self.N_grid = N_grid
        self.N_quad = N_quad
        self.x0_arr = np.linspace(-x_range / 2, x_range / 2, N_grid)
        self.y0_arr = np.linspace(-y_range / 2, y_range / 2, N_grid)
        # meshgrid: filas → y, columnas → x
        self.X0, self.Y0 = np.meshgrid(self.x0_arr, self.y0_arr)

        # instancias de las subclases
        self._hf   = self.HuyghensFresnelDS(self)
        self._fres = self.FresnelDS(self)
        self._dif  = self.DiferenciaDS(self)

    # ------------------------------------------------------------------
    # Utilidad compartida: nodos y pesos GL escalados a [lo, hi]
    # ------------------------------------------------------------------
    @staticmethod
    def _gl_nodes(lo, hi, N):
        """Devuelve nodos y pesos de Gauss-Legendre en el intervalo [lo, hi]."""
        xi, wi = leggauss(N)
        nodes   = 0.5 * (hi - lo) * xi + 0.5 * (hi + lo)
        weights = 0.5 * (hi - lo) * wi
        return nodes, weights

    # ==================================================================
    # Subclase 1 — Huyghens-Fresnel
    # ==================================================================
    class HuyghensFresnelDS:
        """
        Campo difractado mediante la integral de Huyghens-Fresnel (ec. 8):

            U(x0,y0) = (i/λ)² ∫∫_{S2} T2(x2,y2)
                         [∫∫_{S1} T1(x1,y1) e^{ikr12}/r12 dx1dy1]
                         × e^{ikr2}/r2  dx2dy2

        con  r12 = √((x2-x1)²+(y2-y1)²+c²)
             r2  = √((x0-x2)²+(y0-y2)²+d2²)

        La integral interna sobre S1 se vectoriza completamente.
        La integral externa se evalúa fila a fila (fila de y0) para
        mantener el uso de memoria acotado.
        """

        def __init__(self, parent):
            self.p = parent

        # ----------------------------------------------------------------
        def _inner_integral(self, x2, y2, N):
            """
            Calcula I_inner[i,j] = ∫∫_{S1} T1 e^{ikr12}/r12 dx1dy1
            para cada nodo (x2[i], y2[j]) del dominio S2.

            Devuelve array complejo de forma (N2, N2).
            """
            p = self.p
            x1, wx1 = p._gl_nodes(-p.p / 2,  p.p / 2,  N)
            y1, wy1 = p._gl_nodes(-p.q / 2,  p.q / 2,  N)

            # Broadcasting: r12[i,j,m,n] = sqrt((x2[i]-x1[m])²+(y2[j]-y1[n])²+c²)
            dx = x2[:, None, None, None] - x1[None, None, :, None]  # (N2,1,N1,1)
            dy = y2[None, :, None, None] - y1[None, None, None, :]  # (1,N2,1,N1)
            r12 = np.sqrt(dx**2 + dy**2 + p.c**2)                   # (N2,N2,N1,N1)

            # pesos combinados S1
            W1 = wx1[None, None, :, None] * wy1[None, None, None, :]  # (1,1,N1,N1)
            integrand = W1 * np.exp(1j * p.k * r12) / r12             # (N2,N2,N1,N1)
            return integrand.sum(axis=(2, 3))                          # (N2,N2)

        # ----------------------------------------------------------------
        def calcular(self):
            """
            Devuelve el campo complejo U_HF de forma (N_grid, N_grid).
            """
            p = self.p
            N = p.N_quad

            x2, wx2 = p._gl_nodes(p.a - p.p2 / 2, p.a + p.p2 / 2, N)
            y2, wy2 = p._gl_nodes(p.b - p.q2 / 2, p.b + p.q2 / 2, N)

            I_inner = self._inner_integral(x2, y2, N)   # (N2,N2)
            prefactor = (1j / p.lam) ** 2

            U_field = np.zeros((p.N_grid, p.N_grid), dtype=complex)

            # --- integral externa, fila a fila para ahorrar memoria ---
            for m, y_obs in enumerate(p.y0_arr):
                dy2 = y_obs - y2                            # (N2,)
                # dx2[n, i] = x0_arr[n] - x2[i]
                dx2 = p.x0_arr[:, None] - x2[None, :]      # (Ng, N2)

                # r2[n, i, j] = sqrt(dx2[n,i]²+dy2[j]²+d2²)
                r2 = np.sqrt(
                    dx2[:, :, None] ** 2            # (Ng,N2,1)
                    + dy2[None, None, :] ** 2        # (1,1,N2)
                    + p.d2 ** 2
                )                                            # (Ng,N2,N2)

                # pesos S2
                W2 = wx2[None, :, None] * wy2[None, None, :]  # (1,N2,N2)

                outer = I_inner[None, :, :] * np.exp(1j * p.k * r2) / r2 * W2
                U_field[m, :] = prefactor * outer.sum(axis=(1, 2))

            return U_field

        # ----------------------------------------------------------------
        def convergencia(self, x0=0.0, y0=0.0, N_list=None):
            """
            Intensidad en el punto (x0,y0) para distintos N_quad.
            Permite visualizar la convergencia de la cuadratura GL.
            """
            if N_list is None:
                N_list = [4, 6, 8, 10, 14, 18, 22, 28, 35, 44, 55]
            p = self.p
            resultados = []

            for N in N_list:
                x2, wx2 = p._gl_nodes(p.a - p.p2 / 2, p.a + p.p2 / 2, N)
                y2, wy2 = p._gl_nodes(p.b - p.q2 / 2, p.b + p.q2 / 2, N)

                I_inner = self._inner_integral(x2, y2, N)   # (N2,N2)

                dx2 = x0 - x2                                # (N2,)
                dy2 = y0 - y2                                # (N2,)
                r2  = np.sqrt(
                    dx2[:, None] ** 2 + dy2[None, :] ** 2 + p.d2 ** 2
                )                                            # (N2,N2)

                W2 = wx2[:, None] * wy2[None, :]
                U  = (1j / p.lam) ** 2 * (
                    I_inner * np.exp(1j * p.k * r2) / r2 * W2
                ).sum()
                resultados.append(abs(U) ** 2)

            return N_list, resultados

    # ==================================================================
    # Subclase 2 — Fresnel (paraxial, separable en x e y)
    # ==================================================================
    class FresnelDS:
        """
        Campo difractado mediante la integral de Fresnel (ec. 7).

        Gracias a la separabilidad del núcleo paraxial:

            exp(ik/(2c)[(x2-x1)²+(y2-y1)²]) = exp(ik/(2c)(x2-x1)²)
                                               × exp(ik/(2c)(y2-y1)²)

        el campo factoriza como  U_F(x0,y0) ∝ Ux(x0) · Uy(y0),
        reduciendo el costo a dos integrales dobles 1D independientes.

            Ux(x0) = ∫_{S2x} [∫_{S1x} exp(ik(x2-x1)²/2c) dx1]
                              × exp(ik(x0-x2)²/2d2) dx2

        (análogo para y).
        """

        def __init__(self, parent):
            self.p = parent

        # ----------------------------------------------------------------
        def _integral_1d(self, obs_arr, lo1, hi1, lo2, hi2, N):
            """
            Calcula la integral de Fresnel doble 1D para un eje dado.

            Parámetros
            ----------
            obs_arr : puntos de observación en este eje
            lo1,hi1 : límites de integración en S1 (rendija 1)
            lo2,hi2 : límites de integración en S2 (rendija 2)
            N       : puntos GL

            Devuelve array complejo de longitud len(obs_arr).
            """
            p = self.p
            k, c, d2 = p.k, p.c, p.d2

            x1, wx1 = p._gl_nodes(lo1, hi1, N)
            x2, wx2 = p._gl_nodes(lo2, hi2, N)

            # Integral interna: I(x2_i) = Σ_j wx1[j] exp(ik(x2_i-x1_j)²/2c)
            dx_inner = x2[:, None] - x1[None, :]                    # (N2, N1)
            I_inner  = (wx1 * np.exp(1j * k / (2*c) * dx_inner**2)).sum(axis=1)  # (N2,)

            # Integral externa para todos los puntos de observación
            # U(obs_m) = Σ_i wx2[i] I(x2_i) exp(ik(obs_m-x2_i)²/2d2)
            dx_outer = obs_arr[:, None] - x2[None, :]               # (M, N2)
            phase    = np.exp(1j * k / (2 * d2) * dx_outer**2)      # (M, N2)
            return (wx2 * I_inner * phase).sum(axis=1)               # (M,)

        # ----------------------------------------------------------------
        def calcular(self):
            """
            Devuelve el campo complejo U_Fresnel de forma (N_grid, N_grid).

            El prefactor físico 1/(λ²cd2) asegura que la intensidad esté
            en las mismas unidades que U_HF para la comparación.
            """
            p  = self.p
            N  = p.N_quad

            Ux = self._integral_1d(
                p.x0_arr,
                -p.p/2, p.p/2,
                p.a - p.p2/2, p.a + p.p2/2,
                N,
            )  # (N_grid,)

            Uy = self._integral_1d(
                p.y0_arr,
                -p.q/2, p.q/2,
                p.b - p.q2/2, p.b + p.q2/2,
                N,
            )  # (N_grid,)

            # U_F(y_m, x_n) = prefactor * Uy[m] * Ux[n]
            # np.outer(Uy, Ux) → filas = y, columnas = x  ✓
            prefactor = 1.0 / ((1j * p.lam)**2 * p.c * p.d2)
            return prefactor * np.outer(Uy, Ux)

        # ----------------------------------------------------------------
        def convergencia(self, x0=0.0, y0=0.0, N_list=None):
            """
            Intensidad en el punto (x0,y0) para distintos N_quad.
            """
            if N_list is None:
                N_list = [4, 6, 8, 10, 14, 18, 22, 28, 35, 44, 55]
            p = self.p
            k, c, d2 = p.k, p.c, p.d2
            resultados = []

            for N in N_list:
                x1, wx1 = p._gl_nodes(-p.p/2,  p.p/2,  N)
                y1, wy1 = p._gl_nodes(-p.q/2,  p.q/2,  N)
                x2, wx2 = p._gl_nodes(p.a - p.p2/2, p.a + p.p2/2, N)
                y2, wy2 = p._gl_nodes(p.b - p.q2/2, p.b + p.q2/2, N)

                # eje x
                dx_i = x2[:, None] - x1[None, :]
                Ix   = (wx1 * np.exp(1j*k/(2*c)*dx_i**2)).sum(axis=1)
                Ux   = (wx2 * Ix * np.exp(1j*k/(2*d2)*(x0-x2)**2)).sum()

                # eje y
                dy_i = y2[:, None] - y1[None, :]
                Iy   = (wy1 * np.exp(1j*k/(2*c)*dy_i**2)).sum(axis=1)
                Uy   = (wy2 * Iy * np.exp(1j*k/(2*d2)*(y0-y2)**2)).sum()

                pref = 1.0 / ((1j * p.lam)**2 * c * d2)
                U    = pref * Ux * Uy
                resultados.append(abs(U)**2)

            return N_list, resultados

    # ==================================================================
    # Subclase 3 — Diferencia
    # ==================================================================
    class DiferenciaDS:
        """
        Calcula la diferencia entre los patrones normalizados de
        Huyghens-Fresnel y Fresnel.

            ΔI_abs = |I_HF_norm − I_F_norm|
            ΔI_rel = ΔI_abs  [%]  (ya normalizadas, el máximo de HF = 1)
        """

        def __init__(self, parent):
            self.p = parent

        def calcular(self, I_hf, I_fres):
            """
            Recibe las intensidades crudas y devuelve las diferencias
            trabajando sobre versiones normalizadas al máximo de I_hf.
            """
            norm    = I_hf.max()
            I_hf_n  = I_hf  / norm
            I_fres_n = I_fres / norm
            dif_abs = np.abs(I_hf_n - I_fres_n)
            return dif_abs, I_hf_n, I_fres_n

    # ==================================================================
    # Ejecución principal
    # ==================================================================
    def run(self):
        """
        Calcula ambos campos, la diferencia y genera las tres figuras:
          1. Figura Huyghens-Fresnel  (4 paneles)
          2. Figura Fresnel           (4 paneles)
          3. Figura comparativa       (4 paneles)
        """
        print("=" * 55)
        print("  Simulador de difracción por doble rendija")
        print(f"  λ = {self.lam*1e9:.1f} nm   N_quad = {self.N_quad}")
        print(f"  Rendija 1 : {self.p*1e3:.2f}×{self.q*1e3:.2f} mm  en z={self.n*1e3:.1f} mm")
        print(f"  Rendija 2 : {self.p2*1e3:.2f}×{self.q2*1e3:.2f} mm  en ({self.a*1e3:.2f},{self.b*1e3:.2f}) mm")
        print(f"  Separación c = {self.c*1e3:.1f} mm   z_obs = {self.z0*1e3:.1f} mm")
        print("=" * 55)

        # ---------- cálculo de campos ----------
        print("\n[1/4] Calculando Huyghens-Fresnel ...", flush=True)
        U_hf   = self._hf.calcular()
        I_hf   = np.abs(U_hf) ** 2
        print("      listo.")

        print("[2/4] Calculando Fresnel ...", flush=True)
        U_fres = self._fres.calcular()
        I_fres = np.abs(U_fres) ** 2
        print("      listo.")

        # ---------- diferencia ----------
        dif_abs, I_hf_n, I_fres_n = self._dif.calcular(I_hf, I_fres)

        # ---------- convergencia ----------
        print("[3/4] Calculando convergencia numérica ...", flush=True)
        N_list, conv_hf   = self._hf.convergencia()
        _,      conv_fres = self._fres.convergencia()
        print("      listo.")

        # ---------- plots ----------
        print("[4/4] Generando figuras ...")
        self._plot_resultado(
            I_hf,  "Huyghens-Fresnel (exacta)", N_list, conv_hf
        )
        self._plot_resultado(
            I_fres, "Fresnel (paraxial)",        N_list, conv_fres
        )
        self._plot_diferencia(I_hf_n, I_fres_n, dif_abs)
        print("      listo.\n")
        plt.show()

    # ==================================================================
    # Visualización
    # ==================================================================
    def _plot_resultado(self, I, titulo, N_list, conv):
        """
        Figura de 4 paneles para un método de integración:
          [0,0] Patrón 2D de difracción
          [0,1] Perfil de intensidad en X (corte en y ≈ 0)
          [1,0] Perfil de intensidad en Y (corte en x ≈ 0)
          [1,1] Convergencia numérica
        """
        x_mm = self.x0_arr * 1e3
        y_mm = self.y0_arr * 1e3
        ext  = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]

        iy0 = np.argmin(np.abs(self.y0_arr))
        ix0 = np.argmin(np.abs(self.x0_arr))
        I_n = I / I.max()

        fig = plt.figure(figsize=(12, 9))
        fig.suptitle(
            f"Difracción por doble rendija — {titulo}",
            fontsize=13, fontweight="bold",
        )
        gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.34)

        # --- Patrón 2D ---
        ax1 = fig.add_subplot(gs[0, 0])
        im  = ax1.imshow(
            I_n, extent=ext, origin="lower",
            cmap="inferno", aspect="equal", vmin=0, vmax=1,
        )
        plt.colorbar(im, ax=ax1, label="Intensidad norm.")
        ax1.set_xlabel("x  [mm]")
        ax1.set_ylabel("y  [mm]")
        ax1.set_title("Patrón 2D")

        # --- Perfil en X ---
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(x_mm, I_n[iy0, :], color="royalblue", lw=1.6)
        ax2.fill_between(x_mm, I_n[iy0, :], alpha=0.18, color="royalblue")
        ax2.set_xlabel("x  [mm]")
        ax2.set_ylabel("Intensidad norm.")
        ax2.set_title("Perfil en X  (y = 0)")
        ax2.set_ylim(bottom=0)
        ax2.grid(alpha=0.3)

        # --- Perfil en Y ---
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(y_mm, I_n[:, ix0], color="tomato", lw=1.6)
        ax3.fill_between(y_mm, I_n[:, ix0], alpha=0.18, color="tomato")
        ax3.set_xlabel("y  [mm]")
        ax3.set_ylabel("Intensidad norm.")
        ax3.set_title("Perfil en Y  (x = 0)")
        ax3.set_ylim(bottom=0)
        ax3.grid(alpha=0.3)

        # --- Convergencia ---
        ax4 = fig.add_subplot(gs[1, 1])
        conv_arr = np.array(conv, dtype=float)
        ref      = conv_arr[-1]
        error    = np.abs(conv_arr - ref) + 1e-30 * ref   # evitar log(0)
        ax4.semilogy(N_list, error, "o-", color="seagreen", lw=1.6, ms=5)
        ax4.set_xlabel("Puntos de cuadratura  N")
        ax4.set_ylabel(r"$|I(N) - I(N_\mathrm{max})|$")
        ax4.set_title("Convergencia numérica")
        ax4.grid(alpha=0.3, which="both")

        fig.tight_layout()

    # ------------------------------------------------------------------
    def _plot_diferencia(self, I_hf_n, I_fres_n, dif_abs):
        """
        Figura comparativa de 4 paneles:
          [0,0] Patrón normalizado HF
          [0,1] Patrón normalizado Fresnel
          [1,0] Diferencia absoluta  |I_HF - I_F|
          [1,1] Perfil central de ambos métodos superpuestos
        """
        x_mm = self.x0_arr * 1e3
        y_mm = self.y0_arr * 1e3
        ext  = [x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]]
        iy0  = np.argmin(np.abs(self.y0_arr))

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle(
            "Comparación: Huyghens-Fresnel vs Fresnel",
            fontsize=13, fontweight="bold",
        )

        im0 = axes[0, 0].imshow(
            I_hf_n, extent=ext, origin="lower", cmap="inferno",
            vmin=0, vmax=1,
        )
        axes[0, 0].set_title("Huyghens-Fresnel (exacta)")
        plt.colorbar(im0, ax=axes[0, 0], label="I norm.")

        im1 = axes[0, 1].imshow(
            I_fres_n, extent=ext, origin="lower", cmap="inferno",
            vmin=0, vmax=1,
        )
        axes[0, 1].set_title("Fresnel (paraxial)")
        plt.colorbar(im1, ax=axes[0, 1], label="I norm.")

        im2 = axes[1, 0].imshow(
            dif_abs, extent=ext, origin="lower", cmap="plasma",
        )
        axes[1, 0].set_title("Diferencia absoluta  |I_HF − I_F|")
        plt.colorbar(im2, ax=axes[1, 0])

        # perfil central superpuesto
        ax_p = axes[1, 1]
        ax_p.plot(x_mm, I_hf_n[iy0, :],  lw=1.8, label="Huyghens-Fresnel", color="steelblue")
        ax_p.plot(x_mm, I_fres_n[iy0, :], lw=1.8, label="Fresnel", color="tomato",
                  linestyle="--")
        ax_p.fill_between(x_mm,
                           np.minimum(I_hf_n[iy0, :], I_fres_n[iy0, :]),
                           np.maximum(I_hf_n[iy0, :], I_fres_n[iy0, :]),
                           alpha=0.25, color="gray", label="Diferencia")
        ax_p.set_xlabel("x  [mm]")
        ax_p.set_ylabel("Intensidad norm.")
        ax_p.set_title("Perfil en X  —  comparación (y = 0)")
        ax_p.set_ylim(bottom=0)
        ax_p.legend(fontsize=9)
        ax_p.grid(alpha=0.3)

        for ax in axes.flat[:3]:
            ax.set_xlabel("x  [mm]")
            ax.set_ylabel("y  [mm]")

        fig.tight_layout()


# ======================================================================
# Punto de entrada
# ======================================================================
if __name__ == "__main__":
    # Parámetros de ejemplo
    # ---------------------
    # Dos rendijas simétricas de 0.1 mm × 0.1 mm separadas 0.3 mm en x.
    # La primera en z = 10 mm, la segunda en z = 30 mm (c = 20 mm).
    # Plano de observación en z = 230 mm.

    sim = DifraccionDosRendijas(
        p=0.10e-3,  q=0.10e-3,    # rendija 1: 0.10 × 0.10 mm
        p2=0.10e-3, q2=0.10e-3,   # rendija 2: 0.10 × 0.10 mm
        a=0.30e-3,  b=0.0,         # centro rendija 2: (0.3, 0) mm
        c=20e-3,                   # separación entre rendijas: 20 mm
        z0=230e-3,                 # plano de observación: 230 mm
        n=10e-3,                   # posición z rendija 1: 10 mm
        N_grid=64,                 # resolución del plano de observación
        N_quad=30,                 # puntos de cuadratura GL (≥20 es preciso)
        x_range=2.5e-3,            # ventana de observación: ±1.25 mm en x
        y_range=2.5e-3,            #                         ±1.25 mm en y
    )
    sim.run()

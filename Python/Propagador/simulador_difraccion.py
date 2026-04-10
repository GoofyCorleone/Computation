"""
Simulador de difracción por doble rendija — modo 1D en x
=========================================================
Compara la integral de Huyghens-Fresnel (exacta, kernel cilíndrico 2D)
con la integral de Fresnel (aproximación paraxial) para dos rendijas
rectangulares con apertura vertical fija Q_FIJA = 3 cm.

Física
------
La apertura vertical q = q₂ = 3 cm >> λ·z^{1/2} pone el eje y en el
régimen de óptica geométrica: ambos métodos producen la misma respuesta
en y (factor multiplicativo idéntico).  La diferencia HF − Fresnel
proviene únicamente del eje horizontal x, donde las aperturas p, p₂ son
pequeñas (fracción de mm) y el número de Fresnel N_F ≲ 12.

Por ello la comparación se realiza en el perfil 1D de intensidad en x,
usando los kernels:

  Huyghens-Fresnel (exacto 2D):
      exp(ik·r)/√r    con  r = √(Δx² + z²)

  Fresnel (paraxial):
      exp(ik·Δx²/2z)

Geometría
---------
  Rendija 1 : centrada en x = 0,  ancho p,   posición z = n
  Rendija 2 : centrada en x = a,  ancho p₂,  posición z = n + c
  Plano de observación : z = n + c + d₂

  Apertura vertical (fija): q = q₂ = Q_FIJA = 3 cm

Método de integración
---------------------
  Cuadratura de Gauss-Legendre (numpy.polynomial.legendre.leggauss).
  Convergencia exponencial con N_quad para integraciones en x.
  El eje y se excluye de la integración (régimen geométrico).

λ = 632.8 nm (He-Ne)
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss


# ── Constante física global ──────────────────────────────────────────────
Q_FIJA = 3e-2   # apertura vertical fija [m]


# ======================================================================
class DifraccionDosRendijas:
    """
    Simula la difracción por doble rendija en 1D (eje x).

    Parámetros
    ----------
    p      : ancho horizontal de la rendija 1  [m]
    p2     : ancho horizontal de la rendija 2  [m]
    a      : posición x del centro de la rendija 2  [m]
    c      : separación longitudinal entre rendijas  [m]
    d2     : distancia rendija-2 → plano de observación  [m]
    n      : posición z de la rendija 1  [m]
    N_obs  : puntos en el eje x del plano de observación
    N_quad : puntos de cuadratura Gauss-Legendre
    x_range: extensión total en x del plano de observación  [m]
    """

    LAMBDA  = 632.8e-9   # longitud de onda He-Ne [m]
    Q_FIJA  = 3e-2       # apertura vertical fija  [m]

    def __init__(
        self,
        p      = 0.10e-3,
        p2     = 0.10e-3,
        a      = 0.30e-3,
        c      = 20e-3,
        d2     = 150e-3,
        n      = 10e-3,
        N_obs  = 256,
        N_quad = 40,
        x_range= 5e-3,
    ):
        if c <= 0:
            raise ValueError(f"c debe ser positivo (c={c}).")
        if d2 <= 0:
            raise ValueError(f"d2 debe ser positivo (d2={d2}).")

        self.lam = self.LAMBDA
        self.k   = 2.0 * np.pi / self.lam

        self.p   = float(p)
        self.p2  = float(p2)
        self.a   = float(a)
        self.c   = float(c)
        self.d2  = float(d2)
        self.n   = float(n)
        self.z0  = n + c + d2   # posición z del plano de observación

        self.N_obs  = int(N_obs)
        self.N_quad = int(N_quad)
        self.x0_arr = np.linspace(-x_range / 2, x_range / 2, self.N_obs)

    # ------------------------------------------------------------------
    @staticmethod
    def _gl_nodes(lo, hi, N):
        """Nodos y pesos GL escalados al intervalo [lo, hi]."""
        xi, wi = leggauss(N)
        nodes   = 0.5 * (hi - lo) * xi + 0.5 * (hi + lo)
        weights = 0.5 * (hi - lo) * wi
        return nodes, weights

    # ==================================================================
    # Método 1 — Huyghens-Fresnel (kernel cilíndrico exacto 2D en x)
    # ==================================================================
    def calcular_hf(self):
        """
        Campo complejo mediante Huyghens-Fresnel 2D (eje x):

            U(x0) ∝ ∫_{x2} [∫_{x1} exp(ik·r12)/√r12  dx1]
                            × exp(ik·r2)/√r2  dx2

        con  r12 = √((x2−x1)² + c²)
             r2  = √((x0−x2)² + d2²)

        El eje y se excluye (régimen geométrico, q = 3 cm >> λ).

        Devuelve
        --------
        U : ndarray complejo de forma (N_obs,)
        """
        k, c, d2 = self.k, self.c, self.d2
        N = self.N_quad

        x1, wx1 = self._gl_nodes(-self.p / 2,         self.p / 2,         N)
        x2, wx2 = self._gl_nodes(self.a - self.p2 / 2, self.a + self.p2 / 2, N)

        # ── integral interna ─────────────────────────────────────────
        # dx12[i, j] = x2[i] − x1[j]   → (N2, N1)
        dx12 = x2[:, None] - x1[None, :]
        r12  = np.sqrt(dx12**2 + c**2)
        I_inner = (wx1 * np.exp(1j * k * r12) / np.sqrt(r12)).sum(axis=1)  # (N2,)

        # ── integral externa ─────────────────────────────────────────
        # dx_out[m, i] = x0[m] − x2[i]   → (Ng, N2)
        dx_out = self.x0_arr[:, None] - x2[None, :]
        r2     = np.sqrt(dx_out**2 + d2**2)
        U = (wx2 * I_inner * np.exp(1j * k * r2) / np.sqrt(r2)).sum(axis=1)  # (Ng,)

        return (1j / self.lam) * U

    # ==================================================================
    # Método 2 — Fresnel (paraxial, eje x)
    # ==================================================================
    def calcular_fresnel(self):
        """
        Campo complejo mediante Fresnel paraxial 1D (eje x):

            U(x0) ∝ ∫_{x2} [∫_{x1} exp(ik(x2−x1)²/2c) dx1]
                            × exp(ik(x0−x2)²/2d2) dx2

        Devuelve
        --------
        U : ndarray complejo de forma (N_obs,)
        """
        k, c, d2 = self.k, self.c, self.d2
        N = self.N_quad

        x1, wx1 = self._gl_nodes(-self.p / 2,         self.p / 2,         N)
        x2, wx2 = self._gl_nodes(self.a - self.p2 / 2, self.a + self.p2 / 2, N)

        dx_inner = x2[:, None] - x1[None, :]   # (N2, N1)
        I_inner  = (wx1 * np.exp(1j * k / (2 * c) * dx_inner**2)).sum(axis=1)

        dx_outer = self.x0_arr[:, None] - x2[None, :]   # (Ng, N2)
        U = (wx2 * I_inner * np.exp(1j * k / (2 * d2) * dx_outer**2)).sum(axis=1)

        return 1.0 / ((1j * self.lam) * c * d2) * U

    # ==================================================================
    # Ejecución y visualización
    # ==================================================================
    def run(self, guardar_imagen=None):
        """
        Calcula ambos campos, computa la diferencia y genera la figura
        comparativa.  Devuelve el score (diferencia media normalizada).

        Parámetros
        ----------
        guardar_imagen : str o None — ruta para guardar la figura PNG

        Retorna
        -------
        score : float ∈ [0, 1]
        """
        self._imprimir_cabecera()

        U_hf   = self.calcular_hf()
        I_hf   = np.abs(U_hf) ** 2

        U_fres = self.calcular_fresnel()
        I_fres = np.abs(U_fres) ** 2

        norm = max(I_hf.max(), I_fres.max())
        if norm < 1e-50:
            print("  [!] Intensidad nula — revise los parámetros.")
            return 0.0

        I_hf_n   = I_hf   / norm
        I_fres_n = I_fres / norm
        dif      = np.abs(I_hf_n - I_fres_n)
        score    = float(np.mean(dif))

        print(f"  Score (⟨|I_HF − I_F|⟩) = {score:.6f}")

        fig = self._plot(I_hf_n, I_fres_n, dif, score)
        if guardar_imagen:
            fig.savefig(guardar_imagen, dpi=150, bbox_inches="tight")
            print(f"  Figura guardada en '{guardar_imagen}'")

        return score

    # ------------------------------------------------------------------
    def _imprimir_cabecera(self):
        sep = "=" * 58
        print(sep)
        print("  Simulador 1D — doble rendija  (q = q₂ = 3 cm fijo)")
        print(f"  λ = {self.lam*1e9:.1f} nm    N_quad = {self.N_quad}")
        print(f"  Rendija 1 : p  = {self.p*1e3:.4f} mm   z = {self.n*1e3:.1f} mm")
        print(f"  Rendija 2 : p₂ = {self.p2*1e3:.4f} mm   x = {self.a*1e3:.4f} mm")
        print(f"  c  = {self.c*1e3:.2f} mm    d₂ = {self.d2*1e3:.2f} mm")
        print(f"  z_obs = {self.z0*1e3:.2f} mm")
        print(sep)

    # ------------------------------------------------------------------
    def _plot(self, I_hf_n, I_fres_n, dif, score):
        x_mm = self.x0_arr * 1e3

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(
            f"Difracción doble rendija 1D  —  apertura vertical fija q = q₂ = 3 cm\n"
            f"p = {self.p*1e3:.4f} mm,  p₂ = {self.p2*1e3:.4f} mm,  "
            f"a = {self.a*1e3:.4f} mm,  c = {self.c*1e3:.2f} mm,  "
            f"d₂ = {self.d2*1e3:.2f} mm",
            fontsize=10,
        )

        # Panel 1 — HF
        ax = axes[0, 0]
        ax.plot(x_mm, I_hf_n, lw=1.8, color="steelblue", label="Huyghens-Fresnel")
        ax.fill_between(x_mm, I_hf_n, alpha=0.15, color="steelblue")
        ax.set_xlabel("x  [mm]")
        ax.set_ylabel("Intensidad norm.")
        ax.set_title("Huyghens-Fresnel  (exacta 2D)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

        # Panel 2 — Fresnel
        ax = axes[0, 1]
        ax.plot(x_mm, I_fres_n, lw=1.8, color="tomato", label="Fresnel")
        ax.fill_between(x_mm, I_fres_n, alpha=0.15, color="tomato")
        ax.set_xlabel("x  [mm]")
        ax.set_ylabel("Intensidad norm.")
        ax.set_title("Fresnel  (paraxial 1D)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

        # Panel 3 — Diferencia
        ax = axes[1, 0]
        ax.plot(x_mm, dif, lw=1.8, color="purple")
        ax.fill_between(x_mm, dif, alpha=0.2, color="purple")
        ax.set_xlabel("x  [mm]")
        ax.set_ylabel(r"$|I_\mathrm{HF} - I_\mathrm{F}|$ / max")
        ax.set_title(f"Diferencia absoluta  (score = {score:.4f})")
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

        # Panel 4 — Superposición
        ax = axes[1, 1]
        ax.plot(x_mm, I_hf_n, lw=1.8, color="steelblue", label="HF")
        ax.plot(x_mm, I_fres_n, lw=1.8, color="tomato", ls="--", label="Fresnel")
        ax.fill_between(
            x_mm,
            np.minimum(I_hf_n, I_fres_n),
            np.maximum(I_hf_n, I_fres_n),
            alpha=0.30, color="gray", label="Diferencia",
        )
        ax.set_xlabel("x  [mm]")
        ax.set_ylabel("Intensidad norm.")
        ax.set_title("Comparación superpuesta")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

        fig.tight_layout()
        return fig


# ======================================================================
# Punto de entrada
# ======================================================================
if __name__ == "__main__":
    # Ejemplo con parámetros físicamente razonables:
    # rendijas de ~0.1 mm, separadas 0.3 mm en x,
    # c = 20 mm entre planos, observación a 150 mm.
    sim = DifraccionDosRendijas(
        p      = 0.10e-3,   # ancho rendija 1: 0.10 mm
        p2     = 0.10e-3,   # ancho rendija 2: 0.10 mm
        a      = 0.30e-3,   # posición x rendija 2: 0.30 mm
        c      = 20e-3,     # separación z entre rendijas: 20 mm
        d2     = 150e-3,    # distancia rendija 2 → observación: 150 mm
        n      = 10e-3,     # posición z rendija 1: 10 mm
        N_obs  = 256,
        N_quad = 40,
        x_range= 5e-3,
    )
    sim.run(guardar_imagen="simulacion_ejemplo.png")
    plt.show()

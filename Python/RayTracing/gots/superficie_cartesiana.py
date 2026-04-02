"""Superficie cartesiana definida por parámetros GOTS.

Implementa la ecuación paramétrica (Eq. 16-17), el gradiente implícito
(Eq. 63-65) y la normal (Eq. 66) de la tesis de Silva-Lora (2024).
"""

import numpy as np
from .parametros_gots import ParametrosGOTS, calcular_gots


class SuperficieCartesiana:
    """Superficie cartesiana (ovoide de Descartes) en representación GOTS."""

    def __init__(self, params: ParametrosGOTS, n_k: float, n_k1: float):
        """
        Args:
            params: parámetros GOTS de la superficie
            n_k: índice de refracción antes de la superficie
            n_k1: índice de refracción después de la superficie
        """
        self.params = params
        self.G = params.G_k
        self.O = params.O_k
        self.T = params.T_k
        self.S = params.S_k
        self.zeta = params.zeta_k
        self.n_k = n_k
        self.n_k1 = n_k1

        # Coeficiente dentro del radical de Eq. 15: 2S - O²G
        self._radical_coef = 2.0 * self.S - self.O**2 * self.G

        # ρ_max: dominio máximo donde el radical es no-negativo
        # 1 + (2S - O²G)·ρ² ≥ 0  →  ρ² ≤ -1/(2S - O²G) si 2S-O²G < 0
        if self._radical_coef < -1e-15:
            self.rho_max = np.sqrt(-1.0 / self._radical_coef)
        else:
            self.rho_max = np.inf

        # No se aplica signo fijo a la normal; se devuelve ∇f/|∇f| directamente.
        # El signo se ajusta en el trazador para que N̂·û_in > 0 (Eq. 68).

    @classmethod
    def desde_parametros_fisicos(cls, n_k, n_k1, zeta_k, d_k, d_k1):
        """Construye la superficie a partir de los parámetros físicos del sistema."""
        params = calcular_gots(n_k, n_k1, zeta_k, d_k, d_k1)
        return cls(params, n_k, n_k1)

    def z_de_rho(self, rho):
        """Coordenada axial z(ρ) — Eq. 16.

        z(ρ) = ζ + (O + T·ρ²)·ρ² / (1 + S·ρ² + √(1 + (2S - O²G)·ρ²))
        """
        rho2 = np.asarray(rho, dtype=float)**2
        radical = np.sqrt(np.maximum(1.0 + self._radical_coef * rho2, 0.0))
        denominador = 1.0 + self.S * rho2 + radical
        numerador = (self.O + self.T * rho2) * rho2

        # Evitar división por cero en ρ=0
        with np.errstate(divide='ignore', invalid='ignore'):
            resultado = np.where(
                np.abs(denominador) < 1e-30,
                0.0,
                numerador / denominador
            )
        return self.zeta + resultado

    def r_de_rho(self, rho):
        """Coordenada transversal r(ρ) — Eq. 17.

        r(ρ) = √(ρ² - (z(ρ) - ζ)²)
        """
        rho2 = np.asarray(rho, dtype=float)**2
        z = self.z_de_rho(rho)
        tau2 = (z - self.zeta)**2
        return np.sqrt(np.maximum(rho2 - tau2, 0.0))

    def gradiente(self, x, y, z):
        """Gradiente ∇f(x,y,z) de la forma implícita — Eqs. 63-65.

        f(x,y,z) = O·G·τ² - 2(1 + S·ρ²)·τ + (O + T·ρ²)·ρ²
        con τ = z - ζ, ρ² = x² + y² + τ²
        """
        tau = z - self.zeta
        rho2 = x**2 + y**2 + tau**2

        # Eq. 63: ∂f/∂x = 2(O·x - 2S·τ·x + 2T·ρ²·x)
        dfdx = 2.0 * (self.O * x - 2.0 * self.S * tau * x + 2.0 * self.T * rho2 * x)

        # Eq. 64: ∂f/∂y = 2(O·y - 2S·τ·y + 2T·ρ²·y)
        dfdy = 2.0 * (self.O * y - 2.0 * self.S * tau * y + 2.0 * self.T * rho2 * y)

        # Eq. 65: ∂f/∂z = 2(G·O·τ + O·τ - S·(2τ² + ρ²) + 2T·τ·ρ² - 1)
        dfdz = 2.0 * (self.G * self.O * tau + self.O * tau
                       - self.S * (2.0 * tau**2 + rho2)
                       + 2.0 * self.T * tau * rho2 - 1.0)

        return np.array([dfdx, dfdy, dfdz])

    def normal(self, x, y, z):
        """Vector normal unitario en el punto (x,y,z) — Eq. 66.

        Retorna ∇f/|∇f|. El signo se ajusta externamente según la dirección
        del rayo incidente para cumplir con la convención de Eq. 68.
        """
        grad = self.gradiente(x, y, z)
        norma = np.linalg.norm(grad)
        if norma < 1e-15:
            return np.array([0.0, 0.0, 1.0])
        return grad / norma

    def generar_perfil_meridional(self, num_puntos=500, rho_max_frac=0.95):
        """Genera el perfil (r, z) para una gráfica 2D en el plano meridional.

        Returns:
            (z_arr, r_arr): arrays de coordenadas, incluyendo perfil superior e inferior.
        """
        rho_lim = self.rho_max * rho_max_frac if np.isfinite(self.rho_max) else 50.0
        rho = np.linspace(0, rho_lim, num_puntos)
        z = self.z_de_rho(rho)
        r = self.r_de_rho(rho)
        return z, r

    def generar_malla_3d(self, num_rho=100, num_phi=60, rho_max_frac=0.95):
        """Genera una malla 3D por revolución para visualización o STL.

        Returns:
            (X, Y, Z): arrays 2D de forma (num_rho, num_phi).
        """
        rho_lim = self.rho_max * rho_max_frac if np.isfinite(self.rho_max) else 50.0
        rho = np.linspace(0, rho_lim, num_rho)
        phi = np.linspace(0, 2 * np.pi, num_phi, endpoint=False)

        z_perfil = self.z_de_rho(rho)
        r_perfil = self.r_de_rho(rho)

        # Revolución alrededor del eje z
        RHO, PHI = np.meshgrid(r_perfil, phi)
        Z_mesh = np.tile(z_perfil, (num_phi, 1))
        X = RHO * np.cos(PHI)
        Y = RHO * np.sin(PHI)

        return X, Y, Z_mesh

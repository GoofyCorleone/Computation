"""Cálculo de los parámetros de forma GOTS para superficies cartesianas.

Implementa las ecuaciones (10)-(13) de la tesis de Silva-Lora (2024).
Los parámetros se derivan de la forma implícita (Eq. 9):
    O_k·G_k·τ² - 2(1 + S_k·ρ²)·τ + (O_k + T_k·ρ²)·ρ² = 0
dividiendo la Eq. (6) por el factor de normalización (7).
"""

from dataclasses import dataclass


@dataclass
class ParametrosGOTS:
    """Parámetros de forma de una superficie cartesiana.

    G_k: constante de Schwarzschild generalizada (adimensional)
    O_k: curvatura paraxial (1/longitud)
    T_k: parámetro de familia de ovoides (1/longitud³)
    S_k: parámetro de restricción, S² = G·O·T
    zeta_k: posición axial del vértice
    """
    G_k: float
    O_k: float
    T_k: float
    S_k: float
    zeta_k: float

    def verificar(self, tol=1e-6):
        """Verifica la restricción S² = G·O·T."""
        got = self.G_k * self.O_k * self.T_k
        s2 = self.S_k ** 2
        if abs(got) + abs(s2) < 1e-30:
            return True
        error_rel = abs(s2 - got) / max(abs(s2), abs(got), 1e-30)
        return error_rel < tol


def calcular_gots(n_k, n_k1, zeta_k, d_k, d_k1):
    """Calcula los parámetros GOTS a partir de los parámetros físicos.

    Args:
        n_k: índice de refracción del medio objeto
        n_k1: índice de refracción del medio imagen
        zeta_k: posición axial del vértice de la superficie
        d_k: posición axial del punto objeto
        d_k1: posición axial del punto imagen

    Returns:
        ParametrosGOTS con los parámetros G, O, T, S calculados.
    """
    # Distancias desde el vértice (ξ = d_k - ζ, η = d_k1 - ζ)
    xi = d_k - zeta_k
    eta = d_k1 - zeta_k

    # κ = n_k1·η - n_k·ξ  (Eq. 3)
    kappa = n_k1 * eta - n_k * xi

    # O_k  (Eq. 11)
    O_k = (n_k1 * xi - n_k * eta) / (xi * eta * (n_k1 - n_k))

    # T_k  (Eq. 12)
    T_k = (n_k1 - n_k) * (n_k1 + n_k)**2 / (4 * n_k * n_k1 * xi * eta * kappa)

    # S_k  (Eq. 13)
    S_k = (n_k1 + n_k) * (n_k1**2 * eta - n_k**2 * xi) / (2 * n_k * n_k1 * xi * eta * kappa)

    # G_k  (Eq. 10) = (O_k·G_k) / O_k
    OG = (n_k1**2 * eta - n_k**2 * xi)**2 / (n_k * n_k1 * xi * eta * (n_k1 - n_k) * kappa)
    G_k = OG / O_k

    return ParametrosGOTS(G_k=G_k, O_k=O_k, T_k=T_k, S_k=S_k, zeta_k=zeta_k)

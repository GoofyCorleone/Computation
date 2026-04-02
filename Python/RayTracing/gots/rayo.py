"""Rayo y cálculo de intersección con superficies cartesianas.

Implementa las ecuaciones (44)-(56) de la tesis de Silva-Lora (2024)
para encontrar la intersección rayo-superficie resolviendo la cuártica.
"""

from dataclasses import dataclass
import numpy as np
from .utilidades import resolver_cuartica


@dataclass
class Rayo:
    """Rayo óptico definido por un punto de origen y una dirección unitaria."""
    origen: np.ndarray   # (3,)
    direccion: np.ndarray  # (3,) unitario

    def punto(self, t):
        """Punto sobre el rayo a distancia paramétrica t."""
        return self.origen + t * self.direccion


@dataclass
class Interseccion:
    """Resultado de la intersección entre un rayo y una superficie."""
    punto: np.ndarray       # (3,) coordenadas del punto de intersección
    normal: np.ndarray      # (3,) normal unitaria en el punto
    tau: float              # parámetro τ = z - ζ
    superficie_idx: int     # índice de la superficie en el sistema


def coeficientes_cuartica(rayo, superficie):
    """Calcula los coeficientes Q4..Q0 de la Ec. 51.

    El parámetro del polinomio es τ = z - ζ, donde el rayo se parametriza como:
        x' = (u_x/u_z)·τ + b_x        (Eq. 44)
        y' = (u_y/u_z)·τ + b_y        (Eq. 45)
        z' = τ                          (Eq. 46)
    con b_x, b_y dados por Eqs. 47-48.

    Para el caso u_z ≈ 0, se usa una parametrización alternativa.
    """
    ux, uy, uz = rayo.direccion
    xo, yo, zo = rayo.origen
    OG = superficie.OG
    O = superficie.O
    T = superficie.T
    S = superficie.S
    zeta = superficie.zeta

    if abs(uz) > 1e-12:
        # Caso estándar: parametrización por τ = z - ζ
        # Eqs. 47-48
        bx = -(ux / uz) * (zo - zeta) + xo
        by = -(uy / uz) * (zo - zeta) + yo

        mx = ux / uz
        my = uy / uz

        # ρ'² = (mx·τ + bx)² + (my·τ + by)² + τ²
        # = (mx²+my²+1)·τ² + 2(mx·bx+my·by)·τ + (bx²+by²)
        # Definimos:
        #   A = mx²+my²+1 = 1/uz²
        #   B = mx·bx + my·by
        #   C = bx² + by²
        # Entonces ρ'² = A·τ² + 2B·τ + C

        A = mx**2 + my**2 + 1.0
        B = mx * bx + my * by
        C = bx**2 + by**2

        # Coeficientes de la cuártica en τ (Ec. 52-56)
        # f = OG·τ² - 2τ - 2S·ρ²·τ + O·ρ² + T·(ρ²)²
        # con ρ² = A·τ² + 2B·τ + C
        Q4 = T * A**2
        Q3 = 4.0 * T * A * B - 2.0 * S * A
        Q2 = T * (4.0 * B**2 + 2.0 * A * C) - 4.0 * S * B + O * A + OG
        Q1 = 4.0 * T * B * C - 2.0 * S * C + 2.0 * O * B - 2.0
        Q0 = T * C**2 + O * C

        return Q4, Q3, Q2, Q1, Q0, 'tau'

    else:
        # Caso u_z ≈ 0: rayo casi perpendicular al eje óptico
        # Parametrizamos por t (distancia a lo largo del rayo)
        # P = origen + t·û
        # x = xo + ux·t, y = yo + uy·t, z = zo + uz·t ≈ zo
        # τ = z - ζ = zo - ζ + uz·t
        # ρ² = x² + y² + τ²
        #
        # Substituir directamente en f = 0 usando t como variable.
        # Este es un caso menos común; usamos expansión numérica directa.
        tau0 = zo - zeta

        # f(t) = O·G·(τ0+uz·t)² - 2(1+S·ρ²(t))·(τ0+uz·t) + (O+T·ρ²(t))·ρ²(t)
        # donde ρ²(t) = (xo+ux·t)² + (yo+uy·t)² + (τ0+uz·t)²
        #
        # Expandimos ρ²(t):
        # = (ux²+uy²+uz²)·t² + 2(xo·ux+yo·uy+τ0·uz)·t + (xo²+yo²+τ0²)
        # = t² + 2D·t + E    (ya que |û|=1)
        # con D = xo·ux+yo·uy+τ0·uz, E = xo²+yo²+τ0²

        D = xo * ux + yo * uy + tau0 * uz
        E = xo**2 + yo**2 + tau0**2

        # Coeficientes de la cuártica en t (expansión directa de f(t)=0)
        # f = OG·(τ0+uz·t)² - 2(1+S·ρ²)·(τ0+uz·t) + (O+T·ρ²)·ρ²
        # con ρ²(t) = t² + 2D·t + E
        Q4 = T
        Q3 = 4.0 * T * D - 2.0 * S * uz
        Q2 = (T * (4.0 * D**2 + 2.0 * E) - 2.0 * S * (tau0 + 2.0 * D * uz)
               + O + OG * uz**2)
        Q1 = (4.0 * T * D * E - 2.0 * S * (2.0 * D * tau0 + E * uz)
               + 2.0 * O * D + 2.0 * OG * tau0 * uz - 2.0 * uz)
        Q0 = T * E**2 - 2.0 * S * E * tau0 + O * E + OG * tau0**2 - 2.0 * tau0

        return Q4, Q3, Q2, Q1, Q0, 't'


def intersectar(rayo, superficie, superficie_idx=0):
    """Encuentra la intersección entre un rayo y una superficie cartesiana.

    Resuelve la cuártica (Eq. 51) y selecciona la raíz física:
    la raíz con menor |τ| (o t>0) cuyo ρ ≤ ρ_max y t_rayo > 0.

    Returns:
        Interseccion o None si no hay intersección válida.
    """
    Q4, Q3, Q2, Q1, Q0, param_tipo = coeficientes_cuartica(rayo, superficie)
    raices = resolver_cuartica(Q4, Q3, Q2, Q1, Q0)

    if len(raices) == 0:
        return None

    xo, yo, zo = rayo.origen
    ux, uy, uz = rayo.direccion
    zeta = superficie.zeta

    for raiz in raices:
        if param_tipo == 'tau':
            tau = raiz
            z = tau + zeta
            if abs(uz) > 1e-12:
                bx = -(ux / uz) * (zo - zeta) + xo
                by = -(uy / uz) * (zo - zeta) + yo
                x = (ux / uz) * tau + bx
                y = (uy / uz) * tau + by
            else:
                # Esto no debería ocurrir
                continue
        else:  # param_tipo == 't'
            t = raiz
            if t < 1e-12:
                continue  # en esta parametrización t es el parámetro del rayo
            x = xo + ux * t
            y = yo + uy * t
            z = zo + uz * t
            tau = z - zeta

        # Verificar ρ ≤ ρ_max
        rho2 = x**2 + y**2 + tau**2
        rho = np.sqrt(rho2)

        if np.isfinite(superficie.rho_max) and rho > superficie.rho_max * 1.001:
            continue

        # Verificar que el rayo avanza (t > 0 en la dirección del rayo)
        punto = np.array([x, y, z])
        t_rayo = np.dot(punto - rayo.origen, rayo.direccion)
        if t_rayo < 1e-10:
            continue

        normal_vec = superficie.normal(x, y, z)

        # Convención Eq. 68: N̂ = ∇f/|∇f|, con N̂·û < 0 para rayo incidente.
        # ∂f/∂z|vértice = -2, apunta en -z. El rayo viene de -z (+z dirección),
        # así que N̂·û < 0 naturalmente. Si por geometría resulta positivo, flipear.
        if np.dot(normal_vec, rayo.direccion) > 0:
            normal_vec = -normal_vec

        return Interseccion(
            punto=punto,
            normal=normal_vec,
            tau=tau,
            superficie_idx=superficie_idx
        )

    return None

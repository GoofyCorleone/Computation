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
    G = superficie.G
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

        # Ec. 52-56
        Q4 = T * A**2
        Q3 = (4.0 * T * A * B - 2.0 * S * A)
        Q2 = (G * O + O * A + 2.0 * T * (4.0 * B**2 + 2.0 * A * C)
               - 4.0 * S * B)
        # Hmm, let me be more careful. From the thesis:
        # ρ'² = A·τ² + 2B·τ + C as defined above
        # (ρ'²)² = A²·τ⁴ + 4A·B·τ³ + (4B²+2AC)·τ² + 4BC·τ + C²
        #
        # Substituting into f = O·G·τ² - 2(1+S·ρ'²)τ + (O+T·ρ'²)·ρ'²:
        # = O·G·τ² - 2τ - 2S·ρ'²·τ + O·ρ'² + T·(ρ'²)²
        #
        # τ⁴ coeff: T·A²
        # τ³ coeff: 4T·A·B - 2S·A
        # τ² coeff: O·G + T·(4B²+2AC) + O·A - 2S·2B  wait...
        #   -2S·ρ'²·τ has τ² coeff: -2S·A  (from A·τ²·τ → that's τ³, not τ²)
        #   Let me redo systematically:
        #
        # Term: O·G·τ²
        #   τ² → O·G
        #
        # Term: -2τ
        #   τ¹ → -2
        #
        # Term: -2S·ρ'²·τ = -2S·(A·τ² + 2B·τ + C)·τ
        #   = -2S·A·τ³ - 4S·B·τ² - 2S·C·τ
        #
        # Term: O·ρ'² = O·(A·τ² + 2B·τ + C)
        #   = O·A·τ² + 2O·B·τ + O·C
        #
        # Term: T·(ρ'²)² = T·(A²·τ⁴ + 4A·B·τ³ + (4B²+2AC)·τ² + 4BC·τ + C²)
        #
        # Collecting:
        # τ⁴: T·A²
        # τ³: 4T·A·B - 2S·A
        # τ²: T·(4B²+2AC) - 4S·B + O·A + O·G
        # τ¹: 4T·B·C - 2S·C + 2O·B - 2
        # τ⁰: T·C² + O·C

        Q4 = T * A**2
        Q3 = 4.0 * T * A * B - 2.0 * S * A
        Q2 = T * (4.0 * B**2 + 2.0 * A * C) - 4.0 * S * B + O * A + O * G
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

        # τ(t) = τ0 + uz·t
        # (ρ²)² = t⁴ + 4D·t³ + (4D²+2E)·t² + 4DE·t + E²
        #
        # Expanding f(t) and collecting powers of t:
        # τ² = uz²·t² + 2τ0·uz·t + τ0²
        # ρ²·τ = (t²+2D·t+E)·(τ0+uz·t) = uz·t³+(τ0+2D·uz)·t²+(2D·τ0+E·uz)·t+E·τ0

        # τ⁰·τ⁴: T
        # (organized as Q4·t⁴ + Q3·t³ + Q2·t² + Q1·t + Q0)

        Q4 = T  # coeff de (ρ²)² en t⁴ es T·1 = T
        Q3 = 4.0 * T * D - 2.0 * S * uz
        Q2 = (T * (4.0 * D**2 + 2.0 * E) - 4.0 * S * D
               + O + O * G * uz**2 - 2.0 * S * uz * 0)  # hmm let me redo

        # Actually this gets complex. Let me just use a general approach:
        # sample f(t) and find roots numerically via companion matrix.
        # For uz ≈ 0 but not exactly 0, the tau-parametrization with large mx,my
        # still works if we solve the quartic carefully.
        # Let's handle by re-parametrizing with a small uz.
        # Or better: just use the full expansion.

        # Full expansion term by term:
        # Term OG·τ²: OG·(τ0+uz·t)² = OG·uz²·t² + 2·OG·τ0·uz·t + OG·τ0²
        # Term -2τ: -2·(τ0+uz·t) = -2·uz·t - 2·τ0
        # Term -2S·ρ²·τ: -2S·(t²+2D·t+E)·(τ0+uz·t)
        #   = -2S·[uz·t³ + (τ0+2D·uz)·t² + (2D·τ0+E·uz)·t + E·τ0]
        # Term O·ρ²: O·(t²+2D·t+E)
        # Term T·(ρ²)²: T·(t⁴ + 4D·t³ + (4D²+2E)·t² + 4DE·t + E²)

        OG = O * G
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
    la menor τ (o t) positiva cuyo ρ ≤ ρ_max.

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

"""Sistema óptico compuesto por superficies cartesianas.

Provee trazado de rayos secuencial a través de múltiples superficies
y un factory para lentes singletes ovoides estigmáticas (LSOE).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from .superficie_cartesiana import SuperficieCartesiana
from .rayo import Rayo, Interseccion, intersectar
from .snell import refraccion_snell
from .utilidades import normalizar


@dataclass
class ResultadoTrazado:
    """Resultado del trazado de un rayo a través del sistema."""
    puntos: List[np.ndarray]        # puntos de intersección (incluye origen)
    direcciones: List[np.ndarray]   # direcciones del rayo en cada segmento
    rayo_completo: bool             # True si el rayo llegó al final del sistema


class SistemaOptico:
    """Sistema óptico compuesto por un conjunto de superficies cartesianas."""

    def __init__(self):
        self.superficies: List[SuperficieCartesiana] = []

    def agregar_superficie(self, superficie: SuperficieCartesiana):
        """Agrega una superficie al sistema."""
        self.superficies.append(superficie)

    def trazar_rayo(self, rayo: Rayo) -> ResultadoTrazado:
        """Traza un rayo secuencialmente a través de todas las superficies.

        En cada superficie:
        1. Calcula intersección (cuártica, Eq. 51)
        2. Calcula normal (gradiente, Eq. 63-66)
        3. Aplica ley de Snell vectorial (Eq. 68)
        """
        puntos = [rayo.origen.copy()]
        direcciones = [rayo.direccion.copy()]
        rayo_actual = Rayo(rayo.origen.copy(), rayo.direccion.copy())

        for k, sup in enumerate(self.superficies):
            inter = intersectar(rayo_actual, sup, superficie_idx=k)
            if inter is None:
                return ResultadoTrazado(puntos, direcciones, rayo_completo=False)

            puntos.append(inter.punto)

            # Refracción
            u_refractado = refraccion_snell(
                rayo_actual.direccion, inter.normal, sup.n_k, sup.n_k1
            )
            if u_refractado is None:
                # Reflexión total interna
                return ResultadoTrazado(puntos, direcciones, rayo_completo=False)

            direcciones.append(u_refractado)
            rayo_actual = Rayo(inter.punto.copy(), u_refractado.copy())

        return ResultadoTrazado(puntos, direcciones, rayo_completo=True)

    def trazar_abanico(self, fuente, num_rayos=20, angulo_max=0.3,
                        plano='meridional'):
        """Genera y traza un abanico de rayos desde un punto fuente.

        Args:
            fuente: punto fuente (3,) o (x, y, z)
            num_rayos: número de rayos en el abanico
            angulo_max: ángulo máximo respecto al eje óptico (radianes)
            plano: 'meridional' (plano y-z) o 'sagital' (plano x-z)

        Returns:
            Lista de ResultadoTrazado.
        """
        fuente = np.asarray(fuente, dtype=float)
        resultados = []

        angulos = np.linspace(-angulo_max, angulo_max, num_rayos)

        for theta in angulos:
            if plano == 'meridional':
                direccion = normalizar(np.array([0.0, np.sin(theta), np.cos(theta)]))
            else:
                direccion = normalizar(np.array([np.sin(theta), 0.0, np.cos(theta)]))

            rayo = Rayo(fuente.copy(), direccion)
            resultado = self.trazar_rayo(rayo)
            resultados.append(resultado)

        return resultados

    def encontrar_apertura(self):
        """Encuentra la apertura (r_max) donde superficies consecutivas se interceptan.

        Para cada par de superficies consecutivas, busca el ρ donde sus
        perfiles z(ρ) se cruzan. Ese punto define el borde físico de la lente.

        Returns:
            Lista de r_max para cada par de superficies consecutivas.
        """
        aperturas = []
        for k in range(len(self.superficies) - 1):
            sup0 = self.superficies[k]
            sup1 = self.superficies[k + 1]

            rho_lim_0 = sup0.rho_max if np.isfinite(sup0.rho_max) else 200.0
            rho_lim_1 = sup1.rho_max if np.isfinite(sup1.rho_max) else 200.0
            rho_lim = min(rho_lim_0, rho_lim_1)

            rho_test = np.linspace(0.01, rho_lim * 0.99, 2000)
            z0 = sup0.z_de_rho(rho_test)
            z1 = sup1.z_de_rho(rho_test)

            diff = z0 - z1
            cruces = np.where(np.diff(np.sign(diff)))[0]

            if len(cruces) > 0:
                idx = cruces[0]
                f = abs(diff[idx]) / (abs(diff[idx]) + abs(diff[idx + 1]))
                rho_cruce = rho_test[idx] + f * (rho_test[idx + 1] - rho_test[idx])
                r_cruce = sup0.r_de_rho(rho_cruce)
                aperturas.append(float(r_cruce))
            else:
                aperturas.append(min(rho_lim_0, rho_lim_1) * 0.5)

        return aperturas

    @classmethod
    def lsoe(cls, zeta_0, zeta_1, d_0, d_2, n_0, n_1, n_2, sigma=None, d_1=None):
        """Factory para lentes singletes ovoides estigmáticas (LSOE).

        Requiere exactamente uno de sigma o d_1.
        Si se da sigma, calcula d_1 a partir de la Eq. 43.

        Args:
            zeta_0: posición axial del vértice de la primera superficie
            zeta_1: posición axial del vértice de la segunda superficie
            d_0: posición axial del punto objeto
            d_2: posición axial del punto imagen
            n_0: índice de refracción del espacio objeto
            n_1: índice de refracción de la lente
            n_2: índice de refracción del espacio imagen
            sigma: factor de forma (opcional)
            d_1: posición de la imagen intermedia (opcional)

        Returns:
            (SistemaOptico, d_1_usado)
        """
        if sigma is not None and d_1 is None:
            d_1 = _d1_desde_sigma(sigma, zeta_0, zeta_1, d_0, d_2, n_0, n_1, n_2)
        elif d_1 is None:
            raise ValueError("Se debe proporcionar sigma o d_1.")

        sistema = cls()

        sup0 = SuperficieCartesiana.desde_parametros_fisicos(n_0, n_1, zeta_0, d_0, d_1)
        sup1 = SuperficieCartesiana.desde_parametros_fisicos(n_1, n_2, zeta_1, d_1, d_2)

        sistema.agregar_superficie(sup0)
        sistema.agregar_superficie(sup1)

        return sistema, d_1


def _d1_desde_sigma(sigma, zeta_0, zeta_1, d_0, d_2, n_0, n_1, n_2):
    """Calcula d_1 a partir del factor de forma σ — Eqs. 39-43.

    Resuelve la cuadrática C2·d1² + C1·d1 + C0 = 0.
    """
    # Eqs. 40-42
    C2 = (n_2 * (d_0 - zeta_0) * (n_0 - n_1) * (sigma + 1)
          + n_0 * (d_2 - zeta_1) * (n_1 - n_2) * (sigma - 1))

    C1 = (n_1 * (d_0 - zeta_0) * (d_2 - zeta_1)
          * (n_2 * (sigma - 1) - n_0 * (sigma + 1) + 2 * n_1)
          + (zeta_0 + zeta_1) * (n_2 * (n_1 - n_0) * (sigma + 1) * (d_0 - zeta_0)
                                  + n_0 * (n_2 - n_1) * (sigma - 1) * (d_2 - zeta_1)))

    C0 = (n_1 * (d_0 - zeta_0) * (d_2 - zeta_1)
          * (zeta_0 * (n_0 - n_1) * (sigma + 1) + zeta_1 * (n_1 - n_2) * (sigma - 1))
          + zeta_0 * zeta_1 * (n_2 * (d_0 - zeta_0) * (sigma + 1) * (n_0 - n_1)
                                + n_0 * (d_2 - zeta_1) * (sigma - 1) * (n_1 - n_2)))

    discriminante = C1**2 - 4 * C2 * C0
    if discriminante < 0:
        raise ValueError(f"No existe d_1 real para sigma={sigma}.")

    sqrt_disc = np.sqrt(discriminante)

    # Eq. 43: signo - da d_1 ∈ [ζ_0, ζ_1], signo + da d_1 fuera (caso típico)
    d1_minus = (-C1 - sqrt_disc) / (2 * C2)
    d1_plus = (-C1 + sqrt_disc) / (2 * C2)

    # Seleccionar la raíz fuera de [ζ_0, ζ_1] (imagen intermedia virtual/lejana)
    return d1_plus

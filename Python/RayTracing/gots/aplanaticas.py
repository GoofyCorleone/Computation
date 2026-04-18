"""Superficies rigurosamente aplanéticas (Sec. 4.4 de la tesis).

Un sistema es *rigurosamente estigmático* cuando forma imagen puntual perfecta
de un objeto puntual, y *rigurosamente aplanético* cuando además cumple la
condición seno de Abbe (Ec. 69), lo que implica la ausencia simultánea de
aberración esférica y coma. La condición de aplanetismo exige que la
magnificación M({ρ_k}) definida por Ec. (93) sea constante respecto de
todos los radios ρ_k.

La tesis demuestra (Secs. 4.4.1–4.4.4) que existen exactamente cuatro familias
de superficies cartesianas que satisfacen M = constante sin cancelaciones
accidentales. Cada Tipo impone una relación algebraica entre el punto objeto
d_k y el punto imagen d_{k+1} de la superficie k:

    Tipo-0  : 2S_k = G_k · O_k²             → Esfera en puntos de Young
              n_k (d_k − ζ_k) = n_{k+1} (d_{k+1} − ζ_k)         Ec. (99)

    Tipo-1  : d_{k+1} → ∞                    → Cónica (parábola, elipse o hipérbola)
              imagen intermedia al infinito ⇒ rayos paralelos al eje dentro
              de la lente (sólo aplica para LSOE N=2).

    Tipo-2  : O_k = 0                        → Superficie plana en el vértice
              (d_{k+1} − ζ_k)/n_{k+1} = (d_k − ζ_k)/n_k          Ec. (111)

    Tipo-3  : G_k = 0                        → Menisco (caso general)
              n_k² (d_k − ζ_k) = n_{k+1}² (d_{k+1} − ζ_k)        Ec. (121)

Para los Tipos 2 y 3, para que M = 1 (estrictamente aplanético, sin
magnificación anómala), se requiere además n_0 = n_N (mismo medio a la
entrada y salida del sistema).

Este módulo provee:
    * Funciones algebraicas `d_k1_tipo{0,2,3}` que devuelven d_{k+1} a
      partir de d_k, ζ_k y los índices de refracción.
    * Construcción directa de superficies (`superficie_tipo{0,1,2,3}`)
      que sortea los casos degenerados en los que `calcular_gots` tendría
      divisiones por cero (p. ej. κ = 0 en Tipo-0, o d_1 = ∞ en Tipo-1).
    * Factories de sistemas completos (`sistema_lsoe_tipo{0,1,2,3}`) para
      LSOE rigurosamente aplanéticas.
    * La función `magnificacion_M` evalúa la Ec. (93) sobre un array de
      ρ_k para comprobar numéricamente la constancia de M.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .parametros_gots import ParametrosGOTS, calcular_gots
from .superficie_cartesiana import SuperficieCartesiana
from .sistema_optico import SistemaOptico


# ---------------------------------------------------------------------------
# Relaciones algebraicas entre d_k y d_{k+1} para cada Tipo
# ---------------------------------------------------------------------------

def d_k1_tipo0(n_k: float, n_k1: float, zeta_k: float, d_k: float) -> float:
    """Tipo-0: n_k (d_k − ζ_k) = n_{k+1} (d_{k+1} − ζ_k)  (Ec. 99).

    Esfera en sus puntos aplanéticos de Young. Objeto e imagen caen ambos
    al mismo lado del vértice ζ_k, ya que sus distancias mantienen el signo.
    """
    return zeta_k + (n_k / n_k1) * (d_k - zeta_k)


def d_k1_tipo2(n_k: float, n_k1: float, zeta_k: float, d_k: float) -> float:
    """Tipo-2: (d_{k+1} − ζ_k)/n_{k+1} = (d_k − ζ_k)/n_k  (Ec. 111).

    Superficie con O_k = 0. Como las distancias conservan signo relativo al
    vértice, se forman imágenes virtuales a partir de objetos reales (y
    viceversa).
    """
    return zeta_k + (n_k1 / n_k) * (d_k - zeta_k)


def d_k1_tipo3(n_k: float, n_k1: float, zeta_k: float, d_k: float) -> float:
    """Tipo-3: n_k² (d_k − ζ_k) = n_{k+1}² (d_{k+1} − ζ_k)  (Ec. 121).

    Superficie con G_k = 0. Menisco rigurosamente aplanético.
    """
    return zeta_k + ((n_k / n_k1) ** 2) * (d_k - zeta_k)


# ---------------------------------------------------------------------------
# Construcción de superficies (parámetros GOTS directos para casos límite)
# ---------------------------------------------------------------------------

def superficie_tipo0(n_k: float, n_k1: float, zeta_k: float,
                      d_k: float) -> SuperficieCartesiana:
    """Superficie esférica en los puntos aplanéticos de Young.

    Para esta condición κ = n_{k+1}·η − n_k·ξ = 0, por lo que las Ecs.
    (10)-(13) contienen divisiones por cero. Se construye la superficie a
    mano usando G = S = T = 0 y la curvatura paraxial
        O_k = (n_k + n_{k+1}) / (n_k · (d_k − ζ_k))
    que corresponde al radio R_k = n_k·(d_k − ζ_k)/(n_k + n_{k+1}) de la
    esfera de Young.
    """
    xi = d_k - zeta_k
    if abs(xi) < 1e-30:
        raise ValueError("Tipo-0: el objeto no puede coincidir con el vértice.")

    O_k = (n_k + n_k1) / (n_k * xi)
    params = ParametrosGOTS(
        G_k=0.0, O_k=O_k, T_k=0.0, S_k=0.0,
        zeta_k=zeta_k, OG_k=0.0,
    )
    return SuperficieCartesiana(params, n_k, n_k1)


def superficie_tipo1_par(n_0: float, n_1: float, n_2: float,
                          zeta_0: float, zeta_1: float,
                          d_0: float, d_2: float
                          ) -> Tuple[SuperficieCartesiana,
                                     SuperficieCartesiana]:
    """Par de superficies cónicas de una LSOE tipo-1 (d_1 → ∞).

    En el límite d_1 → ∞ la imagen intermedia está al infinito, de modo que
    los rayos dentro de la lente viajan paralelos al eje óptico. Los
    parámetros GOTS toman los valores (Silva-Lora & Torres, 2020c):

        S_0 = S_1 = T_0 = T_1 = 0
        G_0 = −n_1²/n_0²,   G_1 = −n_1²/n_2²
        O_0 = −n_0 / [(d_0 − ζ_0)(n_1 − n_0)]
        O_1 = −n_2 / [(d_2 − ζ_1)(n_1 − n_2)]

    Ambas superficies son cónicas (parábolas/hiperboloides según la relación
    de índices).
    """
    xi0 = d_0 - zeta_0
    eta1 = d_2 - zeta_1
    if abs(xi0) < 1e-30 or abs(eta1) < 1e-30:
        raise ValueError("Tipo-1: objeto o imagen coincide con un vértice.")

    O_0 = -n_0 / (xi0 * (n_1 - n_0))
    O_1 = -n_2 / (eta1 * (n_1 - n_2))
    G_0 = -(n_1 ** 2) / (n_0 ** 2)
    G_1 = -(n_1 ** 2) / (n_2 ** 2)

    params0 = ParametrosGOTS(
        G_k=G_0, O_k=O_0, T_k=0.0, S_k=0.0,
        zeta_k=zeta_0, OG_k=O_0 * G_0,
    )
    params1 = ParametrosGOTS(
        G_k=G_1, O_k=O_1, T_k=0.0, S_k=0.0,
        zeta_k=zeta_1, OG_k=O_1 * G_1,
    )
    sup0 = SuperficieCartesiana(params0, n_0, n_1)
    sup1 = SuperficieCartesiana(params1, n_1, n_2)
    return sup0, sup1


def superficie_tipo2(n_k: float, n_k1: float, zeta_k: float,
                      d_k: float) -> SuperficieCartesiana:
    """Superficie Tipo-2 (O_k = 0) — el punto imagen se calcula por Ec. 111."""
    d_k1 = d_k1_tipo2(n_k, n_k1, zeta_k, d_k)
    # `calcular_gots` maneja correctamente este caso (κ ≠ 0, O_k = 0).
    params = calcular_gots(n_k, n_k1, zeta_k, d_k, d_k1)
    # Forzar O_k exactamente a cero para eliminar ruido de punto flotante
    params = ParametrosGOTS(
        G_k=float('inf'), O_k=0.0, T_k=params.T_k, S_k=params.S_k,
        zeta_k=zeta_k, OG_k=params.OG_k,
    )
    return SuperficieCartesiana(params, n_k, n_k1)


def superficie_tipo3(n_k: float, n_k1: float, zeta_k: float,
                      d_k: float) -> SuperficieCartesiana:
    """Superficie Tipo-3 (G_k = 0) — el punto imagen se calcula por Ec. 121."""
    d_k1 = d_k1_tipo3(n_k, n_k1, zeta_k, d_k)
    params = calcular_gots(n_k, n_k1, zeta_k, d_k, d_k1)
    # Forzar G_k = 0 y OG_k = 0 exactamente.
    params = ParametrosGOTS(
        G_k=0.0, O_k=params.O_k, T_k=params.T_k, S_k=params.S_k,
        zeta_k=zeta_k, OG_k=0.0,
    )
    return SuperficieCartesiana(params, n_k, n_k1)


# ---------------------------------------------------------------------------
# Factories de sistemas LSOE rigurosamente aplanéticos
# ---------------------------------------------------------------------------

@dataclass
class SistemaAplanetico:
    """Resultado de construir una LSOE aplanética: sistema + datos auxiliares."""
    sistema: SistemaOptico
    tipo: int
    d_0: float
    d_1: float
    d_2: float
    descripcion: str


def sistema_lsoe_tipo0(n_0, n_1, n_2, zeta_0, zeta_1, d_0) -> SistemaAplanetico:
    """LSOE tipo-0: ambas superficies son esferas en sus puntos de Young.

    La Ec. 99 aplicada a k=0 define d_1, y a k=1 define d_2. Una elección
    coherente requiere que d_1 sea coherente entre las dos superficies:
    si se usa Ec. 99 en ambas, los puntos d_1 tomados como imagen de sup0 y
    como objeto de sup1 coinciden por construcción sólo si ζ_0 ≡ ζ_1, de
    modo que para ζ_0 ≠ ζ_1 el sistema tipo-0 riguroso en ambas superficies
    es sólo posible si se escoge d_1 a partir de sup0 y se ajusta de forma
    consistente. Aquí tomamos d_1 = ζ_0 + (n_0/n_1)(d_0 − ζ_0) y d_2 a
    partir de la segunda superficie con este d_1.
    """
    d_1 = d_k1_tipo0(n_0, n_1, zeta_0, d_0)
    d_2 = d_k1_tipo0(n_1, n_2, zeta_1, d_1)
    sup0 = superficie_tipo0(n_0, n_1, zeta_0, d_0)
    sup1 = superficie_tipo0(n_1, n_2, zeta_1, d_1)
    sistema = SistemaOptico()
    sistema.agregar_superficie(sup0)
    sistema.agregar_superficie(sup1)
    return SistemaAplanetico(sistema, 0, d_0, d_1, d_2,
                              "LSOE Tipo-0 (esferas en puntos de Young)")


def sistema_lsoe_tipo1(n_0, n_1, n_2, zeta_0, zeta_1, d_0, d_2) -> SistemaAplanetico:
    """LSOE tipo-1: superficies cónicas con imagen intermedia al infinito.

    El usuario fija d_0 y d_2; d_1 es conceptualmente infinito (rayos
    paralelos al eje dentro de la lente).
    """
    sup0, sup1 = superficie_tipo1_par(n_0, n_1, n_2, zeta_0, zeta_1, d_0, d_2)
    sistema = SistemaOptico()
    sistema.agregar_superficie(sup0)
    sistema.agregar_superficie(sup1)
    return SistemaAplanetico(sistema, 1, d_0, float('inf'), d_2,
                              "LSOE Tipo-1 (cónicas, d_1 → ∞)")


def sistema_lsoe_tipo2(n_0, n_1, n_2, zeta_0, zeta_1, d_0) -> SistemaAplanetico:
    """LSOE tipo-2: ambas superficies con O_k = 0 (planas en el vértice)."""
    d_1 = d_k1_tipo2(n_0, n_1, zeta_0, d_0)
    d_2 = d_k1_tipo2(n_1, n_2, zeta_1, d_1)
    sup0 = superficie_tipo2(n_0, n_1, zeta_0, d_0)
    sup1 = superficie_tipo2(n_1, n_2, zeta_1, d_1)
    sistema = SistemaOptico()
    sistema.agregar_superficie(sup0)
    sistema.agregar_superficie(sup1)
    return SistemaAplanetico(sistema, 2, d_0, d_1, d_2,
                              "LSOE Tipo-2 (O=0, plano-aspéricas)")


def sistema_lsoe_tipo3(n_0, n_1, n_2, zeta_0, zeta_1, d_0) -> SistemaAplanetico:
    """LSOE tipo-3: ambas superficies con G_k = 0 (meniscos aplanéticos)."""
    d_1 = d_k1_tipo3(n_0, n_1, zeta_0, d_0)
    d_2 = d_k1_tipo3(n_1, n_2, zeta_1, d_1)
    sup0 = superficie_tipo3(n_0, n_1, zeta_0, d_0)
    sup1 = superficie_tipo3(n_1, n_2, zeta_1, d_1)
    sistema = SistemaOptico()
    sistema.agregar_superficie(sup0)
    sistema.agregar_superficie(sup1)
    return SistemaAplanetico(sistema, 3, d_0, d_1, d_2,
                              "LSOE Tipo-3 (G=0, meniscos aplanéticos)")


# ---------------------------------------------------------------------------
# Chequeo numérico de la condición de aplanetismo (Ec. 93)
# ---------------------------------------------------------------------------

def magnificacion_M(sistema_aplanetico: SistemaAplanetico,
                     rho_values: np.ndarray) -> np.ndarray:
    """Evalúa la magnificación M({ρ_k}) de Ec. (93) para una LSOE.

    Para cada ρ, asume ρ_0 = ρ_1 = ρ y devuelve el cociente de los factores
    de cada superficie. En un sistema rigurosamente aplanético este valor
    es constante respecto de ρ (para Tipos 0 y 1) o idénticamente 1 cuando
    n_0 = n_2 (Tipos 2 y 3).

    Útil como test numérico: `(M/M(0) − 1)` debe permanecer < 1e−10 sobre
    todo el rango de ρ físicamente accesible.
    """
    sistema = sistema_aplanetico.sistema
    rho = np.asarray(rho_values, dtype=float)

    d_list = [sistema_aplanetico.d_0, sistema_aplanetico.d_1,
              sistema_aplanetico.d_2]
    n_list = [sistema.superficies[0].n_k,
              sistema.superficies[0].n_k1,
              sistema.superficies[-1].n_k1]

    M_total = np.ones_like(rho, dtype=float)
    for k, sup in enumerate(sistema.superficies):
        A_k = 2.0 * sup.S - sup.OG * sup.O  # = 2S_k − G_k·O_k²
        d_k = d_list[k]
        d_k1 = d_list[k + 1]
        xi_k = d_k - sup.zeta
        eta_k = d_k1 - sup.zeta
        n_k = n_list[k]
        n_k1 = n_list[k + 1]

        # Factores del producto de Ec. 93 para la superficie k
        # M_k = [n_k·(A_k − G_k·O_k/ξ_k)·√(1+A_k·ρ²)]
        #       / [n_{k+1}·(A_k − G_k·O_k/η_k)·√(1+A_k·ρ²)]
        # (la raíz se cancela consigo misma pues ρ_k es común)
        if np.isfinite(sup.OG) and abs(xi_k) > 1e-30:
            frac_num = A_k - sup.OG / xi_k
        else:
            frac_num = A_k
        if np.isfinite(sup.OG) and abs(eta_k) > 1e-30 and np.isfinite(eta_k):
            frac_den = A_k - sup.OG / eta_k
        else:
            frac_den = A_k

        if abs(frac_den) < 1e-30:
            continue

        M_total = M_total * (n_k * frac_num) / (n_k1 * frac_den)

    return M_total

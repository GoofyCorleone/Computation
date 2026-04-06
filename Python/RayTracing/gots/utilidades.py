"""Funciones utilitarias: normalización de vectores y resolución de cuárticas."""

import numpy as np


def normalizar(v):
    """Retorna el vector unitario de v."""
    norma = np.linalg.norm(v)
    if norma < 1e-15:
        raise ValueError("No se puede normalizar un vector nulo.")
    return v / norma


def resolver_cuartica(Q4, Q3, Q2, Q1, Q0, tol_imag=1e-10):
    """Resuelve Q4*t^4 + Q3*t^3 + Q2*t^2 + Q1*t + Q0 = 0.

    Retorna todas las raíces reales ordenadas por valor absoluto (menor primero).
    Si Q4 ≈ 0 reduce a cúbica/cuadrática automáticamente.
    Si algún coeficiente es NaN o Inf, retorna array vacío.
    """
    # Construir vector de coeficientes eliminando ceros líderes
    coefs = [Q4, Q3, Q2, Q1, Q0]

    # Protección contra coeficientes inválidos (NaN/Inf por parámetros degenerados)
    if any(not np.isfinite(c) for c in coefs):
        return np.array([])

    while len(coefs) > 1 and abs(coefs[0]) < 1e-15:
        coefs.pop(0)

    if len(coefs) <= 1:
        return np.array([])

    raices = np.roots(coefs)

    # Filtrar raíces reales (positivas y negativas)
    reales = []
    for r in raices:
        if abs(r.imag) < tol_imag:
            reales.append(r.real)

    # Ordenar por valor absoluto para que la raíz más cercana al vértice sea primero
    reales.sort(key=abs)
    return np.array(reales)

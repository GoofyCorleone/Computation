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

    Retorna las raíces reales positivas ordenadas de menor a mayor.
    Si Q4 ≈ 0 reduce a cúbica/cuadrática automáticamente.
    """
    # Construir vector de coeficientes eliminando ceros líderes
    coefs = [Q4, Q3, Q2, Q1, Q0]
    while len(coefs) > 1 and abs(coefs[0]) < 1e-15:
        coefs.pop(0)

    if len(coefs) <= 1:
        return np.array([])

    raices = np.roots(coefs)

    # Filtrar raíces reales positivas
    reales = []
    for r in raices:
        if abs(r.imag) < tol_imag and r.real > 1e-12:
            reales.append(r.real)

    return np.sort(np.array(reales))

"""Ley de Snell-Descartes vectorial para superficies cartesianas.

Implementa la Eq. 68 de la tesis de Silva-Lora (2024).
"""

import numpy as np


def refraccion_snell(u_in, normal, n_k, n_k1):
    """Calcula el vector unitario refractado usando la ley de Snell vectorial.

    Eq. 68:
        û' = (n_k/n_k1)·û - [(n_k/n_k1)·(N̂·û) + √(1-(n_k/n_k1)²·(1-(N̂·û)²))]·N̂

    Args:
        u_in: vector unitario incidente (3,)
        normal: vector normal unitario en el punto de incidencia (3,)
        n_k: índice de refracción del medio incidente
        n_k1: índice de refracción del medio refractado

    Returns:
        Vector unitario refractado, o None si hay reflexión total interna (TIR).
    """
    ratio = n_k / n_k1
    cos_i = np.dot(normal, u_in)

    # Argumento del radical
    discriminante = 1.0 - ratio**2 * (1.0 - cos_i**2)

    if discriminante < 0:
        return None  # Reflexión total interna

    cos_t = np.sqrt(discriminante)

    u_out = ratio * u_in - (ratio * cos_i + cos_t) * normal
    # Normalizar por seguridad numérica
    norma = np.linalg.norm(u_out)
    if norma < 1e-15:
        return None
    return u_out / norma

"""
fisica.py — Funciones puras de óptica de polarización (sin dependencias de Manim).
"""

import numpy as np


def jones_desde_angulos(alpha: float, chi: float) -> np.ndarray:
    """
    Vector Jones normalizado para polarización elíptica.

    Parámetros
    ----------
    alpha : orientación de la elipse (rad)  ∈ [0, π]
    chi   : elipticidad (rad)               ∈ [-π/4, π/4]

    Punto resultante en la esfera de Poincaré:
        S₁ = cos(2χ)·cos(2α)
        S₂ = cos(2χ)·sin(2α)
        S₃ = sin(2χ)
    """
    ex = np.cos(alpha) * np.cos(chi) - 1j * np.sin(alpha) * np.sin(chi)
    ey = np.sin(alpha) * np.cos(chi) + 1j * np.cos(alpha) * np.sin(chi)
    return np.array([ex, ey], dtype=complex)


def jones_a_stokes(j: np.ndarray) -> np.ndarray:
    """Vector Jones normalizado → (S₁, S₂, S₃) en la esfera unitaria."""
    ex, ey = j[0], j[1]
    S0 = abs(ex) ** 2 + abs(ey) ** 2
    if S0 < 1e-14:
        return np.zeros(3)
    return np.array([
        (abs(ex) ** 2 - abs(ey) ** 2) / S0,
        2.0 * np.real(ex * np.conj(ey)) / S0,
        2.0 * np.imag(ey * np.conj(ex)) / S0,
    ])


def _norm(j: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(j)
    return j / n if n > 1e-12 else j


def jones_retardador(Gamma: float, theta: float = 0.0) -> np.ndarray:
    """
    Matriz Jones de un retardador birrefringente.

    Parámetros
    ----------
    Gamma : retardancia (rad).  Rota el vector de Stokes alrededor del eje
            (cos 2θ, sin 2θ, 0) un ángulo Gamma.
    theta : ángulo del eje rápido respecto a x (rad).
    """
    c, s = np.cos(theta), np.sin(theta)
    ph   = np.exp( 1j * Gamma / 2)
    pm   = np.exp(-1j * Gamma / 2)
    return np.array([
        [c**2 * ph + s**2 * pm,  c * s * (ph - pm)    ],
        [c * s  * (ph - pm),     s**2 * ph + c**2 * pm],
    ], dtype=complex)


def jones_pol_parcial(p1: float, p2: float, theta: float = 0.0) -> np.ndarray:
    """
    Matriz Jones de un polarizador parcial (diatenuador).

    Parámetros
    ----------
    p1    : transmisión de amplitud del eje rápido ∈ [0, 1].
    p2    : transmisión de amplitud del eje lento  ∈ [0, 1], p2 ≤ p1.
    theta : ángulo del eje rápido respecto a x (rad).
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [p1 * c**2 + p2 * s**2,  (p1 - p2) * c * s      ],
        [(p1 - p2) * c * s,       p1 * s**2 + p2 * c**2 ],
    ], dtype=complex)

"""
esfera_poincare — Librería para animar transformaciones de polarización
en la esfera de Poincaré con Manim Community.

Uso básico
----------
from esfera_poincare import EsferaPoincare, Parametros
from esfera_poincare.escenas import RetardadorFijo, PolarizadorParcial
from esfera_poincare.escenas import RetardadorGiratorio, PolarizadorGiratorio
"""

from .fisica import (
    jones_desde_angulos,
    jones_a_stokes,
    jones_retardador,
    jones_pol_parcial,
)
from .base import EsferaPoincare, Parametros

__all__ = [
    "jones_desde_angulos",
    "jones_a_stokes",
    "jones_retardador",
    "jones_pol_parcial",
    "EsferaPoincare",
    "Parametros",
]

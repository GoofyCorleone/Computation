"""GOTS: Trazado de rayos en superficies cartesianas (ovoides de Descartes).

Basado en la tesis "Aberraciones primarias a partir del estigmatismo riguroso"
de Alberto Silva Lora (UIS, 2024).
"""

from .parametros_gots import ParametrosGOTS, calcular_gots
from .superficie_cartesiana import SuperficieCartesiana
from .rayo import Rayo, Interseccion, intersectar
from .snell import refraccion_snell
from .sistema_optico import SistemaOptico, ResultadoTrazado
from .visualizacion import graficar_seccion_transversal, graficar_3d
from .exportar_stl import exportar_superficie_stl, exportar_sistema_stl
from .utilidades import normalizar, resolver_cuartica

__all__ = [
    'ParametrosGOTS', 'calcular_gots',
    'SuperficieCartesiana',
    'Rayo', 'Interseccion', 'intersectar',
    'refraccion_snell',
    'SistemaOptico', 'ResultadoTrazado',
    'graficar_seccion_transversal', 'graficar_3d',
    'exportar_superficie_stl', 'exportar_sistema_stl',
    'normalizar', 'resolver_cuartica',
]

"""Exportación de superficies cartesianas a formato STL binario.

No requiere dependencias externas. Genera archivos STL binarios estándar:
80 bytes header + uint32 num_triangles + 50 bytes por triángulo.
"""

import struct
import numpy as np


def exportar_superficie_stl(superficie, archivo, rho_max_frac=0.95,
                             num_rho=80, num_phi=60):
    """Exporta una superficie cartesiana a un archivo STL binario.

    Args:
        superficie: SuperficieCartesiana
        archivo: ruta del archivo de salida
        rho_max_frac: fracción de ρ_max a usar
        num_rho: resolución radial
        num_phi: resolución azimutal
    """
    X, Y, Z = superficie.generar_malla_3d(num_rho=num_rho, num_phi=num_phi,
                                            rho_max_frac=rho_max_frac)
    triangulos = _malla_a_triangulos(X, Y, Z)
    _escribir_stl_binario(archivo, triangulos)


def exportar_sistema_stl(sistema, archivo, rho_max_frac=0.95,
                          num_rho=80, num_phi=60):
    """Exporta todas las superficies de un sistema a un solo archivo STL.

    Args:
        sistema: SistemaOptico
        archivo: ruta del archivo de salida
    """
    todos_triangulos = []
    for sup in sistema.superficies:
        X, Y, Z = sup.generar_malla_3d(num_rho=num_rho, num_phi=num_phi,
                                         rho_max_frac=rho_max_frac)
        todos_triangulos.extend(_malla_a_triangulos(X, Y, Z))
    _escribir_stl_binario(archivo, todos_triangulos)


def _malla_a_triangulos(X, Y, Z):
    """Convierte una malla (X, Y, Z) en una lista de triángulos con normales.

    Cada triángulo es (normal, v1, v2, v3) donde cada vértice es (x, y, z).
    """
    n_phi, n_rho = X.shape
    triangulos = []

    for i in range(n_phi):
        i_next = (i + 1) % n_phi
        for j in range(n_rho - 1):
            # Vértices del quad
            v00 = np.array([X[i, j], Y[i, j], Z[i, j]])
            v10 = np.array([X[i_next, j], Y[i_next, j], Z[i_next, j]])
            v01 = np.array([X[i, j + 1], Y[i, j + 1], Z[i, j + 1]])
            v11 = np.array([X[i_next, j + 1], Y[i_next, j + 1], Z[i_next, j + 1]])

            # Triángulo 1: v00, v10, v11
            normal1 = _normal_triangulo(v00, v10, v11)
            triangulos.append((normal1, v00, v10, v11))

            # Triángulo 2: v00, v11, v01
            normal2 = _normal_triangulo(v00, v11, v01)
            triangulos.append((normal2, v00, v11, v01))

    return triangulos


def _normal_triangulo(v1, v2, v3):
    """Calcula la normal de un triángulo definido por tres vértices."""
    edge1 = v2 - v1
    edge2 = v3 - v1
    n = np.cross(edge1, edge2)
    norma = np.linalg.norm(n)
    if norma < 1e-15:
        return np.array([0.0, 0.0, 1.0])
    return n / norma


def _escribir_stl_binario(archivo, triangulos):
    """Escribe una lista de triángulos en formato STL binario."""
    with open(archivo, 'wb') as f:
        # Header: 80 bytes
        header = b'STL GOTS - Superficies Cartesianas'
        header = header.ljust(80, b'\0')
        f.write(header)

        # Número de triángulos: uint32
        f.write(struct.pack('<I', len(triangulos)))

        # Triángulos: 50 bytes cada uno
        for normal, v1, v2, v3 in triangulos:
            # Normal (3 floats)
            f.write(struct.pack('<3f', *normal))
            # Vértice 1 (3 floats)
            f.write(struct.pack('<3f', *v1))
            # Vértice 2 (3 floats)
            f.write(struct.pack('<3f', *v2))
            # Vértice 3 (3 floats)
            f.write(struct.pack('<3f', *v3))
            # Attribute byte count: uint16
            f.write(struct.pack('<H', 0))

"""Exportación de superficies cartesianas a formato STL binario.

No requiere dependencias externas. Genera archivos STL binarios estándar:
80 bytes header + uint32 num_triangles + 50 bytes por triángulo.

Las superficies se recortan a la apertura física de la lente (intersección
entre superficies consecutivas).
"""

import struct
import numpy as np


def exportar_superficie_stl(superficie, archivo, r_max=None,
                             num_rho=80, num_phi=60):
    """Exporta una superficie cartesiana a un archivo STL binario.

    Args:
        superficie: SuperficieCartesiana
        archivo: ruta del archivo de salida
        r_max: radio transversal máximo (si None, usa rho_max * 0.95)
        num_rho: resolución radial
        num_phi: resolución azimutal
    """
    X, Y, Z = _malla_recortada(superficie, r_max, num_rho, num_phi)
    triangulos = _malla_a_triangulos(X, Y, Z)
    _escribir_stl_binario(archivo, triangulos)


def exportar_sistema_stl(sistema, archivo, num_rho=80, num_phi=60):
    """Exporta todas las superficies de un sistema a un solo archivo STL.

    Recorta cada superficie a la apertura de la lente (intersección entre
    superficies consecutivas).

    Args:
        sistema: SistemaOptico
        archivo: ruta del archivo de salida
        num_rho: resolución radial
        num_phi: resolución azimutal
    """
    aperturas = sistema.encontrar_apertura()

    todos_triangulos = []
    for k, sup in enumerate(sistema.superficies):
        # Determinar r_max para esta superficie
        if k < len(aperturas):
            r_max = aperturas[k]
        elif k > 0 and k - 1 < len(aperturas):
            r_max = aperturas[k - 1]
        else:
            r_max = None

        X, Y, Z = _malla_recortada(sup, r_max, num_rho, num_phi)
        todos_triangulos.extend(_malla_a_triangulos(X, Y, Z))

    _escribir_stl_binario(archivo, todos_triangulos)


def _malla_recortada(superficie, r_max, num_rho, num_phi):
    """Genera malla 3D recortada a r_max."""
    if r_max is not None:
        # Encontrar ρ correspondiente a r_max
        rho_lim = superficie.rho_max * 0.99 if np.isfinite(superficie.rho_max) else 200.0
        rho_test = np.linspace(0, rho_lim, 5000)
        r_test = superficie.r_de_rho(rho_test)
        mask = r_test <= r_max * 1.001
        if np.any(mask):
            rho_clip = rho_test[mask][-1]
        else:
            rho_clip = rho_lim * 0.5
    else:
        rho_clip = superficie.rho_max * 0.95 if np.isfinite(superficie.rho_max) else 50.0

    rho = np.linspace(0, rho_clip, num_rho)
    phi = np.linspace(0, 2 * np.pi, num_phi, endpoint=False)

    z_perfil = superficie.z_de_rho(rho)
    r_perfil = superficie.r_de_rho(rho)

    RHO, PHI = np.meshgrid(r_perfil, phi)
    Z_mesh = np.tile(z_perfil, (num_phi, 1))
    X = RHO * np.cos(PHI)
    Y = RHO * np.sin(PHI)

    return X, Y, Z_mesh


def _malla_a_triangulos(X, Y, Z):
    """Convierte una malla (X, Y, Z) en una lista de triángulos con normales.

    Cada triángulo es (normal, v1, v2, v3) donde cada vértice es (x, y, z).
    """
    n_phi, n_rho = X.shape
    triangulos = []

    for i in range(n_phi):
        i_next = (i + 1) % n_phi
        for j in range(n_rho - 1):
            v00 = np.array([X[i, j], Y[i, j], Z[i, j]])
            v10 = np.array([X[i_next, j], Y[i_next, j], Z[i_next, j]])
            v01 = np.array([X[i, j + 1], Y[i, j + 1], Z[i, j + 1]])
            v11 = np.array([X[i_next, j + 1], Y[i_next, j + 1], Z[i_next, j + 1]])

            normal1 = _normal_triangulo(v00, v10, v11)
            triangulos.append((normal1, v00, v10, v11))

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
        header = b'STL GOTS - Superficies Cartesianas'
        header = header.ljust(80, b'\0')
        f.write(header)

        f.write(struct.pack('<I', len(triangulos)))

        for normal, v1, v2, v3 in triangulos:
            f.write(struct.pack('<3f', *normal))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<3f', *v3))
            f.write(struct.pack('<H', 0))

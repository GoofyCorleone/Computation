"""Exportación de superficies cartesianas a formato STL binario sólido.

No requiere dependencias externas. Genera archivos STL binarios estándar:
80 bytes header + uint32 num_triangles + 50 bytes por triángulo.

Las superficies se recortan a la apertura física de la lente (intersección
entre superficies consecutivas). Se agrega un anillo de borde para producir
un sólido cerrado (watertight) apto para impresión 3D.
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
    """Exporta el sistema óptico como sólido cerrado a un archivo STL.

    Recorta cada superficie a la apertura de la lente y añade un anillo
    de borde que conecta las dos superficies, produciendo un sólido
    watertight apto para impresión 3D.

    Args:
        sistema: SistemaOptico
        archivo: ruta del archivo de salida
        num_rho: resolución radial
        num_phi: resolución azimutal
    """
    aperturas = sistema.encontrar_apertura()

    todos_triangulos = []
    mallas = []

    for k, sup in enumerate(sistema.superficies):
        if k < len(aperturas):
            r_max = aperturas[k]
        elif k > 0 and k - 1 < len(aperturas):
            r_max = aperturas[k - 1]
        else:
            r_max = None

        X, Y, Z = _malla_recortada(sup, r_max, num_rho, num_phi)
        todos_triangulos.extend(_malla_a_triangulos(X, Y, Z))
        mallas.append((X, Y, Z))

    # Anillo de borde entre superficies consecutivas → sólido cerrado
    for k in range(len(mallas) - 1):
        X0, Y0, Z0 = mallas[k]
        X1, Y1, Z1 = mallas[k + 1]
        n_phi_rim = min(X0.shape[0], X1.shape[0])
        todos_triangulos.extend(_triangulos_rim(X0, Y0, Z0, X1, Y1, Z1, n_phi_rim))

    _escribir_stl_binario(archivo, todos_triangulos)


# ---------------------------------------------------------------------------
# Funciones internas
# ---------------------------------------------------------------------------

def _rho_clip_ascendente(superficie, r_max):
    """Encuentra ρ máximo en la rama ascendente del óvalo ≤ r_max."""
    rho_lim = superficie.rho_max * 0.99 if np.isfinite(superficie.rho_max) else 200.0
    rho_test = np.linspace(0, rho_lim, 5000)
    r_all = superficie.r_de_rho(rho_test)

    # Fin de la rama ascendente (donde r empieza a decrecer)
    dr = np.diff(r_all)
    idx_desc = np.where(dr < -1e-12)[0]
    idx_max_r = idx_desc[0] if len(idx_desc) > 0 else len(rho_test) - 1

    # Dentro de la rama ascendente, primer ρ donde r excede r_max
    if r_max is not None:
        idx_excede = np.where(r_all[:idx_max_r + 1] > r_max)[0]
        idx_clip = idx_excede[0] if len(idx_excede) > 0 else idx_max_r
    else:
        idx_clip = idx_max_r

    return rho_test[idx_clip] if idx_clip > 0 else rho_test[1]


def _malla_recortada(superficie, r_max, num_rho, num_phi):
    """Genera malla 3D recortada a r_max (solo rama ascendente del óvalo)."""
    rho_clip = _rho_clip_ascendente(superficie, r_max)

    rho = np.linspace(0, rho_clip, num_rho)
    phi = np.linspace(0, 2 * np.pi, num_phi, endpoint=False)

    z_perfil = superficie.z_de_rho(rho)
    r_perfil = superficie.r_de_rho(rho)

    RHO, PHI = np.meshgrid(r_perfil, phi)
    Z_mesh = np.tile(z_perfil, (num_phi, 1))
    X = RHO * np.cos(PHI)
    Y = RHO * np.sin(PHI)

    return X, Y, Z_mesh


def _triangulos_rim(X0, Y0, Z0, X1, Y1, Z1, n_phi):
    """Triángulos del anillo de borde que cierra el sólido entre sup0 y sup1.

    Conecta la última fila de la malla de sup0 con la última fila de sup1,
    produciendo un anillo de triángulos que cierra la malla lateral.
    """
    triangulos = []
    for i in range(n_phi):
        i_next = (i + 1) % n_phi
        v00 = np.array([X0[i, -1],      Y0[i, -1],      Z0[i, -1]])
        v01 = np.array([X0[i_next, -1], Y0[i_next, -1], Z0[i_next, -1]])
        v10 = np.array([X1[i, -1],      Y1[i, -1],      Z1[i, -1]])
        v11 = np.array([X1[i_next, -1], Y1[i_next, -1], Z1[i_next, -1]])

        n1 = _normal_triangulo(v00, v10, v11)
        triangulos.append((n1, v00, v10, v11))
        n2 = _normal_triangulo(v00, v11, v01)
        triangulos.append((n2, v00, v11, v01))

    return triangulos


def _malla_a_triangulos(X, Y, Z):
    """Convierte una malla (X, Y, Z) en una lista de triángulos con normales."""
    n_phi, n_rho = X.shape
    triangulos = []

    for i in range(n_phi):
        i_next = (i + 1) % n_phi
        for j in range(n_rho - 1):
            v00 = np.array([X[i, j],      Y[i, j],      Z[i, j]])
            v10 = np.array([X[i_next, j], Y[i_next, j], Z[i_next, j]])
            v01 = np.array([X[i, j + 1],  Y[i, j + 1],  Z[i, j + 1]])
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

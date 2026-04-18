"""Exportación de superficies cartesianas a formato STL binario.

Genera archivos STL binarios estándar sin dependencias externas
(80 bytes header + uint32 num_triangles + 50 bytes por triángulo).

Para un sistema óptico (lente), se produce un SÓLIDO CERRADO (watertight)
apto para impresión 3D. Concretamente:

    Superficie frontal  +  anillo lateral  +  superficie posterior
          (tapa 0)          (cilindro)          (tapa 1)

La topología está construida con:
  - Un único vértice en el ápice (ρ=0) de cada superficie (sin fan degenerado).
  - n_phi × (num_rho-1) quads por cada casquete + n_phi triángulos en el ápice.
  - n_phi quads laterales que unen los bordes recortados.
  - Orientación consistente: las normales apuntan HACIA AFUERA del sólido
    (winding CCW visto desde el exterior).

Para una sola superficie (sin lente) se exporta solo el casquete, con las
normales igualmente orientadas hacia afuera según el lado del ápice.
"""

import struct
import numpy as np


def exportar_superficie_stl(superficie, archivo, r_max=None,
                             num_rho=80, num_phi=60):
    """Exporta una única superficie cartesiana (casquete abierto) a STL.

    La malla resultante NO es un sólido cerrado (es una sola superficie).
    Para un sólido cerrado apto para impresión 3D, usar `exportar_sistema_stl`.
    """
    rho_clip = _rho_clip_ascendente(superficie, r_max)
    X, Y, Z = _malla_superficie(superficie, rho_clip, num_rho, num_phi)
    triangulos = _triangulos_casquete(X, Y, Z, outward_positive_z=False)
    _escribir_stl_binario(archivo, triangulos)


def exportar_sistema_stl(sistema, archivo, num_rho=80, num_phi=60,
                          r_max=None, espesor_minimo=0.05):
    """Exporta el sistema óptico como SÓLIDO CERRADO watertight a STL.

    Construye la lente uniendo la superficie 0 (frontal), un anillo lateral
    cilíndrico y la superficie 1 (posterior). Las normales se orientan hacia
    afuera del sólido, y la malla es manifold y apta para impresión 3D.

    Args:
        sistema: SistemaOptico con al menos dos superficies.
        archivo: ruta del archivo STL de salida.
        num_rho: resolución radial de cada casquete.
        num_phi: resolución azimutal.
        r_max: radio físico del borde de la lente. Si None, se usa la apertura
            natural del sistema (intersección de los óvalos). Para lentes
            bicóncavas divergentes, `encontrar_apertura` no halla cruce y
            devuelve un radio muy grande; en ese caso conviene acotar r_max.
        espesor_minimo: separación mínima (en z) entre los bordes de las dos
            superficies. Si el espesor en el borde cae por debajo de este valor
            se fuerza r_max menor para que quede un borde con espesor suficiente
            (importante para lentes biconvexas donde las superficies casi se
            tocan en la apertura natural).
    """
    if len(sistema.superficies) < 2:
        raise ValueError("Se requieren al menos dos superficies para un sólido.")

    sup0 = sistema.superficies[0]
    sup1 = sistema.superficies[1]

    # Resolver r_max físico de la lente
    if r_max is None:
        aperturas = sistema.encontrar_apertura()
        if aperturas:
            r_max = aperturas[0]
        else:
            r_max = min(sup0.rho_max * 0.95 if np.isfinite(sup0.rho_max) else 20.0,
                        sup1.rho_max * 0.95 if np.isfinite(sup1.rho_max) else 20.0,
                        20.0)

    # Garantizar espesor mínimo en el borde (evita ápice agudo que rompe manifold)
    r_max = _acotar_por_espesor_minimo(sup0, sup1, r_max, espesor_minimo)

    # Clip de ρ sobre la rama ascendente de cada superficie
    rho_clip_0 = _rho_clip_para_r(sup0, r_max)
    rho_clip_1 = _rho_clip_para_r(sup1, r_max)

    # Mallas (num_phi, num_rho) con ápice en j=0 (todos los ápices coinciden)
    X0, Y0, Z0 = _malla_superficie(sup0, rho_clip_0, num_rho, num_phi)
    X1, Y1, Z1 = _malla_superficie(sup1, rho_clip_1, num_rho, num_phi)

    # Orientación: la normal exterior de la superficie frontal tiene componente -z
    # (apunta fuera del sólido hacia el espacio objeto) y la de la posterior +z.
    # Esto depende de la posición relativa de los ápices:
    apex_z_0 = float(Z0[0, 0])
    apex_z_1 = float(Z1[0, 0])
    frontal_es_sup0 = apex_z_0 <= apex_z_1  # usual: sup0 por detrás en z

    if frontal_es_sup0:
        triangulos = _triangulos_casquete(X0, Y0, Z0, outward_positive_z=False)
        triangulos += _triangulos_casquete(X1, Y1, Z1, outward_positive_z=True)
        triangulos += _triangulos_anillo(X0, Y0, Z0, X1, Y1, Z1)
    else:
        triangulos = _triangulos_casquete(X1, Y1, Z1, outward_positive_z=False)
        triangulos += _triangulos_casquete(X0, Y0, Z0, outward_positive_z=True)
        triangulos += _triangulos_anillo(X1, Y1, Z1, X0, Y0, Z0)

    # Si el volumen signado sale negativo, la forma concreta de la lente hizo
    # que la "frontal" quede del lado equivocado: invertimos winding y normales.
    if _volumen_signado(triangulos) < 0:
        triangulos = [(-n, v1, v3, v2) for (n, v1, v2, v3) in triangulos]

    _escribir_stl_binario(archivo, triangulos)


# ---------------------------------------------------------------------------
# Construcción de la malla
# ---------------------------------------------------------------------------

def _malla_superficie(sup, rho_clip, num_rho, num_phi):
    """Genera una malla 3D (num_phi, num_rho) de la superficie hasta ρ=rho_clip.

    El ápice (j=0) se materializa como un único punto (0, 0, ζ) replicado en
    todas las filas de phi: las triangulaciones posteriores lo tratan como
    un vértice único para evitar fans degenerados.
    """
    rho = np.linspace(0.0, rho_clip, num_rho)
    phi = np.linspace(0.0, 2.0 * np.pi, num_phi, endpoint=False)

    z_perfil = sup.z_de_rho(rho)
    r_perfil = sup.r_de_rho(rho)

    R, PHI = np.meshgrid(r_perfil, phi)
    Z = np.tile(z_perfil, (num_phi, 1))
    X = R * np.cos(PHI)
    Y = R * np.sin(PHI)
    return X, Y, Z


def _triangulos_casquete(X, Y, Z, outward_positive_z):
    """Triangula un casquete con ápice en j=0 y borde en j=-1.

    Si outward_positive_z=True, las normales apuntan hacia +z (lado imagen);
    si False, apuntan hacia -z (lado objeto). El winding se ajusta para que
    el vector normal del triángulo (según regla de la mano derecha) tenga
    la componente z deseada.
    """
    n_phi, n_rho = X.shape
    triangulos = []

    apex = np.array([0.0, 0.0, float(Z[0, 0])])

    # Tapa de ápice: n_phi triángulos (apex, anillo j=1)
    for i in range(n_phi):
        i_next = (i + 1) % n_phi
        v_anillo_i  = np.array([X[i, 1],      Y[i, 1],      Z[i, 1]])
        v_anillo_i1 = np.array([X[i_next, 1], Y[i_next, 1], Z[i_next, 1]])
        if outward_positive_z:
            # Normal ≈ +z: desde +z miramos CCW → (apex, v_i, v_i1)
            triangulos.append(_triangulo_con_normal(apex, v_anillo_i, v_anillo_i1))
        else:
            # Normal ≈ -z: winding opuesto
            triangulos.append(_triangulo_con_normal(apex, v_anillo_i1, v_anillo_i))

    # Cuerpo: quads entre anillos consecutivos (j, j+1)
    for j in range(1, n_rho - 1):
        for i in range(n_phi):
            i_next = (i + 1) % n_phi
            v00 = np.array([X[i, j],          Y[i, j],          Z[i, j]])
            v10 = np.array([X[i_next, j],     Y[i_next, j],     Z[i_next, j]])
            v01 = np.array([X[i, j + 1],      Y[i, j + 1],      Z[i, j + 1]])
            v11 = np.array([X[i_next, j + 1], Y[i_next, j + 1], Z[i_next, j + 1]])
            if outward_positive_z:
                triangulos.append(_triangulo_con_normal(v00, v10, v11))
                triangulos.append(_triangulo_con_normal(v00, v11, v01))
            else:
                triangulos.append(_triangulo_con_normal(v00, v11, v10))
                triangulos.append(_triangulo_con_normal(v00, v01, v11))

    return triangulos


def _triangulos_anillo(X0, Y0, Z0, X1, Y1, Z1):
    """Anillo lateral cerrando el sólido entre los bordes exteriores.

    Se asume que X0,Y0,Z0 es la superficie FRONTAL (normal saliente hacia -z)
    y X1,Y1,Z1 es la POSTERIOR. El borde exterior está en j=-1 y ambos bordes
    están al mismo radio r_max por construcción (se fuerza el clip en r).
    Las normales del cilindro apuntan radialmente hacia afuera.
    """
    n_phi = X0.shape[0]
    triangulos = []

    for i in range(n_phi):
        i_next = (i + 1) % n_phi
        v_front_i  = np.array([X0[i, -1],      Y0[i, -1],      Z0[i, -1]])
        v_front_i1 = np.array([X0[i_next, -1], Y0[i_next, -1], Z0[i_next, -1]])
        v_back_i   = np.array([X1[i, -1],      Y1[i, -1],      Z1[i, -1]])
        v_back_i1  = np.array([X1[i_next, -1], Y1[i_next, -1], Z1[i_next, -1]])

        # Orden CCW visto desde afuera (radialmente). El borde frontal está a
        # menor z que el posterior, así que para una normal saliente radial
        # recorremos: v_front_i → v_front_i1 → v_back_i1 → v_back_i.
        triangulos.append(_triangulo_con_normal(v_front_i,  v_front_i1, v_back_i1))
        triangulos.append(_triangulo_con_normal(v_front_i,  v_back_i1,  v_back_i))

    return triangulos


# ---------------------------------------------------------------------------
# Helpers de geometría
# ---------------------------------------------------------------------------

def _triangulo_con_normal(v1, v2, v3):
    """Calcula normal (regla mano derecha sobre el orden dado) y devuelve tupla."""
    edge1 = v2 - v1
    edge2 = v3 - v1
    n = np.cross(edge1, edge2)
    norma = np.linalg.norm(n)
    if norma < 1e-15:
        n = np.array([0.0, 0.0, 1.0])
    else:
        n = n / norma
    return (n, v1, v2, v3)


def _rho_clip_ascendente(superficie, r_max):
    """Máximo ρ dentro de la rama ascendente y, si aplica, con r(ρ) ≤ r_max."""
    rho_lim = superficie.rho_max * 0.999 if np.isfinite(superficie.rho_max) else 500.0
    rho_test = np.linspace(0.0, rho_lim, 8000)
    r_all = superficie.r_de_rho(rho_test)

    # Fin de la rama ascendente (donde r empieza a decrecer)
    dr = np.diff(r_all)
    idx_desc = np.where(dr < -1e-12)[0]
    idx_asc_end = idx_desc[0] if len(idx_desc) > 0 else len(rho_test) - 1

    if r_max is None:
        return rho_test[idx_asc_end]

    idx_excede = np.where(r_all[:idx_asc_end + 1] >= r_max)[0]
    if len(idx_excede) > 0:
        return rho_test[idx_excede[0]]
    return rho_test[idx_asc_end]


def _rho_clip_para_r(superficie, r_objetivo):
    """Valor de ρ (rama ascendente) tal que r(ρ) ≈ r_objetivo."""
    return _rho_clip_ascendente(superficie, r_objetivo)


def _acotar_por_espesor_minimo(sup0, sup1, r_max, espesor_minimo):
    """Reduce r_max si el espesor local |z1(r)-z0(r)| es < espesor_minimo.

    Evita que la lente termine en un filo agudo donde las dos superficies se
    encuentran (lo cual rompería la manifoldness y el slicing para impresión).
    """
    if espesor_minimo <= 0:
        return r_max

    r_test = np.linspace(r_max * 0.01, r_max, 500)
    rho_grid_0 = np.linspace(0.0,
                              sup0.rho_max * 0.999 if np.isfinite(sup0.rho_max) else 500.0,
                              8000)
    rho_grid_1 = np.linspace(0.0,
                              sup1.rho_max * 0.999 if np.isfinite(sup1.rho_max) else 500.0,
                              8000)

    r0 = sup0.r_de_rho(rho_grid_0)
    z0 = sup0.z_de_rho(rho_grid_0)
    r1 = sup1.r_de_rho(rho_grid_1)
    z1 = sup1.z_de_rho(rho_grid_1)

    # Solo rama ascendente
    idx0 = np.where(np.diff(r0) < -1e-12)[0]
    if len(idx0) > 0:
        r0 = r0[:idx0[0] + 1]; z0 = z0[:idx0[0] + 1]
    idx1 = np.where(np.diff(r1) < -1e-12)[0]
    if len(idx1) > 0:
        r1 = r1[:idx1[0] + 1]; z1 = z1[:idx1[0] + 1]

    z0_i = np.interp(r_test, r0, z0)
    z1_i = np.interp(r_test, r1, z1)
    espesor = np.abs(z1_i - z0_i)

    # Encontrar el primer r donde el espesor cae por debajo del mínimo
    mask = espesor < espesor_minimo
    if not np.any(mask):
        return r_max
    primer = np.argmax(mask)
    # Retroceder hasta encontrar r con espesor ≥ mínimo
    idx_ok = np.where(~mask)[0]
    if len(idx_ok) == 0:
        return r_max * 0.5
    idx_final = idx_ok[idx_ok < primer]
    if len(idx_final) == 0:
        return r_max * 0.5
    return float(r_test[idx_final[-1]])


def _volumen_signado(triangulos):
    """Volumen signado del sólido cerrado (positivo si normales salen hacia afuera)."""
    vol = 0.0
    for _, v1, v2, v3 in triangulos:
        vol += (v1[0] * (v2[1] * v3[2] - v3[1] * v2[2])
                + v2[0] * (v3[1] * v1[2] - v1[1] * v3[2])
                + v3[0] * (v1[1] * v2[2] - v2[1] * v1[2])) / 6.0
    return float(vol)


def _escribir_stl_binario(archivo, triangulos):
    """Escribe una lista de triángulos en formato STL binario."""
    with open(archivo, 'wb') as f:
        header = b'STL GOTS - Solido Watertight'
        header = header.ljust(80, b'\0')
        f.write(header)

        f.write(struct.pack('<I', len(triangulos)))

        for normal, v1, v2, v3 in triangulos:
            f.write(struct.pack('<3f', *normal))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<3f', *v3))
            f.write(struct.pack('<H', 0))

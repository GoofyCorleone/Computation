"""Visualización 2D y 3D de sistemas ópticos con superficies cartesianas."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def graficar_seccion_transversal(sistema, resultados, ax=None, titulo=None,
                                  colores_rayos=None, mostrar=True):
    """Gráfica 2D del plano meridional (r-z): perfiles de superficie + rayos.

    Args:
        sistema: SistemaOptico
        resultados: lista de ResultadoTrazado
        ax: eje matplotlib existente (opcional)
        titulo: título de la gráfica
        colores_rayos: lista de colores para los rayos
        mostrar: si True, llama a plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Dibujar superficies (perfil meridional)
    for k, sup in enumerate(sistema.superficies):
        z_perfil, r_perfil = sup.generar_perfil_meridional()
        # Perfil superior e inferior
        ax.plot(z_perfil, r_perfil, 'k-', linewidth=1.5)
        ax.plot(z_perfil, -r_perfil, 'k-', linewidth=1.5)

        # Cerrar la lente si hay dos superficies consecutivas
        if k > 0:
            sup_prev = sistema.superficies[k - 1]
            z0, r0 = sup_prev.generar_perfil_meridional()
            z1, r1 = sup.generar_perfil_meridional()
            # Borde superior
            r_max_0 = r0[-1]
            r_max_1 = r1[-1]
            z_max_0 = z0[-1]
            z_max_1 = z1[-1]
            ax.plot([z_max_0, z_max_1], [r_max_0, r_max_1], 'k-', linewidth=1.0)
            ax.plot([z_max_0, z_max_1], [-r_max_0, -r_max_1], 'k-', linewidth=1.0)

    # Dibujar rayos
    color_default = 'tab:blue'
    for i, res in enumerate(resultados):
        if colores_rayos and i < len(colores_rayos):
            color = colores_rayos[i]
        else:
            color = color_default

        for j in range(len(res.puntos) - 1):
            p1 = res.puntos[j]
            p2 = res.puntos[j + 1]
            ax.plot([p1[2], p2[2]], [p1[1], p2[1]], color=color,
                    linewidth=0.6, alpha=0.7)

        # Extensión del último segmento
        if res.rayo_completo and len(res.puntos) > 1:
            p_final = res.puntos[-1]
            d_final = res.direcciones[-1]
            # Extender 50 unidades
            extension = 50.0
            p_ext = p_final + extension * d_final
            ax.plot([p_final[2], p_ext[2]], [p_final[1], p_ext[1]],
                    color=color, linewidth=0.6, alpha=0.4, linestyle='--')

    # Eje óptico
    z_min = min(p[2] for res in resultados for p in res.puntos) - 20
    z_max = max(p[2] for res in resultados for p in res.puntos) + 50
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-.')
    ax.set_xlim(z_min, z_max)

    ax.set_xlabel('z (eje óptico)')
    ax.set_ylabel('r (transversal)')
    ax.set_aspect('equal')
    if titulo:
        ax.set_title(titulo)

    if mostrar:
        plt.tight_layout()
        plt.show()

    return ax


def graficar_3d(sistema, resultados=None, ax=None, titulo=None, mostrar=True):
    """Gráfica 3D: superficies en wireframe + rayos.

    Args:
        sistema: SistemaOptico
        resultados: lista de ResultadoTrazado (opcional)
        ax: eje 3D existente (opcional)
        titulo: título de la gráfica
        mostrar: si True, llama a plt.show()
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Dibujar superficies
    for sup in sistema.superficies:
        X, Y, Z = sup.generar_malla_3d()
        ax.plot_wireframe(Z, X, Y, color='steelblue', alpha=0.3,
                          linewidth=0.5, rstride=2, cstride=5)

    # Dibujar rayos
    if resultados:
        for res in resultados:
            zs = [p[2] for p in res.puntos]
            xs = [p[0] for p in res.puntos]
            ys = [p[1] for p in res.puntos]
            ax.plot(zs, xs, ys, color='orange', linewidth=0.5, alpha=0.8)

            # Extensión
            if res.rayo_completo and len(res.puntos) > 1:
                p_f = res.puntos[-1]
                d_f = res.direcciones[-1]
                ext = 50.0
                p_e = p_f + ext * d_f
                ax.plot([p_f[2], p_e[2]], [p_f[0], p_e[0]], [p_f[1], p_e[1]],
                        color='orange', linewidth=0.5, alpha=0.4, linestyle='--')

    ax.set_xlabel('Z (eje óptico)')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    if titulo:
        ax.set_title(titulo)

    if mostrar:
        plt.tight_layout()
        plt.show()

    return ax

"""Visualización 2D y 3D de sistemas ópticos con superficies cartesianas."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _rho_clip_para_r(superficie, r_max):
    """Encuentra el ρ máximo tal que r(ρ) ≤ r_max, solo rama ascendente.

    Las superficies ovoides tienen r(ρ) que crece y luego decrece (el óvalo
    se cierra). Solo usamos la primera rama (ascendente) hasta que r alcanza
    r_max o comienza a decrecer.
    """
    rho_lim = superficie.rho_max * 0.99 if np.isfinite(superficie.rho_max) else 200.0
    rho_test = np.linspace(0, rho_lim, 5000)
    r_all = superficie.r_de_rho(rho_test)

    # Encontrar dónde r empieza a decrecer (fin de la rama ascendente)
    dr = np.diff(r_all)
    idx_decreciente = np.where(dr < -1e-12)[0]
    if len(idx_decreciente) > 0:
        idx_max_r = idx_decreciente[0]
    else:
        idx_max_r = len(rho_test) - 1

    # Dentro de la rama ascendente, encontrar dónde r excede r_max
    idx_excede = np.where(r_all[:idx_max_r + 1] > r_max)[0]
    if len(idx_excede) > 0:
        idx_clip = idx_excede[0]
    else:
        idx_clip = idx_max_r

    if idx_clip == 0:
        return 0.0
    return rho_test[idx_clip]


def _perfil_hasta_r(superficie, r_max, num_puntos=500):
    """Genera perfil (z, r) de la superficie recortado hasta r_max."""
    rho_clip = _rho_clip_para_r(superficie, r_max)
    if rho_clip < 1e-15:
        return np.array([superficie.zeta]), np.array([0.0])
    rho_fine = np.linspace(0, rho_clip, num_puntos)
    z = superficie.z_de_rho(rho_fine)
    r = superficie.r_de_rho(rho_fine)
    return z, r


def _r_max_superficie(sistema, k, aperturas):
    """Determina r_max para la superficie k dadas las aperturas."""
    if k < len(aperturas):
        return aperturas[k]
    elif k > 0 and k - 1 < len(aperturas):
        return aperturas[k - 1]
    else:
        sup = sistema.superficies[k]
        return sup.rho_max * 0.95 if np.isfinite(sup.rho_max) else 50.0


def graficar_seccion_transversal(sistema, resultados, ax=None, titulo=None,
                                  colores_rayos=None, mostrar=True,
                                  z_imagen=None):
    """Gráfica 2D del plano meridional (r-z): perfiles de superficie + rayos.

    Recorta las superficies en su intersección mutua y rellena el cuerpo
    de la lente con un color semitransparente.

    Args:
        sistema: SistemaOptico
        resultados: lista de ResultadoTrazado
        ax: eje matplotlib existente (opcional)
        titulo: título de la gráfica
        colores_rayos: lista de colores para los rayos
        mostrar: si True, llama a plt.show()
        z_imagen: posición z del plano imagen (para extender rayos)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    aperturas = sistema.encontrar_apertura()

    # Dibujar superficies
    for k, sup in enumerate(sistema.superficies):
        r_max = _r_max_superficie(sistema, k, aperturas)
        z_perfil, r_perfil = _perfil_hasta_r(sup, r_max)
        ax.plot(z_perfil, r_perfil, 'k-', linewidth=1.5)
        ax.plot(z_perfil, -r_perfil, 'k-', linewidth=1.5)

    # Rellenar cuerpo de la lente entre superficies consecutivas
    for k in range(len(sistema.superficies) - 1):
        if k >= len(aperturas):
            break
        r_max = aperturas[k]
        z0, r0 = _perfil_hasta_r(sistema.superficies[k], r_max, num_puntos=300)
        z1, r1 = _perfil_hasta_r(sistema.superficies[k + 1], r_max, num_puntos=300)

        z_contorno = np.concatenate([z0, z1[::-1]])
        r_contorno = np.concatenate([r0, r1[::-1]])
        ax.fill(z_contorno, r_contorno, color='lightskyblue', alpha=0.3)
        ax.fill(z_contorno, -r_contorno, color='lightskyblue', alpha=0.3)

        ax.plot([z0[-1], z1[-1]], [r0[-1], r1[-1]], 'k-', linewidth=1.0)
        ax.plot([z0[-1], z1[-1]], [-r0[-1], -r1[-1]], 'k-', linewidth=1.0)

    # Dibujar rayos
    color_default = 'tab:blue'
    for i, res in enumerate(resultados):
        color = colores_rayos[i] if colores_rayos and i < len(colores_rayos) else color_default

        for j in range(len(res.puntos) - 1):
            p1 = res.puntos[j]
            p2 = res.puntos[j + 1]
            ax.plot([p1[2], p2[2]], [p1[1], p2[1]], color=color,
                    linewidth=0.6, alpha=0.7)

        if res.rayo_completo and len(res.puntos) > 1:
            p_final = res.puntos[-1]
            d_final = res.direcciones[-1]
            if z_imagen is not None and abs(d_final[2]) > 1e-12:
                t_ext = (z_imagen - p_final[2]) / d_final[2]
                if t_ext > 0:
                    p_ext = p_final + t_ext * d_final
                    ax.plot([p_final[2], p_ext[2]], [p_final[1], p_ext[1]],
                            color=color, linewidth=0.6, alpha=0.5, linestyle='--')
            else:
                p_ext = p_final + 50.0 * d_final
                ax.plot([p_final[2], p_ext[2]], [p_final[1], p_ext[1]],
                        color=color, linewidth=0.6, alpha=0.4, linestyle='--')

    # Eje óptico y límites
    z_all_pts = [p[2] for res in resultados for p in res.puntos]
    z_min = min(z_all_pts) - 10
    z_max_plot = max(z_all_pts) + 30
    if z_imagen is not None:
        z_max_plot = max(z_max_plot, z_imagen + 20)
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-.')
    ax.set_xlim(z_min, z_max_plot)

    ax.set_xlabel('z (eje óptico)')
    ax.set_ylabel('r (transversal)')
    ax.set_aspect('equal')
    if titulo:
        ax.set_title(titulo)

    if mostrar:
        plt.tight_layout()
        plt.show()

    return ax


def graficar_3d(sistema, resultados=None, ax=None, titulo=None,
                mostrar=True, z_imagen=None, colores_rayos=None,
                elev=18, azim=-55):
    """Gráfica 3D: superficies sólidas semitransparentes + rayos.

    Recorta las superficies a la apertura de la lente y extiende
    los rayos hasta el plano imagen para mostrar la convergencia.

    Args:
        sistema: SistemaOptico
        resultados: lista de ResultadoTrazado (opcional)
        ax: eje 3D existente (opcional)
        titulo: título de la gráfica
        mostrar: si True, llama a plt.show()
        z_imagen: posición z del plano imagen (para extender rayos)
        colores_rayos: lista de colores para cada rayo (opcional)
        elev: elevación de la cámara
        azim: azimut de la cámara
    """
    if ax is None:
        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot(111, projection='3d')

    aperturas = sistema.encontrar_apertura()

    # --- Dibujar superficies como mallas sólidas semitransparentes ---
    num_rho = 50
    num_phi = 48

    for k, sup in enumerate(sistema.superficies):
        r_max = _r_max_superficie(sistema, k, aperturas)
        rho_clip = _rho_clip_para_r(sup, r_max)
        if rho_clip < 1e-15:
            continue

        rho = np.linspace(0, rho_clip, num_rho)
        phi = np.linspace(0, 2 * np.pi, num_phi + 1)

        z_perfil = sup.z_de_rho(rho)
        r_perfil = sup.r_de_rho(rho)

        R_grid, PHI_grid = np.meshgrid(r_perfil, phi)
        Z_grid = np.tile(z_perfil, (num_phi + 1, 1))
        X_grid = R_grid * np.cos(PHI_grid)
        Y_grid = R_grid * np.sin(PHI_grid)

        ax.plot_surface(Z_grid, X_grid, Y_grid,
                        color='cornflowerblue', alpha=0.2,
                        edgecolor='steelblue', linewidth=0.1,
                        rstride=3, cstride=2, shade=True)

    # --- Anillo de borde entre superficies consecutivas ---
    for k in range(len(sistema.superficies) - 1):
        if k >= len(aperturas):
            break
        r_max = aperturas[k]
        sup0 = sistema.superficies[k]
        sup1 = sistema.superficies[k + 1]

        rho_clip0 = _rho_clip_para_r(sup0, r_max)
        rho_clip1 = _rho_clip_para_r(sup1, r_max)
        z_edge0 = sup0.z_de_rho(rho_clip0)
        z_edge1 = sup1.z_de_rho(rho_clip1)
        r_edge0 = sup0.r_de_rho(rho_clip0)
        r_edge1 = sup1.r_de_rho(rho_clip1)

        phi_edge = np.linspace(0, 2 * np.pi, num_phi + 1)
        for i in range(len(phi_edge) - 1):
            c0, s0 = np.cos(phi_edge[i]), np.sin(phi_edge[i])
            c1, s1 = np.cos(phi_edge[i + 1]), np.sin(phi_edge[i + 1])
            verts = [
                [z_edge0, r_edge0 * c0, r_edge0 * s0],
                [z_edge1, r_edge1 * c0, r_edge1 * s0],
                [z_edge1, r_edge1 * c1, r_edge1 * s1],
                [z_edge0, r_edge0 * c1, r_edge0 * s1],
            ]
            poly = Poly3DCollection([verts], alpha=0.12, facecolor='steelblue',
                                     edgecolor='steelblue', linewidth=0.1)
            ax.add_collection3d(poly)

    # --- Dibujar rayos ---
    if resultados:
        color_default = '#E8833A'
        for i, res in enumerate(resultados):
            color = colores_rayos[i] if colores_rayos and i < len(colores_rayos) else color_default

            zs = [p[2] for p in res.puntos]
            xs = [p[0] for p in res.puntos]
            ys = [p[1] for p in res.puntos]
            ax.plot(zs, xs, ys, color=color, linewidth=0.7, alpha=0.8)

            if res.rayo_completo and len(res.puntos) > 1:
                p_f = res.puntos[-1]
                d_f = res.direcciones[-1]
                if z_imagen is not None and abs(d_f[2]) > 1e-12:
                    t_ext = (z_imagen - p_f[2]) / d_f[2]
                    if t_ext > 0:
                        p_e = p_f + t_ext * d_f
                        ax.plot([p_f[2], p_e[2]], [p_f[0], p_e[0]], [p_f[1], p_e[1]],
                                color=color, linewidth=0.6, alpha=0.5, linestyle='--')
                else:
                    p_e = p_f + 50.0 * d_f
                    ax.plot([p_f[2], p_e[2]], [p_f[0], p_e[0]], [p_f[1], p_e[1]],
                            color=color, linewidth=0.5, alpha=0.4, linestyle='--')

    # --- Configuración de ejes ---
    z_pts = [sup.zeta for sup in sistema.superficies]
    xy_pts = [0.0]

    if resultados:
        for res in resultados:
            for p in res.puntos:
                z_pts.append(p[2])
                xy_pts.extend([abs(p[0]), abs(p[1])])
            if res.rayo_completo and z_imagen is not None and len(res.direcciones) > 0:
                p_f = res.puntos[-1]
                d_f = res.direcciones[-1]
                if abs(d_f[2]) > 1e-12:
                    t_ext = (z_imagen - p_f[2]) / d_f[2]
                    if t_ext > 0:
                        p_e = p_f + t_ext * d_f
                        z_pts.append(p_e[2])
                        xy_pts.extend([abs(p_e[0]), abs(p_e[1])])

    z_min, z_max = min(z_pts) - 5, max(z_pts) + 5
    xy_max = max(xy_pts)
    if aperturas:
        xy_max = max(xy_max, max(aperturas) + 2)
    xy_max = max(xy_max, 5.0)

    ax.set_xlim(z_min, z_max)
    ax.set_ylim(-xy_max, xy_max)
    ax.set_zlim(-xy_max, xy_max)

    ax.set_xlabel('Z (eje óptico)', labelpad=8)
    ax.set_ylabel('X', labelpad=8)
    ax.set_zlabel('Y', labelpad=8)

    ax.view_init(elev=elev, azim=azim)

    # Aspecto alargado para que el eje óptico domine visualmente
    z_span = z_max - z_min
    ax.set_box_aspect([z_span, 2 * xy_max, 2 * xy_max])

    # Fondo limpio
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.grid(True, alpha=0.3)

    if titulo:
        ax.set_title(titulo, pad=15)

    if mostrar:
        plt.tight_layout()
        plt.show()

    return ax

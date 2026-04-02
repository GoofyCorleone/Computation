"""Visualización 2D y 3D de sistemas ópticos con superficies cartesianas."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def _encontrar_apertura_lente(sistema):
    """Encuentra la altura transversal máxima (apertura) donde las superficies
    consecutivas se interceptan, formando el borde físico de la lente.

    Returns:
        Lista de r_max para cada par de superficies consecutivas.
        Si no hay intersección, usa un valor por defecto.
    """
    aperturas = []
    for k in range(len(sistema.superficies) - 1):
        sup0 = sistema.superficies[k]
        sup1 = sistema.superficies[k + 1]

        # Buscar ρ donde z_0(ρ) = z_1(ρ) (ambas superficies a la misma altura axial)
        # Barrer ρ hasta encontrar cruce
        rho_lim_0 = sup0.rho_max if np.isfinite(sup0.rho_max) else 200.0
        rho_lim_1 = sup1.rho_max if np.isfinite(sup1.rho_max) else 200.0
        rho_lim = min(rho_lim_0, rho_lim_1)

        rho_test = np.linspace(0.01, rho_lim * 0.99, 2000)
        z0 = sup0.z_de_rho(rho_test)
        z1 = sup1.z_de_rho(rho_test)
        r0 = sup0.r_de_rho(rho_test)
        r1 = sup1.r_de_rho(rho_test)

        # La lente existe donde sup0 está a la izquierda de sup1 (z0 < z1)
        # o al revés. El borde es donde se cruzan.
        diff = z0 - z1
        cruces = np.where(np.diff(np.sign(diff)))[0]

        if len(cruces) > 0:
            # Tomar el primer cruce (apertura de la lente)
            idx = cruces[0]
            # Interpolar para mejor precisión
            f = abs(diff[idx]) / (abs(diff[idx]) + abs(diff[idx + 1]))
            rho_cruce = rho_test[idx] + f * (rho_test[idx + 1] - rho_test[idx])
            r_cruce = sup0.r_de_rho(rho_cruce)
            aperturas.append(float(r_cruce))
        else:
            # Sin cruce, usar un valor razonable
            aperturas.append(min(rho_lim_0, rho_lim_1) * 0.5)

    return aperturas


def _perfil_hasta_r(superficie, r_max, num_puntos=500):
    """Genera perfil (z, r) de la superficie recortado hasta r_max."""
    # Buscar el ρ que da r = r_max
    rho_lim = superficie.rho_max * 0.99 if np.isfinite(superficie.rho_max) else 200.0
    rho_test = np.linspace(0, rho_lim, 5000)
    z_all = superficie.z_de_rho(rho_test)
    r_all = superficie.r_de_rho(rho_test)

    # Recortar donde r <= r_max
    mask = r_all <= r_max * 1.001
    if not np.any(mask):
        return np.array([superficie.zeta]), np.array([0.0])

    rho_clip = rho_test[mask]
    rho_fine = np.linspace(0, rho_clip[-1], num_puntos)
    z = superficie.z_de_rho(rho_fine)
    r = superficie.r_de_rho(rho_fine)

    return z, r


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

    # Encontrar aperturas donde las superficies se interceptan
    aperturas = _encontrar_apertura_lente(sistema)

    # Dibujar superficies y rellenar lente
    for k, sup in enumerate(sistema.superficies):
        # Determinar r_max para esta superficie
        if k < len(aperturas):
            r_max = aperturas[k]
        elif k > 0 and k - 1 < len(aperturas):
            r_max = aperturas[k - 1]
        else:
            r_lim = sup.rho_max * 0.95 if np.isfinite(sup.rho_max) else 50.0
            r_max = r_lim

        z_perfil, r_perfil = _perfil_hasta_r(sup, r_max)

        # Perfil superior e inferior
        ax.plot(z_perfil, r_perfil, 'k-', linewidth=1.5)
        ax.plot(z_perfil, -r_perfil, 'k-', linewidth=1.5)

    # Rellenar cuerpo de la lente entre superficies consecutivas
    for k in range(len(sistema.superficies) - 1):
        if k >= len(aperturas):
            break
        r_max = aperturas[k]
        sup0 = sistema.superficies[k]
        sup1 = sistema.superficies[k + 1]

        z0, r0 = _perfil_hasta_r(sup0, r_max, num_puntos=300)
        z1, r1 = _perfil_hasta_r(sup1, r_max, num_puntos=300)

        # Contorno cerrado: sup0 de abajo a arriba, borde, sup1 de arriba a abajo, borde
        # Perfil superior
        z_contorno = np.concatenate([z0, z1[::-1]])
        r_contorno = np.concatenate([r0, r1[::-1]])
        ax.fill(z_contorno, r_contorno, color='lightskyblue', alpha=0.3)
        ax.fill(z_contorno, -r_contorno, color='lightskyblue', alpha=0.3)

        # Borde superior e inferior de la lente
        ax.plot([z0[-1], z1[-1]], [r0[-1], r1[-1]], 'k-', linewidth=1.0)
        ax.plot([z0[-1], z1[-1]], [-r0[-1], -r1[-1]], 'k-', linewidth=1.0)

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

        # Extensión del último segmento hasta z_imagen o distancia fija
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
                extension = 50.0
                p_ext = p_final + extension * d_final
                ax.plot([p_final[2], p_ext[2]], [p_final[1], p_ext[1]],
                        color=color, linewidth=0.6, alpha=0.4, linestyle='--')

    # Eje óptico
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


def graficar_3d(sistema, resultados=None, ax=None, titulo=None, mostrar=True):
    """Gráfica 3D: superficies en wireframe + rayos.

    Recorta las superficies a la apertura de la lente.

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

    aperturas = _encontrar_apertura_lente(sistema)

    # Dibujar superficies recortadas
    for k, sup in enumerate(sistema.superficies):
        if k < len(aperturas):
            r_max = aperturas[k]
        elif k > 0 and k - 1 < len(aperturas):
            r_max = aperturas[k - 1]
        else:
            r_max = sup.rho_max * 0.95 if np.isfinite(sup.rho_max) else 50.0

        # Buscar ρ_max correspondiente a r_max
        rho_lim = sup.rho_max * 0.99 if np.isfinite(sup.rho_max) else 200.0
        rho_test = np.linspace(0, rho_lim, 2000)
        r_test = sup.r_de_rho(rho_test)
        mask = r_test <= r_max * 1.001
        if np.any(mask):
            rho_clip = rho_test[mask][-1]
        else:
            rho_clip = rho_lim * 0.5

        num_rho = 60
        num_phi = 40
        rho = np.linspace(0, rho_clip, num_rho)
        phi = np.linspace(0, 2 * np.pi, num_phi, endpoint=False)

        z_perfil = sup.z_de_rho(rho)
        r_perfil = sup.r_de_rho(rho)

        RHO, PHI = np.meshgrid(r_perfil, phi)
        Z_mesh = np.tile(z_perfil, (num_phi, 1))
        X = RHO * np.cos(PHI)
        Y = RHO * np.sin(PHI)

        ax.plot_surface(Z_mesh, X, Y, color='lightskyblue', alpha=0.3,
                        edgecolor='steelblue', linewidth=0.2,
                        rstride=2, cstride=3)

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

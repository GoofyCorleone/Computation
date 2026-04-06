"""Ejemplo: Trazado de rayos en distintas LSOE estigmáticas (Fig. 10 de la tesis).

Muestra un panel 4x3 con diferentes configuraciones de lentes LSOE convergentes
y divergentes, con el comportamiento de los rayos al atravesarlas.

- Fila 1: Lentes biconvexas (objeto real → imagen real)
- Fila 2: Menisco y plano-convexa
- Fila 3: Convexa-plano, objeto virtual, imagen lejana
- Fila 4: Lentes DIVERGENTES (imagen virtual)

Ejes como en Fig. 10 de la tesis: r en horizontal, z en vertical.

Uso:
    python ejemplo_fig10.py          # sin exportar STL (por defecto)
    python ejemplo_fig10.py --stl    # exportar STL de cada lente
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from gots import SistemaOptico, Rayo
from gots import exportar_sistema_stl
from gots.visualizacion import _perfil_hasta_r
from gots.utilidades import normalizar


# --- Configuraciones LSOE: (d0, d2, ζ0, ζ1, n0, n1, n2, sigma, descripción) ---
# Filas 1-3: lentes convergentes (imagen real, d2 > ζ1)
CONVERGENTES = [
    # Fila 1: Biconvexas
    (0, 150, 60, 80, 1.0, 1.7, 1.0, -0.3, "Biconvexa\nσ=−0.3"),
    (0, 150, 60, 80, 1.0, 1.7, 1.0,  0.0, "Biconvexa sim.\nσ=0"),
    (0, 150, 60, 80, 1.0, 1.7, 1.0,  0.3, "Biconvexa\nσ=+0.3"),
    # Fila 2: Menisco / plano-convexa
    (0, 150, 60, 80, 1.0, 1.7, 1.0, -0.8, "Menisco\nσ=−0.8"),
    (0, 150, 60, 80, 1.0, 1.7, 1.0, -1.0, "Plano-convexa\nσ=−1"),
    (0, 150, 60, 80, 1.0, 1.7, 1.0,  0.8, "Menisco inv.\nσ=+0.8"),
    # Fila 3: Convexa-plano / objeto virtual / imagen lejana
    (0, 150, 60, 80, 1.0, 1.7, 1.0,  1.0, "Convexa-plano\nσ=+1"),
    (-60, 150, 60, 80, 1.0, 1.7, 1.0, 0.0, "Obj. virtual\nσ=0"),
    (0, 300, 60, 80, 1.0, 1.7, 1.0,  0.0, "Imagen lejana\nσ=0"),
]

# Fila 4: lentes divergentes (imagen virtual: d2 < ζ0)
# sigma=±1 degeneran con d2=30 (d1 cae exactamente en un vértice de superficie)
DIVERGENTES = [
    (0,  30, 60, 80, 1.0, 1.7, 1.0, -0.5, "Menisco cóncavo\nσ=−0.5 (div.)"),
    (0,  30, 60, 80, 1.0, 1.7, 1.0,  0.0, "Bicóncava\nσ=0 (div.)"),
    (0,  30, 60, 80, 1.0, 1.7, 1.0,  0.5, "Cóncava menisco\nσ=+0.5 (div.)"),
]

CONFIGURACIONES = CONVERGENTES + DIVERGENTES


def _es_divergente(config):
    d_0, d_2, zeta_0 = config[0], config[1], config[2]
    return d_2 < zeta_0


def trazar_y_dibujar(ax, config, exportar_stl_flag=False):
    """Traza rayos a través de una LSOE y dibuja en el eje dado."""
    d_0, d_2, zeta_0, zeta_1, n_0, n_1, n_2, sigma, desc = config
    divergente = _es_divergente(config)

    # Construir sistema
    try:
        sistema, d1 = SistemaOptico.lsoe(
            zeta_0=zeta_0, zeta_1=zeta_1, d_0=d_0, d_2=d_2,
            n_0=n_0, n_1=n_1, n_2=n_2, sigma=sigma
        )
    except Exception as e:
        ax.text(0.5, 0.5, f'Error LSOE:\n{e}', ha='center', va='center',
                transform=ax.transAxes, fontsize=7, color='red')
        ax.set_title(desc, fontsize=9)
        return None

    # Dibujar y trazar (envuelto en try/except para configs degeneradas)
    try:
        aperturas = sistema.encontrar_apertura()
        r_max = aperturas[0] if aperturas else 25.0

        sup0 = sistema.superficies[0]
        sup1 = sistema.superficies[1]
        z0, r0 = _perfil_hasta_r(sup0, r_max)
        z1, r1 = _perfil_hasta_r(sup1, r_max)

        # Perfil completo (±r) — ejes: r horizontal, z vertical (como Fig. 10)
        ax.plot(r0, z0, 'k-', linewidth=1.2)
        ax.plot(-r0, z0, 'k-', linewidth=1.2)
        ax.plot(r1, z1, 'k-', linewidth=1.2)
        ax.plot(-r1, z1, 'k-', linewidth=1.2)

        # Rellenar cuerpo de la lente
        color_lente = 'lightskyblue' if not divergente else 'lightyellow'
        z_c = np.concatenate([z0, z1[::-1]])
        r_c = np.concatenate([r0, r1[::-1]])
        ax.fill(r_c, z_c, color=color_lente, alpha=0.5)
        ax.fill(-r_c, z_c, color=color_lente, alpha=0.5)

        # Borde de la lente (arista)
        ax.plot([r0[-1], r1[-1]], [z0[-1], z1[-1]], 'k-', linewidth=0.9)
        ax.plot([-r0[-1], -r1[-1]], [z0[-1], z1[-1]], 'k-', linewidth=0.9)

        # Trazar rayos
        fuente = np.array([0.0, 0.0, d_0])
        num_rayos = 13
        angulo_max = 0.10
        color_rayo = '#1a6eb5' if not divergente else '#b55a1a'

        for theta in np.linspace(-angulo_max, angulo_max, num_rayos):
            direccion = normalizar(np.array([0.0, np.sin(theta), np.cos(theta)]))
            rayo = Rayo(fuente.copy(), direccion)
            res = sistema.trazar_rayo(rayo)

            # Segmentos dentro del sistema
            for j in range(len(res.puntos) - 1):
                p1, p2 = res.puntos[j], res.puntos[j + 1]
                ax.plot([p1[1], p2[1]], [p1[2], p2[2]],
                        color=color_rayo, linewidth=0.5, alpha=0.8)

            # Extensión después del sistema
            if res.rayo_completo and len(res.puntos) > 1:
                pf = res.puntos[-1]
                df = res.direcciones[-1]
                if not divergente:
                    # Convergente: extender hasta la imagen (d_2)
                    if abs(df[2]) > 1e-12:
                        t_ext = (d_2 - pf[2]) / df[2]
                        if t_ext > 0:
                            pe = pf + t_ext * df
                            ax.plot([pf[1], pe[1]], [pf[2], pe[2]],
                                    color=color_rayo, linewidth=0.5,
                                    alpha=0.5, linestyle='--')
                else:
                    # Divergente: extender 60 unidades para mostrar divergencia
                    pe = pf + 60.0 * df
                    ax.plot([pf[1], pe[1]], [pf[2], pe[2]],
                            color=color_rayo, linewidth=0.5,
                            alpha=0.5, linestyle='--')

        # Ejes
        ax.axhline(y=d_0, color='gray', linewidth=0.3, linestyle=':')
        ax.axvline(x=0, color='gray', linewidth=0.3, linestyle='-.')
        ax.set_title(desc, fontsize=8.5)
        ax.set_aspect('equal')

        # Exportar STL
        if exportar_stl_flag:
            nombre = (desc.split('\n')[0]
                      .replace(' ', '_').replace('.', '').replace('−', 'n'))
            stl_path = os.path.join(os.path.dirname(__file__),
                                    f'stl_{nombre}_sigma{sigma:.1f}.stl')
            try:
                exportar_sistema_stl(sistema, stl_path)
            except Exception:
                pass

    except Exception as e:
        ax.text(0.5, 0.5, f'Error trazado:\n{e}', ha='center', va='center',
                transform=ax.transAxes, fontsize=7, color='red')
        ax.set_title(desc, fontsize=9)
        return None

    return sistema


def main():
    exportar_stl_flag = '--stl' in sys.argv

    fig, axes = plt.subplots(4, 3, figsize=(13, 18))
    fig.suptitle('Trazado de rayos — LSOE estigmáticas (convergentes y divergentes)',
                 fontsize=13, y=0.995)

    # Etiquetas de fila
    filas = ['Biconvexas', 'Menisco / Plano-convexa',
             'Convexa-plano / Especiales', 'Divergentes (imagen virtual)']
    for row, etiqueta in enumerate(filas):
        axes[row, 0].set_ylabel(etiqueta, fontsize=8.5, labelpad=4)

    for i, config in enumerate(CONFIGURACIONES):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        trazar_y_dibujar(ax, config, exportar_stl_flag)

    for ax in axes[-1, :]:
        ax.set_xlabel('r', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.993])

    # Guardar imagen en docs/
    out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'fig10_lsoe.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Figura guardada en {os.path.normpath(out)}")

    if exportar_stl_flag:
        print("Archivos STL exportados en ejemplos/")
    else:
        print("(Ejecutar con --stl para exportar STL)")

    plt.show()


if __name__ == "__main__":
    main()

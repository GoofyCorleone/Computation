"""Ejemplo: Trazado de rayos en distintas LSOE estigmáticas (Fig. 10 de la tesis).

Muestra un panel 3x3 con diferentes configuraciones de lentes LSOE y el
comportamiento de los rayos al atravesarlas. Incluye:
- Fila 1: Lentes biconvexas (objeto real → imagen real)
- Fila 2: Lentes con una superficie cóncava y otra convexa
- Fila 3: Lentes que forman imágenes virtuales

Los ejes muestran z (vertical) vs r (horizontal), como en la Fig. 10 de la tesis.

Uso:
    python ejemplo_fig10.py             # sin STL
    python ejemplo_fig10.py --no-stl    # deshabilitar STL
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from gots import SistemaOptico, SuperficieCartesiana, Rayo, calcular_gots
from gots import exportar_sistema_stl
from gots.visualizacion import _perfil_hasta_r
from gots.utilidades import normalizar


# --- 9 configuraciones de LSOE (Fig. 10) ---
# Cada tupla: (d_0, d_2, zeta_0, zeta_1, n_0, n_1, n_2, sigma, descripcion)
CONFIGURACIONES = [
    # Fila 1: Biconvexas, objeto real → imagen real
    (0,   150,  60, 80, 1.0, 1.7, 1.0, -0.3, "Biconvexa\nσ=-0.3"),
    (0,   150,  60, 80, 1.0, 1.7, 1.0,  0.0, "Biconvexa sim.\nσ=0"),
    (0,   150,  60, 80, 1.0, 1.7, 1.0,  0.3, "Biconvexa\nσ=+0.3"),

    # Fila 2: Menisco / cóncavo-convexa
    (0,   150,  60, 80, 1.0, 1.7, 1.0, -0.8, "Menisco\nσ=-0.8"),
    (0,   150,  60, 80, 1.0, 1.7, 1.0, -1.0, "Plano-convexa\nσ=-1"),
    (-60, 150,  60, 80, 1.0, 1.7, 1.0,  0.0, "Obj. virtual\nσ=0"),

    # Fila 3: Configuraciones con imagen lejana / colimación
    (0,   300,  60, 80, 1.0, 1.7, 1.0,  0.0, "Imagen lejana\nσ=0"),
    (0,   150,  60, 80, 1.0, 1.7, 1.0,  0.8, "Menisco inv.\nσ=+0.8"),
    (0,   150,  60, 80, 1.0, 1.7, 1.0,  1.0, "Convexa-plano\nσ=+1"),
]


def trazar_y_dibujar(ax, config, exportar_stl_flag=True):
    """Traza rayos a través de una LSOE y dibuja en el eje dado."""
    d_0, d_2, zeta_0, zeta_1, n_0, n_1, n_2, sigma, desc = config

    try:
        sistema, d1 = SistemaOptico.lsoe(
            zeta_0=zeta_0, zeta_1=zeta_1, d_0=d_0, d_2=d_2,
            n_0=n_0, n_1=n_1, n_2=n_2, sigma=sigma
        )
    except Exception as e:
        ax.text(0.5, 0.5, f'Error:\n{e}', ha='center', va='center',
                transform=ax.transAxes, fontsize=8)
        ax.set_title(desc, fontsize=9)
        return None

    aperturas = sistema.encontrar_apertura()
    r_max = aperturas[0] if aperturas else 25.0

    # Dibujar superficies
    sup0 = sistema.superficies[0]
    sup1 = sistema.superficies[1]
    z0, r0 = _perfil_hasta_r(sup0, r_max)
    z1, r1 = _perfil_hasta_r(sup1, r_max)

    # Nota: ejes como en Fig. 10 → z vertical, r horizontal
    ax.plot(r0, z0, 'k-', linewidth=1.2)
    ax.plot(-r0, z0, 'k-', linewidth=1.2)
    ax.plot(r1, z1, 'k-', linewidth=1.2)
    ax.plot(-r1, z1, 'k-', linewidth=1.2)

    # Rellenar
    z_c = np.concatenate([z0, z1[::-1]])
    r_c = np.concatenate([r0, r1[::-1]])
    ax.fill(r_c, z_c, color='lightgray', alpha=0.5)
    ax.fill(-r_c, z_c, color='lightgray', alpha=0.5)
    ax.plot([r0[-1], r1[-1]], [z0[-1], z1[-1]], 'k-', linewidth=0.8)
    ax.plot([-r0[-1], -r1[-1]], [z0[-1], z1[-1]], 'k-', linewidth=0.8)

    # Trazar rayos desde el punto objeto
    fuente = np.array([0.0, 0.0, d_0])
    num_rayos = 15
    angulo_max = 0.12

    for theta in np.linspace(-angulo_max, angulo_max, num_rayos):
        direccion = normalizar(np.array([0.0, np.sin(theta), np.cos(theta)]))
        rayo = Rayo(fuente.copy(), direccion)
        res = sistema.trazar_rayo(rayo)

        # Dibujar segmentos (r=y, z=z, en ejes: x→r, y→z)
        for j in range(len(res.puntos) - 1):
            p1, p2 = res.puntos[j], res.puntos[j + 1]
            ax.plot([p1[1], p2[1]], [p1[2], p2[2]], 'k-', linewidth=0.4, alpha=0.7)

        # Extensión después de la lente
        if res.rayo_completo and len(res.puntos) > 1:
            pf = res.puntos[-1]
            df = res.direcciones[-1]
            ext = 40.0
            pe = pf + ext * df
            ax.plot([pf[1], pe[1]], [pf[2], pe[2]], 'k-', linewidth=0.4, alpha=0.5)

    ax.axhline(y=0, color='gray', linewidth=0.3)
    ax.axvline(x=0, color='gray', linewidth=0.3, linestyle='-.')
    ax.set_title(desc, fontsize=9)
    ax.set_aspect('equal')

    # Exportar STL individual
    if exportar_stl_flag:
        nombre = desc.split('\n')[0].replace(' ', '_').replace('.', '')
        stl_path = os.path.join(os.path.dirname(__file__),
                                f'stl_{nombre}_sigma{sigma}.stl')
        try:
            exportar_sistema_stl(sistema, stl_path)
        except Exception:
            pass

    return sistema


def main():
    exportar_stl_flag = '--no-stl' not in sys.argv

    fig, axes = plt.subplots(3, 3, figsize=(14, 16))
    fig.suptitle('Trazado de rayos en distintas LSOE (Fig. 10)', fontsize=14, y=0.98)

    for i, config in enumerate(CONFIGURACIONES):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        trazar_y_dibujar(ax, config, exportar_stl_flag)

    # Etiquetas comunes
    for ax in axes[-1, :]:
        ax.set_xlabel('r')
    for ax in axes[:, 0]:
        ax.set_ylabel('z')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    if exportar_stl_flag:
        print("Archivos STL exportados en ejemplos/")
    else:
        print("(Usar sin --no-stl para exportar STL)")


if __name__ == "__main__":
    main()

"""Ejemplo: Formas de LSOE para distintos factores de forma σ (Fig. 5 de la tesis).

Genera un panel con 5 lentes LSOE para σ = -1, -0.5, 0, 0.5, 1
usando los parámetros: ζ₀=60, ζ₁=80, d₀=0, d₂=150, n₀=n₂=1, n₁=1.7.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from gots import SistemaOptico, calcular_gots, SuperficieCartesiana
from gots.visualizacion import _perfil_hasta_r


def main():
    zeta_0, zeta_1 = 60.0, 80.0
    d_0, d_2 = 0.0, 150.0
    n_0, n_1, n_2 = 1.0, 1.7, 1.0

    sigmas = [-1.0, -0.5, 0.0, 0.5, 1.0]

    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)
    fig.suptitle('Formas de LSOE vs factor de forma σ (Fig. 5)', fontsize=14)

    for i, sigma in enumerate(sigmas):
        ax = axes[i]
        sistema, d1 = SistemaOptico.lsoe(
            zeta_0=zeta_0, zeta_1=zeta_1, d_0=d_0, d_2=d_2,
            n_0=n_0, n_1=n_1, n_2=n_2, sigma=sigma
        )

        aperturas = sistema.encontrar_apertura()
        r_max = aperturas[0] if aperturas else 25.0

        sup0 = sistema.superficies[0]
        sup1 = sistema.superficies[1]

        z0, r0 = _perfil_hasta_r(sup0, r_max)
        z1, r1 = _perfil_hasta_r(sup1, r_max)

        # Superficie 0 (azul)
        ax.plot(z0, r0, 'b-', linewidth=1.5, label='Σ₀')
        ax.plot(z0, -r0, 'b-', linewidth=1.5)
        # Superficie 1 (gris)
        ax.plot(z1, r1, color='gray', linewidth=1.5, label='Σ₁')
        ax.plot(z1, -r1, color='gray', linewidth=1.5)

        # Rellenar lente
        z_c = np.concatenate([z0, z1[::-1]])
        r_c = np.concatenate([r0, r1[::-1]])
        ax.fill(z_c, r_c, color='lightskyblue', alpha=0.3)
        ax.fill(z_c, -r_c, color='lightskyblue', alpha=0.3)

        # Bordes
        ax.plot([z0[-1], z1[-1]], [r0[-1], r1[-1]], 'k-', linewidth=0.8)
        ax.plot([z0[-1], z1[-1]], [-r0[-1], -r1[-1]], 'k-', linewidth=0.8)

        ax.set_title(f'σ = {sigma:+.1f}\nd₁ = {d1:.1f}', fontsize=10)
        ax.set_xlabel('coordenada axial')
        ax.axhline(0, color='gray', linewidth=0.3, linestyle='-.')
        ax.set_aspect('equal')
        if i == 0:
            ax.set_ylabel('coordenada transversal')
            ax.legend(fontsize=8, loc='lower left')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

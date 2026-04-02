"""Ejemplo: Lente singlete ovoide estigmática (LSOE).

Replica la configuración de la Tabla 4 de la tesis de Silva-Lora (2024):
- Objeto en z=0, imagen en z=200
- Primera superficie en ζ_0=80, segunda en ζ_1=90
- n_0=1, n_1=1.6, n_2=1
- d_0=0, d_1=400, d_2=200

Genera visualizaciones 2D y 3D (Figs. 11-12 de la tesis).

Uso:
    python ejemplo_lsoe.py              # sin STL
    python ejemplo_lsoe.py --stl        # exporta STL recortado
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gots import (
    SistemaOptico, SuperficieCartesiana,
    calcular_gots, graficar_seccion_transversal, graficar_3d,
    exportar_sistema_stl
)


def main():
    exportar_stl = '--stl' in sys.argv

    # --- Parámetros de la Tabla 4 ---
    zeta_0 = 80.0
    zeta_1 = 90.0
    d_0 = 0.0
    d_1 = 400.0
    d_2 = 200.0
    n_0 = 1.0
    n_1 = 1.6
    n_2 = 1.0

    # --- Calcular GOTS y verificar contra Tabla 4 ---
    p0 = calcular_gots(n_0, n_1, zeta_0, d_0, d_1)
    p1 = calcular_gots(n_1, n_2, zeta_1, d_1, d_2)

    print("=== Parámetros GOTS (comparar con Tabla 4) ===")
    print(f"Superficie 0: G={p0.G_k:.6f}, O={p0.O_k:.7f}, "
          f"T={p0.T_k:.6e}, S={p0.S_k:.6e}")
    print(f"  Verificación S²=GOT: {p0.verificar()}")
    print(f"Superficie 1: G={p1.G_k:.6f}, O={p1.O_k:.6f}, "
          f"T={p1.T_k:.6e}, S={p1.S_k:.6e}")
    print(f"  Verificación S²=GOT: {p1.verificar()}")

    print("\nTabla 4 esperada:")
    print("  Sup 0: G=-1.905429, O=0.0291667, T=-4.181733e-8, S=-4.820787e-5")
    print("  Sup 1: G=-5.646665, O=-0.006549, T=4.814778e-8, S=4.219721e-5")

    # --- Construir sistema ---
    sistema = SistemaOptico()
    sup0 = SuperficieCartesiana(p0, n_0, n_1)
    sup1 = SuperficieCartesiana(p1, n_1, n_2)
    sistema.agregar_superficie(sup0)
    sistema.agregar_superficie(sup1)

    # --- Trazado de rayos: objeto sobre el eje óptico ---
    fuente = np.array([0.0, 0.0, d_0])
    print("\n=== Trazado de rayos (objeto en eje) ===")
    resultados_eje = sistema.trazar_abanico(fuente, num_rayos=21, angulo_max=0.08)

    completos = sum(1 for r in resultados_eje if r.rayo_completo)
    print(f"Rayos trazados: {len(resultados_eje)}, completos: {completos}")

    # Verificar convergencia al punto imagen
    if completos > 0:
        puntos_imagen = []
        for r in resultados_eje:
            if r.rayo_completo:
                p_final = r.puntos[-1]
                d_final = r.direcciones[-1]
                if abs(d_final[2]) > 1e-12:
                    t_img = (d_2 - p_final[2]) / d_final[2]
                    p_img = p_final + t_img * d_final
                    puntos_imagen.append(p_img)

        if puntos_imagen:
            puntos_img = np.array(puntos_imagen)
            r_img = np.sqrt(puntos_img[:, 0]**2 + puntos_img[:, 1]**2)
            print(f"Dispersión en plano imagen: max(r) = {np.max(r_img):.6e}")
            print(f"  (debe ser ≈ 0 para estigmatismo riguroso)")

    # --- Trazado de rayos: objeto fuera de eje ---
    H = 7.5  # campo máximo en mm
    fuente_off = np.array([0.0, H, d_0])
    resultados_off = sistema.trazar_abanico(fuente_off, num_rayos=21, angulo_max=0.08)

    print(f"\n=== Trazado fuera de eje (H={H}mm) ===")
    completos_off = sum(1 for r in resultados_off if r.rayo_completo)
    print(f"Rayos completos: {completos_off}")

    # --- Visualización 2D ---
    print("\n=== Generando gráfica 2D (Fig. 12) ===")
    todos_resultados = resultados_eje + resultados_off
    colores = ['tab:blue'] * len(resultados_eje) + ['tab:red'] * len(resultados_off)
    graficar_seccion_transversal(sistema, todos_resultados,
                                  titulo='LSOE - Sección Transversal (Tabla 4)',
                                  colores_rayos=colores,
                                  z_imagen=d_2,
                                  mostrar=True)

    # --- Visualización 3D ---
    print("=== Generando gráfica 3D (Fig. 11) ===")
    graficar_3d(sistema, todos_resultados,
                titulo='LSOE - Vista 3D',
                colores_rayos=colores,
                z_imagen=d_2,
                mostrar=True)

    # --- Exportar STL (opcional) ---
    if exportar_stl:
        stl_path = os.path.join(os.path.dirname(__file__), 'lsoe.stl')
        exportar_sistema_stl(sistema, stl_path)
        size_kb = os.path.getsize(stl_path) / 1024
        print(f"\nSTL exportado: {stl_path} ({size_kb:.1f} KB)")
    else:
        print("\n(Usar --stl para exportar la lente a STL)")

    # --- LSOE con factor de forma ---
    print("\n=== LSOE con factor de forma σ ===")
    for sigma in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        try:
            _, d1_calc = SistemaOptico.lsoe(
                zeta_0=60, zeta_1=80, d_0=0, d_2=150,
                n_0=1.0, n_1=1.7, n_2=1.0, sigma=sigma
            )
            print(f"  σ={sigma:+.1f} → d_1={d1_calc:.6f}")
        except Exception as e:
            print(f"  σ={sigma:+.1f} → Error: {e}")


if __name__ == "__main__":
    main()

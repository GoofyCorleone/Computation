"""Ejemplo: LSOEs rigurosamente aplanéticas (Sec. 4.4 de la tesis).

Muestra los cuatro tipos de sistemas rigurosamente aplanéticos derivados
en la tesis de Silva-Lora (2024):

    Tipo-0: esferas en los puntos aplanéticos de Young (2S = G·O²).
    Tipo-1: cónicas con imagen intermedia al infinito (d_1 → ∞).
    Tipo-2: superficies con O_k = 0 (planas en el vértice).
    Tipo-3: meniscos con G_k = 0 (menisco rigurosamente aplanético).

Para cada Tipo traza un abanico de rayos desde un objeto puntual, dibuja
la sección meridional y verifica numéricamente:

    (i) estigmatismo riguroso → convergencia exacta al punto imagen,
    (ii) aplanetismo riguroso → M({ρ_k}) = constante en ρ (Ec. 93).

Uso:
    python ejemplos/ejemplo_aplanaticas.py
    python ejemplos/ejemplo_aplanaticas.py --stl    # exporta un STL sólido por tipo
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from gots import (
    sistema_lsoe_tipo0, sistema_lsoe_tipo1,
    sistema_lsoe_tipo2, sistema_lsoe_tipo3,
    magnificacion_M, exportar_sistema_stl,
)


# --- Helpers de dibujo -----------------------------------------------------

def _perfil_ascendente(sup, num=800):
    rho_lim = sup.rho_max * 0.99 if np.isfinite(sup.rho_max) else 60.0
    rho = np.linspace(0.0, rho_lim, num)
    r = sup.r_de_rho(rho)
    z = sup.z_de_rho(rho)
    dr = np.diff(r)
    idx_desc = np.where(dr < -1e-12)[0]
    if len(idx_desc) > 0:
        i = idx_desc[0] + 1
        r, z = r[:i], z[:i]
    return r, z


def _dibujar_lente(ax, sa, r_max_cap=18.0):
    """Dibuja las dos superficies recortadas a un radio común visualmente razonable."""
    sup0, sup1 = sa.sistema.superficies
    r0, z0 = _perfil_ascendente(sup0)
    r1, z1 = _perfil_ascendente(sup1)
    r_max = min(r_max_cap, r0.max(), r1.max())

    mask0 = r0 <= r_max
    mask1 = r1 <= r_max
    ax.plot(z0[mask0], r0[mask0], color='black', lw=1.2)
    ax.plot(z0[mask0], -r0[mask0], color='black', lw=1.2)
    ax.plot(z1[mask1], r1[mask1], color='black', lw=1.2)
    ax.plot(z1[mask1], -r1[mask1], color='black', lw=1.2)

    # Relleno sombreado entre superficies (si ambas existen al mismo r)
    r_comun = np.linspace(0, r_max, 200)
    z0c = np.interp(r_comun, r0[mask0], z0[mask0])
    z1c = np.interp(r_comun, r1[mask1], z1[mask1])
    ax.fill_betweenx(np.concatenate([r_comun, -r_comun[::-1]]),
                      np.concatenate([z0c, z0c[::-1]]),
                      np.concatenate([z1c, z1c[::-1]]),
                      color='tab:blue', alpha=0.08)
    return r_max


def _dibujar_rayos(ax, resultados, sa, color='tab:red'):
    for r in resultados:
        if not r.rayo_completo:
            continue
        pts = np.array(r.puntos)
        # Segmentos: objeto → sup0 → sup1 → plano imagen
        ax.plot(pts[:, 2], pts[:, 1], color=color, lw=0.6, alpha=0.85)
        p_last = pts[-1]
        d_last = r.direcciones[-1]
        if abs(d_last[2]) > 1e-12:
            if np.isfinite(sa.d_2):
                t = (sa.d_2 - p_last[2]) / d_last[2]
                if t > 0:
                    p_end = p_last + t * d_last
                    ax.plot([p_last[2], p_end[2]], [p_last[1], p_end[1]],
                            color=color, lw=0.6, alpha=0.85)
                else:
                    # imagen virtual: prolongar hacia atrás
                    p_end = p_last + t * d_last
                    ax.plot([p_last[2], p_end[2]], [p_last[1], p_end[1]],
                            color=color, lw=0.5, alpha=0.4, linestyle=':')
                    # y también hacia adelante (divergencia real del rayo)
                    p_fwd = p_last + 40.0 * d_last
                    ax.plot([p_last[2], p_fwd[2]], [p_last[1], p_fwd[1]],
                            color=color, lw=0.5, alpha=0.6, linestyle='--')


def _verificar(sa):
    fuente = np.array([0.0, 0.0, sa.d_0])
    res = sa.sistema.trazar_abanico(fuente, num_rayos=11, angulo_max=0.05)
    completos = [r for r in res if r.rayo_completo]
    # Convergencia (estigmatismo)
    err_estig = 0.0
    if completos:
        pts_img = []
        for r in completos:
            p = r.puntos[-1]; d = r.direcciones[-1]
            if abs(d[2]) > 1e-12:
                t = (sa.d_2 - p[2]) / d[2]
                pts_img.append(p + t * d)
        if pts_img:
            pts_img = np.array(pts_img)
            err_estig = float(np.max(np.sqrt(pts_img[:, 0]**2 + pts_img[:, 1]**2)))
    # Aplanatismo M(ρ) constante
    M_vals = magnificacion_M(sa, np.linspace(0.2, 3.0, 8))
    err_aplan = float(np.max(np.abs(M_vals - M_vals[0])))
    return completos, err_estig, err_aplan


# --- Configuraciones de demostración ---------------------------------------

def configuraciones():
    return [
        ("Tipo-0\nEsferas (Young)", lambda: sistema_lsoe_tipo0(
            n_0=1.0, n_1=1.5, n_2=1.0, zeta_0=60, zeta_1=70, d_0=0)),
        ("Tipo-1\nCónicas (d₁→∞)", lambda: sistema_lsoe_tipo1(
            n_0=1.0, n_1=1.7, n_2=1.0, zeta_0=60, zeta_1=78, d_0=0, d_2=200)),
        ("Tipo-2\nO=0 (planas)", lambda: sistema_lsoe_tipo2(
            n_0=1.0, n_1=1.5, n_2=1.0, zeta_0=60, zeta_1=80, d_0=-60)),
        ("Tipo-3\nG=0 (meniscos)", lambda: sistema_lsoe_tipo3(
            n_0=1.0, n_1=1.5, n_2=1.0, zeta_0=60, zeta_1=80, d_0=-60)),
    ]


def main():
    exportar_stl = '--stl' in sys.argv

    cfgs = configuraciones()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    for ax, (titulo, fn) in zip(axes.flat, cfgs):
        sa = fn()
        completos, err_estig, err_aplan = _verificar(sa)

        # Trazar un abanico propio del tipo (ángulo adaptado al diámetro)
        fuente = np.array([0.0, 0.0, sa.d_0])
        res = sa.sistema.trazar_abanico(fuente, num_rayos=11, angulo_max=0.07)

        r_max_cap = 18.0 if sa.tipo != 1 else 22.0
        r_max = _dibujar_lente(ax, sa, r_max_cap=r_max_cap)
        _dibujar_rayos(ax, res, sa)

        # Plano objeto e imagen
        ax.axvline(sa.d_0, color='tab:blue',  lw=0.8, ls='--', alpha=0.6)
        if np.isfinite(sa.d_2):
            ax.axvline(sa.d_2, color='tab:orange', lw=0.8, ls='--', alpha=0.6)

        ax.set_title(f"{titulo}\n"
                     f"d₀={sa.d_0:.2f}  d₁={sa.d_1 if not np.isinf(sa.d_1) else '∞'}  "
                     f"d₂={sa.d_2:.2f}\n"
                     f"estig. max|r|={err_estig:.1e}   aplan. ΔM={err_aplan:.1e}",
                     fontsize=9)
        ax.set_xlabel('z'); ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.25)

        print(f"{titulo.splitlines()[0]}:")
        print(f"  d_0={sa.d_0}, d_1={sa.d_1}, d_2={sa.d_2}")
        print(f"  rayos completos: {len(completos)}/11")
        print(f"  estigmatismo max|r_img| = {err_estig:.3e}  (debe ≈ 0)")
        print(f"  aplanatismo  max|ΔM|   = {err_aplan:.3e}  (debe ≈ 0)")

        if exportar_stl:
            outdir = os.path.join(os.path.dirname(__file__), 'stl_aplanaticas')
            os.makedirs(outdir, exist_ok=True)
            path = os.path.join(outdir, f"lsoe_tipo{sa.tipo}.stl")
            try:
                exportar_sistema_stl(sa.sistema, path, r_max=r_max,
                                      espesor_minimo=0.08)
                print(f"  STL sólido: {path} ({os.path.getsize(path)/1024:.1f} KB)")
            except Exception as e:
                print(f"  STL: ERROR {type(e).__name__}: {e}")
        print()

    fig.suptitle("LSOEs rigurosamente aplanéticas — cuatro tipos (Sec. 4.4 de la tesis)",
                  fontsize=12)
    fig.tight_layout()

    docs = os.path.join(os.path.dirname(__file__), '..', 'docs')
    os.makedirs(docs, exist_ok=True)
    out_png = os.path.join(docs, 'aplanaticas_cuatro_tipos.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Figura guardada: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()

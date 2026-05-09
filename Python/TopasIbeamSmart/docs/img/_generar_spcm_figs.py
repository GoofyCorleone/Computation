"""
Genera figuras de ejemplo (estilo del SPCM50A/M) para el README.
Ejecutar:
    cd Python/TopasIbeamSmart
    python docs/img/_generar_spcm_figs.py

Salidas:
    docs/img/spcm_alignment.png   — tasa CPS vs tiempo (live view)
    docs/img/spcm_graph.png       — counts por bin (pestaña Graph)
    docs/img/spcm_bar.png         — histograma agrupado (pestaña Bar)
    docs/img/spcm_hist.png        — distribución Poisson de cuentas/bin
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as pl

# ── Paleta dark Catppuccin Mocha (igual a la GUI) ───────────────────────────
BG       = "#1e1e2e"
AX_BG    = "#181825"
FG       = "#cdd6f4"
GRID_COL = "#45475a"
C_VERDE  = "#a6e3a1"
C_AZUL   = "#89b4fa"
C_LILA   = "#cba6f7"
C_AMBAR  = "#f9e2af"
C_ROSA   = "#f38ba8"


def _aplicar_estilo(ax, fig):
    fig.set_facecolor(BG)
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_color(GRID_COL)
    ax.tick_params(colors=FG, labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.6)


def fig_alignment(out: Path):
    rng = np.random.default_rng(42)
    t   = np.linspace(0, 60, 600)
    # Tasa media ~2200 cps con deriva lenta + ruido Poisson en cada bin de 0.1 s
    cps_med = 2200 + 80 * np.sin(2 * np.pi * t / 25) + 30 * np.cos(2 * np.pi * t / 7)
    cps     = rng.poisson(cps_med * 0.1) / 0.1   # bins de 100 ms

    fig, ax = pl.subplots(figsize=(7.6, 3.2))
    _aplicar_estilo(ax, fig)
    ax.plot(t, cps, color=C_VERDE, lw=1.0, alpha=0.85)
    ax.axhline(cps.mean(), color=C_AMBAR, ls="--", lw=0.9,
               label=f"Media = {cps.mean():.0f} cps")
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Tasa de conteo [cps]")
    ax.set_title("SPCM50A/M — pestaña Alignment (live view, ventana 60 s)")
    ax.legend(loc="upper right", labelcolor=FG, facecolor=AX_BG,
              edgecolor=GRID_COL, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=BG)
    pl.close(fig)
    print(f"  → {out}")


def fig_graph(out: Path):
    rng = np.random.default_rng(7)
    n_bins = 1000
    bins   = np.arange(n_bins)
    base   = 2.2 + 0.3 * np.sin(2 * np.pi * bins / 250)   # cuentas/bin
    counts = rng.poisson(base)

    fig, ax = pl.subplots(figsize=(7.6, 3.2))
    _aplicar_estilo(ax, fig)
    ax.plot(bins, counts, color=C_AZUL, lw=0.7, alpha=0.85)
    ax.set_xlabel("Bin number")
    ax.set_ylabel("Counts per bin")
    ax.set_title("SPCM50A/M — pestaña Graph (Counts vs Bin Number, 1 ms/bin)")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=BG)
    pl.close(fig)
    print(f"  → {out}")


def fig_bar(out: Path):
    rng = np.random.default_rng(11)
    n_bins = 10_000
    base   = 2.2 + 0.3 * np.sin(2 * np.pi * np.arange(n_bins) / 1500)
    counts = rng.poisson(base)
    grupos = 200
    grupo_size = n_bins // grupos
    agrup  = counts.reshape(grupos, grupo_size).sum(axis=1)
    xs = np.arange(grupos) * grupo_size

    fig, ax = pl.subplots(figsize=(7.6, 3.2))
    _aplicar_estilo(ax, fig)
    ax.bar(xs, agrup, width=grupo_size * 0.9, color=C_LILA, edgecolor=AX_BG, lw=0.2)
    ax.set_xlabel(f"Bin number  (agrupado por {grupo_size} bins)")
    ax.set_ylabel("Counts (suma por grupo)")
    ax.set_title("SPCM50A/M — pestaña Bar (10 000 bins agrupados a 200 barras)")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=BG)
    pl.close(fig)
    print(f"  → {out}")


def fig_histograma(out: Path):
    rng = np.random.default_rng(99)
    counts = rng.poisson(2.2, size=20_000)
    bordes = np.arange(counts.min(), counts.max() + 2) - 0.5

    fig, ax = pl.subplots(figsize=(7.6, 3.2))
    _aplicar_estilo(ax, fig)
    ax.hist(counts, bins=bordes, color=C_ROSA, edgecolor=AX_BG, lw=0.5)
    media = counts.mean()
    ax.axvline(media, color=C_AMBAR, ls="--", lw=1.0,
               label=f"⟨n⟩ = {media:.2f}  (Poisson, λ = 2.2)")
    ax.set_xlabel("Counts per bin")
    ax.set_ylabel("Frecuencia")
    ax.set_title("SPCM50A/M — distribución de cuentas/bin (estadística Poisson)")
    ax.legend(loc="upper right", labelcolor=FG, facecolor=AX_BG,
              edgecolor=GRID_COL, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=BG)
    pl.close(fig)
    print(f"  → {out}")


def main():
    aqui = Path(__file__).resolve().parent
    print("Generando figuras de ejemplo del SPCM50A/M ...")
    fig_alignment(aqui / "spcm_alignment.png")
    fig_graph    (aqui / "spcm_graph.png")
    fig_bar      (aqui / "spcm_bar.png")
    fig_histograma(aqui / "spcm_hist.png")
    print("Listo.")


if __name__ == "__main__":
    main()

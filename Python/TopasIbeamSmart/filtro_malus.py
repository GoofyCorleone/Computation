"""
Filtrado de ruido de fuente y ajuste de ley de Malus por conteo de fotones.

Contexto: fuente láser TOPTICA iBeam Smart atenuada hasta régimen de fotones
individuales. Las oscilaciones de potencia de la fuente distorsionan el ajuste
N(θ) ∝ cos²(θ − θ₀). Este script aplica tres correcciones en cascada:

  1. Normalización por potencia de fuente (si se registró P(t) simultáneamente).
  2. Rechazo σ de puntos estadísticamente atípicos (sigma clipping).
  3. Promediado por bin angular cuando hay múltiples repeticiones en el mismo ángulo.

Ajuste final: A · cos²(θ − θ₀) + B, con B = conteo de fondo (oscuridad).

Entrada CSV (cabecera obligatoria):
  angulo_deg, conteos [, potencia_mW] [, tiempo_s]

Uso:
  python filtro_malus.py datos.csv
  python filtro_malus.py datos.csv --sin-potencia
  python filtro_malus.py datos.csv --umbral-sigma 2.5 --sin-potencia
  python filtro_malus.py          # datos sintéticos de ejemplo
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ── Constantes ───────────────────────────────────────────────────────────────

UMBRAL_SIGMA_DEFAULT = 3.0
COLOR_CRUDO   = "#5a8fc4"
COLOR_FILTRADO = "#4caf78"
COLOR_AJUSTE  = "#e05252"
COLOR_RUIDO   = "#f0a030"


# ── Modelo de Malus ──────────────────────────────────────────────────────────

def malus(theta_deg, A, theta0_deg, B):
    """A · cos²(θ − θ₀) + B."""
    return A * np.cos(np.deg2rad(theta_deg - theta0_deg))**2 + B


# ── Utilidades ───────────────────────────────────────────────────────────────

def _semilla_inicial(angulos, conteos):
    """Estimación burda de [A, θ₀, B] para inicializar curve_fit."""
    B0 = np.percentile(conteos, 5)
    A0 = conteos.max() - B0
    idx_max = np.argmax(conteos)
    theta0_0 = angulos[idx_max]
    return [A0, theta0_0, B0]


def ajustar(angulos, conteos, etiqueta=""):
    """Ajuste cos² con reporte en consola. Devuelve (popt, perr) o (None, None)."""
    p0 = _semilla_inicial(angulos, conteos)
    try:
        popt, pcov = curve_fit(
            malus, angulos, conteos, p0=p0,
            bounds=([0, -720, 0], [np.inf, 720, np.inf]),
            maxfev=20000,
        )
        perr = np.sqrt(np.diag(pcov))
        A, theta0, B = popt
        residuos = conteos - malus(angulos, *popt)
        # χ² reducido (varianza Poisson ≈ N)
        varianza = np.maximum(conteos, 1.0)
        chi2r = np.sum(residuos**2 / varianza) / max(len(conteos) - 3, 1)
        visibilidad = A / (A + 2 * B) if (A + 2 * B) > 0 else float("nan")

        print(f"\n{'─'*48}")
        print(f" Ajuste  {etiqueta}")
        print(f"{'─'*48}")
        print(f"  A      = {A:10.2f}  ±  {perr[0]:.2f}   (conteos pico)")
        print(f"  θ₀     = {theta0:10.3f}° ±  {perr[1]:.3f}°")
        print(f"  B      = {B:10.2f}  ±  {perr[2]:.2f}   (fondo)")
        print(f"  χ²_r   = {chi2r:10.4f}")
        print(f"  Visib. = {visibilidad:10.4f}  (A/(A+2B))")
        return popt, perr
    except Exception as exc:
        print(f"\n  [!] Ajuste '{etiqueta}' fallido: {exc}")
        return None, None


# ── Normalización por potencia ────────────────────────────────────────────────

def normalizar_por_potencia(conteos, potencia_mW):
    """
    Devuelve conteos corregidos: N_corr = N · <P> / P(t).
    Conserva la escala absoluta (media sin cambio).
    """
    p_ref = np.mean(potencia_mW)
    factor = p_ref / potencia_mW
    conteos_corr = conteos * factor
    desviacion_p = 100.0 * potencia_mW.std() / potencia_mW.mean()
    print(f"\nNormalización por potencia:")
    print(f"  <P>  = {p_ref:.4f} mW")
    print(f"  σ_P  = {potencia_mW.std():.5f} mW  ({desviacion_p:.2f} %)")
    print(f"  factor min/max = {factor.min():.4f} / {factor.max():.4f}")
    return conteos_corr


# ── Rechazo σ ────────────────────────────────────────────────────────────────

def rechazo_sigma(angulos, conteos, umbral, max_iter=5):
    """
    Sigma clipping iterativo: ajusta, calcula residuos, elimina puntos
    con |residuo| > umbral·σ. Repite hasta convergencia.
    """
    mascara = np.ones(len(angulos), dtype=bool)
    for _ in range(max_iter):
        a, c = angulos[mascara], conteos[mascara]
        if len(a) < 4:
            break
        p0 = _semilla_inicial(a, c)
        try:
            popt, _ = curve_fit(
                malus, a, c, p0=p0,
                bounds=([0, -720, 0], [np.inf, 720, np.inf]),
                maxfev=10000,
            )
        except Exception:
            break
        res = c - malus(a, *popt)
        sigma = res.std()
        nuevos_malos = np.abs(res) > umbral * sigma
        if not nuevos_malos.any():
            break
        indices_globales = np.where(mascara)[0]
        mascara[indices_globales[nuevos_malos]] = False

    n_rechazados = (~mascara).sum()
    print(f"\nRechazo {umbral}σ (iterativo): {n_rechazados} punto(s) "
          f"eliminado(s) de {len(angulos)}.")
    return mascara


# ── Promediado por bin angular ────────────────────────────────────────────────

def promediar_por_angulo(angulos, conteos, tolerancia_deg=0.5):
    """
    Agrupa mediciones en el mismo ángulo (dentro de `tolerancia_deg`) y
    calcula media y error estándar de la media para cada grupo.
    Devuelve (angulos_unicos, conteos_medio, errores).
    """
    angulos_unicos = np.unique(np.round(angulos / tolerancia_deg) * tolerancia_deg)
    medias, errores = [], []
    for a in angulos_unicos:
        idx = np.abs(angulos - a) <= tolerancia_deg / 2
        grupo = conteos[idx]
        medias.append(grupo.mean())
        sem = grupo.std() / np.sqrt(len(grupo)) if len(grupo) > 1 else np.sqrt(grupo.mean())
        errores.append(sem)
    return angulos_unicos, np.array(medias), np.array(errores)


# ── Carga de datos ────────────────────────────────────────────────────────────

def cargar_csv(ruta, usar_potencia):
    datos = np.genfromtxt(ruta, delimiter=",", names=True)
    nombres = datos.dtype.names
    angulos = datos["angulo_deg"].astype(float)
    conteos = datos["conteos"].astype(float)
    potencia = None
    if usar_potencia and "potencia_mW" in nombres:
        potencia = datos["potencia_mW"].astype(float)
        print("  Columna 'potencia_mW' encontrada — se usará para normalizar.")
    elif usar_potencia:
        print("  Columna 'potencia_mW' no encontrada — se omite normalización.")
    return angulos, conteos, potencia


def datos_sinteticos():
    """Genera datos sintéticos con ruido de potencia y outliers para demostración."""
    rng = np.random.default_rng(7)
    angulos = np.arange(0, 360, 5, dtype=float)
    A_true, theta0_true, B_true = 8000.0, 37.0, 120.0
    # Potencia con fluctuaciones del 4 %
    potencia = 0.05 * (1 + 0.04 * rng.standard_normal(len(angulos)))
    potencia = np.clip(potencia, 0.03, 0.08)
    conteos_ideales = malus(angulos, A_true, theta0_true, B_true)
    escala = potencia / potencia.mean()
    conteos = rng.poisson(conteos_ideales * escala).astype(float)
    # 5 outliers
    idx_out = rng.choice(len(conteos), size=5, replace=False)
    conteos[idx_out] *= rng.uniform(1.8, 3.5, size=5)
    print("Datos sintéticos: "
          f"A={A_true}, θ₀={theta0_true}°, B={B_true}, ruido_P≈4 %, 5 outliers.")
    return angulos, conteos, potencia


# ── Gráficas ──────────────────────────────────────────────────────────────────

def graficar(angulos_raw, conteos_raw, potencia,
             angulos_fil, conteos_fil, errores_fil,
             popt_raw, popt_fil):

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Ley de Malus — Conteo de fotones\n"
                 "Filtrado de ruido de fuente", fontsize=13, fontweight="bold")

    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)
    ax_crudo  = fig.add_subplot(gs[0, 0])
    ax_filtro = fig.add_subplot(gs[0, 1])
    ax_comp   = fig.add_subplot(gs[0, 2])
    ax_pot    = fig.add_subplot(gs[1, 0])
    ax_res    = fig.add_subplot(gs[1, 1])
    ax_hist   = fig.add_subplot(gs[1, 2])

    theta_fino = np.linspace(angulos_raw.min(), angulos_raw.max(), 1000)

    # ── Panel 1: datos crudos ─────────────────────────────────────────────────
    ax_crudo.scatter(angulos_raw, conteos_raw, s=14, alpha=0.55,
                     color=COLOR_CRUDO, label="Mediciones crudas")
    if popt_raw is not None:
        ax_crudo.plot(theta_fino, malus(theta_fino, *popt_raw),
                      color=COLOR_AJUSTE, lw=1.8,
                      label=f"cos² (θ₀={popt_raw[1]:.1f}°)")
    ax_crudo.set_xlabel("Ángulo [°]"); ax_crudo.set_ylabel("Conteos")
    ax_crudo.set_title("Sin filtrar")
    ax_crudo.legend(fontsize=7); ax_crudo.grid(True, alpha=0.3)

    # ── Panel 2: datos filtrados ──────────────────────────────────────────────
    ax_filtro.errorbar(angulos_fil, conteos_fil, yerr=errores_fil,
                       fmt="o", ms=4, color=COLOR_FILTRADO, alpha=0.8,
                       elinewidth=0.8, capsize=2, label="Filtrados")
    if popt_fil is not None:
        ax_filtro.plot(theta_fino, malus(theta_fino, *popt_fil),
                       color=COLOR_AJUSTE, lw=1.8,
                       label=f"cos² (θ₀={popt_fil[1]:.1f}°)")
    ax_filtro.set_xlabel("Ángulo [°]"); ax_filtro.set_ylabel("Conteos")
    ax_filtro.set_title("Filtrado + promediado")
    ax_filtro.legend(fontsize=7); ax_filtro.grid(True, alpha=0.3)

    # ── Panel 3: comparación superpuesta ─────────────────────────────────────
    ax_comp.scatter(angulos_raw, conteos_raw / conteos_raw.max(),
                    s=10, alpha=0.35, color=COLOR_CRUDO, label="Crudo (norm.)")
    ax_comp.scatter(angulos_fil, conteos_fil / conteos_fil.max(),
                    s=14, alpha=0.75, color=COLOR_FILTRADO, label="Filtrado (norm.)")
    if popt_fil is not None:
        y_fit = malus(theta_fino, *popt_fil)
        ax_comp.plot(theta_fino, y_fit / y_fit.max(),
                     color=COLOR_AJUSTE, lw=1.8, label="cos²")
    ax_comp.set_xlabel("Ángulo [°]"); ax_comp.set_ylabel("Conteos (norm.)")
    ax_comp.set_title("Comparación")
    ax_comp.legend(fontsize=7); ax_comp.grid(True, alpha=0.3)

    # ── Panel 4: potencia de fuente ───────────────────────────────────────────
    if potencia is not None:
        ax_pot.scatter(angulos_raw, potencia, s=12, alpha=0.65,
                       color=COLOR_RUIDO)
        p_med = potencia.mean()
        ax_pot.axhline(p_med, color="gray", ls="--", lw=0.9,
                       label=f"<P> = {p_med:.4f} mW")
        ax_pot.fill_between(
            [angulos_raw.min(), angulos_raw.max()],
            p_med - potencia.std(), p_med + potencia.std(),
            alpha=0.15, color="gray", label="±σ")
        ax_pot.set_xlabel("Ángulo [°]"); ax_pot.set_ylabel("Potencia [mW]")
        ax_pot.set_title("Potencia de fuente durante medición")
        ax_pot.legend(fontsize=7); ax_pot.grid(True, alpha=0.3)
    else:
        ax_pot.text(0.5, 0.5, "Sin datos\nde potencia",
                    transform=ax_pot.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")
        ax_pot.set_title("Potencia de fuente")

    # ── Panel 5: residuos del ajuste filtrado ─────────────────────────────────
    if popt_fil is not None:
        res = conteos_fil - malus(angulos_fil, *popt_fil)
        ax_res.scatter(angulos_fil, res, s=14, alpha=0.75,
                       color=COLOR_FILTRADO)
        ax_res.axhline(0, color="gray", ls="--", lw=0.9)
        sr = res.std()
        ax_res.axhline( sr, color="red", ls=":", lw=0.8, label=f"±σ = {sr:.1f}")
        ax_res.axhline(-sr, color="red", ls=":", lw=0.8)
        ax_res.set_xlabel("Ángulo [°]"); ax_res.set_ylabel("Residuo [conteos]")
        ax_res.set_title("Residuos — ajuste filtrado")
        ax_res.legend(fontsize=7); ax_res.grid(True, alpha=0.3)

    # ── Panel 6: histograma de residuos ──────────────────────────────────────
    if popt_fil is not None:
        ax_hist.hist(res, bins=15, color=COLOR_FILTRADO, alpha=0.7,
                     edgecolor="white", linewidth=0.4)
        ax_hist.axvline(0, color="gray", ls="--", lw=0.9)
        ax_hist.set_xlabel("Residuo [conteos]")
        ax_hist.set_ylabel("Frecuencia")
        ax_hist.set_title("Distribución de residuos")
        ax_hist.grid(True, alpha=0.3)

    ruta_fig = Path("malus_filtrado.png")
    plt.savefig(ruta_fig, dpi=150, bbox_inches="tight")
    print(f"\nGráfica guardada: {ruta_fig.resolve()}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Filtrado de ruido de fuente — Ley de Malus por conteo de fotones"
    )
    ap.add_argument(
        "datos", nargs="?", default=None,
        help="CSV con columnas angulo_deg, conteos [, potencia_mW]"
    )
    ap.add_argument(
        "--sin-potencia", action="store_true",
        help="Ignorar columna potencia_mW aunque exista"
    )
    ap.add_argument(
        "--umbral-sigma", type=float, default=UMBRAL_SIGMA_DEFAULT,
        help=f"Umbral de rechazo σ (default: {UMBRAL_SIGMA_DEFAULT})"
    )
    ap.add_argument(
        "--tolerancia-bin", type=float, default=0.5,
        help="Tolerancia en grados para agrupar ángulos iguales (default: 0.5)"
    )
    ap.add_argument(
        "--sin-grafica", action="store_true",
        help="No mostrar ventana de gráfica (solo guardar PNG)"
    )
    args = ap.parse_args()

    usar_potencia = not args.sin_potencia

    # ── Carga de datos ────────────────────────────────────────────────────────
    if args.datos is None:
        print("Sin archivo CSV — usando datos sintéticos de demostración.\n")
        angulos_raw, conteos_raw, potencia = datos_sinteticos()
    else:
        print(f"Cargando '{args.datos}' …")
        angulos_raw, conteos_raw, potencia = cargar_csv(args.datos, usar_potencia)
        print(f"  {len(angulos_raw)} puntos cargados.")

    angulos_raw = np.asarray(angulos_raw, dtype=float)
    conteos_raw = np.asarray(conteos_raw, dtype=float)

    # ── Ajuste sin filtrar ────────────────────────────────────────────────────
    popt_raw, _ = ajustar(angulos_raw, conteos_raw, "SIN FILTRAR")

    # ── Normalización por potencia ────────────────────────────────────────────
    conteos_norm = conteos_raw.copy()
    if potencia is not None and usar_potencia:
        potencia = np.asarray(potencia, dtype=float)
        conteos_norm = normalizar_por_potencia(conteos_raw, potencia)
    else:
        potencia = None   # asegura coherencia en gráficas

    # ── Rechazo σ ─────────────────────────────────────────────────────────────
    mascara = rechazo_sigma(angulos_raw, conteos_norm, args.umbral_sigma)
    ang_fil  = angulos_raw[mascara]
    cnt_fil  = conteos_norm[mascara]

    # ── Promediado por bin angular ────────────────────────────────────────────
    ang_prom, cnt_prom, err_prom = promediar_por_angulo(
        ang_fil, cnt_fil, tolerancia_deg=args.tolerancia_bin
    )
    print(f"\nPromediado por bin: {len(ang_prom)} ángulos únicos "
          f"(tolerancia {args.tolerancia_bin}°).")

    # ── Ajuste filtrado ───────────────────────────────────────────────────────
    popt_fil, _ = ajustar(ang_prom, cnt_prom, "FILTRADO + PROMEDIADO")

    # ── Gráficas ──────────────────────────────────────────────────────────────
    if args.sin_grafica:
        plt.switch_backend("Agg")
    graficar(angulos_raw, conteos_raw, potencia,
             ang_prom, cnt_prom, err_prom,
             popt_raw, popt_fil)


if __name__ == "__main__":
    main()

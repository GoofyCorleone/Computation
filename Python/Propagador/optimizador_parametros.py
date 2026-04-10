"""
Optimizador de parámetros — máxima diferencia HF − Fresnel
===========================================================

Problema
--------
Apertura vertical fija q = q₂ = 3 cm (régimen geométrico en y).
Encontrar (p, p₂, a, c, d₂) que maximicen

    score = ⟨|I_HF(norm) − I_F(norm)|⟩  ∈ [0, 1]

donde las intensidades están normalizadas al máximo global.

Parámetros de búsqueda
----------------------
  p    : ancho horizontal rendija 1  [m]
  p₂   : ancho horizontal rendija 2  [m]
  a    : posición x del centro de la rendija 2  [m]
  c    : separación longitudinal entre rendijas  [m]
  d₂   : distancia rendija 2 → plano de observación  [m]

Método
------
Evolución Diferencial (DE, scipy) + refinamiento Nelder-Mead.
workers=-1 → paralelismo completo (Apple Silicon M-series).

Restricciones numéricas
-----------------------
La cuadratura GL con N_quad puntos resuelve correctamente integrales
con N_F = (dim/2)² / (λ·z) ≲ N_F_MAX oscilaciones.
Pares fuera de este rango se descartan (score = 0).

Uso
---
    python optimizador_parametros.py
"""

import json
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, minimize

from simulador_difraccion import DifraccionDosRendijas

# ── Constantes ──────────────────────────────────────────────────────────
LAMBDA  = DifraccionDosRendijas.LAMBDA   # 632.8 nm
Q_FIJA  = DifraccionDosRendijas.Q_FIJA   # 3 cm  (fijo)
N_FIJA  = 10e-3                          # posición z rendija 1 [m]

# ── Resolución durante la optimización ──────────────────────────────────
N_OBS_OPT  = 128
N_QUAD_OPT = 40

# ── Restricción numérica de Fresnel (eje x) ─────────────────────────────
N_F_MAX = 12.0

# ── Resolución alta para validación final ───────────────────────────────
N_OBS_VAL  = 256
N_QUAD_VAL = 60

# ── Espacio de búsqueda ──────────────────────────────────────────────────
NAMES = ["p", "p2", "a", "c", "d2"]
UNITS = ["mm"] * 5

BOUNDS = [
    (0.04e-3,  0.80e-3),   # p   [m]  ancho horizontal rendija 1
    (0.04e-3,  0.80e-3),   # p2  [m]  ancho horizontal rendija 2
    (0.00e-3,  3.00e-3),   # a   [m]  posición x rendija 2 (relativa a rendija 1)
    (2.00e-3, 80.00e-3),   # c   [m]  separación longitudinal entre rendijas
    (5.00e-3, 200.0e-3),   # d2  [m]  distancia rendija 2 → plano de observación
]


# ══════════════════════════════════════════════════════════════════════════
# Funciones auxiliares
# ══════════════════════════════════════════════════════════════════════════

def _fresnel_number(half_aperture, z):
    """N_F = (a/2)² / (λ·z)."""
    return half_aperture ** 2 / (LAMBDA * z)


def _es_valido(p, p2, a, c, d2):
    """
    True si los parámetros satisfacen:
    1. N_F_x ≤ N_F_MAX  (precisión numérica en eje x)
    2. c > 0, d2 > 0
    """
    if c <= 0 or d2 <= 0:
        return False
    nf1 = _fresnel_number(p / 2,  c)
    nf2 = _fresnel_number(p2 / 2, d2)
    return nf1 <= N_F_MAX and nf2 <= N_F_MAX


def _ventana(p, p2, a, d2):
    """
    Ventana de observación adaptativa en x.
    Cubre el lóbulo principal de difracción más el desplazamiento a.
    """
    spread = LAMBDA * d2 / max(p2, 1e-9)
    half_x = 3.0 * spread + abs(a) + max(p, p2)
    return float(np.clip(2.0 * half_x, 1.0e-3, 20e-3))


def _calcular_score(params, N_obs, N_quad):
    """Construye el simulador y devuelve el score."""
    p, p2, a, c, d2 = params
    x_range = _ventana(p, p2, a, d2)

    sim = DifraccionDosRendijas(
        p=p, p2=p2, a=a, c=c, d2=d2,
        n=N_FIJA, N_obs=N_obs, N_quad=N_quad, x_range=x_range,
    )
    I_hf  = np.abs(sim.calcular_hf()) ** 2
    I_f   = np.abs(sim.calcular_fresnel()) ** 2

    max_val = max(I_hf.max(), I_f.max())
    if max_val < 1e-50:
        return 0.0

    return float(np.mean(np.abs(I_hf / max_val - I_f / max_val)))


# ══════════════════════════════════════════════════════════════════════════
# Función objetivo (module-level para pickle en multiprocessing)
# ══════════════════════════════════════════════════════════════════════════

def objetivo(params):
    """Devuelve −score (DE minimiza → maximizamos la diferencia)."""
    p, p2, a, c, d2 = params
    if not _es_valido(p, p2, a, c, d2):
        return 0.0
    try:
        return -_calcular_score(params, N_OBS_OPT, N_QUAD_OPT)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════
# Callback de progreso
# ══════════════════════════════════════════════════════════════════════════

class _Progreso:
    def __init__(self):
        self.historial = []
        self.t0        = time.time()

    def __call__(self, intermediate_result):
        if hasattr(intermediate_result, "fun"):
            score = -float(intermediate_result.fun)
        else:
            score = -objetivo(np.asarray(intermediate_result))
        self.historial.append(score)
        elapsed = time.time() - self.t0
        gen     = len(self.historial)
        print(
            f"  gen {gen:>4d}  |  score = {score:.6f}"
            f"  |  {elapsed:6.1f} s",
            flush=True,
        )


# ══════════════════════════════════════════════════════════════════════════
# Optimización principal
# ══════════════════════════════════════════════════════════════════════════

def optimizar(popsize=12, maxiter=120, seed=42, workers=-1):
    """
    Búsqueda global (DE) + refinamiento local (Nelder-Mead).

    Parámetros
    ----------
    popsize  : factor de tamaño de población (total = popsize × 5)
    maxiter  : generaciones máximas de DE
    seed     : semilla para reproducibilidad
    workers  : núcleos (-1 = todos)

    Retorna
    -------
    params_opt : ndarray  (p, p2, a, c, d2) en metros
    score_opt  : float
    historial  : list — mejor score por generación
    """
    n = len(BOUNDS)
    pop_total = popsize * n

    print("=" * 64)
    print("  Optimizador: máxima diferencia HF − Fresnel (1D en x)")
    print(f"  Apertura vertical fija: q = q₂ = {Q_FIJA*1e2:.0f} cm")
    print(f"  Parámetros: {NAMES}")
    print(f"  Población total: {pop_total}   Generaciones: {maxiter}")
    print(f"  N_obs = {N_OBS_OPT},  N_quad = {N_QUAD_OPT},  N_F_max = {N_F_MAX}")
    print(f"  Workers: {workers}")
    print("=" * 64)

    progreso = _Progreso()
    t0 = time.time()

    res = differential_evolution(
        objetivo,
        bounds       = BOUNDS,
        seed         = seed,
        popsize      = popsize,
        maxiter      = maxiter,
        mutation     = (0.5, 1.0),
        recombination= 0.9,
        tol          = 1e-6,
        polish       = False,
        callback     = progreso,
        disp         = False,
        workers      = workers,
        updating     = "deferred",
    )
    t_de = time.time() - t0
    score_de = -res.fun
    print(f"\n  DE completado en {t_de:.1f} s   score = {score_de:.6f}")

    # ── Nelder-Mead ────────────────────────────────────────────────────
    print("  Refinando con Nelder-Mead ...", flush=True)
    t1 = time.time()
    res_nm = minimize(
        objetivo,
        x0     = res.x,
        method = "Nelder-Mead",
        options = {"maxiter": 3000, "xatol": 1e-9, "fatol": 1e-8, "disp": False},
    )
    t_nm = time.time() - t1

    if res_nm.fun < res.fun:
        params_opt = res_nm.x
        score_opt  = -res_nm.fun
    else:
        params_opt = res.x
        score_opt  = score_de

    print(f"  Nelder-Mead en {t_nm:.1f} s   score = {score_opt:.6f}")

    # ── Resumen ────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  RESULTADO ÓPTIMO   score = {score_opt:.6f}")
    print(f"{'='*64}")
    p, p2, a, c, d2 = params_opt
    nf_p  = _fresnel_number(p  / 2, c)
    nf_p2 = _fresnel_number(p2 / 2, d2)
    print(f"    p  = {p*1e3:8.4f} mm   (N_F = {nf_p:.2f})")
    print(f"    p2 = {p2*1e3:8.4f} mm   (N_F = {nf_p2:.2f})")
    print(f"    a  = {a*1e3:8.4f} mm")
    print(f"    c  = {c*1e3:8.4f} mm")
    print(f"    d2 = {d2*1e3:8.4f} mm")
    print()
    return params_opt, score_opt, progreso.historial


# ══════════════════════════════════════════════════════════════════════════
# Validación con alta resolución
# ══════════════════════════════════════════════════════════════════════════

def validar(params, show_plots=True, guardar=None):
    """
    Recalcula con alta resolución y genera la figura comparativa.

    Parámetros
    ----------
    params      : (p, p2, a, c, d2) en metros
    show_plots  : bool — mostrar ventana de matplotlib
    guardar     : str o None — ruta PNG donde guardar la figura
    """
    p, p2, a, c, d2 = params
    x_range = _ventana(p, p2, a, d2)

    print(f"\n  Validando con N_obs={N_OBS_VAL}, N_quad={N_QUAD_VAL} ...")
    sim = DifraccionDosRendijas(
        p=p, p2=p2, a=a, c=c, d2=d2,
        n=N_FIJA, N_obs=N_OBS_VAL, N_quad=N_QUAD_VAL, x_range=x_range,
    )
    score = sim.run(guardar_imagen=guardar)
    if show_plots:
        plt.show()
    return score


# ══════════════════════════════════════════════════════════════════════════
# Búsqueda de familia de soluciones
# ══════════════════════════════════════════════════════════════════════════

def buscar_familia(n_soluciones=6, seeds=None, popsize=12, maxiter=80):
    """
    Ejecuta la optimización con distintas semillas para obtener una
    familia de configuraciones que maximizan la diferencia HF − Fresnel.

    Parámetros
    ----------
    n_soluciones : número de semillas a probar
    seeds        : lista de semillas (por defecto 0..n_soluciones-1)
    popsize      : tamaño de población por semilla
    maxiter      : generaciones por semilla

    Retorna
    -------
    familia : lista de dict {params, score}  ordenada de mayor a menor score
    """
    if seeds is None:
        seeds = list(range(n_soluciones))

    familia = []
    for i, s in enumerate(seeds):
        print(f"\n{'─'*50}")
        print(f"  Semilla {s}  ({i+1}/{len(seeds)})")
        print(f"{'─'*50}")
        try:
            params, score, _ = optimizar(popsize=popsize, maxiter=maxiter,
                                         seed=s, workers=-1)
            familia.append({"params": params, "score": score, "semilla": s})
        except Exception as e:
            print(f"  Error con semilla {s}: {e}")

    familia.sort(key=lambda x: x["score"], reverse=True)
    print(f"\n{'='*64}")
    print(f"  FAMILIA DE SOLUCIONES  (ordenadas por score)")
    print(f"{'='*64}")
    for i, sol in enumerate(familia):
        p, p2, a, c, d2 = sol["params"]
        print(f"  [{i+1}] score={sol['score']:.6f}  semilla={sol['semilla']}")
        print(f"       p={p*1e3:.4f} mm  p2={p2*1e3:.4f} mm  "
              f"a={a*1e3:.4f} mm  c={c*1e3:.3f} mm  d2={d2*1e3:.3f} mm")
    return familia


# ══════════════════════════════════════════════════════════════════════════
# Persistencia
# ══════════════════════════════════════════════════════════════════════════

def guardar_json(params, score, archivo="parametros_optimos.json"):
    p, p2, a, c, d2 = params
    datos = {
        "p_mm":  round(float(p)  * 1e3, 6),
        "p2_mm": round(float(p2) * 1e3, 6),
        "a_mm":  round(float(a)  * 1e3, 6),
        "c_mm":  round(float(c)  * 1e3, 6),
        "d2_mm": round(float(d2) * 1e3, 6),
        "q_fija_cm": float(Q_FIJA * 1e2),
        "score":     float(score),
        "unidades":  "mm (excepto q_fija_cm [cm] y score [adim])",
        "N_F_max":   N_F_MAX,
        "N_quad_opt": N_QUAD_OPT,
    }
    with open(archivo, "w") as f:
        json.dump(datos, f, indent=2)
    print(f"  Parámetros guardados en '{archivo}'")


def cargar_json(archivo="parametros_optimos.json"):
    with open(archivo) as f:
        datos = json.load(f)
    params = np.array([
        datos["p_mm"]  * 1e-3,
        datos["p2_mm"] * 1e-3,
        datos["a_mm"]  * 1e-3,
        datos["c_mm"]  * 1e-3,
        datos["d2_mm"] * 1e-3,
    ])
    return params, datos["score"]


# ══════════════════════════════════════════════════════════════════════════
# Visualización de la convergencia
# ══════════════════════════════════════════════════════════════════════════

def plot_convergencia(historial, params_opt, guardar=None):
    """Figura 2 paneles: convergencia DE + barras de parámetros."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Optimización — max diferencia HF − Fresnel   (q = q₂ = 3 cm)",
        fontsize=13,
    )

    # convergencia
    ax = axes[0]
    gens = np.arange(1, len(historial) + 1)
    ax.plot(gens, historial, lw=1.5, color="steelblue")
    ax.fill_between(gens, historial, alpha=0.2, color="steelblue")
    ax.set_xlabel("Generación DE")
    ax.set_ylabel(r"Score $\langle|I_\mathrm{HF} - I_\mathrm{F}|\rangle$")
    ax.set_title("Convergencia de la búsqueda global")
    ax.grid(alpha=0.3)

    # barras normalizadas
    ax2 = axes[1]
    etiquetas = [f"{n}\n({v*1e3:.3f} mm)" for n, v in zip(NAMES, params_opt)]
    vals_norm = [(v - lo) / (hi - lo) for v, (lo, hi) in zip(params_opt, BOUNDS)]
    colores   = plt.cm.viridis(np.linspace(0.15, 0.85, len(NAMES)))
    bars = ax2.barh(etiquetas, vals_norm, color=colores, edgecolor="white")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Posición relativa en el rango  [0 = mín, 1 = máx]")
    ax2.set_title("Parámetros óptimos")
    ax2.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, vals_norm):
        ax2.text(
            min(val + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=8,
        )

    fig.tight_layout()
    if guardar:
        fig.savefig(guardar, dpi=150, bbox_inches="tight")
        print(f"  Figura de convergencia guardada en '{guardar}'")
    return fig


# ══════════════════════════════════════════════════════════════════════════
# Punto de entrada
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # ── 1. Búsqueda global + refinamiento ─────────────────────────────
    params_opt, score_opt, historial = optimizar(
        popsize = 12,    # población = 12 × 5 = 60 individuos
        maxiter = 120,   # 120 generaciones  →  ~7 320 evaluaciones
        seed    = 42,
        workers = -1,
    )

    # ── 2. Guardar resultado ──────────────────────────────────────────
    guardar_json(params_opt, score_opt)

    # ── 3. Figura de convergencia ─────────────────────────────────────
    plot_convergencia(historial, params_opt, guardar="convergencia_optimizacion.png")

    # ── 4. Validación con alta resolución (genera figura y la guarda) ─
    validar(params_opt, show_plots=False, guardar="simulacion_optima.png")

    # ── 5. Familia de soluciones (3 semillas adicionales) ─────────────
    familia = buscar_familia(
        n_soluciones=3,
        seeds=[1, 7, 13],
        popsize=10,
        maxiter=80,
    )

    # Guardar familia en JSON
    familia_datos = []
    for sol in familia:
        p, p2, a, c, d2 = sol["params"]
        familia_datos.append({
            "semilla": int(sol["semilla"]),
            "score":   float(sol["score"]),
            "p_mm":    round(float(p)  * 1e3, 6),
            "p2_mm":   round(float(p2) * 1e3, 6),
            "a_mm":    round(float(a)  * 1e3, 6),
            "c_mm":    round(float(c)  * 1e3, 6),
            "d2_mm":   round(float(d2) * 1e3, 6),
        })
    with open("familia_parametros.json", "w") as f:
        json.dump(familia_datos, f, indent=2)
    print("  Familia guardada en 'familia_parametros.json'")

    plt.show()

"""
Optimizador de parámetros para maximizar la diferencia
entre la integral de Huyghens-Fresnel y la de Fresnel
======================================================

Problema
--------
Encontrar (p, q, p2, q2, a, b, c, d2) que maximicen

    score = mean_píxeles |I_HF(norm) − I_F(norm)|  ∈ [0, 1]

donde I_HF e I_F están normalizadas al máximo de I_HF.

Método
------
Evolución Diferencial (DE, scipy) + refinamiento Nelder-Mead.

  * DE: búsqueda global, robusta ante mínimos locales, sin gradiente.
    Con workers=-1 evalúa la población entera en paralelo (Pool).
  * Nelder-Mead: refinamiento local desde la mejor solución de DE.

¿Por qué no Bayesiana (GP)?
  Para funciones que tardan ~15-30 ms, DE con ~3 000-5 000 evaluaciones
  converge en 1-3 min usando todos los núcleos del M-series Mac,
  explorando el espacio mucho más densamente que un GP con 150 puntos.

Restricción de precisión numérica
----------------------------------
La cuadratura GL con N_quad = 30 puntos resuelve correctamente
integrales con hasta N_F ≈ 12 oscilaciones en el integrando
(Número de Fresnel N_F = (a/2)² / (λ z)).  Pares (p, c) o (p2, d2)
con N_F > N_F_MAX se rechazan silenciosamente (score = 0).

Uso
---
    .venv/bin/python optimizador_parametros.py

Para usar con varios núcleos en Mac se necesita el bloque
`if __name__ == "__main__"` (que ya está al final del archivo).
"""

import json
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, minimize

from simulador_difraccion import DifraccionDosRendijas

# ── Constantes físicas ──────────────────────────────────────────────────
LAMBDA   = DifraccionDosRendijas.LAMBDA   # 632.8 nm
N_FIXED  = 10e-3                          # posición z rendija 1 [m] (fija)

# ── Resolución durante la optimización (rápida) ─────────────────────────
N_GRID_OPT = 48     # puntos en cada eje del plano de observación
N_QUAD_OPT = 30     # puntos GL por dimensión

# ── Restricción de oscilaciones (garantiza precisión con N_QUAD_OPT=30) ─
# N_F = (dim/2)² / (λ · z) < N_F_MAX  →  ≲ 9 oscilaciones en el integrando
N_F_MAX = 12.0

# ── Restricción de resolvibilidad del patrón ────────────────────────────
# El lóbulo principal λ·d2/p2 debe ser ≥ MIN_PIXELS_LOBE × pixel_size
# para que la métrica capture física real y no artefactos de discretización.
MIN_PIXELS_LOBE = 2.0

# ── Resolución alta para la validación final ────────────────────────────
N_GRID_VAL = 72
N_QUAD_VAL = 45

# ── Nombres de parámetros ───────────────────────────────────────────────
NAMES  = ["p", "q", "p2", "q2", "a", "b", "c", "d2"]
UNITS  = ["mm"] * 8

# ── Límites del espacio de búsqueda ─────────────────────────────────────
# p, q, p2, q2 limitados a 0.35 mm para que el lóbulo de difracción
# λ·d2/p2 ≥ ~2.5 píxeles en la grilla de optimización (N_grid=48)
# y así evitar artefactos de muestreo en la métrica.
BOUNDS = [
    (0.05e-3,  0.35e-3),   # p   [m]
    (0.05e-3,  0.35e-3),   # q   [m]
    (0.05e-3,  0.35e-3),   # p2  [m]
    (0.05e-3,  0.35e-3),   # q2  [m]
    (0.00e-3,  2.50e-3),   # a   [m]  desplazamiento x rendija 2
    (0.00e-3,  1.20e-3),   # b   [m]  desplazamiento y rendija 2
    (2.00e-3, 60.00e-3),   # c   [m]  separación entre rendijas
    (3.00e-3,  80.0e-3),   # d2  [m]  distancia rendija 2 → observación
]


# ══════════════════════════════════════════════════════════════════════════
# Funciones auxiliares
# ══════════════════════════════════════════════════════════════════════════

def _fresnel_number(half_aperture, z):
    """Número de Fresnel N_F = (a/2)² / (λ·z)."""
    return half_aperture ** 2 / (LAMBDA * z)


def _es_valido(p, q, p2, q2, a, b, c, d2):
    """
    True si los parámetros satisfacen dos condiciones:

    1. Precisión numérica : N_F = (dim/2)²/(λ·z) ≤ N_F_MAX
       Garantiza que el integrando no oscile más veces de lo que
       N_QUAD_OPT puntos GL pueden resolver.

    2. Resolvibilidad del patrón : el lóbulo de difracción principal
       de la rendija 2 (semianchura λ·d2/p2) debe cubrir al menos
       MIN_PIXELS_LOBE píxeles en la grilla de optimización.
       Si es subpíxel, el score dependería de artefactos de muestreo,
       no de física real.
    """
    # --- condición 1: números de Fresnel ---
    nf1 = max(_fresnel_number(p / 2, c),   _fresnel_number(q / 2, c))
    nf2 = max(_fresnel_number(p2 / 2, d2), _fresnel_number(q2 / 2, d2))
    if nf1 > N_F_MAX or nf2 > N_F_MAX:
        return False

    # --- condición 2: patrón resolvible en la grilla de optimización ---
    x_range, y_range = _ventana(p, q, p2, q2, a, b, d2)
    pixel_x = x_range / N_GRID_OPT
    pixel_y = y_range / N_GRID_OPT
    lobe_x  = LAMBDA * d2 / p2
    lobe_y  = LAMBDA * d2 / q2
    if lobe_x < MIN_PIXELS_LOBE * pixel_x or lobe_y < MIN_PIXELS_LOBE * pixel_y:
        return False

    return True


def _ventana(p, q, p2, q2, a, b, d2):
    """
    Calcula la ventana de observación adaptativa en x e y.

    La ventana se basa en el spread del SEGUNDO paso de propagación
    (rendija 2 → plano de observación), que es el que define la escala
    del patrón observable.  Incluye el desplazamiento geométrico (a, b)
    para que el patrón siempre quede dentro del plano de observación.
    """
    # Lóbulo principal de difracción desde la rendija 2 al plano de observación
    spread_x = LAMBDA * d2 / max(p2, 1e-9)
    spread_y = LAMBDA * d2 / max(q2, 1e-9)

    # Ventana simétrica que cubre el origen (x=0) y el centro de la rendija 2 (x=a)
    # con margen = 3 × spread
    half_x = 3.0 * spread_x + abs(a) + max(p, p2)
    half_y = 3.0 * spread_y + abs(b) + max(q, q2)

    x_range = float(np.clip(2.0 * half_x, 2.0e-3, 12e-3))
    y_range = float(np.clip(2.0 * half_y, 2.0e-3, 12e-3))
    return x_range, y_range


def _calcular_score(params, N_grid, N_quad):
    """
    Construye el simulador, calcula ambos campos y devuelve el score.
    Se llama tanto durante la optimización (baja resolución) como
    en la validación final (alta resolución).
    """
    p, q, p2, q2, a, b, c, d2 = params
    z0 = N_FIXED + c + d2
    x_range, y_range = _ventana(p, q, p2, q2, a, b, d2)

    sim = DifraccionDosRendijas(
        p=p, q=q, p2=p2, q2=q2,
        a=a, b=b, c=c, z0=z0, n=N_FIXED,
        N_grid=N_grid, N_quad=N_quad,
        x_range=x_range, y_range=y_range,
    )
    I_hf  = np.abs(sim._hf.calcular()) ** 2
    I_f   = np.abs(sim._fres.calcular()) ** 2

    # Normalizar al mayor de los dos máximos → score ∈ [0, 1].
    max_val = max(I_hf.max(), I_f.max())
    if max_val < 1e-50:
        return 0.0   # ambos patrones ausentes: configuración inválida

    score = float(np.mean(np.abs(I_hf / max_val - I_f / max_val)))
    return score


# ══════════════════════════════════════════════════════════════════════════
# Función objetivo  (debe ser module-level para que multiprocessing la
# pueda serializar con pickle cuando workers=-1)
# ══════════════════════════════════════════════════════════════════════════

def objetivo(params):
    """
    Devuelve −score para que differential_evolution minimice.
    Devuelve 0 (= score 0) si los parámetros son inválidos o
    generan una excepción numérica.
    """
    p, q, p2, q2, a, b, c, d2 = params
    if not _es_valido(p, q, p2, q2, a, b, c, d2):
        return 0.0
    try:
        return -_calcular_score(params, N_GRID_OPT, N_QUAD_OPT)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════
# Callback de progreso
# ══════════════════════════════════════════════════════════════════════════

class _Progreso:
    """Registra el mejor score en cada generación de DE."""

    def __init__(self):
        self.historial  = []   # mejor score por generación
        self.t_inicio   = time.time()

    def __call__(self, intermediate_result):
        """
        scipy >= 1.11 pasa un OptimizeResult;
        versiones anteriores pasan (xk, convergence).
        """
        if hasattr(intermediate_result, "fun"):
            score = -float(intermediate_result.fun)
        else:
            # fallback: intermediate_result es xk
            score = -objetivo(np.asarray(intermediate_result))

        self.historial.append(score)
        elapsed = time.time() - self.t_inicio
        gen     = len(self.historial)
        print(
            f"  gen {gen:>4d}  |  score = {score:.6f}"
            f"  |  {elapsed:6.1f} s",
            flush=True,
        )


# ══════════════════════════════════════════════════════════════════════════
# Optimización principal
# ══════════════════════════════════════════════════════════════════════════

def optimizar(
    popsize  = 10,
    maxiter  = 100,
    seed     = 42,
    workers  = -1,
):
    """
    Ejecuta la optimización global (DE) seguida de refinamiento local.

    Parámetros
    ----------
    popsize  : factor del tamaño de población  (total = popsize × n_params)
    maxiter  : generaciones máximas de DE
    seed     : semilla para reproducibilidad
    workers  : núcleos a usar (-1 = todos)

    Retorna
    -------
    params_opt : ndarray  — parámetros óptimos encontrados
    score_opt  : float    — score en esos parámetros (N_grid_OPT, N_quad_OPT)
    historial  : list     — mejor score por generación de DE
    """
    n_params   = len(BOUNDS)
    pop_size   = popsize * n_params
    n_eval_est = pop_size * (maxiter + 1)

    print("=" * 64)
    print("  Optimización: max diferencia Huyghens-Fresnel vs Fresnel")
    print(f"  Método        : Evolución Diferencial + Nelder-Mead")
    print(f"  Parámetros    : {n_params}   (p, q, p2, q2, a, b, c, d2)")
    print(f"  Población     : {pop_size}  (popsize={popsize} × {n_params})")
    print(f"  Generaciones  : {maxiter}")
    print(f"  Evaluaciones  : ~{n_eval_est:,}")
    print(f"  Resolución    : N_grid={N_GRID_OPT}, N_quad={N_QUAD_OPT}")
    print(f"  N_F_max       : {N_F_MAX}  (restricción numérica)")
    print(f"  Workers       : {workers}")
    print("=" * 64)

    progreso = _Progreso()

    t0  = time.time()
    res = differential_evolution(
        objetivo,
        bounds       = BOUNDS,
        seed         = seed,
        popsize      = popsize,
        maxiter      = maxiter,
        mutation     = (0.5, 1.0),
        recombination= 0.9,
        tol          = 1e-6,
        polish       = False,   # haremos el pulido con Nelder-Mead
        callback     = progreso,
        disp         = False,
        workers      = workers,
        updating     = "deferred",  # necesario para workers != 1
    )
    t_de = time.time() - t0

    score_de = -res.fun
    print(f"\n  DE completado en {t_de:.1f} s  |  score = {score_de:.6f}")

    # ── Refinamiento local (Nelder-Mead) ──────────────────────────────────
    print("  Refinando con Nelder-Mead ...", flush=True)
    t1 = time.time()
    res_nm = minimize(
        objetivo,
        x0     = res.x,
        method = "Nelder-Mead",
        options = {
            "maxiter": 2000,
            "xatol"  : 1e-9,
            "fatol"  : 1e-8,
            "disp"   : False,
        },
    )
    t_nm = time.time() - t1

    # Tomar la mejor de las dos soluciones
    if res_nm.fun < res.fun:
        params_opt = res_nm.x
        score_opt  = -res_nm.fun
    else:
        params_opt = res.x
        score_opt  = score_de

    print(f"  Nelder-Mead en {t_nm:.1f} s  |  score = {score_opt:.6f}")

    # ── Resumen ──────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  RESULTADO ÓPTIMO   score = {score_opt:.6f}")
    print(f"{'='*64}")
    for name, val in zip(NAMES, params_opt):
        nf_str = ""
        if name == "p" or name == "q":
            nf = _fresnel_number(val / 2, params_opt[NAMES.index("c")])
            nf_str = f"  (N_F = {nf:.2f})"
        elif name == "p2" or name == "q2":
            nf = _fresnel_number(val / 2, params_opt[NAMES.index("d2")])
            nf_str = f"  (N_F = {nf:.2f})"
        print(f"    {name:>4s} = {val * 1e3:9.4f} mm{nf_str}")
    print()

    return params_opt, score_opt, progreso.historial


# ══════════════════════════════════════════════════════════════════════════
# Validación con alta resolución
# ══════════════════════════════════════════════════════════════════════════

def validar(params, show_plots=True):
    """
    Recalcula el score con alta resolución y, opcionalmente,
    lanza los plots completos del simulador.
    """
    print(f"\n  Validando con N_grid={N_GRID_VAL}, N_quad={N_QUAD_VAL} ...")
    score_val = _calcular_score(params, N_GRID_VAL, N_QUAD_VAL)
    print(f"  Score validado : {score_val:.6f}")

    if show_plots:
        p, q, p2, q2, a, b, c, d2 = params
        z0 = N_FIXED + c + d2
        x_range, y_range = _ventana(p, q, p2, q2, a, b, d2)

        sim = DifraccionDosRendijas(
            p=p, q=q, p2=p2, q2=q2,
            a=a, b=b, c=c, z0=z0, n=N_FIXED,
            N_grid=N_GRID_VAL, N_quad=N_QUAD_VAL,
            x_range=x_range, y_range=y_range,
        )
        sim.run()

    return score_val


# ══════════════════════════════════════════════════════════════════════════
# Persistencia
# ══════════════════════════════════════════════════════════════════════════

def guardar(params, score, archivo="parametros_optimos.json"):
    datos = {
        name: round(float(val) * 1e3, 6)   # en mm
        for name, val in zip(NAMES, params)
    }
    datos["score"]      = float(score)
    datos["unidades"]   = "mm (excepto score)"
    datos["N_F_max"]    = N_F_MAX
    datos["N_quad_opt"] = N_QUAD_OPT
    with open(archivo, "w") as f:
        json.dump(datos, f, indent=2)
    print(f"  Guardado en '{archivo}'")


def cargar(archivo="parametros_optimos.json"):
    with open(archivo) as f:
        datos = json.load(f)
    params = np.array([datos[n] * 1e-3 for n in NAMES])  # mm → m
    return params, datos["score"]


# ══════════════════════════════════════════════════════════════════════════
# Plots de la optimización
# ══════════════════════════════════════════════════════════════════════════

def plot_convergencia(historial, params_opt):
    """Figura de 2 paneles: convergencia DE + radar de parámetros óptimos."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Optimización: máxima diferencia HF − Fresnel", fontsize=13)

    # Panel 1 — Evolución del score
    ax = axes[0]
    generaciones = np.arange(1, len(historial) + 1)
    ax.plot(generaciones, historial, lw=1.5, color="steelblue")
    ax.fill_between(generaciones, historial, alpha=0.2, color="steelblue")
    ax.set_xlabel("Generación DE")
    ax.set_ylabel(r"Score $\langle|I_\mathrm{HF} - I_\mathrm{F}|\rangle$")
    ax.set_title("Convergencia de la búsqueda global")
    ax.grid(alpha=0.3)

    # Panel 2 — Barras de los parámetros óptimos
    ax2 = axes[1]
    etiquetas = [f"{n}\n({v*1e3:.3f} mm)" for n, v in zip(NAMES, params_opt)]
    # Normalizar cada parámetro en su rango
    vals_norm = [(v - lo) / (hi - lo) for v, (lo, hi) in zip(params_opt, BOUNDS)]
    colores   = plt.cm.viridis(np.linspace(0.15, 0.85, len(NAMES)))
    bars = ax2.barh(etiquetas, vals_norm, color=colores, edgecolor="white")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Posición relativa en el rango de búsqueda  [0 = mín, 1 = máx]")
    ax2.set_title("Parámetros óptimos")
    ax2.grid(axis="x", alpha=0.3)

    # Añadir etiquetas con los valores numéricos
    for bar, val in zip(bars, vals_norm):
        ax2.text(
            min(val + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=8,
        )

    fig.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
# Punto de entrada
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Guardia obligatoria para multiprocessing en macOS (spawn) ─────────
    import multiprocessing
    multiprocessing.freeze_support()

    params_opt, score_opt, historial = optimizar(
        popsize = 10,     # población = 10 × 8 = 80 individuos
        maxiter = 100,    # 100 generaciones  →  ~8 080 evaluaciones
        seed    = 42,
        workers = -1,     # todos los núcleos del Mac (M-series: 10+)
    )

    guardar(params_opt, score_opt)
    plot_convergencia(historial, params_opt)
    validar(params_opt, show_plots=True)

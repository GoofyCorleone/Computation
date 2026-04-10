"""
SENSOR DE FRENTE DE ONDA SHACK-HARTMANN — MÓDULO DE ANÁLISIS
================================================================================
Reconstrucción del frente de onda, espectro de polinomios de Zernike y
caracterización de aberraciones ópticas primarias (Seidel).

Convención de polinomios: OSA/ANSI (índice j, columnas Order=n, Frequency=m).
Los archivos CSV de Thorlabs WFS exportan los coeficientes en este orden.

Unidades: µm para aberraciones, mm para coordenadas de pupila.

Uso rápido
----------
    from SensorWavefront import SensorShackHartmann
    sensor = SensorShackHartmann('lenteThorlabs.csv', guardar=True)
    resultados = sensor.analizar()
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from scipy.special import factorial as _fact
import chardet
import os
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Polinomios de Zernike  (convención OSA/ANSI, normalizados sobre pupila)
# ─────────────────────────────────────────────────────────────────────────────

def _radial(n, m, rho):
    """Polinomio radial R_n^|m|(ρ)."""
    m_abs = abs(m)
    R = np.zeros_like(rho, dtype=np.float64)
    for s in range((n - m_abs) // 2 + 1):
        coef = ((-1) ** s
                * int(_fact(n - s, exact=True))
                // (int(_fact(s, exact=True))
                    * int(_fact((n + m_abs) // 2 - s, exact=True))
                    * int(_fact((n - m_abs) // 2 - s, exact=True))))
        R += coef * rho ** (n - 2 * s)
    return R


def zernike_nm(n, m, rho, theta):
    """
    Polinomio de Zernike normalizado Z_n^m(ρ,θ)  — convención OSA/ANSI.

    Normalización:  ∫_pupila |Z|² dA / π = 1   (RMS unitario sobre pupila circular).

    Parámetros
    ----------
    n     : int  — orden radial  (n ≥ 0)
    m     : int  — frecuencia azimutal  (−n ≤ m ≤ n, pasos de 2)
    rho   : ndarray — radio normalizado [0, 1]
    theta : ndarray — ángulo [rad]
    """
    N = np.sqrt(2 * (n + 1)) if m != 0 else np.sqrt(n + 1)
    R = _radial(n, m, rho)
    if m > 0:
        return N * R * np.cos(m * theta)
    elif m < 0:
        return N * R * np.sin(-m * theta)
    else:
        return N * R


def nombre_zernike(n, m):
    """Nombre físico del término (n, m) en español."""
    _tabla = {
        (0,  0): "Pistón",
        (1, -1): "Inclinación Y (Tilt)",
        (1,  1): "Inclinación X (Tip)",
        (2, -2): "Astigmatismo 45°",
        (2,  0): "Desenfoque",
        (2,  2): "Astigmatismo 0°",
        (3, -3): "Trébol Y",
        (3, -1): "Coma Y",
        (3,  1): "Coma X",
        (3,  3): "Trébol X",
        (4, -4): "Tetratrébol Y",
        (4, -2): "Astigmatismo 2.° 45°",
        (4,  0): "Esférica Primaria",
        (4,  2): "Astigmatismo 2.° 0°",
        (4,  4): "Tetratrébol X",
        (5, -5): "Pentafolio Y",
        (5, -3): "Trébol 2.° Y",
        (5, -1): "Coma 2.ª Y",
        (5,  1): "Coma 2.ª X",
        (5,  3): "Trébol 2.° X",
        (5,  5): "Pentafolio X",
        (6,  0): "Esférica Secundaria",
    }
    return _tabla.get((n, m), f"Z({n},{m:+d})")


# ─────────────────────────────────────────────────────────────────────────────
# Clase principal
# ─────────────────────────────────────────────────────────────────────────────

class SensorShackHartmann:
    """
    Análisis completo de aberraciones ópticas para datos de sensor Shack-Hartmann
    (archivos CSV de Thorlabs WFS).

    Pipeline
    --------
    1. Lectura del CSV con detección automática de codificación.
    2. Extracción de coeficientes de Zernike con índices (n, m) explícitos del CSV.
    3. Extracción de la malla medida W_medido(x, y).
    4. Reconstrucción  W_rec(x, y) = Σ c_j · Z_n^m(ρ, θ)  sobre la pupila.
    5. Aberraciones de Seidel: mapas por grupo + coeficientes clásicos W_klm.
    6. Tres figuras: mapas del frente de onda · espectro Zernike · aberraciones.
    7. Exportación opcional a PDF multipágina y TXT.

    Parámetros
    ----------
    archivo_csv     : str   — ruta al CSV del sensor
    guardar         : bool  — exporta PDF y TXT si True
    carpeta_salida  : str   — directorio de salida (por defecto: junto al CSV)
    angulo          : float — ángulo de inclinación [°], para análisis multi-ángulo
    guardar_archivos: bool  — alias de `guardar` (compatibilidad con versión anterior)
    nombre_carpeta  : str   — alias de `carpeta_salida` (compatibilidad)
    """

    LAMBDA_REF = 0.6328   # µm — He-Ne, referencia para estimación de Strehl

    def __init__(self, archivo_csv,
                 guardar=False, carpeta_salida=None, angulo=None,
                 # Alias para compatibilidad con ModuloAberraciones.py
                 guardar_archivos=None, nombre_carpeta=None):

        self.archivo_csv    = archivo_csv
        self.guardar        = guardar if guardar_archivos is None else guardar_archivos
        self.carpeta_salida = carpeta_salida if nombre_carpeta is None else nombre_carpeta
        self.angulo         = angulo

        # Datos internos (se rellenan en el pipeline)
        self._lineas   = None
        self.terminos  = []      # lista de {'n': int, 'm': int, 'c': float}

        self.radio_pupila = None          # mm
        self.X_med = self.Y_med = self.W_med = None  # malla medida
        self.X_rec = self.Y_rec = self.W_rec = None  # malla reconstruida

        # Nombre base para archivos de salida
        base = os.path.splitext(os.path.basename(archivo_csv))[0]
        if angulo is not None:
            base += f"_{angulo:+.0f}deg"
        self.nombre_base = base

        if self.carpeta_salida:
            os.makedirs(self.carpeta_salida, exist_ok=True)

    # ── I / O ─────────────────────────────────────────────────────────────────

    def _leer_csv(self):
        """Lee el CSV con detección automática de codificación."""
        try:
            with open(self.archivo_csv, 'rb') as fh:
                enc = chardet.detect(fh.read())['encoding'] or 'latin-1'
        except Exception:
            enc = 'latin-1'

        for encoding in [enc, 'latin-1', 'iso-8859-1', 'cp1252', 'utf-8']:
            try:
                with open(self.archivo_csv, 'r', encoding=encoding, errors='replace') as fh:
                    self._lineas = fh.readlines()
                return True
            except Exception:
                continue
        return False

    @staticmethod
    def _limpiar(s):
        """Elimina caracteres corruptos típicos de CSV Thorlabs."""
        return s.strip().replace('Á', '').replace('\ufffd', '').replace('?', '')

    # ── Parseo del CSV ────────────────────────────────────────────────────────

    def _parsear_coeficientes(self):
        """
        Extrae los términos de Zernike de la sección *** ZERNIKE FIT ***.

        Columnas del CSV: Index | Order (n) | Frequency (m) | Coefficient [µm]
        Almacena en self.terminos como lista de {'n', 'm', 'c'}.
        """
        terminos = []
        en_seccion = False

        for linea in self._lineas:
            if '*** ZERNIKE FIT ***' in linea:
                en_seccion = True
                continue
            if en_seccion and linea.startswith('***'):
                break
            if not en_seccion or ',' not in linea or 'Index' in linea:
                continue

            partes = linea.split(',')
            if len(partes) < 4:
                continue
            try:
                n = int(self._limpiar(partes[1]))
                m = int(self._limpiar(partes[2]))
                c = float(self._limpiar(partes[3]))
                terminos.append({'n': n, 'm': m, 'c': c})
            except ValueError:
                continue

        self.terminos = terminos

    def _parsear_wavefront(self):
        """
        Extrae la malla del frente de onda de la sección *** WAVEFRONT ***.

        Primera fila de datos: coordenadas X [mm]  (cabecera "y / x [mm]").
        Primera columna: coordenadas Y [mm].
        Interior: W(y, x) en µm. Celdas vacías o no numéricas → NaN.
        """
        filas_datos = []
        en_seccion  = False
        x_coords    = None

        for linea in self._lineas:
            if '*** WAVEFRONT ***' in linea:
                en_seccion = True
                continue
            if en_seccion and linea.startswith('***'):
                break
            if not en_seccion or ',' not in linea:
                continue

            if 'y / x [mm]' in linea:
                partes = linea.split(',')
                x_coords = []
                for x in partes[1:]:
                    try:
                        x_coords.append(float(self._limpiar(x)))
                    except ValueError:
                        continue
                continue

            if x_coords is None:
                continue

            partes = linea.split(',')
            try:
                y = float(self._limpiar(partes[0]))
            except ValueError:
                continue

            fila_z = []
            for val in partes[1:len(x_coords) + 1]:
                v = self._limpiar(val)
                try:
                    fila_z.append(float(v))
                except ValueError:
                    fila_z.append(np.nan)

            while len(fila_z) < len(x_coords):
                fila_z.append(np.nan)

            filas_datos.append((y, fila_z[:len(x_coords)]))

        if not filas_datos or x_coords is None:
            return False

        y_coords = np.array([f[0] for f in filas_datos])
        x_arr    = np.array(x_coords)
        Z_med    = np.array([f[1] for f in filas_datos])

        self.X_med, self.Y_med = np.meshgrid(x_arr, y_coords)
        self.W_med              = Z_med.astype(float)
        self.radio_pupila       = max(np.abs(x_arr).max(), np.abs(y_coords).max())
        return True

    # ── Física ────────────────────────────────────────────────────────────────

    def reconstruir_wavefront(self):
        """
        Reconstruye W_rec(x, y) = Σ_j  c_j · Z_{n_j}^{m_j}(ρ, θ).

        Usa la misma malla que el frente medido. Los puntos fuera de la pupila
        unitaria (ρ > 1) se enmascaran con NaN tanto en W_rec como en W_med.
        """
        if self.radio_pupila is None:
            self.radio_pupila = 1.5  # fallback si no hay datos de malla

        if self.X_med is not None:
            X, Y = self.X_med, self.Y_med
        else:
            xi   = np.linspace(-self.radio_pupila, self.radio_pupila, 101)
            X, Y = np.meshgrid(xi, xi)

        rho   = np.sqrt(X ** 2 + Y ** 2) / self.radio_pupila
        theta = np.arctan2(Y, X)

        W_rec = np.zeros_like(rho, dtype=np.float64)
        for t in self.terminos:
            W_rec += t['c'] * zernike_nm(t['n'], t['m'], rho, theta)

        # Enmascarar fuera de pupila
        fuera = rho > 1.0
        W_rec[fuera] = np.nan
        if self.W_med is not None:
            W_m             = self.W_med.astype(float)
            W_m[fuera]      = np.nan
            self.W_med      = W_m

        self.X_rec, self.Y_rec, self.W_rec = X, Y, W_rec

    def calcular_aberraciones(self):
        """
        Calcula mapas y métricas de cada grupo de aberración, y los coeficientes
        de Seidel clásicos derivados de los coeficientes de Zernike.

        Retorna
        -------
        dict con claves:
          'grupos'       — {nombre: {'mapa', 'rms', 'pv', 'coefs'}}
          'W_ho'         — frente de onda sin pistón ni inclinación
          'metricas'     — {'rms_total', 'pv_total', 'strehl', 'diametro_pupila'}
          'tabla_seidel' — lista de (nombre, c_Z, W_seidel, forma_funcional)
        """
        if self.W_rec is None:
            self.reconstruir_wavefront()

        X, Y = self.X_rec, self.Y_rec
        rho   = np.sqrt(X ** 2 + Y ** 2) / self.radio_pupila
        theta = np.arctan2(Y, X)
        valida = (rho <= 1.0) & ~np.isnan(rho)

        def _mapa_grupo(pares_nm):
            W = np.zeros_like(rho, dtype=np.float64)
            for n, m in pares_nm:
                c = next((t['c'] for t in self.terminos
                          if t['n'] == n and t['m'] == m), 0.0)
                W += c * zernike_nm(n, m, rho, theta)
            W[~valida] = np.nan
            return W

        def _stats(W):
            v = W[valida & ~np.isnan(W)]
            if v.size == 0:
                return 0.0, 0.0
            return float(np.std(v)), float(v.max() - v.min())

        # Grupos de aberración (pares (n, m))
        grupos_def = {
            'Pistón':       [(0,  0)],
            'Inclinación':  [(1, -1), (1,  1)],
            'Desenfoque':   [(2,  0)],
            'Astigmatismo': [(2, -2), (2,  2)],
            'Coma':         [(3, -1), (3,  1)],
            'Trébol':       [(3, -3), (3,  3)],
            'Esférica':     [(4,  0)],
            'Ast. 2.°':     [(4, -2), (4,  2)],
            'Tetratrébol':  [(4, -4), (4,  4)],
        }

        grupos = {}
        for nombre, pares in grupos_def.items():
            W_g    = _mapa_grupo(pares)
            rms_g, pv_g = _stats(W_g)
            coefs = {f"Z({n},{m:+d})": next(
                        (t['c'] for t in self.terminos
                         if t['n'] == n and t['m'] == m), 0.0)
                     for n, m in pares}
            grupos[nombre] = {'mapa': W_g, 'rms': rms_g, 'pv': pv_g, 'coefs': coefs}

        # Frente de onda de alto orden (sin pistón ni inclinación)
        W_piston = _mapa_grupo([(0, 0)])
        W_tilt   = _mapa_grupo([(1, -1), (1, 1)])
        W_ho     = np.where(valida, self.W_rec - W_piston - W_tilt, np.nan)
        vals_ho  = W_ho[valida & ~np.isnan(W_ho)]
        rms_ho   = float(np.std(vals_ho))   if vals_ho.size else 0.0
        pv_ho    = float(vals_ho.max() - vals_ho.min()) if vals_ho.size else 0.0
        strehl   = float(np.exp(-(2 * np.pi * rms_ho / self.LAMBDA_REF) ** 2))

        # ── Coeficientes de Seidel clásicos (pupila normalizada) ─────────
        # Los términos de Zernike balanceados se relacionan con los Seidel via:
        #   Z(2,0)  = √3(2ρ²−1)          → W_020 = 2√3 · c_20  (defocus ρ²)
        #   Z(2,±2) = √6·ρ²cos/sin(2θ)   → W_222 = 2√6 · |c_astig|  (astig ρ²cos²θ)
        #   Z(3,±1) = √8(3ρ²−2)ρcos/sinθ → W_131 = 3√8 · |c_coma|   (coma ρ³cosθ)
        #   Z(4,0)  = √5(6ρ⁴−6ρ²+1)      → W_040 = 6√5 · c_40       (esférica ρ⁴)
        def _c(n, m):
            return next((t['c'] for t in self.terminos
                         if t['n'] == n and t['m'] == m), 0.0)

        c_def  = _c(2,  0)
        c_ast  = np.hypot(_c(2, -2), _c(2,  2))
        c_coma = np.hypot(_c(3, -1), _c(3,  1))
        c_sph  = _c(4,  0)

        W020 = 2 * np.sqrt(3) * c_def   # coeficiente del término ρ²
        W222 = 2 * np.sqrt(6) * c_ast   # coeficiente del término ρ²cos²θ
        W131 = 3 * np.sqrt(8) * c_coma  # coeficiente del término ρ³cosθ
        W040 = 6 * np.sqrt(5) * c_sph   # coeficiente del término ρ⁴

        tabla_seidel = [
            ("Desenfoque",        c_def,  W020, "W₀₂₀ ρ²"),
            ("Astigmatismo",      c_ast,  W222, "W₂₂₂ ρ²cos²θ"),
            ("Coma",              c_coma, W131, "W₁₃₁ ρ³cosθ"),
            ("Esférica Primaria", c_sph,  W040, "W₀₄₀ ρ⁴"),
        ]

        return {
            'grupos':       grupos,
            'W_ho':         W_ho,
            'metricas': {
                'rms_total':       rms_ho,
                'pv_total':        pv_ho,
                'strehl':          strehl,
                'diametro_pupila': 2 * self.radio_pupila,
            },
            'tabla_seidel': tabla_seidel,
        }

    # ── Figuras ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cbar(fig, ax, im, etiqueta='µm'):
        cb = fig.colorbar(im, ax=ax, shrink=0.80, pad=0.03)
        cb.set_label(etiqueta, fontsize=8)
        cb.ax.tick_params(labelsize=7)

    @staticmethod
    def _circulo_pupila(ax, radio):
        th = np.linspace(0, 2 * np.pi, 360)
        ax.plot(radio * np.cos(th), radio * np.sin(th),
                'k-', lw=0.9, alpha=0.7)

    def figura_wavefront(self):
        """
        Figura 1 — Mapas del frente de onda.

        Fila superior : mapas 2D — Medido | Reconstruido (Zernike) | Residuo
        Fila inferior : Medido 3D | Reconstruido 3D | Perfil radial
        """
        if self.W_rec is None:
            self.reconstruir_wavefront()

        W_med = self.W_med
        W_rec = self.W_rec
        W_res = (W_med - W_rec) if (W_med is not None) else None

        X, Y = self.X_rec, self.Y_rec
        cmap  = 'RdYlBu_r'

        # Escala de color compartida para medido y reconstruido
        vmax = float(np.nanpercentile(np.abs(W_med if W_med is not None else W_rec), 99))
        vmax = max(vmax, 0.1)

        fig = plt.figure(figsize=(18, 10))
        gs  = GridSpec(2, 3, figure=fig,
                       hspace=0.42, wspace=0.34,
                       left=0.06, right=0.97, top=0.92, bottom=0.06)

        # ── Fila 0: mapas 2D ──────────────────────────────────────────────
        configs_2d = [
            (W_med, 'Frente de Onda Medido',                  -vmax, vmax,   cmap),
            (W_rec, 'Frente de Onda Reconstruido\n(Zernike)', -vmax, vmax,   cmap),
            (W_res, 'Residuo  (Medido − Reconstruido)',        None, None, 'seismic'),
        ]
        for col, (W, titulo, vmin, vmax_c, cm) in enumerate(configs_2d):
            ax = fig.add_subplot(gs[0, col])
            if W is None:
                ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center',
                        transform=ax.transAxes, fontsize=11)
                ax.set_title(titulo, fontsize=10)
                continue
            if vmin is None:
                vr  = max(float(np.nanpercentile(np.abs(W), 99)), 0.05)
                vmin, vmax_c = -vr, vr
            im = ax.contourf(X, Y, W, levels=60, cmap=cm, vmin=vmin, vmax=vmax_c)
            ax.contour(X, Y, W, levels=10, colors='k', alpha=0.2, linewidths=0.4)
            self._circulo_pupila(ax, self.radio_pupila)
            ax.set_aspect('equal')
            ax.set_xlabel('X [mm]', fontsize=9)
            ax.set_ylabel('Y [mm]', fontsize=9)
            ax.set_title(titulo, fontsize=10)
            ax.tick_params(labelsize=8)
            self._cbar(fig, ax, im)

        # ── Fila 1 izquierda / centro: superficies 3D ─────────────────────
        for col, (W, titulo) in enumerate([
            (W_med, 'Medido 3D'),
            (W_rec, 'Reconstruido 3D'),
        ]):
            ax3 = fig.add_subplot(gs[1, col], projection='3d')
            if W is not None:
                ax3.plot_surface(X, Y, W,
                                 cmap=cmap, rstride=1, cstride=1,
                                 linewidth=0, antialiased=False, alpha=0.92)
            ax3.set_xlabel('X [mm]', fontsize=8, labelpad=3)
            ax3.set_ylabel('Y [mm]', fontsize=8, labelpad=3)
            ax3.set_zlabel('µm',     fontsize=8, labelpad=3)
            ax3.set_title(titulo, fontsize=10)
            ax3.view_init(elev=28, azim=225)
            ax3.tick_params(labelsize=7)

        # ── Fila 1 derecha: perfil radial ─────────────────────────────────
        ax_r = fig.add_subplot(gs[1, 2])
        rho_flat = (np.sqrt(X ** 2 + Y ** 2) / self.radio_pupila).flatten()
        if W_med is not None:
            m_flat = W_med.flatten()
            r_flat = W_rec.flatten()
            ok     = ~np.isnan(m_flat) & ~np.isnan(r_flat)
            ax_r.scatter(rho_flat[ok], m_flat[ok],
                         s=2, alpha=0.35, color='steelblue', label='Medido')
            ax_r.scatter(rho_flat[ok], r_flat[ok],
                         s=2, alpha=0.35, color='tomato',    label='Reconstruido')
        ax_r.set_xlabel('ρ (radio normalizado)', fontsize=9)
        ax_r.set_ylabel('Desviación [µm]',       fontsize=9)
        ax_r.set_title('Perfil Radial',           fontsize=10)
        ax_r.legend(fontsize=8, markerscale=5)
        ax_r.grid(True, alpha=0.3)
        ax_r.tick_params(labelsize=8)

        titulo_fig = f'Frente de Onda — {self.nombre_base}'
        if self.angulo is not None:
            titulo_fig += f'  (ángulo {self.angulo:+.0f}°)'
        fig.suptitle(titulo_fig, fontsize=13, fontweight='bold')
        return fig

    def figura_zernike(self):
        """
        Figura 2 — Espectro de coeficientes de Zernike.

        Barras coloreadas por orden radial n, con anotaciones para términos
        dominantes. Los coeficientes se identifican por su (n, m) correcto
        extraído del CSV, no por un índice asumido.
        """
        if not self.terminos:
            return None

        etiquetas = [nombre_zernike(t['n'], t['m']) for t in self.terminos]
        valores   = [t['c']                         for t in self.terminos]
        n_list    = [t['n']                          for t in self.terminos]

        palette = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2',
                   '#59a14f', '#edc948', '#b07aa1', '#ff9da7']
        colores = [palette[min(n, len(palette) - 1)] for n in n_list]
        # Resaltar barras con |c| > 1 µm en rojo
        colores_fin = ['firebrick' if abs(v) > 1.0 else c
                       for v, c in zip(valores, colores)]

        fig, ax = plt.subplots(figsize=(14, 5))
        x      = np.arange(len(etiquetas))
        barras = ax.bar(x, valores,
                        color=colores_fin, alpha=0.85,
                        edgecolor='white', linewidth=0.6)

        # Anotaciones para términos significativos
        for bar, val in zip(barras, valores):
            if abs(val) > 0.3:
                despl = np.sign(val) * (abs(val) * 0.04 + 0.2)
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + despl,
                        f'{val:.3f}',
                        ha='center',
                        va='bottom' if val >= 0 else 'top',
                        fontsize=7.5, fontweight='bold')

        ax.axhline(0, color='k', lw=0.8)
        ax.axhline( 1.0, color='firebrick', lw=0.7, ls='--', alpha=0.5, label='|c| = 1 µm')
        ax.axhline(-1.0, color='firebrick', lw=0.7, ls='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(etiquetas, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Coeficiente c_j [µm]', fontsize=10)
        ax.set_title('Espectro de Coeficientes de Zernike  (convención OSA/ANSI)',
                     fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='y', labelsize=9)

        # Leyenda de órdenes radiales
        handles = [Patch(color=palette[min(n, len(palette) - 1)],
                         label=f'Orden n = {n}')
                   for n in sorted(set(n_list))]
        handles.append(plt.Line2D([0], [0], color='firebrick', ls='--',
                                  label='Umbral 1 µm'))
        ax.legend(handles=handles, fontsize=8, loc='upper right')

        plt.tight_layout()
        return fig

    def figura_aberraciones(self, aberraciones=None):
        """
        Figura 3 — Mapas de aberraciones individuales y tabla de Seidel.

        Muestra los mapas 2D de cada grupo (ordenados por RMS descendente)
        y un panel inferior con las métricas de calidad y los coeficientes
        de Seidel calculados desde los coeficientes de Zernike.
        """
        if aberraciones is None:
            aberraciones = self.calcular_aberraciones()

        grupos   = aberraciones['grupos']
        metricas = aberraciones['metricas']
        tabla    = aberraciones['tabla_seidel']

        # Seleccionar grupos con RMS apreciable (sin pistón), máx 6 mapas
        mostrar = [(n, g) for n, g in grupos.items()
                   if n != 'Pistón' and g['rms'] > 1e-3]
        mostrar.sort(key=lambda x: -x[1]['rms'])
        mostrar = mostrar[:6]

        n_mapas = len(mostrar)
        ncols   = 3
        nfilas_mapas = max(1, (n_mapas + ncols - 1) // ncols)
        nfilas  = nfilas_mapas + 1   # +1 para tabla inferior

        fig = plt.figure(figsize=(16, 4.2 * nfilas))
        gs  = GridSpec(nfilas, ncols, figure=fig,
                       hspace=0.50, wspace=0.36,
                       left=0.06, right=0.97, top=0.93, bottom=0.04)

        cmap = 'RdYlBu_r'
        X, Y = self.X_rec, self.Y_rec

        for idx, (nombre, grupo) in enumerate(mostrar):
            fila, col = divmod(idx, ncols)
            ax  = fig.add_subplot(gs[fila, col])
            W   = grupo['mapa']
            vmax = max(float(np.nanpercentile(np.abs(W), 99)), 0.01)
            im  = ax.contourf(X, Y, W, levels=50, cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.contour(X, Y, W, levels=8, colors='k', alpha=0.2, linewidths=0.4)
            self._circulo_pupila(ax, self.radio_pupila)
            ax.set_aspect('equal')
            ax.set_xlabel('X [mm]', fontsize=8)
            ax.set_ylabel('Y [mm]', fontsize=8)
            ax.set_title(f'{nombre}\nRMS = {grupo["rms"]:.4f} µm   '
                         f'PV = {grupo["pv"]:.4f} µm',
                         fontsize=9)
            ax.tick_params(labelsize=7)
            self._cbar(fig, ax, im)

        # ── Panel inferior: métricas + tabla de Seidel ────────────────────
        ax_tab = fig.add_subplot(gs[nfilas - 1, :])
        ax_tab.axis('off')

        rms    = metricas['rms_total']
        strehl = metricas['strehl']
        if rms < 0.045:
            calidad = "Limitada por difracción  (RMS < λ/14)"
        elif rms < 0.50:
            calidad = "Excelente"
        elif rms < 1.00:
            calidad = "Buena"
        elif rms < 2.00:
            calidad = "Aceptable"
        else:
            calidad = "Pobre — considerar recalibración"

        texto_met = (
            f"Métricas (sin pistón / inclinación):\n"
            f"  PV = {metricas['pv_total']:.4f} µm     "
            f"RMS = {rms:.4f} µm     "
            f"Strehl ≈ {strehl:.4f}     "
            f"Ø pupila = {metricas['diametro_pupila']:.3f} mm     "
            f"Calidad: {calidad}"
        )
        ax_tab.text(0.0, 0.97, texto_met,
                    transform=ax_tab.transAxes,
                    fontsize=9.5, va='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.9))

        # Tabla de coeficientes de Seidel
        encabezados = ['Aberración',
                       'c_Z [µm]\n(RMS Zernike)',
                       'W_Seidel [µm]\n(coef. ρⁿ)',
                       'Forma funcional',
                       'Interpretación']
        interp = {
            "Desenfoque":        "Error de enfoque",
            "Astigmatismo":      "Imagen en dos focos separados",
            "Coma":              "Cola de cometa en imagen puntual",
            "Esférica Primaria": "Enfoque diferente centro / borde",
        }
        filas_tab = [
            [nombre,
             f'{c_z:.4f}',
             f'{ws:.4f}',
             forma,
             interp.get(nombre, '—')]
            for nombre, c_z, ws, forma in tabla
        ]

        tbl = ax_tab.table(
            cellText=filas_tab,
            colLabels=encabezados,
            cellLoc='center',
            loc='bottom',
            bbox=[0.0, 0.0, 1.0, 0.72],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        for (row, _col), cell in tbl.get_celld().items():
            cell.set_edgecolor('#cccccc')
            if row == 0:
                cell.set_facecolor('#2c5f8a')
                cell.set_text_props(color='white', fontweight='bold')
            elif row % 2 == 0:
                cell.set_facecolor('#eef4fb')

        fig.suptitle('Aberraciones Ópticas — Análisis de Seidel',
                     fontsize=13, fontweight='bold')
        return fig

    # ── Exportación ──────────────────────────────────────────────────────────

    def _ruta(self, sufijo):
        """Construye la ruta de salida en la carpeta correcta."""
        nombre = self.nombre_base + sufijo
        if self.carpeta_salida:
            return os.path.join(self.carpeta_salida, nombre)
        directorio = os.path.dirname(os.path.abspath(self.archivo_csv))
        return os.path.join(directorio, nombre)

    def exportar_pdf(self, figs):
        """Guarda la lista de figuras como PDF multi-página."""
        ruta = self._ruta('_analisis.pdf')
        try:
            with PdfPages(ruta) as pdf:
                for fig in figs:
                    if fig is not None:
                        pdf.savefig(fig, dpi=150, bbox_inches='tight')
            print(f"  ✓ PDF → {ruta}")
        except Exception as e:
            print(f"  ✗ Error PDF: {e}")

    def exportar_txt(self, aberraciones):
        """Escribe un reporte de texto con todos los resultados numéricos."""
        ruta   = self._ruta('_reporte.txt')
        met    = aberraciones['metricas']
        tabla  = aberraciones['tabla_seidel']
        rms    = met['rms_total']

        lineas_calidad = {
            rms < 0.045: "Limitada por difracción (RMS < λ/14 ≈ 0.045 µm)",
            rms < 0.50:  "Excelente",
            rms < 1.00:  "Buena",
            rms < 2.00:  "Aceptable",
            True:        "Pobre — considerar recalibración",
        }
        calidad = next(v for k, v in lineas_calidad.items() if k)

        try:
            with open(ruta, 'w', encoding='utf-8') as f:
                sep = "=" * 62
                f.write(sep + "\n")
                f.write("  ANÁLISIS DE ABERRACIONES — SENSOR SHACK-HARTMANN\n")
                f.write(sep + "\n")
                f.write(f"  Archivo : {self.archivo_csv}\n")
                f.write(f"  Fecha   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if self.angulo is not None:
                    f.write(f"  Ángulo  : {self.angulo:+.1f}°\n")
                f.write("\n")

                f.write("MÉTRICAS DEL FRENTE DE ONDA (sin pistón ni inclinación)\n")
                f.write("-" * 45 + "\n")
                f.write(f"  PV       = {met['pv_total']:.4f} µm\n")
                f.write(f"  RMS      = {rms:.4f} µm\n")
                f.write(f"  Strehl   ≈ {met['strehl']:.4f}  "
                        f"(Maréchal, λ = {self.LAMBDA_REF} µm)\n")
                f.write(f"  Ø pupila = {met['diametro_pupila']:.3f} mm\n")
                f.write(f"  Calidad  : {calidad}\n\n")

                f.write("COEFICIENTES DE ZERNIKE (OSA/ANSI, columnas CSV: n, m, c)\n")
                f.write("-" * 55 + "\n")
                f.write(f"  {'j':>3}  {'n':>3}  {'m':>4}  "
                        f"{'Nombre':<26}  {'c [µm]':>10}  {'Nota'}\n")
                f.write("  " + "-" * 58 + "\n")
                for j, t in enumerate(self.terminos):
                    nota = ("← DOMINANTE"  if abs(t['c']) > 3.0 else
                            "← significat." if abs(t['c']) > 1.0 else "")
                    f.write(f"  {j+1:>3}  {t['n']:>3}  {t['m']:>4}  "
                            f"{nombre_zernike(t['n'], t['m']):<26}  "
                            f"{t['c']:>10.4f}  {nota}\n")
                f.write("\n")

                f.write("ABERRACIONES DE SEIDEL\n")
                f.write("-" * 55 + "\n")
                f.write(f"  {'Aberración':<22}  {'c_Z [µm]':>10}  "
                        f"{'W_Seidel [µm]':>14}  {'Forma'}\n")
                f.write("  " + "-" * 58 + "\n")
                for nombre, c_z, w_s, forma in tabla:
                    f.write(f"  {nombre:<22}  {c_z:>10.4f}  {w_s:>14.4f}  {forma}\n")
                f.write("\n")

                f.write("INTERPRETACIÓN\n")
                f.write("-" * 45 + "\n")
                interp_reglas = [
                    (lambda: aberraciones['grupos']['Astigmatismo']['rms'] > 0.5,
                     "Astigmatismo significativo — diferente potencia en meridianos ortogonales."),
                    (lambda: aberraciones['grupos']['Coma']['rms'] > 0.5,
                     "Coma presente — asimetría en la imagen de un punto fuente."),
                    (lambda: aberraciones['grupos']['Esférica']['rms'] > 0.5,
                     "Aberración esférica — rayos marginales y paraxiales no convergen."),
                    (lambda: aberraciones['grupos']['Desenfoque']['rms'] > 1.0,
                     "Desenfoque elevado — verificar posición del plano imagen."),
                ]
                for condicion, mensaje in interp_reglas:
                    try:
                        if condicion():
                            f.write(f"  • {mensaje}\n")
                    except Exception:
                        pass
                if all(g['rms'] < 0.5
                       for k, g in aberraciones['grupos'].items()
                       if k not in ('Pistón', 'Inclinación')):
                    f.write("  • No se detectaron aberraciones significativas (> 0.5 µm).\n")

            print(f"  ✓ Reporte → {ruta}")
        except Exception as e:
            print(f"  ✗ Error TXT: {e}")

    # ── Pipeline principal ────────────────────────────────────────────────────

    def analizar(self, mostrar=True):
        """
        Ejecuta el pipeline completo y devuelve un dict con todos los resultados.

        Parámetros
        ----------
        mostrar : bool — llama plt.show() si True

        Retorna
        -------
        dict con claves: 'terminos', 'W_medido', 'W_rec', 'X', 'Y',
                         'aberraciones', 'figuras'
        """
        print(f"\n{'─'*60}")
        print(f"  Análisis: {self.archivo_csv}")
        print(f"{'─'*60}")

        if not self._leer_csv():
            raise IOError(f"No se pudo leer: {self.archivo_csv}")

        self._parsear_coeficientes()
        print(f"  Coeficientes Zernike : {len(self.terminos)}")

        tiene_wavefront = self._parsear_wavefront()
        if tiene_wavefront:
            print(f"  Malla wavefront      : {self.W_med.shape}  "
                  f"(Ø = {2 * self.radio_pupila:.2f} mm)")
        else:
            print("  Advertencia: sin datos de malla wavefront en el CSV.")
            self.radio_pupila = self.radio_pupila or 1.5

        self.reconstruir_wavefront()
        aber = self.calcular_aberraciones()
        met  = aber['metricas']

        print(f"  PV  = {met['pv_total']:.4f} µm   "
              f"RMS = {met['rms_total']:.4f} µm   "
              f"Strehl ≈ {met['strehl']:.4f}")

        fig1 = self.figura_wavefront()
        fig2 = self.figura_zernike()
        fig3 = self.figura_aberraciones(aber)

        if self.guardar:
            self.exportar_pdf([fig1, fig2, fig3])
            self.exportar_txt(aber)

        if mostrar:
            plt.show()

        return {
            'terminos':     self.terminos,
            'W_medido':     self.W_med,
            'W_rec':        self.W_rec,
            'X':            self.X_rec,
            'Y':            self.Y_rec,
            'aberraciones': aber,
            'figuras':      [fig1, fig2, fig3],
        }

    def main(self):
        """Alias de analizar() para compatibilidad con Analisis.py."""
        return self.analizar(mostrar=not self.guardar)


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución directa
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    archivo = sys.argv[1] if len(sys.argv) > 1 else 'lenteThorlabs.csv'
    sensor  = SensorShackHartmann(archivo, guardar=True)
    sensor.analizar()

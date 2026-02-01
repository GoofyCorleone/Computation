import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from numba import njit, prange
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PARÁMETROS FÍSICOS Y GEOMÉTRICOS
# ============================================================================
# Longitud de onda (en metros)
lambda_ = 632.8e-9  # Luz roja de He-Ne

# Dimensiones de las rendijas (en metros)
p = 0.5e-3  # Ancho en x
q = 0.5e-3  # Alto en y

# Posiciones de las rendijas
n = 50e-3  # Posición z de la primera rendija (m)
c = 10e-3  # Separación z entre rendijas (m)

# Desplazamiento de la segunda rendija respecto a la primera
a = 1.0e-3  # Desplazamiento en x (m)
b = 0.0e-3  # Desplazamiento en y (m)

# Plano de observación
z_obs = 200e-3 # Distancia de observación (m)

# Número de onda
k = 2 * np.pi / lambda_

# ============================================================================
# FUNCIONES AUXILIARES OPTIMIZADAS
# ============================================================================
@njit
def rect(x, width):
    """Función rectángulo."""
    return 1.0 if np.abs(x) <= width/2 else 0.0

@njit
def A1(x1, y1, p, q):
    """Transmitancia de la primera rendija."""
    return rect(x1, p) * rect(y1, q)

@njit
def A2(x2, y2, p, q, a, b):
    """Transmitancia de la segunda rendija."""
    return rect(x2 - a, p) * rect(y2 - b, q)

@njit
def spherical_wave_propagator(x, y, xi, eta, dz, k):
    """Propagador de onda esférica (sin factor 1/(i*lambda))."""
    r = np.sqrt((x - xi)**2 + (y - eta)**2 + dz**2)
    return np.exp(1j * k * r) / r

# ============================================================================
# MÉTODO DE SIMPSON ADAPTATIVO EN 2D
# ============================================================================
def simpson_2d_adaptive(f, xlim, ylim, tol=1e-6, max_depth=8, *args):
    """
    Método de Simpson adaptativo para integral doble.
    
    Parámetros:
    f: función f(x, y, *args)
    xlim: (x_min, x_max)
    ylim: (y_min, y_max)
    tol: tolerancia
    max_depth: profundidad máxima de recursión
    *args: argumentos adicionales para f
    
    Retorna:
    integral, error_estimate
    """
    
    def _simpson_2d_recursive(xa, xb, ya, yb, depth, fa, fb, fc, fd, *args):
        # Puntos medios
        xm = (xa + xb) / 2
        ym = (ya + yb) / 2
        
        # Evaluar en puntos adicionales
        f1 = f(xm, ya, *args)
        f2 = f(xb, ym, *args)
        f3 = f(xm, yb, *args)
        f4 = f(xa, ym, *args)
        f5 = f(xm, ym, *args)
        
        # Regla de Simpson compuesta para un subrectángulo
        hx = (xb - xa) / 2
        hy = (yb - ya) / 2
        
        # Integral aproximada con 4 subrectángulos
        I1 = (hx * hy / 9) * (fa + fb + fc + fd + 4*(f1 + f2 + f3 + f4) + 16*f5)
        
        # Subdividir si es necesario
        if depth < max_depth:
            # Calcular subdivisiones
            I2_sum = 0
            for i in range(4):
                if i == 0:  # Subrectángulo inferior izquierdo
                    sub_I, _ = _simpson_2d_recursive(xa, xm, ya, ym, depth+1,
                                                     fa, f5, f1, f4, *args)
                elif i == 1:  # Subrectángulo inferior derecho
                    sub_I, _ = _simpson_2d_recursive(xm, xb, ya, ym, depth+1,
                                                     f5, fb, f2, f1, *args)
                elif i == 2:  # Subrectángulo superior izquierdo
                    sub_I, _ = _simpson_2d_recursive(xa, xm, ym, yb, depth+1,
                                                     f4, f1, f5, f3, *args)
                else:  # Subrectángulo superior derecho
                    sub_I, _ = _simpson_2d_recursive(xm, xb, ym, yb, depth+1,
                                                     f1, f2, f3, f5, *args)
                I2_sum += sub_I
            
            # Estimación del error
            error = np.abs(I1 - I2_sum)
            
            if error < tol * np.abs(I2_sum):
                return I2_sum, error
            else:
                return I1, error
        else:
            return I1, 0.0
    
    # Evaluar en las esquinas
    xa, xb = xlim
    ya, yb = ylim
    fa = f(xa, ya, *args)
    fb = f(xb, ya, *args)
    fc = f(xa, yb, *args)
    fd = f(xb, yb, *args)
    
    return _simpson_2d_recursive(xa, xb, ya, yb, 0, fa, fb, fc, fd, *args)

# ============================================================================
# CÁLCULO DEL CAMPO DIFRACTADO
# ============================================================================
@njit(parallel=True)
def compute_diffraction_pattern(x_grid, y_grid, lambda_, k, p, q, n, c, a, b, z_obs):
    """
    Calcula el patrón de difracción usando el método de Huygens-Fresnel.
    Usa una aproximación más eficiente para evitar la integral cuádruple.
    """
    # Distancias de propagación
    dz1 = c  # Primera rendija -> segunda rendija
    dz2 = z_obs - (n + c)  # Segunda rendija -> plano de observación
    
    # Tamaño de la grilla
    nx, ny = x_grid.shape[1], x_grid.shape[0]
    
    # Campo resultante
    U = np.zeros((ny, nx), dtype=np.complex128)
    
    # Constante de propagación
    const = 1 / (1j * lambda_)**2
    
    # Definir los límites de las rendijas
    x1_min, x1_max = -p/2, p/2
    y1_min, y1_max = -q/2, q/2
    
    x2_min, x2_max = a - p/2, a + p/2
    y2_min, y2_max = b - q/2, b + q/2
    
    # Número de puntos de integración (reducido para simulación)
    n_points = 20
    
    # Crear puntos de integración para cada rendija
    x1_pts = np.linspace(x1_min, x1_max, n_points)
    y1_pts = np.linspace(y1_min, y1_max, n_points)
    x2_pts = np.linspace(x2_min, x2_max, n_points)
    y2_pts = np.linspace(y2_min, y2_max, n_points)
    
    # Pesos para integración trapezoidal simple
    w1 = (x1_max - x1_min) * (y1_max - y1_min) / (n_points * n_points)
    w2 = (x2_max - x2_min) * (y2_max - y2_min) / (n_points * n_points)
    
    # Precalcular contribuciones de la rendija 1 a la rendija 2
    U2_before = np.zeros((n_points, n_points), dtype=np.complex128)
    
    for i in prange(n_points):
        for j in prange(n_points):
            x2 = x2_pts[i]
            y2 = y2_pts[j]
            
            # Integrar sobre la rendija 1
            sum_val = 0j
            for ii in range(n_points):
                for jj in range(n_points):
                    x1 = x1_pts[ii]
                    y1 = y1_pts[jj]
                    
                    if rect(x1, p) * rect(y1, q) > 0:  # Si está dentro de la rendija
                        r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + dz1**2)
                        sum_val += np.exp(1j * k * r12) / r12
            
            U2_before[i, j] = sum_val * w1
    
    # Multiplicar por la transmitancia de la rendija 2
    U2_after = np.zeros((n_points, n_points), dtype=np.complex128)
    for i in prange(n_points):
        for j in prange(n_points):
            x2 = x2_pts[i]
            y2 = y2_pts[j]
            U2_after[i, j] = U2_before[i, j] * rect(x2 - a, p) * rect(y2 - b, q)
    
    # Propagación al plano de observación
    for i in prange(ny):
        for j in prange(nx):
            x = x_grid[i, j]
            y = y_grid[i, j]
            
            # Integrar sobre la rendija 2
            sum_val = 0j
            for ii in range(n_points):
                for jj in range(n_points):
                    x2 = x2_pts[ii]
                    y2 = y2_pts[jj]
                    
                    if rect(x2 - a, p) * rect(y2 - b, q) > 0:  # Si está dentro de la rendija 2
                        r2 = np.sqrt((x - x2)**2 + (y - y2)**2 + dz2**2)
                        sum_val += U2_after[ii, jj] * np.exp(1j * k * r2) / r2
            
            U[i, j] = const * sum_val * w2
    
    return U

# ============================================================================
# CONFIGURACIÓN DE LA SIMULACIÓN
# ============================================================================
print("Simulación de difracción de Huygens-Fresnel para dos rendijas rectangulares")
print("=" * 70)
print(f"Longitud de onda: {lambda_*1e9:.1f} nm")
print(f"Dimensiones de rendijas: {p*1e3:.2f} mm × {q*1e3:.2f} mm")
print(f"Posiciones: Rendija 1 en z={n:.2f} m, Rendija 2 en z={n+c:.2f} m")
print(f"Desplazamiento de la rendija 2: ({a*1e3:.2f} mm, {b*1e3:.2f} mm)")
print(f"Plano de observación en z={z_obs:.2f} m")

# Tamaño de la ventana de observación (en metros)
window_size = 10e-3
# Resolución espacial
resolution = 100

# Crear grilla de observación
x = np.linspace(-window_size/2, window_size/2, resolution)
y = np.linspace(-window_size/2, window_size/2, resolution)
X, Y = np.meshgrid(x, y)

# ============================================================================
# EJECUCIÓN DE LA SIMULACIÓN
# ============================================================================
print("\nCalculando patrón de difracción...")
start_time = time.time()

# Calcular campo difractado
U = compute_diffraction_pattern(X, Y, lambda_, k, p, q, n, c, a, b, z_obs)

# Calcular intensidad
I = np.abs(U)**2
I_normalized = I / np.max(I)  # Normalizar

print(f"Tiempo de cálculo: {time.time() - start_time:.2f} segundos")

# ============================================================================
# VISUALIZACIÓN
# ============================================================================
fig = plt.figure(figsize=(15, 10))

# 1. Patrón de difracción completo
ax1 = plt.subplot(2, 3, 1)
im = ax1.imshow(I_normalized, 
                extent=[-window_size/2*1e3, window_size/2*1e3, 
                        -window_size/2*1e3, window_size/2*1e3],
                cmap='hot', origin='lower')
ax1.set_xlabel('x (mm)')
ax1.set_ylabel('y (mm)')
ax1.set_title('Patrón de difracción Huygens-Fresnel')
plt.colorbar(im, ax=ax1, label='Intensidad normalizada')

# Dibujar las posiciones de las rendijas en el plano de observación
rect1 = Rectangle((-p/2*1e3, -q/2*1e3), p*1e3, q*1e3,
                  linewidth=1, edgecolor='cyan', facecolor='none', 
                  linestyle='--', label='Rendija 1 (proyectada)')
rect2 = Rectangle(((a-p/2)*1e3, (b-q/2)*1e3), p*1e3, q*1e3,
                  linewidth=1, edgecolor='lime', facecolor='none',
                  linestyle='--', label='Rendija 2 (proyectada)')
ax1.add_patch(rect1)
ax1.add_patch(rect2)
ax1.legend(loc='upper right', fontsize=8)

# 2. Perfil horizontal (corte en y=0)
ax2 = plt.subplot(2, 3, 2)
y_idx = np.argmin(np.abs(y))  # Índice para y=0
I_horizontal = I_normalized[y_idx, :]
ax2.plot(x*1e3, I_horizontal, 'b-', linewidth=2)
ax2.set_xlabel('x (mm)')
ax2.set_ylabel('Intensidad normalizada')
ax2.set_title('Perfil horizontal (y=0)')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-window_size/2*1e3, window_size/2*1e3])

# Marcar posiciones de las rendijas
ax2.axvline(x=-p/2*1e3, color='cyan', linestyle='--', alpha=0.5)
ax2.axvline(x=p/2*1e3, color='cyan', linestyle='--', alpha=0.5)
ax2.axvline(x=(a-p/2)*1e3, color='lime', linestyle='--', alpha=0.5)
ax2.axvline(x=(a+p/2)*1e3, color='lime', linestyle='--', alpha=0.5)

# 3. Perfil vertical (corte en x=0)
ax3 = plt.subplot(2, 3, 3)
x_idx = np.argmin(np.abs(x))  # Índice para x=0
I_vertical = I_normalized[:, x_idx]
ax3.plot(y*1e3, I_vertical, 'r-', linewidth=2)
ax3.set_xlabel('y (mm)')
ax3.set_ylabel('Intensidad normalizada')
ax3.set_title('Perfil vertical (x=0)')
ax3.grid(True, alpha=0.3)
ax3.set_xlim([-window_size/2*1e3, window_size/2*1e3])

# Marcar posiciones de las rendijas
ax3.axvline(x=-q/2*1e3, color='cyan', linestyle='--', alpha=0.5)
ax3.axvline(x=q/2*1e3, color='cyan', linestyle='--', alpha=0.5)
ax3.axvline(x=(b-q/2)*1e3, color='lime', linestyle='--', alpha=0.5)
ax3.axvline(x=(b+q/2)*1e3, color='lime', linestyle='--', alpha=0.5)

# 4. Visualización 3D del patrón de difracción
from mpl_toolkits.mplot3d import Axes3D
ax4 = plt.subplot(2, 3, 4, projection='3d')
# Submuestrear para visualización 3D
step = resolution // 20
X_small = X[::step, ::step]
Y_small = Y[::step, ::step]
I_small = I_normalized[::step, ::step]
surf = ax4.plot_surface(X_small*1e3, Y_small*1e3, I_small, 
                       cmap='hot', alpha=0.8, linewidth=0)
ax4.set_xlabel('x (mm)')
ax4.set_ylabel('y (mm)')
ax4.set_zlabel('Intensidad')
ax4.set_title('Superficie 3D del patrón')

# 5. Diagrama de fase
ax5 = plt.subplot(2, 3, 5)
phase = np.angle(U)
phase_im = ax5.imshow(phase, 
                     extent=[-window_size/2*1e3, window_size/2*1e3,
                             -window_size/2*1e3, window_size/2*1e3],
                     cmap='hsv', origin='lower', vmin=-np.pi, vmax=np.pi)
ax5.set_xlabel('x (mm)')
ax5.set_ylabel('y (mm)')
ax5.set_title('Fase del campo difractado')
plt.colorbar(phase_im, ax=ax5, label='Fase (rad)')

# 6. Configuración geométrica
ax6 = plt.subplot(2, 3, 6)
# Diagrama esquemático de la configuración
z_positions = [0, n, n+c, z_obs]
labels = ['Origen', 'Rendija 1', 'Rendija 2', 'Observación']
colors = ['gray', 'cyan', 'lime', 'red']

for i, (z, label, color) in enumerate(zip(z_positions, labels, colors)):
    ax6.plot([-1, 1], [z, z], color=color, linewidth=2)
    ax6.text(1.2, z, label, verticalalignment='center', color=color)
    
    # Dibujar rendijas
    if label == 'Rendija 1':
        ax6.add_patch(Rectangle((-0.5, z-0.01), 1, 0.02, 
                              facecolor=color, alpha=0.3))
    elif label == 'Rendija 2':
        ax6.add_patch(Rectangle((a*10-0.5, z-0.01), 1, 0.02, 
                              facecolor=color, alpha=0.3))

ax6.set_xlabel('Posición x (escala arbitraria)')
ax6.set_ylabel('Posición z (m)')
ax6.set_title('Configuración geométrica')
ax6.grid(True, alpha=0.3)
ax6.set_xlim([-2, 2])
ax6.set_ylim([-0.1, z_obs + 0.1])

plt.suptitle('Difracción de Huygens-Fresnel: Dos rendijas rectangulares', fontsize=14, y=0.98)
plt.tight_layout()
plt.show()

# ============================================================================
# INFORMACIÓN ADICIONAL
# ============================================================================
print("\n" + "="*70)
print("ANÁLISIS DEL PATRÓN DE DIFRACCIÓN")
print("="*70)

# Calcular algunas características del patrón
max_intensity = np.max(I)
mean_intensity = np.mean(I)
contrast = (max_intensity - np.min(I)) / (max_intensity + np.min(I))

print(f"Intensidad máxima: {max_intensity:.2e}")
print(f"Intensidad media: {mean_intensity:.2e}")
print(f"Contraste: {contrast:.3f}")

# Encontrar máximos locales en el perfil horizontal
from scipy.signal import find_peaks
peaks, properties = find_peaks(I_horizontal, height=0.1)
if len(peaks) > 0:
    print(f"\nMáximos encontrados en el perfil horizontal:")
    for i, peak in enumerate(peaks[:5]):  # Mostrar solo los primeros 5
        x_pos = x[peak] * 1e3
        intensity = I_horizontal[peak]
        print(f"  Máximo {i+1}: x = {x_pos:.2f} mm, Intensidad = {intensity:.3f}")

# ============================================================================
# OPCIÓN PARA GUARDAR LOS RESULTADOS
# ============================================================================
save_results = input("\n¿Desea guardar los resultados? (s/n): ")
if save_results.lower() == 's':
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"huygens_fresnel_diffraction_{timestamp}.png"
    plt.figure(fig.number)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Resultados guardados como {filename}")
    
    # Guardar datos numéricos
    data = {
        'x_mm': x * 1e3,
        'y_mm': y * 1e3,
        'intensity': I,
        'intensity_normalized': I_normalized,
        'phase': phase,
        'parameters': {
            'lambda_nm': lambda_ * 1e9,
            'p_mm': p * 1e3,
            'q_mm': q * 1e3,
            'n_m': n,
            'c_m': c,
            'a_mm': a * 1e3,
            'b_mm': b * 1e3,
            'z_obs_m': z_obs
        }
    }
    np.savez(f"diffraction_data_{timestamp}.npz", **data)
    print(f"Datos numéricos guardados como diffraction_data_{timestamp}.npz")

print("\nSimulación completada.")
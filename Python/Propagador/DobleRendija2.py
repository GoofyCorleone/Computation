import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

class HuygensFresnelDiffractionSimulator:
    def __init__(self, wavelength=632.8e-9):
        """
        Inicializa el simulador de difracción de Huygens-Fresnel
        
        Parámetros:
        -----------
        wavelength : float
            Longitud de onda en metros (default: 632.8 nm, luz He-Ne)
        """
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength  # Número de onda
        
    def simpson_adaptive(self, func, a, b, tol=1e-6, max_depth=20):
        """
        Método de Simpson adaptativo recursivo para integración 1D
        
        Parámetros:
        -----------
        func : function
            Función a integrar
        a, b : float
            Límites de integración
        tol : float
            Tolerancia relativa
        max_depth : int
            Profundidad máxima de recursión
        
        Retorna:
        --------
        float : Valor de la integral
        """
        def recursive_simpson(left, right, f_left, f_right, f_mid, whole, depth):
            mid = (left + right) / 2
            f_left_mid = func((left + mid) / 2)
            f_right_mid = func((mid + right) / 2)
            
            left_half = (mid - left) * (f_left + 4 * f_left_mid + f_mid) / 6
            right_half = (right - mid) * (f_mid + 4 * f_right_mid + f_right) / 6
            total = left_half + right_half
            
            if depth >= max_depth or abs(total - whole) <= 15 * tol:
                return total + (total - whole) / 15
            
            return (recursive_simpson(left, mid, f_left, f_mid, f_left_mid, left_half, depth + 1) +
                    recursive_simpson(mid, right, f_mid, f_right, f_right_mid, right_half, depth + 1))
        
        f_a = func(a)
        f_b = func(b)
        f_mid = func((a + b) / 2)
        whole = (b - a) * (f_a + 4 * f_mid + f_b) / 6
        
        return recursive_simpson(a, b, f_a, f_b, f_mid, whole, 0)
    
    def huygens_kernel_3d(self, x0, y0, x, y, d):
        """
        Kernel de Huygens-Fresnel 3D (onda esférica)
        
        Parámetros:
        -----------
        x0, y0 : float
            Punto de observación
        x, y : float
            Punto en la apertura
        d : float
            Distancia de propagación
        
        Retorna:
        --------
        complex : Valor del kernel (exp(i*k*r)/r)
        """
        if abs(d) < 1e-12:
            return 0
        # Distancia radial para propagación 3D (onda esférica)
        r = np.sqrt((x0 - x)**2 + (y0 - y)**2 + d**2)
        return np.exp(1j * self.k * r) / r
    
    def rect_function(self, x, center, width):
        """
        Función rectangular (apertura)
        
        Parámetros:
        -----------
        x : float
            Posición
        center : float
            Centro de la rendija
        width : float
            Ancho de la rendija
        
        Retorna:
        --------
        float : 1 si está dentro, 0 fuera
        """
        return 1.0 if abs(x - center) <= width/2 else 0.0
    
    def simulate_double_slit_huygens(self, params, Nx=100, Ny=100, x_range=0.01, y_range=0.01):
        """
        Simula el patrón de difracción 2D para dos rendijas usando Huygens-Fresnel
        
        Parámetros:
        -----------
        params : dict
            Diccionario con los parámetros del sistema:
            - p: ancho en x (m)
            - q: alto en y (m)
            - a: desplazamiento en x de la segunda rendija (m)
            - b: desplazamiento en y de la segunda rendija (m)
            - n: posición z de la primera rendija (m)
            - c: distancia entre rendijas (m)
            - z0: posición del plano de observación (m)
        Nx, Ny : int
            Número de puntos en x e y
        x_range, y_range : float
            Rango de observación en metros
        
        Retorna:
        --------
        dict : Resultados de la simulación
        """
        # Extraer parámetros
        p = params['p']
        q = params['q']
        a = params['a']
        b = params['b']
        n = params['n']
        c = params['c']
        z0 = params['z0']
        
        # Distancias de propagación
        d1 = c  # Primera rendija a segunda rendija
        d2 = z0 - (n + c)  # Segunda rendija a plano de observación
        
        # Crear mallas de observación
        x_obs = np.linspace(-x_range/2, x_range/2, Nx)
        y_obs = np.linspace(-y_range/2, y_range/2, Ny)
        X_obs, Y_obs = np.meshgrid(x_obs, y_obs)
        
        # Campo en el plano de observación
        U_2d = np.zeros((Ny, Nx), dtype=complex)
        
        # Puntos de integración para las rendijas
        # (reducir estos números para mayor velocidad, aumentar para mayor precisión)
        N_integration = 20
        
        # Definir límites de integración
        x1_min, x1_max = -p/2, p/2
        y1_min, y1_max = -q/2, q/2
        x2_min, x2_max = a - p/2, a + p/2
        y2_min, y2_max = b - q/2, b + q/2
        
        # Crear puntos de integración
        x1_pts = np.linspace(x1_min, x1_max, N_integration)
        y1_pts = np.linspace(y1_min, y1_max, N_integration)
        x2_pts = np.linspace(x2_min, x2_max, N_integration)
        y2_pts = np.linspace(y2_min, y2_max, N_integration)
        
        # Pesos de integración (trapezoidal simple)
        dx1 = x1_pts[1] - x1_pts[0]
        dy1 = y1_pts[1] - y1_pts[0]
        dx2 = x2_pts[1] - x2_pts[0]
        dy2 = y2_pts[1] - y2_pts[0]
        
        print("Calculando campo difractado...")
        print(f"Puntos de integración: {N_integration} por dimensión")
        start_time = time.time()
        
        # Precalcular campo en la segunda rendija
        print("Precalculando campo en la segunda rendija...")
        U_slit2 = np.zeros((N_integration, N_integration), dtype=complex)
        
        for i in range(N_integration):
            for j in range(N_integration):
                x2 = x2_pts[i]
                y2 = y2_pts[j]
                
                # Verificar si está dentro de la segunda rendija
                if (abs(x2 - a) <= p/2 and abs(y2 - b) <= q/2):
                    # Integrar sobre la primera rendija
                    sum_val = 0j
                    for k in range(N_integration):
                        for l in range(N_integration):
                            x1 = x1_pts[k]
                            y1 = y1_pts[l]
                            
                            # Verificar si está dentro de la primera rendija
                            if (abs(x1) <= p/2 and abs(y1) <= q/2):
                                # Distancia de la primera a la segunda rendija
                                r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + d1**2)
                                # Contribución de la primera rendija
                                sum_val += np.exp(1j * self.k * r12) / r12 * dx1 * dy1
                    
                    U_slit2[i, j] = sum_val
        
        # Factor de Huygens-Fresnel para la primera propagación
        prefactor1 = 1.0 / (1j * self.wavelength)
        
        # Propagación al plano de observación
        print("Propagando al plano de observación...")
        
        for i in range(Ny):
            for j in range(Nx):
                x0 = X_obs[i, j]
                y0 = Y_obs[i, j]
                
                # Integrar sobre la segunda rendija
                sum_val = 0j
                for m in range(N_integration):
                    for n_idx in range(N_integration):
                        x2 = x2_pts[m]
                        y2 = y2_pts[n_idx]
                        
                        # Verificar si está dentro de la segunda rendija
                        if (abs(x2 - a) <= p/2 and abs(y2 - b) <= q/2):
                            # Distancia de la segunda rendija al punto de observación
                            r20 = np.sqrt((x0 - x2)**2 + (y0 - y2)**2 + d2**2)
                            # Contribución total
                            sum_val += U_slit2[m, n_idx] * (np.exp(1j * self.k * r20) / r20) * dx2 * dy2
                
                # Factor de Huygens-Fresnel completo: (1/(iλ))^2
                prefactor = prefactor1 * (1.0 / (1j * self.wavelength))
                U_2d[i, j] = prefactor * sum_val
        
        print(f"Tiempo total de cálculo: {time.time() - start_time:.2f} s")
        
        # Calcular intensidad
        I_2d = np.abs(U_2d)**2
        
        # Normalizar intensidad
        if np.max(I_2d) > 0:
            I_2d = I_2d / np.max(I_2d)
        
        # Perfiles de intensidad
        I_x_profile = I_2d[Ny//2, :]
        I_y_profile = I_2d[:, Nx//2]
        
        # Campo complejo en los perfiles
        U_x_profile = U_2d[Ny//2, :]
        U_y_profile = U_2d[:, Nx//2]
        
        return {
            'x_obs': x_obs,
            'y_obs': y_obs,
            'U_2d': U_2d,
            'I_2d': I_2d,
            'I_x_profile': I_x_profile,
            'I_y_profile': I_y_profile,
            'U_x_profile': U_x_profile,
            'U_y_profile': U_y_profile,
            'params': params
        }

def plot_huygens_fresnel_results(results):
    """
    Grafica los resultados de la simulación de Huygens-Fresnel
    
    Parámetros:
    -----------
    results : dict
        Resultados de la simulación
    """
    x_obs = results['x_obs']
    y_obs = results['y_obs']
    I_2d = results['I_2d']
    I_x_profile = results['I_x_profile']
    I_y_profile = results['I_y_profile']
    params = results['params']
    
    # Crear figura con subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Patrón de difracción 2D
    ax1 = plt.subplot(2, 3, 1)
    im = ax1.imshow(I_2d, 
                    extent=[x_obs[0]*1000, x_obs[-1]*1000, 
                            y_obs[0]*1000, y_obs[-1]*1000], 
                    origin='lower', aspect='auto', cmap='hot')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.set_title('Patrón de difracción Huygens-Fresnel')
    plt.colorbar(im, ax=ax1, label='Intensidad normalizada')
    
    # Dibujar las posiciones de las rendijas proyectadas
    p_mm = params['p'] * 1000
    q_mm = params['q'] * 1000
    a_mm = params['a'] * 1000
    b_mm = params['b'] * 1000
    
    rect1 = Rectangle((-p_mm/2, -q_mm/2), p_mm, q_mm,
                      linewidth=1, edgecolor='cyan', facecolor='none', 
                      linestyle='--', label='Rendija 1')
    rect2 = Rectangle((a_mm - p_mm/2, b_mm - q_mm/2), p_mm, q_mm,
                      linewidth=1, edgecolor='lime', facecolor='none',
                      linestyle='--', label='Rendija 2')
    ax1.add_patch(rect1)
    ax1.add_patch(rect2)
    ax1.legend(loc='upper right', fontsize=8)
    
    # 2. Perfil de intensidad en x (corte en y=0)
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(x_obs * 1000, I_x_profile, 'b-', linewidth=2)
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('Intensidad normalizada')
    ax2.set_title('Perfil horizontal (y=0)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    # Marcar posiciones de las rendijas
    ax2.axvline(x=-p_mm/2, color='cyan', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axvline(x=p_mm/2, color='cyan', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axvline(x=a_mm - p_mm/2, color='lime', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axvline(x=a_mm + p_mm/2, color='lime', linestyle='--', alpha=0.5, linewidth=1)
    
    # 3. Perfil de intensidad en y (corte en x=0)
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(y_obs * 1000, I_y_profile, 'r-', linewidth=2)
    ax3.set_xlabel('y (mm)')
    ax3.set_ylabel('Intensidad normalizada')
    ax3.set_title('Perfil vertical (x=0)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.1])
    
    # Marcar posiciones de las rendijas
    ax3.axvline(x=-q_mm/2, color='cyan', linestyle='--', alpha=0.5, linewidth=1)
    ax3.axvline(x=q_mm/2, color='cyan', linestyle='--', alpha=0.5, linewidth=1)
    ax3.axvline(x=b_mm - q_mm/2, color='lime', linestyle='--', alpha=0.5, linewidth=1)
    ax3.axvline(x=b_mm + q_mm/2, color='lime', linestyle='--', alpha=0.5, linewidth=1)
    
    # 4. Diagrama de fase
    ax4 = plt.subplot(2, 3, 4)
    phase = np.angle(results['U_2d'])
    phase_im = ax4.imshow(phase, 
                         extent=[x_obs[0]*1000, x_obs[-1]*1000,
                                 y_obs[0]*1000, y_obs[-1]*1000],
                         origin='lower', aspect='auto', cmap='hsv',
                         vmin=-np.pi, vmax=np.pi)
    ax4.set_xlabel('x (mm)')
    ax4.set_ylabel('y (mm)')
    ax4.set_title('Fase del campo difractado')
    plt.colorbar(phase_im, ax=ax4, label='Fase (rad)')
    
    # 5. Diagrama de la configuración geométrica
    ax5 = plt.subplot(2, 3, 5)
    
    # Parámetros en mm
    p_mm = params['p'] * 1000
    q_mm = params['q'] * 1000
    a_mm = params['a'] * 1000
    b_mm = params['b'] * 1000
    n_mm = params['n'] * 1000
    c_mm = params['c'] * 1000
    z0_mm = params['z0'] * 1000
    
    # Dibujar rendijas
    rect1 = Rectangle((-p_mm/2, -q_mm/2), p_mm, q_mm, linewidth=2, edgecolor='blue', 
                      facecolor='lightblue', alpha=0.7, label=f'Rendija 1 (z={n_mm:.1f} mm)')
    ax5.add_patch(rect1)
    
    rect2 = Rectangle((a_mm - p_mm/2, b_mm - q_mm/2), p_mm, q_mm, linewidth=2, edgecolor='red', 
                      facecolor='pink', alpha=0.7, label=f'Rendija 2 (z={n_mm + c_mm:.1f} mm)')
    ax5.add_patch(rect2)
    
    # Indicar plano de observación
    obs_y = 0  # Representación en 2D
    ax5.plot([-x_obs[-1]*1000*0.8, x_obs[-1]*1000*0.8], [obs_y, obs_y], 
             color='green', linestyle='--', linewidth=2, 
             label=f'Observación (z={z0_mm:.1f} mm)')
    
    ax5.set_xlim([-max(abs(a_mm)+p_mm, p_mm)*1.2, max(abs(a_mm)+p_mm, p_mm)*1.2])
    ax5.set_ylim([-max(abs(b_mm)+q_mm, q_mm)*1.2, max(abs(b_mm)+q_mm, q_mm)*1.2])
    ax5.set_xlabel('x (mm)')
    ax5.set_ylabel('y (mm)')
    ax5.set_title('Configuración de rendijas (vista desde z)')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # 6. Información de parámetros y análisis
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calcular algunas propiedades
    I_2d = results['I_2d']
    contrast = (np.max(I_2d) - np.min(I_2d)) / (np.max(I_2d) + np.min(I_2d) + 1e-10)
    
    # Número de Fresnel para referencia
    d_total = params['z0'] - params['n']
    Fresnel_number = (params['p']**2) / (params['wavelength'] * d_total)
    
    param_text = (
        f"Parámetros de simulación:\n\n"
        f"λ = {params.get('wavelength', 632.8e-9)*1e9:.1f} nm\n"
        f"p = {p_mm:.3f} mm\n"
        f"q = {q_mm:.3f} mm\n"
        f"a = {a_mm:.3f} mm\n"
        f"b = {b_mm:.3f} mm\n"
        f"n = {n_mm:.1f} mm\n"
        f"c = {c_mm:.1f} mm\n"
        f"z0 = {z0_mm:.1f} mm\n"
        f"d1 = {c_mm:.1f} mm\n"
        f"d2 = {z0_mm - (n_mm + c_mm):.1f} mm\n\n"
        f"Propiedades del patrón:\n"
        f"Contraste = {contrast:.3f}\n"
        f"N° Fresnel = {Fresnel_number:.3f}\n"
    )
    
    if Fresnel_number > 1:
        region_text = "Región: Campo cercano"
    else:
        region_text = "Región: Campo lejano"
    
    ax6.text(0.1, 0.5, param_text + region_text, fontsize=10, 
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Difracción de Huygens-Fresnel: Dos rendijas rectangulares', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()

def main():
    """
    Función principal para ejecutar la simulación de Huygens-Fresnel
    """
    # Definir parámetros del sistema (en metros)
    params = {
        'p': 0.5e-3,      # 0.5 mm de ancho en x
        'q': 0.5e-3,      # 0.5 mm de alto en y
        'a': 1.0e-3,      # 1 mm desplazamiento en x de la segunda rendija
        'b': 0.0e-3,      # 0 mm desplazamiento en y de la segunda rendija
        'n': 50e-3,       # 50 mm desde el origen a la primera rendija
        'c': 10e-3,       # 10 mm entre rendijas
        'z0': 200e-3,     # 200 mm al plano de observación
        'wavelength': 632.8e-9  # Longitud de onda de luz He-Ne
    }
    
    print("SIMULACIÓN DE DIFRACCIÓN DE HUYGENS-FRESNEL")
    print("=" * 70)
    print(f"Longitud de onda: {params['wavelength']*1e9:.1f} nm")
    print(f"Dimensiones rendijas: {params['p']*1000:.2f} mm x {params['q']*1000:.2f} mm")
    print(f"Posición rendija 1: (0, 0, {params['n']*1000:.1f}) mm")
    print(f"Posición rendija 2: ({params['a']*1000:.2f}, {params['b']*1000:.2f}, "
          f"{params['n']*1000 + params['c']*1000:.1f}) mm")
    print(f"Plano observación: z = {params['z0']*1000:.1f} mm")
    print("=" * 70)
    
    # Crear simulador
    simulator = HuygensFresnelDiffractionSimulator(wavelength=params['wavelength'])
    
    # Ejecutar simulación
    print("\nIniciando cálculo de difracción...")
    
    results = simulator.simulate_double_slit_huygens(
        params, 
        Nx=100,           # Número de puntos en x
        Ny=100,           # Número de puntos en y
        x_range=10e-3,    # 10 mm de rango en x
        y_range=10e-3     # 10 mm de rango en y
    )
    
    # Graficar resultados
    plot_huygens_fresnel_results(results)
    
    # Mostrar información adicional
    print("\n" + "="*70)
    print("ANÁLISIS DEL PATRÓN DE DIFRACCIÓN")
    print("="*70)
    
    I_2d = results['I_2d']
    max_intensity = np.max(I_2d)
    mean_intensity = np.mean(I_2d)
    min_intensity = np.min(I_2d)
    
    print(f"Intensidad máxima: {max_intensity:.4f}")
    print(f"Intensidad mínima: {min_intensity:.4f}")
    print(f"Intensidad promedio: {mean_intensity:.4f}")
    
    # Calcular contraste
    contrast = (max_intensity - min_intensity) / (max_intensity + min_intensity)
    print(f"Contraste del patrón: {contrast:.3f}")
    
    # Calcular número de Fresnel
    d_total = params['z0'] - params['n']
    Fresnel_number = (params['p']**2) / (params['wavelength'] * d_total)
    print(f"\nNúmero de Fresnel: {Fresnel_number:.3f}")
    
    if Fresnel_number > 1:
        print("Región: Campo cercano (difracción de Fresnel)")
    else:
        print("Región: Campo lejano (difracción de Fraunhofer)")
    
    # Encontrar máximos en el perfil horizontal
    print("\nAnálisis del perfil horizontal:")
    I_x = results['I_x_profile']
    x_mm = results['x_obs'] * 1000
    
    # Usar un umbral simple para encontrar máximos
    threshold = 0.1
    maxima_indices = []
    for i in range(1, len(I_x)-1):
        if I_x[i] > I_x[i-1] and I_x[i] > I_x[i+1] and I_x[i] > threshold:
            maxima_indices.append(i)
    
    if len(maxima_indices) > 0:
        print(f"Se encontraron {len(maxima_indices)} máximos principales:")
        for i, idx in enumerate(maxima_indices[:5]):  # Mostrar solo los primeros 5
            print(f"  Máximo {i+1}: x = {x_mm[idx]:.2f} mm, Intensidad = {I_x[idx]:.3f}")
    else:
        print("No se encontraron máximos significativos en el perfil horizontal")
    
    # Opción para guardar resultados
    save_option = input("\n¿Desea guardar los resultados? (s/n): ")
    if save_option.lower() == 's':
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"huygens_fresnel_results_{timestamp}.png"
        plt.figure(fig.number)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Resultados guardados como {filename}")
        
        # Guardar datos numéricos
        data = {
            'x_mm': x_mm,
            'y_mm': results['y_obs'] * 1000,
            'intensity': results['I_2d'],
            'intensity_normalized': I_2d,
            'phase': np.angle(results['U_2d']),
            'parameters': {
                'lambda_nm': params['wavelength'] * 1e9,
                'p_mm': params['p'] * 1e3,
                'q_mm': params['q'] * 1e3,
                'n_mm': params['n'] * 1e3,
                'c_mm': params['c'] * 1e3,
                'a_mm': params['a'] * 1e3,
                'b_mm': params['b'] * 1e3,
                'z0_mm': params['z0'] * 1e3
            }
        }
        np.savez(f"huygens_data_{timestamp}.npz", **data)
        print(f"Datos numéricos guardados como huygens_data_{timestamp}.npz")

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time

class FresnelDiffractionSimulator:
    def __init__(self, wavelength=632.8e-9):
        """
        Inicializa el simulador de difracción de Fresnel
        
        Parámetros:
        -----------
        wavelength : float
            Longitud de onda en metros (default: 632.8 nm, luz He-Ne)
        """
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength  # Número de onda
        
    def simpson_adaptive_1d(self, func, a, b, tol=1e-6, max_depth=15):
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
            if depth >= max_depth:
                return whole
            
            mid = (left + right) / 2
            f_left_mid = func((left + mid) / 2)
            f_right_mid = func((mid + right) / 2)
            
            left_half = (mid - left) * (f_left + 4 * f_left_mid + f_mid) / 6
            right_half = (right - mid) * (f_mid + 4 * f_right_mid + f_right) / 6
            total = left_half + right_half
            
            if abs(total - whole) <= 15 * tol:
                return total + (total - whole) / 15
            
            left_result = recursive_simpson(left, mid, f_left, f_mid, f_left_mid, left_half, depth + 1)
            right_result = recursive_simpson(mid, right, f_mid, f_right, f_right_mid, right_half, depth + 1)
            
            return left_result + right_result
        
        f_a = func(a)
        f_b = func(b)
        f_mid = func((a + b) / 2)
        whole = (b - a) * (f_a + 4 * f_mid + f_b) / 6
        
        return recursive_simpson(a, b, f_a, f_b, f_mid, whole, 0)
    
    def simpson_adaptive_2d(self, func, ax, bx, ay, by, tol=1e-4, max_depth=8):
        """
        Método de Simpson adaptativo para integración 2D (anidado)
        
        Parámetros:
        -----------
        func : function(x, y)
            Función a integrar en 2D
        ax, bx : float
            Límites de integración en x
        ay, by : float
            Límites de integración en y
        tol : float
            Tolerancia relativa
        max_depth : int
            Profundidad máxima de recursión
        
        Retorna:
        --------
        complex : Valor de la integral 2D
        """
        # Función auxiliar que integra en y para un x fijo
        def integrate_y_for_fixed_x(x):
            def func_y(y):
                return func(x, y)
            return self.simpson_adaptive_1d(func_y, ay, by, tol=tol/10, max_depth=max_depth-2)
        
        # Integrar en x, llamando a la integración en y para cada x
        return self.simpson_adaptive_1d(integrate_y_for_fixed_x, ax, bx, tol=tol, max_depth=max_depth)
    
    def fresnel_kernel_2d(self, x0, y0, x, y, d):
        """
        Kernel de Fresnel 2D completo
        
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
        complex : Valor del kernel 2D
        """
        if abs(d) < 1e-12:
            return 0
        r2 = (x0 - x)**2 + (y0 - y)**2
        return np.exp(1j * self.k * r2 / (2 * d))
    
    def huygens_fresnel_kernel_2d(self, x0, y0, x, y, d):
        """
        Kernel de Huygens-Fresnel 2D completo (versión estabilizada)
        
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
        complex : Valor del kernel 2D
        """
        if abs(d) < 1e-12:
            return 0
        
        # Distancia exacta
        dx = x0 - x
        dy = y0 - y
        r = np.sqrt(dx**2 + dy**2 + d**2)
        
        # Para evitar problemas numéricos con distancias pequeñas
        if r < 1e-10:
            return 0
        
        # Factor de oblicuidad (aproximación para ángulos pequeños)
        cos_theta = d / r
        
        # Kernel de Huygens-Fresnel
        return (cos_theta / r) * np.exp(1j * self.k * r)
    
    def rect_function_2d(self, x, y, center_x, center_y, width_x, width_y):
        """
        Función rectangular 2D (apertura)
        
        Parámetros:
        -----------
        x, y : float
            Posición
        center_x, center_y : float
            Centro de la rendija
        width_x, width_y : float
            Ancho y alto de la rendija
        
        Retorna:
        --------
        float : 1 si está dentro, 0 fuera
        """
        if abs(x - center_x) <= width_x/2 and abs(y - center_y) <= width_y/2:
            return 1.0
        return 0.0
    
    def calculate_field_at_slit2_2d(self, x2, y2, params, method='fresnel'):
        """
        Calcula el campo en la segunda rendija debido a la primera rendija (2D)
        
        Parámetros:
        -----------
        x2, y2 : float
            Punto en la segunda rendija
        params : dict
            Parámetros del sistema
        method : str
            'fresnel' o 'huygens_fresnel'
        
        Retorna:
        --------
        complex : Campo complejo en (x2, y2)
        """
        p = params['p']
        q = params['q']
        c = params['c']
        
        # Seleccionar kernel según el método
        if method == 'fresnel':
            kernel_func = lambda x1, y1: self.fresnel_kernel_2d(x2, y2, x1, y1, c)
            prefactor = np.exp(1j * self.k * c) / (1j * self.wavelength * c)
        else:  # huygens_fresnel
            kernel_func = lambda x1, y1: self.huygens_fresnel_kernel_2d(x2, y2, x1, y1, c)
            prefactor = 1 / (1j * self.wavelength)
        
        # Función a integrar sobre la primera rendija
        def integrand(x1, y1):
            # Transmitancia de la primera rendija
            transmission = self.rect_function_2d(x1, y1, 0, 0, p, q)
            if transmission == 0:
                return 0
            return transmission * kernel_func(x1, y1)
        
        # Integrar sobre la primera rendija
        integral = self.simpson_adaptive_2d(
            integrand, 
            -p/2, p/2, 
            -q/2, q/2,
            tol=1e-5, 
            max_depth=6
        )
        
        return prefactor * integral
    
    def calculate_field_observation_2d(self, x0, y0, params, method='fresnel'):
        """
        Calcula el campo en el plano de observación (2D)
        
        Parámetros:
        -----------
        x0, y0 : float
            Punto de observación
        params : dict
            Parámetros del sistema
        method : str
            'fresnel' o 'huygens_fresnel'
        
        Retorna:
        --------
        complex : Campo complejo en (x0, y0)
        """
        p = params['p']
        q = params['q']
        a = params['a']
        b = params['b']
        n = params['n']
        c = params['c']
        z0 = params['z0']
        d2 = z0 - (n + c)
        
        # Seleccionar kernel según el método
        if method == 'fresnel':
            kernel_func = lambda x2, y2: self.fresnel_kernel_2d(x0, y0, x2, y2, d2)
            prefactor = np.exp(1j * self.k * d2) / (1j * self.wavelength * d2)
        else:  # huygens_fresnel
            kernel_func = lambda x2, y2: self.huygens_fresnel_kernel_2d(x0, y0, x2, y2, d2)
            prefactor = 1 / (1j * self.wavelength)
        
        # Función a integrar sobre la segunda rendija
        def integrand(x2, y2):
            # Transmitancia de la segunda rendija
            transmission = self.rect_function_2d(x2, y2, a, b, p, q)
            if transmission == 0:
                return 0
            
            # Campo en la segunda rendija
            field_at_slit2 = self.calculate_field_at_slit2_2d(x2, y2, params, method)
            
            return field_at_slit2 * transmission * kernel_func(x2, y2)
        
        # Integrar sobre la segunda rendija
        integral = self.simpson_adaptive_2d(
            integrand,
            a - p/2, a + p/2,
            b - q/2, b + q/2,
            tol=1e-5,
            max_depth=6
        )
        
        return prefactor * integral
    
    def simulate_2d_diffraction_complete(self, params, Nx=100, Ny=100, x_range=0.01, y_range=0.01):
        """
        Simula el patrón de difracción 2D completo para ambos métodos
        
        Parámetros:
        -----------
        params : dict
            Diccionario con los parámetros del sistema
        Nx, Ny : int
            Número de puntos en x e y
        x_range, y_range : float
            Rango de observación en metros
        
        Retorna:
        --------
        dict : Resultados de la simulación para ambos métodos
        """
        # Crear mallas de observación
        x_obs = np.linspace(-x_range/2, x_range/2, Nx)
        y_obs = np.linspace(-y_range/2, y_range/2, Ny)
        
        # Precalcular campos para Fresnel
        print("Calculando patrón de Fresnel (2D completo)...")
        start_time = time.time()
        
        I_fresnel = np.zeros((Nx, Ny))
        U_fresnel = np.zeros((Nx, Ny), dtype=complex)
        
        for i in range(Nx):
            if i % 10 == 0:
                print(f"  Progreso Fresnel: {i}/{Nx} puntos en x")
            for j in range(Ny):
                U = self.calculate_field_observation_2d(x_obs[i], y_obs[j], params, 'fresnel')
                U_fresnel[i, j] = U
                I_fresnel[i, j] = np.abs(U)**2
        
        fresnel_time = time.time() - start_time
        print(f"Tiempo cálculo Fresnel: {fresnel_time:.2f} s")
        
        # Precalcular campos para Huygens-Fresnel
        print("\nCalculando patrón de Huygens-Fresnel (2D completo)...")
        start_time = time.time()
        
        I_hf = np.zeros((Nx, Ny))
        U_hf = np.zeros((Nx, Ny), dtype=complex)
        
        for i in range(Nx):
            if i % 10 == 0:
                print(f"  Progreso Huygens-Fresnel: {i}/{Nx} puntos en x")
            for j in range(Ny):
                U = self.calculate_field_observation_2d(x_obs[i], y_obs[j], params, 'huygens_fresnel')
                U_hf[i, j] = U
                I_hf[i, j] = np.abs(U)**2
        
        hf_time = time.time() - start_time
        print(f"Tiempo cálculo Huygens-Fresnel: {hf_time:.2f} s")
        
        # Normalizar intensidades
        I_fresnel_norm = I_fresnel / np.max(I_fresnel) if np.max(I_fresnel) > 0 else I_fresnel
        I_hf_norm = I_hf / np.max(I_hf) if np.max(I_hf) > 0 else I_hf
        
        # Perfiles en x (y=0)
        center_y_idx = Ny // 2
        I_x_fresnel = I_fresnel_norm[:, center_y_idx]
        I_x_hf = I_hf_norm[:, center_y_idx]
        
        return {
            'x_obs': x_obs,
            'y_obs': y_obs,
            'I_fresnel': I_fresnel_norm,
            'I_hf': I_hf_norm,
            'I_x_fresnel': I_x_fresnel,
            'I_x_hf': I_x_hf,
            'U_fresnel': U_fresnel,
            'U_hf': U_hf,
            'params': params,
            'times': {'fresnel': fresnel_time, 'huygens_fresnel': hf_time}
        }

def plot_results_4panels(results):
    """
    Grafica los resultados en 4 paneles como solicitaste
    
    Fila 1: Patrones 2D (Fresnel y Huygens-Fresnel)
    Fila 2: Perfiles en x (Fresnel y Huygens-Fresnel)
    """
    x_obs = results['x_obs']
    y_obs = results['y_obs']
    I_fresnel = results['I_fresnel']
    I_hf = results['I_hf']
    I_x_fresnel = results['I_x_fresnel']
    I_x_hf = results['I_x_hf']
    params = results['params']
    
    # Crear figura con 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Patrón de Fresnel 2D
    ax1 = axes[0, 0]
    im1 = ax1.imshow(I_fresnel.T, 
                     extent=[x_obs[0]*1000, x_obs[-1]*1000, 
                             y_obs[0]*1000, y_obs[-1]*1000], 
                     origin='lower', aspect='auto', cmap='hot',
                     vmin=0, vmax=1)
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.set_title('Difracción de Fresnel (2D completo)')
    plt.colorbar(im1, ax=ax1, label='Intensidad normalizada')
    
    # 2. Patrón de Huygens-Fresnel 2D
    ax2 = axes[0, 1]
    im2 = ax2.imshow(I_hf.T, 
                     extent=[x_obs[0]*1000, x_obs[-1]*1000,
                             y_obs[0]*1000, y_obs[-1]*1000], 
                     origin='lower', aspect='auto', cmap='hot',
                     vmin=0, vmax=1)
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('y (mm)')
    ax2.set_title('Difracción de Huygens-Fresnel (2D completo)')
    plt.colorbar(im2, ax=ax2, label='Intensidad normalizada')
    
    # 3. Perfil en x para Fresnel
    ax3 = axes[1, 0]
    ax3.plot(x_obs * 1000, I_x_fresnel, 'b-', linewidth=2)
    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('Intensidad normalizada')
    ax3.set_title('Perfil en x (y=0) - Fresnel')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.1])
    
    # 4. Perfil en x para Huygens-Fresnel
    ax4 = axes[1, 1]
    ax4.plot(x_obs * 1000, I_x_hf, 'r-', linewidth=2)
    ax4.set_xlabel('x (mm)')
    ax4.set_ylabel('Intensidad normalizada')
    ax4.set_title('Perfil en x (y=0) - Huygens-Fresnel')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.1])
    
    # Añadir información de parámetros
    param_text = (
        f"Parámetros:\n"
        f"λ = {params['wavelength']*1e9:.1f} nm\n"
        f"Rendija: {params['p']*1000:.2f}×{params['q']*1000:.2f} mm\n"
        f"Desplazamiento: ({params['a']*1000:.2f}, {params['b']*1000:.2f}) mm\n"
        f"d1 = {params['c']*1000:.1f} mm, d2 = {params['z0'] - params['n'] - params['c']:.1f} mm\n"
        f"Tiempo Fresnel: {results['times']['fresnel']:.1f} s\n"
        f"Tiempo H-F: {results['times']['huygens_fresnel']:.1f} s"
    )
    
    plt.figtext(0.02, 0.02, param_text, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    # Gráfica comparativa adicional
    fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
    ax_comp.plot(x_obs * 1000, I_x_fresnel, 'b-', linewidth=2, label='Fresnel')
    ax_comp.plot(x_obs * 1000, I_x_hf, 'r--', linewidth=2, label='Huygens-Fresnel')
    ax_comp.set_xlabel('x (mm)')
    ax_comp.set_ylabel('Intensidad normalizada')
    ax_comp.set_title('Comparación de perfiles en x (y=0)')
    ax_comp.grid(True, alpha=0.3)
    ax_comp.legend()
    ax_comp.set_ylim([0, 1.1])
    
    # Calcular diferencia
    diff = np.abs(I_x_fresnel - I_x_hf)
    avg_diff = np.mean(diff)
    max_diff = np.max(diff)
    
    diff_text = f"Diferencia promedio: {avg_diff:.4f}\nDiferencia máxima: {max_diff:.4f}"
    ax_comp.text(0.02, 0.98, diff_text, transform=ax_comp.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Función principal para ejecutar la simulación 2D completa
    """
    # Crear simulador
    simulator = FresnelDiffractionSimulator(wavelength=632.8e-9)
    
    # Definir parámetros (valores más pequeños para prueba rápida)
    params = {
        'p': 0.2e-3,      # 0.2 mm de ancho en x (reducido)
        'q': 0.2e-3,      # 0.2 mm de alto en y (reducido)
        'a': 0.5e-3,      # 0.5 mm desplazamiento en x
        'b': 0.3e-3,      # 0.3 mm desplazamiento en y
        'n': 20e-3,       # 20 mm desde origen a primera rendija
        'c': 5e-3,        # 5 mm entre rendijas
        'z0': 100e-3,     # 100 mm al plano de observación
        'wavelength': 632.8e-9
    }
    
    print("=" * 70)
    print("SIMULACIÓN 2D COMPLETA: FRESNEL vs HUYGENS-FRESNEL")
    print("=" * 70)
    print(f"Longitud de onda: {params['wavelength']*1e9:.1f} nm")
    print(f"Dimensiones rendijas: {params['p']*1000:.2f} × {params['q']*1000:.2f} mm")
    print(f"Posición rendija 1: (0, 0, {params['n']*1000:.1f}) mm")
    print(f"Posición rendija 2: ({params['a']*1000:.2f}, {params['b']*1000:.2f}, "
          f"{params['n']*1000 + params['c']*1000:.1f}) mm")
    print(f"Plano observación: z = {params['z0']*1000:.1f} mm")
    print("=" * 70)
    
    print("\nNOTA: Esta simulación 2D completa puede tomar varios minutos.")
    print("      Se usan integración adaptativa 2D y pocos puntos para velocidad.\n")
    
    # Ejecutar simulación
    start_time = time.time()
    
    results = simulator.simulate_2d_diffraction_complete(
        params,
        Nx=100,           # Puntos en x (reducido por tiempo)
        Ny=100,           # Puntos en y (reducido por tiempo)
        x_range=8e-3,    # 8 mm de rango
        y_range=8e-3     # 8 mm de rango
    )
    
    total_time = time.time() - start_time
    print(f"\nTiempo total de simulación: {total_time:.2f} segundos")
    
    # Graficar resultados
    plot_results_4panels(results)
    
    # Información adicional
    print("\n" + "=" * 50)
    print("RESUMEN DE RESULTADOS")
    print("=" * 50)
    print(f"Máxima intensidad Fresnel: {np.max(results['I_fresnel']):.4f}")
    print(f"Máxima intensidad Huygens-Fresnel: {np.max(results['I_hf']):.4f}")
    
    # Número de Fresnel
    d_total = params['z0'] - params['n']
    Fresnel_number = (params['p']**2) / (params['wavelength'] * d_total)
    print(f"\nNúmero de Fresnel: {Fresnel_number:.3f}")
    
    if Fresnel_number > 1:
        print("Región: Difracción de Fresnel (campo cercano)")
        print("Las diferencias entre métodos pueden ser notables.")
    else:
        print("Región: Difracción de Fraunhofer (campo lejano)")
        print("Ambos métodos deberían converger.")
    
    # Diferencia entre métodos
    diff_total = np.mean(np.abs(results['I_fresnel'] - results['I_hf']))
    print(f"\nDiferencia promedio total: {diff_total:.4f}")
    
    # Mostrar consejo para simulaciones más rápidas
    print("\n" + "=" * 50)
    print("CONSEJOS PARA SIMULACIONES MÁS RÁPIDAS:")
    print("=" * 50)
    print("1. Reducir Nx y Ny (ej: 40x40)")
    print("2. Reducir el rango de observación")
    print("3. Usar valores más grandes de tolerancia en simpson_adaptive_2d")
    print("4. Considerar usar métodos FFT para cálculos 2D completos")

if __name__ == "__main__":
    main()
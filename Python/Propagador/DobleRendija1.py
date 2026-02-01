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
    
    def fresnel_kernel_1d(self, x0, x, d):
        """
        Kernel de Fresnel 1D (sin prefactor)
        
        Parámetros:
        -----------
        x0 : float
            Punto de observación
        x : float
            Punto en la apertura
        d : float
            Distancia de propagación
        
        Retorna:
        --------
        complex : Valor del kernel
        """
        if abs(d) < 1e-12:
            return 0
        return np.exp(1j * self.k * (x0 - x)**2 / (2 * d))
    
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
    
    def calculate_field_at_slit2(self, x2, p, c):
        """
        Calcula el campo en la segunda rendija debido a la primera rendija
        
        Parámetros:
        -----------
        x2 : float
            Punto en la segunda rendija
        p : float
            Ancho de las rendijas en x
        c : float
            Distancia entre rendijas
        
        Retorna:
        --------
        complex : Campo complejo en x2
        """
        # Integral sobre la primera rendija (centrada en 0)
        def integrand(x1):
            return self.rect_function(x1, 0, p) * self.fresnel_kernel_1d(x2, x1, c)
        
        # Usar Simpson adaptativo para integrar sobre la primera rendija
        integral = self.simpson_adaptive(integrand, -p/2, p/2, tol=1e-8)
        
        # Factor de propagación para 1D
        prefactor = np.exp(1j * self.k * c) / np.sqrt(1j * self.wavelength * c)
        
        return prefactor * integral
    
    def calculate_field_observation(self, x0, p, a, c, d2):
        """
        Calcula el campo en el plano de observación
        
        Parámetros:
        -----------
        x0 : float
            Punto de observación
        p : float
            Ancho de las rendijas en x
        a : float
            Desplazamiento en x de la segunda rendija
        c : float
            Distancia entre rendijas
        d2 : float
            Distancia desde la segunda rendija al plano de observación
        
        Retorna:
        --------
        complex : Campo complejo en x0
        """
        # Integral sobre la segunda rendija
        def integrand(x2):
            # Campo en la segunda rendija
            field_at_slit2 = self.calculate_field_at_slit2(x2, p, c)
            # Transmitancia de la segunda rendija
            transmission = self.rect_function(x2, a, p)
            # Kernel para la segunda propagación
            kernel = self.fresnel_kernel_1d(x0, x2, d2)
            
            return field_at_slit2 * transmission * kernel
        
        # Usar Simpson adaptativo para integrar sobre la segunda rendija
        integral = self.simpson_adaptive(integrand, a - p/2, a + p/2, tol=1e-8)
        
        # Factor de propagación para 1D
        prefactor = np.exp(1j * self.k * d2) / np.sqrt(1j * self.wavelength * d2)
        
        return prefactor * integral
    
    def simulate_2d_diffraction(self, params, Nx=200, Ny=200, x_range=0.01, y_range=0.01):
        """
        Simula el patrón de difracción 2D completo
        
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
        
        # Distancia de la segunda rendija al plano de observación
        d2 = z0 - (n + c)
        
        # Crear mallas de observación
        x_obs = np.linspace(-x_range/2, x_range/2, Nx)
        y_obs = np.linspace(-y_range/2, y_range/2, Ny)
        
        # Calcular campos en x
        print("Calculando campo en dirección x...")
        start_time = time.time()
        
        U_x = np.zeros(Nx, dtype=complex)
        for i, x0 in enumerate(x_obs):
            U_x[i] = self.calculate_field_observation(x0, p, a, c, d2)
        
        print(f"Tiempo cálculo en x: {time.time() - start_time:.2f} s")
        
        # Calcular campos en y
        print("Calculando campo en dirección y...")
        start_time = time.time()
        
        U_y = np.zeros(Ny, dtype=complex)
        for j, y0 in enumerate(y_obs):
            # Para y, usamos los mismos parámetros pero con q y b
            def integrand(y2):
                # Campo en la segunda rendija en y (similar a x)
                def y_integrand(y1):
                    return self.rect_function(y1, 0, q) * self.fresnel_kernel_1d(y2, y1, c)
                
                integral = self.simpson_adaptive(y_integrand, -q/2, q/2, tol=1e-8)
                field_at_slit2_y = np.exp(1j * self.k * c) / np.sqrt(1j * self.wavelength * c) * integral
                
                transmission = self.rect_function(y2, b, q)
                kernel = self.fresnel_kernel_1d(y0, y2, d2)
                
                return field_at_slit2_y * transmission * kernel
            
            integral = self.simpson_adaptive(integrand, b - q/2, b + q/2, tol=1e-8)
            prefactor = np.exp(1j * self.k * d2) / np.sqrt(1j * self.wavelength * d2)
            U_y[j] = prefactor * integral
        
        print(f"Tiempo cálculo en y: {time.time() - start_time:.2f} s")
        
        # Combinar para obtener campo 2D (aproximación separable)
        print("Combinando para campo 2D...")
        U_2d = np.outer(U_x, U_y)
        
        # Factor global 2D (corrección por separabilidad)
        # Para 2D, el factor correcto sería: exp(i*k*(z0-n)) / (i*λ*(z0-n))
        # Pero nuestras integrales 1D ya incluyen factores 1/sqrt(i*λ*d)
        # El producto de dos factores 1/sqrt(i*λ*d) = 1/(i*λ*d)
        global_factor = np.exp(1j * self.k * (z0 - n)) / (1j * self.wavelength * (z0 - n))
        U_2d *= global_factor
        
        # Calcular intensidad
        I_2d = np.abs(U_2d)**2
        
        # Normalizar intensidad
        I_2d = I_2d / np.max(I_2d)
        
        # Perfiles de intensidad
        I_x_profile = np.abs(U_x)**2
        I_x_profile = I_x_profile / np.max(I_x_profile) if np.max(I_x_profile) > 0 else I_x_profile
        
        I_y_profile = np.abs(U_y)**2
        I_y_profile = I_y_profile / np.max(I_y_profile) if np.max(I_y_profile) > 0 else I_y_profile
        
        return {
            'x_obs': x_obs,
            'y_obs': y_obs,
            'U_x': U_x,
            'U_y': U_y,
            'U_2d': U_2d,
            'I_2d': I_2d,
            'I_x_profile': I_x_profile,
            'I_y_profile': I_y_profile,
            'params': params
        }

def plot_results(results):
    """
    Grafica los resultados de la simulación
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
    im = ax1.imshow(I_2d, extent=[x_obs[0]*1000, x_obs[-1]*1000, y_obs[0]*1000, y_obs[-1]*1000], 
                    origin='lower', aspect='auto', cmap='hot')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.set_title('Patrón de difracción 2D')
    plt.colorbar(im, ax=ax1, label='Intensidad (normalizada)')
    
    # 2. Perfil de intensidad en x
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(x_obs * 1000, I_x_profile, 'b-', linewidth=2)
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('Intensidad (normalizada)')
    ax2.set_title('Perfil de intensidad en x')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    # 3. Perfil de intensidad en y
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(y_obs * 1000, I_y_profile, 'r-', linewidth=2)
    ax3.set_xlabel('y (mm)')
    ax3.set_ylabel('Intensidad (normalizada)')
    ax3.set_title('Perfil de intensidad en y')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.1])
    
    # 4. Diagrama de fase 2D
    ax4 = plt.subplot(2, 3, 4)
    phase = np.angle(results['U_2d'])
    im2 = ax4.imshow(phase, extent=[x_obs[0]*1000, x_obs[-1]*1000, y_obs[0]*1000, y_obs[-1]*1000], 
                     origin='lower', aspect='auto', cmap='hsv')
    ax4.set_xlabel('x (mm)')
    ax4.set_ylabel('y (mm)')
    ax4.set_title('Fase del campo (radianes)')
    plt.colorbar(im2, ax=ax4)
    
    # 5. Diagrama de la configuración
    ax5 = plt.subplot(2, 3, 5)
    
    # Parámetros
    p = params['p'] * 1000  # a mm
    q = params['q'] * 1000  # a mm
    a = params['a'] * 1000  # a mm
    b = params['b'] * 1000  # a mm
    n = params['n'] * 1000  # a mm
    c = params['c'] * 1000  # a mm
    z0 = params['z0'] * 1000  # a mm
    
    # Dibujar rendijas
    # Rendija 1 en z = n
    rect1 = Rectangle((-p/2, -q/2), p, q, linewidth=2, edgecolor='blue', 
                      facecolor='lightblue', alpha=0.7, label='Rendija 1')
    ax5.add_patch(rect1)
    
    # Rendija 2 en z = n + c
    rect2 = Rectangle((a - p/2, b - q/2), p, q, linewidth=2, edgecolor='red', 
                      facecolor='pink', alpha=0.7, label='Rendija 2')
    ax5.add_patch(rect2)
    
    # Plano de observación en z = z0
    ax5.axhline(y=0, xmin=-0.5, xmax=0.5, color='green', linestyle='--', 
                linewidth=2, label='Plano observación')
    
    ax5.set_xlim([-max(abs(a)+p, p)*1.2, max(abs(a)+p, p)*1.2])
    ax5.set_ylim([-max(abs(b)+q, q)*1.2, max(abs(b)+q, q)*1.2])
    ax5.set_xlabel('x (mm)')
    ax5.set_ylabel('y (mm)')
    ax5.set_title('Configuración de rendijas (vista desde z)')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # 6. Información de parámetros
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    param_text = (
        f"Parámetros de simulación:\n\n"
        f"λ = {params.get('wavelength', 632.8e-9)*1e9:.1f} nm\n"
        f"p = {p:.3f} mm\n"
        f"q = {q:.3f} mm\n"
        f"a = {a:.3f} mm\n"
        f"b = {b:.3f} mm\n"
        f"n = {n:.1f} mm\n"
        f"c = {c:.1f} mm\n"
        f"z0 = {z0:.1f} mm\n"
        f"d2 = {z0 - (n + c):.1f} mm"
    )
    ax6.text(0.1, 0.5, param_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Función principal para ejecutar la simulación
    """
    # Crear simulador
    simulator = FresnelDiffractionSimulator(wavelength=632.8e-9)  # Luz He-Ne
    
    # Definir parámetros del sistema (en metros)
    params = {
        'p': 0.5e-3,      # 0.5 mm de ancho en x
        'q': 0.5e-3,      # 0.5 mm de alto en y
        'a': 1.0e-3,      # 1 mm desplazamiento en x de la segunda rendija
        'b': 0.0e-3,      # 0 mm desplazamiento en y de la segunda rendija
        'n': 50e-3,       # 50 mm desde el origen a la primera rendija
        'c': 10e-3,       # 10 mm entre rendijas
        'z0': 200e-3,     # 200 mm al plano de observación
        'wavelength': 632.8e-9
    }
    
    print("Simulación de difracción de Fresnel para dos rendijas rectangulares")
    print("=" * 70)
    print(f"Longitud de onda: {params['wavelength']*1e9:.1f} nm")
    print(f"Dimensiones rendijas: {params['p']*1000:.2f} mm x {params['q']*1000:.2f} mm")
    print(f"Posición rendija 1: (0, 0, {params['n']*1000:.1f}) mm")
    print(f"Posición rendija 2: ({params['a']*1000:.2f}, {params['b']*1000:.2f}, "
          f"{params['n']*1000 + params['c']*1000:.1f}) mm")
    print(f"Plano observación: z = {params['z0']*1000:.1f} mm")
    print("=" * 70)
    
    # Ejecutar simulación
    print("\nIniciando cálculo de difracción...")
    start_time = time.time()
    
    results = simulator.simulate_2d_diffraction(
        params, 
        Nx=150,           # Número de puntos en x (reducido para velocidad)
        Ny=150,           # Número de puntos en y
        x_range=10e-3,    # 10 mm de rango en x
        y_range=10e-3     # 10 mm de rango en y
    )
    
    print(f"\nTiempo total de simulación: {time.time() - start_time:.2f} segundos")
    
    # Graficar resultados
    plot_results(results)
    
    # Mostrar información adicional
    print("\nInformación adicional:")
    print(f"Máxima intensidad: {np.max(results['I_2d']):.4f}")
    print(f"Intensidad promedio: {np.mean(results['I_2d']):.4f}")
    
    # Calcular y mostrar el número de Fresnel
    d_total = params['z0'] - params['n']
    Fresnel_number = (params['p']**2) / (params['wavelength'] * d_total)
    print(f"Número de Fresnel aproximado: {Fresnel_number:.3f}")
    
    if Fresnel_number > 1:
        print("Región: Difracción de Fresnel (campo cercano)")
    else:
        print("Región: Difracción de Fraunhofer (campo lejano)")

if __name__ == "__main__":
    main()
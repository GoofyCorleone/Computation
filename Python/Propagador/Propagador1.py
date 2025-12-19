import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple

# ============================================================================
# 1. MÉTODO DE INTEGRACIÓN NUMÉRICA
# ============================================================================

def simpson_adaptativo(f: Callable, a: float, b: float, tol: float = 1e-8, max_depth: int = 20) -> Tuple[complex, float]:
    """
    Método de Simpson auto-adaptativo recursivo para integración numérica.
    
    Parámetros:
    -----------
    f : función a integrar
    a, b : límites de integración
    tol : tolerancia de error
    max_depth : profundidad máxima de recursión
    
    Retorna:
    --------
    (integral, error_estimado)
    """
    def _simpson_rec(a: float, b: float, fa: complex, fb: complex, fc: complex, tol: float, depth: int) -> complex:
        # Regla de Simpson en el intervalo [a, b]
        h = (b - a) / 2
        m = a + h
        fd = f(a + h/2)
        fe = f(b - h/2)
        
        S_ab = h/3 * (fa + 4*fc + fb)  # Simpson en intervalo completo
        S_ac = h/6 * (fa + 4*fd + fc)  # Simpson en primera mitad
        S_cb = h/6 * (fc + 4*fe + fb)  # Simpson en segunda mitad
        S_total = S_ac + S_cb
        
        # Estimación del error
        error = abs(S_total - S_ab) / 15
        
        if error < tol or depth >= max_depth:
            return S_total + (S_total - S_ab) / 15  # Corrección de Richardson
        else:
            # Subdividir recursivamente
            left = _simpson_rec(a, m, fa, fc, fd, tol/2, depth+1)
            right = _simpson_rec(m, b, fc, fb, fe, tol/2, depth+1)
            return left + right
    
    fa, fb, fc = f(a), f(b), f((a+b)/2)
    integral = _simpson_rec(a, b, fa, fb, fc, tol, 0)
    return integral

# ============================================================================
# 2. PARÁMETROS FÍSICOS Y FUNCIONES DE DIFRACCIÓN
# ============================================================================

class DifraccionRendija:
    """Clase para simular difracción de una rendija unidimensional."""
    
    def __init__(self, lambda_: float = 500e-9, ancho: float = 1e-3):
        """
        Parámetros:
        -----------
        lambda_ : longitud de onda [m]
        ancho : ancho de la rendija [m]
        """
        self.lambda_ = lambda_
        self.ancho = ancho
        self.k = 2 * np.pi / lambda_  # número de onda
    
    def set_numero_fresnel(self, NF: float) -> None:
        """Establece la distancia z a partir del número de Fresnel."""
        self.z = self.ancho**2 / (self.lambda_ * NF)
        self.NF = NF
    
    def funcion_abertura(self, x_prime: float) -> float:
        """Función de abertura: 1 dentro de la rendija, 0 fuera."""
        return 1.0 if abs(x_prime) <= self.ancho/2 else 0.0
    
    def integrando_fresnel(self, x: float, x_prime: float) -> complex:
        """Integrando para la aproximación de Fresnel."""
        return np.exp(-1j * self.k * (x - x_prime)**2 / (2 * self.z))
    
    def integrando_huygens_fresnel(self, x: float, x_prime: float) -> complex:
        """Integrando para el principio de Huygens-Fresnel."""
        r = np.sqrt(self.z**2 + (x - x_prime)**2)
        factor = self.z / r
        return factor * np.exp(-1j * self.k * r) / np.sqrt(r)
    
    def campo_fresnel(self, x: float) -> complex:
        """Campo difractado usando aproximación de Fresnel."""
        factor = np.sqrt(1j / (self.lambda_ * self.z)) * np.exp(-1j * self.k * self.z)
        
        def integrando(xp):
            return self.funcion_abertura(xp) * self.integrando_fresnel(x, xp)
        
        integral = simpson_adaptativo(
            lambda xp: integrando(xp),
            -self.ancho/2, self.ancho/2,
            tol=1e-8
        )
        return factor * integral
    
    def campo_huygens_fresnel(self, x: float) -> complex:
        """Campo difractado usando principio de Huygens-Fresnel."""
        factor = np.sqrt(1j / self.lambda_)
        
        def integrando(xp):
            return (self.funcion_abertura(xp) * 
                    self.integrando_huygens_fresnel(x, xp))
        
        integral = simpson_adaptativo(
            lambda xp: integrando(xp),
            -self.ancho/2, self.ancho/2,
            tol=1e-8
        )
        return factor * integral
    
    def calcular_patron(self, x_vals: np.ndarray, metodo: str = 'fresnel') -> np.ndarray:
        """Calcula el patrón de difracción para un array de posiciones x."""
        campo = np.zeros_like(x_vals, dtype=complex)
        
        if metodo.lower() == 'fresnel':
            for i, x in enumerate(x_vals):
                campo[i] = self.campo_fresnel(x)
        elif metodo.lower() == 'huygens-fresnel':
            for i, x in enumerate(x_vals):
                campo[i] = self.campo_huygens_fresnel(x)
        else:
            raise ValueError("Método debe ser 'fresnel' o 'huygens-fresnel'")
        
        return np.abs(campo)**2  # Intensidad

# ============================================================================
# 3. SIMULACIÓN Y VISUALIZACIÓN
# ============================================================================

def simulacion_rendija_unidimensional():
    """Simulación completa para una rendija unidimensional."""
    
    # Crear instancia de difracción
    dif = DifraccionRendija(lambda_=500e-9, ancho=1e-3)
    
    # Números de Fresnel a explorar (como en la Figura 11)
    numeros_fresnel = [0.1, 1.0, 10.0]
    
    # Rango de posiciones transversales (en mm)
    x_vals_mm = np.linspace(-5, 5, 401)
    x_vals = x_vals_mm * 1e-3  # Convertir a metros
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Patrones de Difracción de una Rendija Unidimensional', fontsize=16)
    
    for col, NF in enumerate(numeros_fresnel):
        # Configurar número de Fresnel
        dif.set_numero_fresnel(NF)
        
        # Calcular patrones
        intensidad_fresnel = dif.calcular_patron(x_vals, metodo='fresnel')
        intensidad_hf = dif.calcular_patron(x_vals, metodo='huygens-fresnel')
        
        # Normalizar intensidades
        intensidad_fresnel /= np.max(intensidad_fresnel)
        intensidad_hf /= np.max(intensidad_hf)
        
        # Gráficas superiores: Aproximación de Fresnel
        ax = axes[0, col]
        ax.plot(x_vals_mm, intensidad_fresnel, 'b-', linewidth=2)
        ax.set_xlabel('Posición x (mm)')
        ax.set_ylabel('Intensidad (normalizada)')
        ax.set_title(f'Fresnel, NF = {NF}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        # Gráficas inferiores: Huygens-Fresnel
        ax = axes[1, col]
        ax.plot(x_vals_mm, intensidad_hf, 'r-', linewidth=2)
        ax.set_xlabel('Posición x (mm)')
        ax.set_ylabel('Intensidad (normalizada)')
        ax.set_title(f'Huygens-Fresnel, NF = {NF}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.show()
    
    # Comparación detallada para un NF específico
    print("="*60)
    print("COMPARACIÓN DE MÉTODOS PARA NF = 1.0")
    print("="*60)
    
    dif.set_numero_fresnel(1.0)
    
    # Calcular en el centro (x=0)
    x_centro = 0.0
    campo_f = dif.campo_fresnel(x_centro)
    campo_hf = dif.campo_huygens_fresnel(x_centro)
    
    print(f"Campo en centro (x=0):")
    print(f"  Fresnel:         {campo_f:.4e}")
    print(f"  Huygens-Fresnel: {campo_hf:.4e}")
    print(f"  Diferencia:      {abs(campo_f - campo_hf):.4e}")
    
    # Calcular la intensidad máxima
    intensidad_f = np.abs(campo_f)**2
    intensidad_hf = np.abs(campo_hf)**2
    print(f"\nIntensidad en centro:")
    print(f"  Fresnel:         {intensidad_f:.4e}")
    print(f"  Huygens-Fresnel: {intensidad_hf:.4e}")
    print(f"  Relación:        {intensidad_f/intensidad_hf:.4f}")

# ============================================================================
# 4. EXTENSIÓN A ABERTURA RECTANGULAR 2D (USANDO SEPARABILIDAD)
# ============================================================================

class DifraccionRectangulo2D:
    """Clase para simular difracción de una abertura rectangular 2D."""
    
    def __init__(self, lambda_: float = 500e-9, ancho_x: float = 1e-3, ancho_y: float = 1e-3):
        self.lambda_ = lambda_
        self.ancho_x = ancho_x
        self.ancho_y = ancho_y
        self.k = 2 * np.pi / lambda_
        
        # Instancias para cada dimensión
        self.dif_x = DifraccionRendija(lambda_, ancho_x)
        self.dif_y = DifraccionRendija(lambda_, ancho_y)
    
    def set_numero_fresnel(self, NF: float) -> None:
        """Usamos el ancho en x como referencia para NF."""
        self.z = self.ancho_x**2 / (self.lambda_ * NF)
        self.NF = NF
        self.dif_x.z = self.z
        self.dif_y.z = self.z
    
    def campo_fresnel_2d(self, x: float, y: float) -> complex:
        """Campo difractado usando separabilidad (aproximación de Fresnel)."""
        campo_x = self.dif_x.campo_fresnel(x)
        campo_y = self.dif_y.campo_fresnel(y)
        
        # Factor de corrección por separabilidad
        factor = 1j / (self.lambda_ * self.z) * np.exp(-1j * self.k * self.z)
        
        return campo_x * campo_y * factor
    
    def intensidad_2d(self, x_vals: np.ndarray, y_vals: np.ndarray) -> np.ndarray:
        """Calcula la intensidad en una malla 2D."""
        intensidad = np.zeros((len(y_vals), len(x_vals)))
        
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                campo = self.campo_fresnel_2d(x, y)
                intensidad[i, j] = np.abs(campo)**2
        
        return intensidad

def simulacion_rectangular_2d():
    """Simulación para abertura rectangular 2D."""
    
    # Crear instancia
    dif_2d = DifraccionRectangulo2D(lambda_=500e-9, ancho_x=1e-3, ancho_y=0.5e-3)
    
    # Configurar NF
    NF = 1.0
    dif_2d.set_numero_fresnel(NF)
    
    # Crear malla de puntos
    x_range = np.linspace(-3e-3, 3e-3, 201)
    y_range = np.linspace(-2e-3, 2e-3, 151)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Calcular intensidad
    I = dif_2d.intensidad_2d(x_range, y_range)
    
    # Visualización
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(I, extent=[x_range.min()*1e3, x_range.max()*1e3, 
                          y_range.min()*1e3, y_range.max()*1e3],
               origin='lower', aspect='auto', cmap='hot')
    plt.colorbar(label='Intensidad')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title(f'Patrón de Difracción - Abertura Rectangular 2D\nNF = {NF}')
    
    plt.subplot(1, 2, 2)
    plt.contour(X*1e3, Y*1e3, I, levels=20, cmap='viridis')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title('Curvas de nivel')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 5. ABERTURA TRIANGULAR 2D (INTEGRAL DOBLE)
# ============================================================================

class DifraccionTriangulo2D:
    """Clase para simular difracción de una abertura triangular 2D."""
    
    def __init__(self, lambda_: float = 500e-9, base: float = 1e-3, altura: float = 1e-3):
        """
        Triángulo rectángulo con vértices en:
        (0,0), (base,0), (0,altura)
        """
        self.lambda_ = lambda_
        self.base = base
        self.altura = altura
        self.k = 2 * np.pi / lambda_
    
    def set_numero_fresnel(self, NF: float) -> None:
        """Usamos la base como referencia para NF."""
        self.z = self.base**2 / (self.lambda_ * NF)
        self.NF = NF
    
    def dentro_triangulo(self, xp: float, yp: float) -> bool:
        """Verifica si el punto (xp, yp) está dentro del triángulo."""
        if xp < 0 or yp < 0:
            return False
        if yp > self.altura * (1 - xp/self.base):
            return False
        return True
    
    def integrando_fresnel_triangulo(self, x: float, y: float, xp: float, yp: float) -> complex:
        """Integrando para aproximación de Fresnel en 2D."""
        return np.exp(-1j * self.k * ((x - xp)**2 + (y - yp)**2) / (2 * self.z))
    
    def campo_fresnel_triangulo(self, x: float, y: float, tol: float = 1e-6) -> complex:
        """Campo difractado para abertura triangular (integral doble)."""
        factor = (1j/(self.lambda_ * self.z)) * np.exp(-1j * self.k * self.z)
        
        # Integral doble sobre el triángulo
        # Primero en x', luego en y' (con límites variables)
        def integral_en_x(xp):
            # Para xp fijo, yp va de 0 a yp_max
            yp_max = self.altura * (1 - xp/self.base)
            
            if yp_max <= 0:
                return 0.0
            
            # Función para integrar en y'
            def f_yp(yp):
                return self.integrando_fresnel_triangulo(x, y, xp, yp)
            
            # Integrar en y' usando Simpson adaptativo
            integral_yp = simpson_adaptativo(f_yp, 0, yp_max, tol=tol)
            return integral_yp
        
        # Integrar en x' de 0 a base
        integral_total = simpson_adaptativo(
            lambda xp: integral_en_x(xp),
            0, self.base,
            tol=tol
        )
        
        return factor * integral_total
    
    def calcular_patron_triangular(self, x_vals: np.ndarray, y_vals: np.ndarray, 
                                   resolucion: int = 50) -> np.ndarray:
        """Calcula el patrón de difracción para malla 2D (con resolución reducida)."""
        intensidad = np.zeros((len(y_vals), len(x_vals)))
        
        # Submuestreo para cálculos intensivos
        paso_x = max(1, len(x_vals) // resolucion)
        paso_y = max(1, len(y_vals) // resolucion)
        
        indices_x = range(0, len(x_vals), paso_x)
        indices_y = range(0, len(y_vals), paso_y)
        
        print(f"Calculando patrón triangular (NF={self.NF})...")
        print(f"Puntos a calcular: {len(indices_x)} x {len(indices_y)}")
        
        for i in indices_y:
            y = y_vals[i]
            for j in indices_x:
                x = x_vals[j]
                campo = self.campo_fresnel_triangulo(x, y)
                intensidad[i, j] = np.abs(campo)**2
        
        # Interpolación para suavizar
        from scipy import interpolate
        if paso_x > 1 or paso_y > 1:
            x_fine = np.linspace(x_vals.min(), x_vals.max(), len(x_vals))
            y_fine = np.linspace(y_vals.min(), y_vals.max(), len(y_vals))
            X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
            
            # Extraer puntos calculados
            X_calc, Y_calc = np.meshgrid(x_vals[indices_x], y_vals[indices_y])
            I_calc = intensidad[indices_y[0]:indices_y[-1]+1:paso_y, 
                               indices_x[0]:indices_x[-1]+1:paso_x]
            
            # Interpolación bicúbica
            interp = interpolate.RectBivariateSpline(
                y_vals[indices_y], x_vals[indices_x], I_calc
            )
            intensidad = interp(y_fine, x_fine)
        
        return intensidad

def simulacion_triangular_2d():
    """Simulación para abertura triangular 2D (como en Figura 11)."""
    
    # Parámetros similares a la Figura 11
    base = 1e-3  # 1 mm
    altura = 1e-3  # 1 mm
    
    # Crear instancia
    dif_tri = DifraccionTriangulo2D(lambda_=500e-9, base=base, altura=altura)
    
    # Números de Fresnel a explorar
    numeros_fresnel = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    # Rango de observación
    x_vals = np.linspace(-3e-3, 3e-3, 151)
    y_vals = np.linspace(-3e-3, 3e-3, 151)
    
    # Crear figura
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Patrones de Difracción - Abertura Triangular 2D', fontsize=16)
    
    axes = axes.flatten()
    
    for idx, NF in enumerate(numeros_fresnel):
        if idx >= len(axes):
            break
            
        # Configurar NF
        dif_tri.set_numero_fresnel(NF)
        
        # Calcular patrón
        I = dif_tri.calcular_patron_triangular(x_vals, y_vals, resolucion=40)
        
        # Normalizar
        I_norm = I / np.max(I)
        
        # Graficar
        ax = axes[idx]
        im = ax.imshow(I_norm, 
                       extent=[x_vals.min()*1e3, x_vals.max()*1e3, 
                               y_vals.min()*1e3, y_vals.max()*1e3],
                       origin='lower', 
                       aspect='auto',
                       cmap='hot',
                       vmin=0, vmax=1)
        
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(f'NF = {NF}')
        
        # Agregar barra de color para el último gráfico
        if idx == len(axes) - 1:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Intensidad (normalizada)')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 6. EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("SIMULACIONES DE DIFRACCIÓN ESCALAR")
    print("="*60)
    
    while True:
        print("\nSeleccione una simulación:")
        print("1. Rendija unidimensional (Fresnel vs Huygens-Fresnel)")
        print("2. Abertura rectangular 2D")
        print("3. Abertura triangular 2D (como Figura 11)")
        print("4. Todas las simulaciones")
        print("5. Salir")
        
        opcion = input("\nOpción (1-5): ").strip()
        
        if opcion == '1':
            print("\nEjecutando simulación de rendija unidimensional...")
            simulacion_rendija_unidimensional()
        
        elif opcion == '2':
            print("\nEjecutando simulación de abertura rectangular 2D...")
            simulacion_rectangular_2d()
        
        elif opcion == '3':
            print("\nEjecutando simulación de abertura triangular 2D...")
            print("Nota: Esta simulación puede tomar varios minutos.")
            simulacion_triangular_2d()
        
        elif opcion == '4':
            print("\nEjecutando todas las simulaciones...")
            print("\n1. Rendija unidimensional:")
            simulacion_rendija_unidimensional()
            
            print("\n2. Abertura rectangular 2D:")
            simulacion_rectangular_2d()
            
            print("\n3. Abertura triangular 2D:")
            simulacion_triangular_2d()
        
        elif opcion == '5':
            print("¡Hasta luego!")
            break
        
        else:
            print("Opción no válida. Intente de nuevo.")

# ============================================================================
# 7. FUNCIONES ADICIONALES PARA ANÁLISIS
# ============================================================================

def analizar_convergencia():
    """Analiza la convergencia del método de Simpson auto-adaptativo."""
    
    # Test de convergencia con función conocida
    def f(x):
        return np.sin(x)
    
    a, b = 0, np.pi
    valor_real = 2.0  # ∫sin(x)dx de 0 a π
    
    tolerancias = [1e-4, 1e-6, 1e-8, 1e-10]
    
    print("\nANÁLISIS DE CONVERGENCIA DEL MÉTODO DE SIMPSON")
    print("-"*50)
    
    for tol in tolerancias:
        integral, _ = simpson_adaptativo(f, a, b, tol=tol)
        error = abs(integral - valor_real)
        print(f"Tol={tol:.0e}: Integral={integral:.10f}, Error={error:.2e}")
    
    # Test con función oscilatoria (más relevante para difracción)
    print("\nTest con función oscilatoria (similar a difracción):")
    
    k_test = 2*np.pi/500e-9
    z_test = 0.1
    def f_oscilatoria(x):
        return np.exp(-1j * k_test * x**2 / (2*z_test))
    
    for tol in [1e-6, 1e-8]:
        integral, _ = simpson_adaptativo(f_oscilatoria, -1e-3, 1e-3, tol=tol)
        print(f"Tol={tol:.0e}: Integral={integral:.6e}")

# Para ejecutar análisis de convergencia:
# analizar_convergencia()
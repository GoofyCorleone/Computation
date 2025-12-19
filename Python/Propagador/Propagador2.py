import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional
import time
from scipy import interpolate

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
# 2. DIFRACCIÓN RECTANGULAR 2D CON COMPARACIÓN FRESNEL vs HUYGENS-FRESNEL
# ============================================================================

class DifraccionRectangular2DCompleta:
    """
    Clase para simular difracción de una abertura rectangular 2D
    con comparación entre Fresnel y Huygens-Fresnel.
    """
    
    def __init__(self, lambda_: float = 500e-9, ancho_x: float = 1e-3, ancho_y: float = 1e-3):
        """
        Parámetros:
        -----------
        lambda_ : longitud de onda [m]
        ancho_x : ancho en dirección x [m]
        ancho_y : ancho en dirección y [m]
        """
        self.lambda_ = lambda_
        self.ancho_x = ancho_x
        self.ancho_y = ancho_y
        self.k = 2 * np.pi / lambda_  # número de onda
        self.z = None
        self.NF = None
    
    def set_numero_fresnel(self, NF: float) -> None:
        """Establece la distancia z a partir del número de Fresnel."""
        # Usamos el promedio de las dimensiones para NF
        a_prom = (self.ancho_x + self.ancho_y) / 2
        self.z = a_prom**2 / (self.lambda_ * NF)
        self.NF = NF
    
    def set_distancia(self, z: float) -> None:
        """Establece la distancia z directamente."""
        self.z = z
        # Calcula NF usando el promedio de las dimensiones
        a_prom = (self.ancho_x + self.ancho_y) / 2
        self.NF = a_prom**2 / (self.lambda_ * z)
    
    def funcion_abertura_2d(self, x_prime: float, y_prime: float) -> float:
        """Función de abertura 2D: 1 dentro del rectángulo, 0 fuera."""
        if abs(x_prime) <= self.ancho_x/2 and abs(y_prime) <= self.ancho_y/2:
            return 1.0
        return 0.0
    
    def _integrando_fresnel_2d(self, x: float, y: float, x_prime: float, y_prime: float) -> complex:
        """Integrando para aproximación de Fresnel en 2D."""
        return np.exp(-1j * self.k * ((x - x_prime)**2 + (y - y_prime)**2) / (2 * self.z))
    
    def _integrando_huygens_fresnel_2d(self, x: float, y: float, x_prime: float, y_prime: float) -> complex:
        """Integrando para principio de Huygens-Fresnel en 2D."""
        r = np.sqrt(self.z**2 + (x - x_prime)**2 + (y - y_prime)**2)
        factor = self.z / r
        return factor * np.exp(-1j * self.k * r) / r
    
    # ========================= MÉTODOS PARA CÁLCULO 1D (PERFILES) ========================
    
    def perfil_x_fresnel(self, x_vals: np.ndarray, y_fijo: float = 0.0) -> np.ndarray:
        """Calcula el perfil de intensidad en X usando aproximación de Fresnel."""
        intensidad = np.zeros_like(x_vals, dtype=float)
        factor = (1j/(self.lambda_ * self.z)) * np.exp(-1j * self.k * self.z)
        
        for i, x in enumerate(x_vals):
            # Integral doble separada (por ser rectangular)
            def integral_en_y(yp):
                if abs(yp) <= self.ancho_y/2:
                    return np.exp(-1j * self.k * (y_fijo - yp)**2 / (2 * self.z))
                return 0.0
            
            def integral_en_x(xp):
                if abs(xp) <= self.ancho_x/2:
                    # Integral en y para este xp
                    int_y = simpson_adaptativo(
                        lambda yp: integral_en_y(yp),
                        -self.ancho_y/2, self.ancho_y/2,
                        tol=1e-8
                    )
                    return np.exp(-1j * self.k * (x - xp)**2 / (2 * self.z)) * int_y
                return 0.0
            
            # Integral en x
            int_total = simpson_adaptativo(
                lambda xp: integral_en_x(xp),
                -self.ancho_x/2, self.ancho_x/2,
                tol=1e-8
            )
            
            campo = factor * int_total
            intensidad[i] = np.abs(campo)**2
        
        return intensidad
    
    def perfil_x_huygens_fresnel(self, x_vals: np.ndarray, y_fijo: float = 0.0) -> np.ndarray:
        """Calcula el perfil de intensidad en X usando Huygens-Fresnel."""
        intensidad = np.zeros_like(x_vals, dtype=float)
        factor = (1j/self.lambda_)
        
        for i, x in enumerate(x_vals):
            def integral_en_y(yp):
                if abs(yp) <= self.ancho_y/2:
                    r = np.sqrt(self.z**2 + (x - 0)**2 + (y_fijo - yp)**2)  # x'=0 para perfil central
                    return (self.z/r) * np.exp(-1j * self.k * r) / r
                return 0.0
            
            def integral_en_x(xp):
                if abs(xp) <= self.ancho_x/2:
                    # Integral en y para este xp
                    int_y = simpson_adaptativo(
                        lambda yp: integral_en_y(yp),
                        -self.ancho_y/2, self.ancho_y/2,
                        tol=1e-8
                    )
                    r = np.sqrt(self.z**2 + (x - xp)**2 + (y_fijo - 0)**2)  # y'=0 para perfil central
                    return (self.z/r) * np.exp(-1j * self.k * r) / r * int_y
                return 0.0
            
            # Integral en x
            int_total = simpson_adaptativo(
                lambda xp: integral_en_x(xp),
                -self.ancho_x/2, self.ancho_x/2,
                tol=1e-8
            )
            
            campo = factor * int_total
            intensidad[i] = np.abs(campo)**2
        
        return intensidad
    
    def perfil_y_fresnel(self, y_vals: np.ndarray, x_fijo: float = 0.0) -> np.ndarray:
        """Calcula el perfil de intensidad en Y usando aproximación de Fresnel."""
        intensidad = np.zeros_like(y_vals, dtype=float)
        factor = (1j/(self.lambda_ * self.z)) * np.exp(-1j * self.k * self.z)
        
        for i, y in enumerate(y_vals):
            # Integral doble separada (por ser rectangular)
            def integral_en_x(xp):
                if abs(xp) <= self.ancho_x/2:
                    return np.exp(-1j * self.k * (x_fijo - xp)**2 / (2 * self.z))
                return 0.0
            
            def integral_en_y(yp):
                if abs(yp) <= self.ancho_y/2:
                    # Integral en x para este yp
                    int_x = simpson_adaptativo(
                        lambda xp: integral_en_x(xp),
                        -self.ancho_x/2, self.ancho_x/2,
                        tol=1e-8
                    )
                    return np.exp(-1j * self.k * (y - yp)**2 / (2 * self.z)) * int_x
                return 0.0
            
            # Integral en y
            int_total = simpson_adaptativo(
                lambda yp: integral_en_y(yp),
                -self.ancho_y/2, self.ancho_y/2,
                tol=1e-8
            )
            
            campo = factor * int_total
            intensidad[i] = np.abs(campo)**2
        
        return intensidad
    
    def perfil_y_huygens_fresnel(self, y_vals: np.ndarray, x_fijo: float = 0.0) -> np.ndarray:
        """Calcula el perfil de intensidad en Y usando Huygens-Fresnel."""
        intensidad = np.zeros_like(y_vals, dtype=float)
        factor = (1j/self.lambda_)
        
        for i, y in enumerate(y_vals):
            def integral_en_x(xp):
                if abs(xp) <= self.ancho_x/2:
                    r = np.sqrt(self.z**2 + (x_fijo - xp)**2 + (y - 0)**2)  # y'=0 para perfil central
                    return (self.z/r) * np.exp(-1j * self.k * r) / r
                return 0.0
            
            def integral_en_y(yp):
                if abs(yp) <= self.ancho_y/2:
                    # Integral en x para este yp
                    int_x = simpson_adaptativo(
                        lambda xp: integral_en_x(xp),
                        -self.ancho_x/2, self.ancho_x/2,
                        tol=1e-8
                    )
                    r = np.sqrt(self.z**2 + (x_fijo - 0)**2 + (y - yp)**2)  # x'=0 para perfil central
                    return (self.z/r) * np.exp(-1j * self.k * r) / r * int_x
                return 0.0
            
            # Integral en y
            int_total = simpson_adaptativo(
                lambda yp: integral_en_y(yp),
                -self.ancho_y/2, self.ancho_y/2,
                tol=1e-8
            )
            
            campo = factor * int_total
            intensidad[i] = np.abs(campo)**2
        
        return intensidad
    
    # ========================= MÉTODOS PARA CÁLCULO 2D (MAPAS) ========================
    
    def calcular_mapa_fresnel_2d(self, x_vals: np.ndarray, y_vals: np.ndarray) -> np.ndarray:
        """Calcula el mapa de intensidad 2D usando aproximación de Fresnel."""
        print(f"Calculando mapa Fresnel 2D (NF={self.NF:.2f})...")
        inicio = time.time()
        
        intensidad = np.zeros((len(y_vals), len(x_vals)))
        factor = (1j/(self.lambda_ * self.z)) * np.exp(-1j * self.k * self.z)
        
        # Usamos separabilidad para eficiencia
        # Precalculamos las integrales 1D
        campos_x = np.zeros(len(x_vals), dtype=complex)
        campos_y = np.zeros(len(y_vals), dtype=complex)
        
        # Calcular campos en x
        for i, x in enumerate(x_vals):
            def integrando_x(xp):
                if abs(xp) <= self.ancho_x/2:
                    return np.exp(-1j * self.k * (x - xp)**2 / (2 * self.z))
                return 0.0
            
            campos_x[i] = simpson_adaptativo(
                lambda xp: integrando_x(xp),
                -self.ancho_x/2, self.ancho_x/2,
                tol=1e-8
            )
        
        # Calcular campos en y
        for j, y in enumerate(y_vals):
            def integrando_y(yp):
                if abs(yp) <= self.ancho_y/2:
                    return np.exp(-1j * self.k * (y - yp)**2 / (2 * self.z))
                return 0.0
            
            campos_y[j] = simpson_adaptativo(
                lambda yp: integrando_y(yp),
                -self.ancho_y/2, self.ancho_y/2,
                tol=1e-8
            )
        
        # Combinar usando separabilidad
        for i in range(len(x_vals)):
            for j in range(len(y_vals)):
                campo = factor * campos_x[i] * campos_y[j]
                intensidad[j, i] = np.abs(campo)**2
        
        tiempo = time.time() - inicio
        print(f"  Tiempo de cálculo: {tiempo:.2f} segundos")
        
        return intensidad
    
    def calcular_mapa_huygens_fresnel_2d(self, x_vals: np.ndarray, y_vals: np.ndarray, 
                                         resolucion_integracion: int = 30) -> np.ndarray:
        """Calcula el mapa de intensidad 2D usando Huygens-Fresnel."""
        print(f"Calculando mapa Huygens-Fresnel 2D (NF={self.NF:.2f})...")
        inicio = time.time()
        
        intensidad = np.zeros((len(y_vals), len(x_vals)))
        factor = (1j/self.lambda_)
        
        # Puntos de integración (reducidos para eficiencia)
        xp_vals = np.linspace(-self.ancho_x/2, self.ancho_x/2, resolucion_integracion)
        yp_vals = np.linspace(-self.ancho_y/2, self.ancho_y/2, resolucion_integracion)
        
        # Pesos de Simpson para integración numérica
        def pesos_simpson(n):
            """Genera pesos para la regla compuesta de Simpson."""
            h = 1.0 / (n - 1)
            weights = np.ones(n)
            weights[1:-1:2] = 4
            weights[2:-1:2] = 2
            return h * weights / 3
        
        wx = pesos_simpson(resolucion_integracion) * self.ancho_x
        wy = pesos_simpson(resolucion_integracion) * self.ancho_y
        
        # Para cada punto de observación
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                # Crear malla 2D para integración
                Xp, Yp = np.meshgrid(xp_vals, yp_vals)
                
                # Calcular distancias
                r = np.sqrt(self.z**2 + (x - Xp)**2 + (y - Yp)**2)
                
                # Integrando
                integrando = (self.z/r) * np.exp(-1j * self.k * r) / r
                
                # Integración 2D usando regla de Simpson compuesta
                integral = np.sum(wx[None, :] * wy[:, None] * integrando)
                
                campo = factor * integral
                intensidad[i, j] = np.abs(campo)**2
        
        tiempo = time.time() - inicio
        print(f"  Tiempo de cálculo: {tiempo:.2f} segundos")
        
        return intensidad

# ============================================================================
# 3. VISUALIZACIÓN COMPLETA
# ============================================================================

def visualizar_comparacion_completa(dif: DifraccionRectangular2DCompleta, 
                                    x_range_mm: Tuple[float, float] = (-5, 5),
                                    y_range_mm: Tuple[float, float] = (-5, 5),
                                    resolucion_2d: int = 100,
                                    guardar_imagen: bool = False,
                                    nombre_archivo: str = None):
    """
    Visualización completa con mapas 2D y perfiles 1D comparando ambos métodos.
    
    Parámetros:
    -----------
    dif : instancia de DifraccionRectangular2DCompleta
    x_range_mm, y_range_mm : rangos de visualización en mm
    resolucion_2d : resolución para mapas 2D
    guardar_imagen : si True, guarda la figura
    nombre_archivo : nombre del archivo para guardar
    """
    
    # Convertir rangos a metros
    x_min, x_max = x_range_mm[0] * 1e-3, x_range_mm[1] * 1e-3
    y_min, y_max = y_range_mm[0] * 1e-3, y_range_mm[1] * 1e-3
    
    # Crear mallas
    x_vals = np.linspace(x_min, x_max, resolucion_2d)
    y_vals = np.linspace(y_min, y_max, resolucion_2d)
    x_vals_mm = x_vals * 1e3
    y_vals_mm = y_vals * 1e3
    
    # Calcular mapas 2D
    I_fresnel_2d = dif.calcular_mapa_fresnel_2d(x_vals, y_vals)
    I_hf_2d = dif.calcular_mapa_huygens_fresnel_2d(x_vals, y_vals, resolucion_integracion=40)
    
    # Calcular perfiles 1D
    print("\nCalculando perfiles 1D...")
    
    # Perfil en X (y=0)
    perfil_x_f = dif.perfil_x_fresnel(x_vals, y_fijo=0.0)
    perfil_x_hf = dif.perfil_x_huygens_fresnel(x_vals, y_fijo=0.0)
    
    # Perfil en Y (x=0)
    perfil_y_f = dif.perfil_y_fresnel(y_vals, x_fijo=0.0)
    perfil_y_hf = dif.perfil_y_huygens_fresnel(y_vals, x_fijo=0.0)
    
    # Normalizar todos los resultados
    max_2d = max(np.max(I_fresnel_2d), np.max(I_hf_2d))
    I_fresnel_2d_norm = I_fresnel_2d / max_2d
    I_hf_2d_norm = I_hf_2d / max_2d
    
    max_perfil_x = max(np.max(perfil_x_f), np.max(perfil_x_hf))
    perfil_x_f_norm = perfil_x_f / max_perfil_x
    perfil_x_hf_norm = perfil_x_hf / max_perfil_x
    
    max_perfil_y = max(np.max(perfil_y_f), np.max(perfil_y_hf))
    perfil_y_f_norm = perfil_y_f / max_perfil_y
    perfil_y_hf_norm = perfil_y_hf / max_perfil_y
    
    # Crear figura
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Difracción Rectangular 2D - Comparación Fresnel vs Huygens-Fresnel\n'
                 f'NF = {dif.NF:.2f}, z = {dif.z*100:.2f} cm, '
                 f'λ = {dif.lambda_*1e9:.0f} nm, '
                 f'Ancho X = {dif.ancho_x*1000:.1f} mm, Ancho Y = {dif.ancho_y*1000:.1f} mm',
                 fontsize=14, fontweight='bold')
    
    # ===================== MAPAS 2D =====================
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(I_fresnel_2d_norm, 
                     extent=[x_vals_mm.min(), x_vals_mm.max(), 
                             y_vals_mm.min(), y_vals_mm.max()],
                     origin='lower', 
                     aspect='auto',
                     cmap='hot',
                     vmin=0, vmax=1)
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.set_title('Aproximación de Fresnel - Mapa 2D')
    ax1.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(im1, ax=ax1, label='Intensidad (normalizada)')
    
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(I_hf_2d_norm, 
                     extent=[x_vals_mm.min(), x_vals_mm.max(), 
                             y_vals_mm.min(), y_vals_mm.max()],
                     origin='lower', 
                     aspect='auto',
                     cmap='hot',
                     vmin=0, vmax=1)
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('y (mm)')
    ax2.set_title('Huygens-Fresnel - Mapa 2D')
    ax2.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(im2, ax=ax2, label='Intensidad (normalizada)')
    
    # ===================== DIFERENCIA ENTRE MÉTODOS =====================
    ax3 = plt.subplot(2, 3, 3)
    diferencia = np.abs(I_fresnel_2d_norm - I_hf_2d_norm)
    im3 = ax3.imshow(diferencia, 
                     extent=[x_vals_mm.min(), x_vals_mm.max(), 
                             y_vals_mm.min(), y_vals_mm.max()],
                     origin='lower', 
                     aspect='auto',
                     cmap='viridis')
    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('y (mm)')
    ax3.set_title(f'Diferencia Absoluta\nMáx: {np.max(diferencia):.4f}')
    ax3.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(im3, ax=ax3, label='|Fresnel - Huygens-Fresnel|')
    
    # ===================== PERFIL EN X =====================
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(x_vals_mm, perfil_x_f_norm, 'b-', linewidth=2, label='Fresnel')
    ax4.plot(x_vals_mm, perfil_x_hf_norm, 'r--', linewidth=2, label='Huygens-Fresnel')
    ax4.set_xlabel('x (mm)')
    ax4.set_ylabel('Intensidad (normalizada)')
    ax4.set_title(f'Perfil en X (y=0)\nNF = {dif.NF:.2f}')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.1)
    
    # Calcular y mostrar diferencia máxima en perfil X
    diff_x = np.max(np.abs(perfil_x_f_norm - perfil_x_hf_norm))
    ax4.text(0.02, 0.98, f'Diff máx: {diff_x:.4f}', 
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ===================== PERFIL EN Y =====================
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(y_vals_mm, perfil_y_f_norm, 'b-', linewidth=2, label='Fresnel')
    ax5.plot(y_vals_mm, perfil_y_hf_norm, 'r--', linewidth=2, label='Huygens-Fresnel')
    ax5.set_xlabel('y (mm)')
    ax5.set_ylabel('Intensidad (normalizada)')
    ax5.set_title(f'Perfil en Y (x=0)\nNF = {dif.NF:.2f}')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1.1)
    
    # Calcular y mostrar diferencia máxima en perfil Y
    diff_y = np.max(np.abs(perfil_y_f_norm - perfil_y_hf_norm))
    ax5.text(0.02, 0.98, f'Diff máx: {diff_y:.4f}', 
             transform=ax5.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ===================== COMPARACIÓN DIRECTA EN UN PUNTO =====================
    ax6 = plt.subplot(2, 3, 6)
    
    # Calcular valores en el centro (0,0)
    centro_x_idx = np.argmin(np.abs(x_vals))
    centro_y_idx = np.argmin(np.abs(y_vals))
    
    I_centro_fresnel = I_fresnel_2d_norm[centro_y_idx, centro_x_idx]
    I_centro_hf = I_hf_2d_norm[centro_y_idx, centro_x_idx]
    
    # Gráfico de barras comparativo
    metodos = ['Fresnel', 'Huygens-Fresnel']
    valores = [I_centro_fresnel, I_centro_hf]
    colores = ['blue', 'red']
    
    bars = ax6.bar(metodos, valores, color=colores, alpha=0.7)
    ax6.set_ylabel('Intensidad en centro (0,0)')
    ax6.set_title('Comparación en centro')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_ylim(0, 1.1)
    
    # Añadir valores encima de las barras
    for bar, valor in zip(bars, valores):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{valor:.4f}', ha='center', va='bottom')
    
    # Información adicional
    ax6.text(0.02, 0.85, f'NF = {dif.NF:.2f}', transform=ax6.transAxes)
    ax6.text(0.02, 0.78, f'z = {dif.z*100:.1f} cm', transform=ax6.transAxes)
    ax6.text(0.02, 0.71, f'λ = {dif.lambda_*1e9:.0f} nm', transform=ax6.transAxes)
    
    plt.tight_layout()
    
    if guardar_imagen:
        if nombre_archivo is None:
            nombre_archivo = f"difraccion_rectangular_NF{dif.NF:.2f}.png"
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
        print(f"Figura guardada como '{nombre_archivo}'")
    
    plt.show()
    
    # Imprimir resumen estadístico
    print("\n" + "="*70)
    print("RESUMEN ESTADÍSTICO")
    print("="*70)
    print(f"Número de Fresnel: {dif.NF:.4f}")
    print(f"Distancia z: {dif.z:.4f} m ({dif.z*100:.2f} cm)")
    print(f"Intensidad máxima Fresnel 2D: {np.max(I_fresnel_2d):.4e}")
    print(f"Intensidad máxima Huygens-Fresnel 2D: {np.max(I_hf_2d):.4e}")
    print(f"Diferencia máxima en mapa 2D: {np.max(diferencia):.6f}")
    print(f"Diferencia máxima en perfil X: {diff_x:.6f}")
    print(f"Diferencia máxima en perfil Y: {diff_y:.6f}")
    print(f"Intensidad en centro - Fresnel: {I_centro_fresnel:.6f}")
    print(f"Intensidad en centro - Huygens-Fresnel: {I_centro_hf:.6f}")
    print(f"Relación Fresnel/HF en centro: {I_centro_fresnel/I_centro_hf:.6f}")

# ============================================================================
# 4. SIMULACIONES PARA DIFERENTES NÚMEROS DE FRESNEL
# ============================================================================

def simulacion_varios_NF():
    """Ejecuta simulaciones para diferentes números de Fresnel."""
    
    print("SIMULACIÓN DE DIFRACCIÓN RECTANGULAR 2D")
    print("="*60)
    
    # Parámetros fijos
    lambda_ = 632.8e-9  # Luz He-Ne
    ancho_x = 1.0e-3    # 1 mm
    ancho_y = 0.5e-3    # 0.5 mm (rectángulo asimétrico)
    
    # Números de Fresnel a explorar (régimen cercano, intermedio y lejano)
    numeros_fresnel = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for NF in numeros_fresnel:
        print(f"\n{'='*60}")
        print(f"SIMULACIÓN PARA NF = {NF}")
        print(f"{'='*60}")
        
        # Crear instancia
        dif = DifraccionRectangular2DCompleta(lambda_=lambda_, 
                                              ancho_x=ancho_x, 
                                              ancho_y=ancho_y)
        dif.set_numero_fresnel(NF)
        
        # Determinar rango de visualización basado en NF
        # Para NF pequeño (campo lejano), el patrón se expande más
        factor_expansion = 2.0 + 5.0/NF if NF > 0.1 else 20.0
        rango = 5 * factor_expansion
        
        # Visualizar
        visualizar_comparacion_completa(
            dif,
            x_range_mm=(-rango, rango),
            y_range_mm=(-rango, rango),
            resolucion_2d=80,  # Reducido para velocidad
            guardar_imagen=True,
            nombre_archivo=f"rectangular_NF{NF:.1f}.png"
        )

# ============================================================================
# 5. INTERFAZ INTERACTIVA
# ============================================================================

def interfaz_interactiva():
    """Interfaz interactiva para explorar diferentes parámetros."""
    
    print("INTERFAZ INTERACTIVA DE DIFRACCIÓN RECTANGULAR 2D")
    print("="*60)
    
    while True:
        print("\nConfiguración de parámetros:")
        
        # Longitud de onda
        lambda_nm = float(input("Longitud de onda (nm, por defecto 632.8): ") or "632.8")
        lambda_ = lambda_nm * 1e-9
        
        # Dimensiones
        ancho_x_mm = float(input("Ancho en X (mm, por defecto 1.0): ") or "1.0")
        ancho_y_mm = float(input("Ancho en Y (mm, por defecto 0.5): ") or "0.5")
        ancho_x = ancho_x_mm * 1e-3
        ancho_y = ancho_y_mm * 1e-3
        
        # Crear instancia
        dif = DifraccionRectangular2DCompleta(lambda_=lambda_, 
                                              ancho_x=ancho_x, 
                                              ancho_y=ancho_y)
        
        # Modo de operación
        print("\nModo de operación:")
        print("1. Especificar número de Fresnel (NF)")
        print("2. Especificar distancia z")
        modo = input("Seleccione (1 o 2): ").strip()
        
        if modo == '1':
            NF = float(input("Número de Fresnel (por defecto 1.0): ") or "1.0")
            dif.set_numero_fresnel(NF)
        elif modo == '2':
            z_cm = float(input("Distancia z (cm, por defecto 10.0): ") or "10.0")
            z = z_cm * 0.01
            dif.set_distancia(z)
        else:
            print("Opción no válida, usando NF=1.0 por defecto")
            dif.set_numero_fresnel(1.0)
        
        # Rango de visualización
        rango_mm = float(input("Rango de visualización (±mm, por defecto 5.0): ") or "5.0")
        
        # Resolución
        resolucion = int(input("Resolución 2D (puntos, por defecto 80): ") or "80")
        
        # Ejecutar simulación
        print(f"\nEjecutando simulación con:")
        print(f"  λ = {lambda_nm} nm")
        print(f"  Ancho X = {ancho_x_mm} mm, Ancho Y = {ancho_y_mm} mm")
        print(f"  NF = {dif.NF:.4f}")
        print(f"  z = {dif.z*100:.2f} cm")
        print(f"  Rango = ±{rango_mm} mm")
        
        visualizar_comparacion_completa(
            dif,
            x_range_mm=(-rango_mm, rango_mm),
            y_range_mm=(-rango_mm, rango_mm),
            resolucion_2d=resolucion,
            guardar_imagen=False
        )
        
        # Preguntar si continuar
        continuar = input("\n¿Realizar otra simulación? (s/n): ").strip().lower()
        if continuar != 's':
            print("¡Hasta luego!")
            break

# ============================================================================
# 6. ANÁLISIS DE CONVERGENCIA Y PRECISIÓN
# ============================================================================

def analizar_precision(dif: DifraccionRectangular2DCompleta, 
                       tolerancias: List[float] = [1e-4, 1e-6, 1e-8, 1e-10]):
    """Analiza la precisión del cálculo en función de la tolerancia."""
    
    print("\nANÁLISIS DE PRECISIÓN")
    print("="*60)
    
    # Punto de prueba (centro)
    x_test = 0.0
    y_test = 0.0
    
    resultados_fresnel = []
    resultados_hf = []
    
    for tol in tolerancias:
        print(f"\nTolerancia: {tol:.1e}")
        
        # Modificar función simpson_adaptativo para usar tolerancia específica
        def simpson_tol(f, a, b):
            return simpson_adaptativo(f, a, b, tol=tol)
        
        # Guardar función original temporalmente
        import types
        dif_temp = types.SimpleNamespace()
        dif_temp.lambda_ = dif.lambda_
        dif_temp.ancho_x = dif.ancho_x
        dif_temp.ancho_y = dif.ancho_y
        dif_temp.k = dif.k
        dif_temp.z = dif.z
        
        # Calcular con tolerancia específica (simplificado)
        # En una implementación real, necesitaríamos modificar la clase
        
        print(f"  Cálculo con tolerancia {tol:.1e}...")
    
    print("\nNota: Para un análisis detallado de convergencia,")
    print("se necesitaría modificar la clase para aceptar diferentes tolerancias.")

# ============================================================================
# 7. EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("DIFRACCIÓN RECTANGULAR 2D - FRESNEL vs HUYGENS-FRESNEL")
    print("="*60)
    
    while True:
        print("\nSeleccione una opción:")
        print("1. Simulación con parámetros por defecto (NF=1.0)")
        print("2. Simulación para varios números de Fresnel")
        print("3. Interfaz interactiva (personalizar parámetros)")
        print("4. Salir")
        
        opcion = input("\nOpción (1-4): ").strip()
        
        if opcion == '1':
            print("\nSimulación con parámetros por defecto...")
            
            # Parámetros por defecto
            lambda_ = 632.8e-9  # Luz He-Ne
            ancho_x = 1.0e-3    # 1 mm
            ancho_y = 0.5e-3    # 0.5 mm
            
            dif = DifraccionRectangular2DCompleta(lambda_=lambda_, 
                                                  ancho_x=ancho_x, 
                                                  ancho_y=ancho_y)
            dif.set_numero_fresnel(1.0)
            
            visualizar_comparacion_completa(
                dif,
                x_range_mm=(-5, 5),
                y_range_mm=(-5, 5),
                resolucion_2d=80
            )
        
        elif opcion == '2':
            print("\nEjecutando simulaciones para varios NF...")
            simulacion_varios_NF()
        
        elif opcion == '3':
            interfaz_interactiva()
        
        elif opcion == '4':
            print("¡Hasta luego!")
            break
        
        else:
            print("Opción no válida. Intente de nuevo.")
            
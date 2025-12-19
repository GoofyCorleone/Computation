import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Dict, Optional
import time

# ============================================================================
# 1. MÉTODO DE INTEGRACIÓN NUMÉRICA CON REGISTRO DE PUNTOS
# ============================================================================

class SimpsonAdaptativo:
    """
    Método de Simpson auto-adaptativo que registra los puntos de evaluación.
    """
    
    def __init__(self, registrar_puntos: bool = False):
        """
        Parámetros:
        -----------
        registrar_puntos : si True, registra todos los puntos de evaluación
        """
        self.registrar_puntos = registrar_puntos
        self.puntos_evaluados = []  # Lista de (x, f(x))
        self.llamadas_funcion = 0
        self.profundidad_maxima = 0
    
    def simpson(self, f: Callable, a: float, b: float, 
                tol: float = 1e-8, max_depth: int = 20) -> Tuple[complex, Dict]:
        """
        Método de Simpson auto-adaptativo recursivo.
        
        Retorna:
        --------
        (integral, info) donde info contiene estadísticas
        """
        self.puntos_evaluados.clear()
        self.llamadas_funcion = 0
        self.profundidad_maxima = 0
        
        def _simpson_rec(a: float, b: float, fa: complex, fb: complex, 
                        fc: complex, tol: float, depth: int) -> complex:
            # Actualizar profundidad máxima
            self.profundidad_maxima = max(self.profundidad_maxima, depth)
            
            # Regla de Simpson en el intervalo [a, b]
            h = (b - a) / 2
            m = a + h
            
            # Evaluar en puntos medios
            fd = f(a + h/2)
            fe = f(b - h/2)
            
            # Registrar puntos si está habilitado
            if self.registrar_puntos:
                for punto in [(a + h/2, fd), (b - h/2, fe)]:
                    if punto not in self.puntos_evaluados:
                        self.puntos_evaluados.append(punto)
            
            self.llamadas_funcion += 2
            
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
        
        # Evaluaciones iniciales
        fa, fb, fc = f(a), f(b), f((a+b)/2)
        
        # Registrar puntos iniciales si está habilitado
        if self.registrar_puntos:
            self.puntos_evaluados.extend([(a, fa), (b, fb), ((a+b)/2, fc)])
        
        self.llamadas_funcion += 3
        
        integral = _simpson_rec(a, b, fa, fb, fc, tol, 0)
        
        # Información de estadísticas
        info = {
            'llamadas_funcion': self.llamadas_funcion,
            'profundidad_maxima': self.profundidad_maxima,
            'num_puntos': len(self.puntos_evaluados),
            'puntos_evaluados': sorted(self.puntos_evaluados, key=lambda x: x[0])
        }
        
        return integral, info

# Instancia global para uso general
simpson_global = SimpsonAdaptativo(registrar_puntos=False)

def simpson_adaptativo(f: Callable, a: float, b: float, 
                       tol: float = 1e-8, max_depth: int = 20) -> Tuple[complex, Dict]:
    """Wrapper para compatibilidad con código existente."""
    return simpson_global.simpson(f, a, b, tol, max_depth)

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
        
        # Para almacenar información de integración
        self.info_integracion = {}
    
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
    
    # ========================= MÉTODOS PARA CÁLCULO 1D (PERFILES) ========================
    
    def perfil_x_fresnel(self, x_vals: np.ndarray, y_fijo: float = 0.0, 
                         registrar_puntos: bool = False) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Calcula el perfil de intensidad en X usando aproximación de Fresnel.
        
        Parámetros:
        -----------
        registrar_puntos : si True, registra puntos de integración para x=0
        """
        intensidad = np.zeros_like(x_vals, dtype=float)
        factor = (1j/(self.lambda_ * self.z)) * np.exp(-1j * self.k * self.z)
        
        info_integracion = None
        
        for i, x in enumerate(x_vals):
            # Integral doble separada (por ser rectangular)
            def integral_en_y(yp):
                if abs(yp) <= self.ancho_y/2:
                    return np.exp(-1j * self.k * (y_fijo - yp)**2 / (2 * self.z))
                return 0.0
            
            def integral_en_x(xp):
                if abs(xp) <= self.ancho_x/2:
                    # Integral en y para este xp
                    int_y, _ = simpson_adaptativo(
                        lambda yp: integral_en_y(yp),
                        -self.ancho_y/2, self.ancho_y/2,
                        tol=1e-8
                    )
                    return np.exp(-1j * self.k * (x - xp)**2 / (2 * self.z)) * int_y
                return 0.0
            
            # Integral en x
            if registrar_puntos and abs(x) < 1e-10:  # Solo para x=0
                integrador = SimpsonAdaptativo(registrar_puntos=True)
                int_total, info = integrador.simpson(
                    lambda xp: integral_en_x(xp),
                    -self.ancho_x/2, self.ancho_x/2,
                    tol=1e-8
                )
                info_integracion = info
            else:
                int_total, _ = simpson_adaptativo(
                    lambda xp: integral_en_x(xp),
                    -self.ancho_x/2, self.ancho_x/2,
                    tol=1e-8
                )
            
            campo = factor * int_total
            intensidad[i] = np.abs(campo)**2
        
        return intensidad, info_integracion
    
    def perfil_x_huygens_fresnel(self, x_vals: np.ndarray, y_fijo: float = 0.0) -> np.ndarray:
        """Calcula el perfil de intensidad en X usando Huygens-Fresnel."""
        intensidad = np.zeros_like(x_vals, dtype=float)
        factor = (1j/self.lambda_)
        
        for i, x in enumerate(x_vals):
            def integral_en_y(yp):
                if abs(yp) <= self.ancho_y/2:
                    r = np.sqrt(self.z**2 + (x - 0)**2 + (y_fijo - yp)**2)
                    return (self.z/r) * np.exp(-1j * self.k * r) / r
                return 0.0
            
            def integral_en_x(xp):
                if abs(xp) <= self.ancho_x/2:
                    # Integral en y para este xp
                    int_y, _ = simpson_adaptativo(
                        lambda yp: integral_en_y(yp),
                        -self.ancho_y/2, self.ancho_y/2,
                        tol=1e-8
                    )
                    r = np.sqrt(self.z**2 + (x - xp)**2 + (y_fijo - 0)**2)
                    return (self.z/r) * np.exp(-1j * self.k * r) / r * int_y
                return 0.0
            
            # Integral en x
            int_total, _ = simpson_adaptativo(
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
                    int_x, _ = simpson_adaptativo(
                        lambda xp: integral_en_x(xp),
                        -self.ancho_x/2, self.ancho_x/2,
                        tol=1e-8
                    )
                    return np.exp(-1j * self.k * (y - yp)**2 / (2 * self.z)) * int_x
                return 0.0
            
            # Integral en y
            int_total, _ = simpson_adaptativo(
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
                    r = np.sqrt(self.z**2 + (x_fijo - xp)**2 + (y - 0)**2)
                    return (self.z/r) * np.exp(-1j * self.k * r) / r
                return 0.0
            
            def integral_en_y(yp):
                if abs(yp) <= self.ancho_y/2:
                    # Integral en x para este yp
                    int_x, _ = simpson_adaptativo(
                        lambda xp: integral_en_x(xp),
                        -self.ancho_x/2, self.ancho_x/2,
                        tol=1e-8
                    )
                    r = np.sqrt(self.z**2 + (x_fijo - 0)**2 + (y - yp)**2)
                    return (self.z/r) * np.exp(-1j * self.k * r) / r * int_x
                return 0.0
            
            # Integral en y
            int_total, _ = simpson_adaptativo(
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
            
            campos_x[i], _ = simpson_adaptativo(
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
            
            campos_y[j], _ = simpson_adaptativo(
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
# 3. VISUALIZACIÓN COMPLETA SIN MALLA DE INTEGRACIÓN
# ============================================================================

def visualizar_comparacion_completa(dif: DifraccionRectangular2DCompleta, 
                                    x_range_mm: Tuple[float, float] = (-5, 5),
                                    y_range_mm: Tuple[float, float] = (-5, 5),
                                    resolucion_2d: int = 100,
                                    guardar_imagen: bool = False,
                                    nombre_archivo: str = None):
    """
    Visualización completa con mapas 2D y perfiles 1D comparando ambos métodos.
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
    inicio_perfiles = time.time()
    
    # Perfil en X (y=0) - NO registrar puntos (para velocidad)
    perfil_x_f, _ = dif.perfil_x_fresnel(x_vals, y_fijo=0.0, registrar_puntos=False)
    perfil_x_hf = dif.perfil_x_huygens_fresnel(x_vals, y_fijo=0.0)
    
    # Perfil en Y (x=0)
    perfil_y_f = dif.perfil_y_fresnel(y_vals, x_fijo=0.0)
    perfil_y_hf = dif.perfil_y_huygens_fresnel(y_vals, x_fijo=0.0)
    
    tiempo_perfiles = time.time() - inicio_perfiles
    print(f"  Tiempo cálculo perfiles: {tiempo_perfiles:.2f} segundos")
    
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
    
    # Crear figura con disposición 2x2
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'Difracción Rectangular 2D - Comparación Fresnel vs Huygens-Fresnel\n'
                 f'NF = {dif.NF:.2f}, z = {dif.z*100:.2f} cm, '
                 f'λ = {dif.lambda_*1e9:.0f} nm, '
                 f'Ancho X = {dif.ancho_x*1000:.1f} mm, Ancho Y = {dif.ancho_y*1000:.1f} mm',
                 fontsize=14, fontweight='bold')
    
    # ===================== MAPA FRESNEL 2D =====================
    ax1 = plt.subplot(2, 2, 1)
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
    
    # ===================== MAPA HUYGENS-FRESNEL 2D =====================
    ax2 = plt.subplot(2, 2, 2)
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
    
    # ===================== PERFIL EN X =====================
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(x_vals_mm, perfil_x_f_norm, 'b-', linewidth=2, label='Fresnel')
    ax3.plot(x_vals_mm, perfil_x_hf_norm, 'r--', linewidth=2, label='Huygens-Fresnel')
    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('Intensidad (normalizada)')
    ax3.set_title(f'Perfil en X (y=0)\nNF = {dif.NF:.2f}')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)
    
    # Calcular y mostrar diferencia máxima en perfil X
    diff_x = np.max(np.abs(perfil_x_f_norm - perfil_x_hf_norm))
    ax3.text(0.02, 0.98, f'Diff máx: {diff_x:.4f}', 
             transform=ax3.transAxes, verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ===================== PERFIL EN Y =====================
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(y_vals_mm, perfil_y_f_norm, 'b-', linewidth=2, label='Fresnel')
    ax4.plot(y_vals_mm, perfil_y_hf_norm, 'r--', linewidth=2, label='Huygens-Fresnel')
    ax4.set_xlabel('y (mm)')
    ax4.set_ylabel('Intensidad (normalizada)')
    ax4.set_title(f'Perfil en Y (x=0)\nNF = {dif.NF:.2f}')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.1)
    
    # Calcular y mostrar diferencia máxima en perfil Y
    diff_y = np.max(np.abs(perfil_y_f_norm - perfil_y_hf_norm))
    ax4.text(0.02, 0.98, f'Diff máx: {diff_y:.4f}', 
             transform=ax4.transAxes, verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if guardar_imagen:
        if nombre_archivo is None:
            nombre_archivo = f"difraccion_comparacion_NF{dif.NF:.2f}.png"
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
    print(f"Diferencia máxima en perfil X: {diff_x:.6f}")
    print(f"Diferencia máxima en perfil Y: {diff_y:.6f}")

# ============================================================================
# 4. FUNCIÓN PARA VISUALIZAR SOLO LA MALLA DE INTEGRACIÓN
# ============================================================================

def visualizar_malla_integracion(dif: DifraccionRectangular2DCompleta, 
                                 x_observacion: float = 0.0,
                                 y_observacion: float = 0.0):
    """
    Visualización detallada de la malla de integración para un punto específico.
    """
    
    print(f"\nGenerando malla de integración para (x={x_observacion*1000:.1f} mm, y={y_observacion*1000:.1f} mm)")
    print("Calculando...")
    
    # Crear un integrador con registro de puntos
    integrador = SimpsonAdaptativo(registrar_puntos=True)
    
    # Definir el integrando para Fresnel en el punto dado
    def integrando_x(xp):
        if abs(xp) <= dif.ancho_x/2:
            # Integral en y para este xp
            def integrando_y(yp):
                if abs(yp) <= dif.ancho_y/2:
                    return np.exp(-1j * dif.k * ((x_observacion - xp)**2 + (y_observacion - yp)**2) / (2 * dif.z))
                return 0.0
            
            # Usar Simpson para integral en y
            int_y, _ = integrador.simpson(
                integrando_y,
                -dif.ancho_y/2, dif.ancho_y/2,
                tol=1e-8
            )
            return int_y
        return 0.0
    
    # Calcular la integral en x
    inicio = time.time()
    integral, info = integrador.simpson(
        integrando_x,
        -dif.ancho_x/2, dif.ancho_x/2,
        tol=1e-8
    )
    tiempo = time.time() - inicio
    
    # Extraer puntos de la malla
    puntos = np.array([p[0] for p in info['puntos_evaluados']])
    valores = np.array([p[1] for p in info['puntos_evaluados']])
    
    # Crear figura para visualización detallada
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Malla de Integración Simpson Adaptativo\n'
                f'NF = {dif.NF:.2f}, Punto de observación: ({x_observacion*1000:.1f}, {y_observacion*1000:.1f}) mm',
                fontsize=14, fontweight='bold')
    
    # Gráfico 1: Puntos de evaluación
    ax1 = axes[0, 0]
    ax1.scatter(puntos * 1000, np.arange(len(puntos)), 
               color='blue', alpha=0.6, s=50)
    ax1.set_xlabel('x\' (mm)')
    ax1.set_ylabel('Índice del punto')
    ax1.set_title(f'Distribución de {len(puntos)} puntos de evaluación')
    ax1.grid(True, alpha=0.3)
    ax1.axvspan(-dif.ancho_x/2 * 1000, dif.ancho_x/2 * 1000, 
               alpha=0.1, color='green', label='Abertura')
    ax1.legend()
    
    # Gráfico 2: Valor del integrando en cada punto
    ax2 = axes[0, 1]
    orden = np.argsort(puntos)
    ax2.plot(puntos[orden] * 1000, np.abs(valores[orden]), 
            'b-', linewidth=2, label='|Integrando|')
    ax2.scatter(puntos[orden] * 1000, np.abs(valores[orden]), 
               color='red', s=30, alpha=0.6, label='Puntos')
    ax2.set_xlabel('x\' (mm)')
    ax2.set_ylabel('|Integrando|')
    ax2.set_title('Valor del integrando en los puntos de evaluación')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Gráfico 3: Parte real e imaginaria
    ax3 = axes[1, 0]
    ax3.plot(puntos[orden] * 1000, np.real(valores[orden]), 
            'b-', linewidth=2, label='Parte Real')
    ax3.plot(puntos[orden] * 1000, np.imag(valores[orden]), 
            'r--', linewidth=2, label='Parte Imaginaria')
    ax3.set_xlabel('x\' (mm)')
    ax3.set_ylabel('Valor')
    ax3.set_title('Parte Real e Imaginaria del Integrando')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Gráfico 4: Histograma de espaciado entre puntos
    ax4 = axes[1, 1]
    if len(puntos) > 1:
        espaciados = np.diff(np.sort(puntos)) * 1000  # En mm
        ax4.hist(espaciados, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_xlabel('Espaciado entre puntos (mm)')
        ax4.set_ylabel('Frecuencia')
        ax4.set_title(f'Distribución de espaciados\nMedia: {np.mean(espaciados):.4f} mm')
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No hay suficientes puntos\npara histograma',
                horizontalalignment='center', verticalalignment='center',
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Distribución de espaciados')
    
    # Información estadística
    stats_text = (
        f'Estadísticas de Integración:\n'
        f'• Puntos de evaluación: {info["num_puntos"]}\n'
        f'• Evaluaciones totales: {info["llamadas_funcion"]}\n'
        f'• Profundidad máxima: {info["profundidad_maxima"]}\n'
        f'• Tiempo de cálculo: {tiempo:.3f} s\n'
        f'• Valor integral: {integral:.4e}\n'
        f'• |Integral|: {np.abs(integral):.4e}\n'
        f'• Rango x\': [{puntos.min()*1000:.3f}, {puntos.max()*1000:.3f}] mm'
    )
    
    fig.text(0.02, 0.02, stats_text, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"malla_integracion_NF{dif.NF:.2f}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nInformación de la malla:")
    print(f"  Puntos de evaluación: {info['num_puntos']}")
    print(f"  Evaluaciones totales: {info['llamadas_funcion']}")
    print(f"  Profundidad máxima de recursión: {info['profundidad_maxima']}")
    print(f"  Tiempo de cálculo: {tiempo:.3f} segundos")
    print(f"  Valor de la integral: {integral:.6e}")
    print(f"  Módulo de la integral: {np.abs(integral):.6e}")

# ============================================================================
# 5. SIMULACIÓN PARA VARIOS NÚMEROS DE FRESNEL
# ============================================================================

def simulacion_varios_NF():
    """Ejecuta simulaciones para diferentes números de Fresnel."""
    
    print("SIMULACIÓN DE DIFRACCIÓN RECTANGULAR 2D")
    print("="*60)
    
    # Parámetros fijos
    lambda_ = 632.8e-9  # Luz He-Ne
    ancho_x = 1.0e-3    # 1 mm
    ancho_y = 0.5e-3    # 0.5 mm (rectángulo asimétrico)
    
    # Números de Fresnel a explorar
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
        factor_expansion = 2.0 + 5.0/NF if NF > 0.1 else 20.0
        rango = 5 * factor_expansion
        
        # Visualizar sin malla (para velocidad)
        visualizar_comparacion_completa(
            dif,
            x_range_mm=(-rango, rango),
            y_range_mm=(-rango, rango),
            resolucion_2d=70,
            guardar_imagen=True,
            nombre_archivo=f"rectangular_NF{NF:.1f}.png"
        )

# ============================================================================
# 6. MENÚ PRINCIPAL MEJORADO
# ============================================================================

def menu_principal():
    """Menú principal del programa."""
    
    print("DIFRACCIÓN RECTANGULAR 2D - FRESNEL vs HUYGENS-FRESNEL")
    print("="*60)
    print("Simulaciones numéricas usando método de Simpson auto-adaptativo")
    print("="*60)
    
    while True:
        print("\n" + "="*50)
        print("MENÚ PRINCIPAL")
        print("="*50)
        print("1. Simulación completa (NF=1.0)")
        print("2. Visualizar malla de integración (detallada)")
        print("3. Simulación para varios números de Fresnel")
        print("4. Interfaz interactiva")
        print("5. Salir")
        
        opcion = input("\nSeleccione una opción (1-5): ").strip()
        
        if opcion == '1':
            print("\n" + "="*50)
            print("SIMULACIÓN COMPLETA (NF=1.0)")
            print("="*50)
            print("λ = 632.8 nm, Ancho X = 1.0 mm, Ancho Y = 0.5 mm, NF = 1.0")
            
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
                resolucion_2d=80,
                guardar_imagen=True,
                nombre_archivo=f"completa_NF{dif.NF:.2f}.png"
            )
        
        elif opcion == '2':
            print("\n" + "="*50)
            print("VISUALIZACIÓN DETALLADA DE MALLA DE INTEGRACIÓN")
            print("="*50)
            
            # Configurar parámetros
            lambda_ = 632.8e-9
            ancho_x = 1.0e-3
            ancho_y = 0.5e-3
            
            dif = DifraccionRectangular2DCompleta(lambda_=lambda_, 
                                                  ancho_x=ancho_x, 
                                                  ancho_y=ancho_y)
            
            # Pedir NF
            NF = float(input("Número de Fresnel (por defecto 1.0): ") or "1.0")
            dif.set_numero_fresnel(NF)
            
            # Pedir punto de observación
            x_mm = float(input("Coordenada x del punto de observación (mm, por defecto 0.0): ") or "0.0")
            y_mm = float(input("Coordenada y del punto de observación (mm, por defecto 0.0): ") or "0.0")
            
            x_obs = x_mm * 1e-3
            y_obs = y_mm * 1e-3
            
            visualizar_malla_integracion(dif, x_observacion=x_obs, y_observacion=y_obs)
        
        elif opcion == '3':
            print("\n" + "="*50)
            print("SIMULACIÓN PARA VARIOS NÚMEROS DE FRESNEL")
            print("="*50)
            print("NF = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]")
            print("λ = 632.8 nm, Ancho X = 1.0 mm, Ancho Y = 0.5 mm")
            print("\nNota: Esta simulación puede tomar varios minutos...")
            
            confirmar = input("¿Continuar? (s/n): ").strip().lower()
            if confirmar == 's':
                simulacion_varios_NF()
        
        elif opcion == '4':
            print("\n" + "="*50)
            print("INTERFAZ INTERACTIVA")
            print("="*50)
            interfaz_interactiva()
        
        elif opcion == '5':
            print("\n¡Gracias por usar el simulador de difracción!")
            print("¡Hasta luego!")
            break
        
        else:
            print("\nOpción no válida. Por favor, seleccione 1-5.")

# ============================================================================
# 7. INTERFAZ INTERACTIVA
# ============================================================================

def interfaz_interactiva():
    """Interfaz interactiva para explorar diferentes parámetros."""
    
    print("\n" + "="*50)
    print("INTERFAZ INTERACTIVA")
    print("="*50)
    
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
        print("2. Especificar distancia z (en cm)")
        
        modo = input("Seleccione (1 o 2): ").strip()
        
        if modo == '1':
            NF = float(input("Número de Fresnel (por defecto 1.0): ") or "1.0")
            dif.set_numero_fresnel(NF)
        elif modo == '2':
            z_cm = float(input("Distancia z (cm, por defecto 10.0): ") or "10.0")
            z = z_cm * 0.01
            dif.set_distancia(z)
        else:
            print("Usando NF=1.0 por defecto")
            dif.set_numero_fresnel(1.0)
        
        # Rango de visualización
        rango_mm = float(input("Rango de visualización (±mm, por defecto 5.0): ") or "5.0")
        
        # Resolución
        resolucion = int(input("Resolución 2D (puntos, por defecto 80): ") or "80")
        
        # ¿Mostrar malla de integración?
        mostrar_malla = input("\n¿Desea visualizar la malla de integración? (s/n): ").strip().lower()
        
        print(f"\n{'='*50}")
        print("EJECUTANDO SIMULACIÓN")
        print("="*50)
        print(f"  λ = {lambda_nm} nm")
        print(f"  Ancho X = {ancho_x_mm} mm, Ancho Y = {ancho_y_mm} mm")
        print(f"  NF = {dif.NF:.4f}")
        print(f"  z = {dif.z*100:.2f} cm")
        print(f"  Rango = ±{rango_mm} mm")
        print(f"  Resolución = {resolucion}×{resolucion}")
        
        # Ejecutar simulación principal
        visualizar_comparacion_completa(
            dif,
            x_range_mm=(-rango_mm, rango_mm),
            y_range_mm=(-rango_mm, rango_mm),
            resolucion_2d=resolucion,
            guardar_imagen=True,
            nombre_archivo=f"interactiva_NF{dif.NF:.2f}.png"
        )
        
        # Si se solicitó, mostrar malla de integración
        if mostrar_malla == 's':
            visualizar_malla_integracion(dif, x_observacion=0.0, y_observacion=0.0)
        
        # Preguntar si continuar
        continuar = input("\n¿Realizar otra simulación? (s/n): ").strip().lower()
        if continuar != 's':
            break

# ============================================================================
# 8. EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Ejecutar menú principal
    menu_principal()
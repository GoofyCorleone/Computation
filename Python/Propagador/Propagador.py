import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Dict, Optional
import time

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
        self.puntos_evaluados = [] 
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
            
            self.profundidad_maxima = max(self.profundidad_maxima, depth)
            h = (b - a) / 2
            m = a + h
            
            fd = f(a + h/2)
            fe = f(b - h/2)
            
            if self.registrar_puntos:
                for punto in [(a + h/2, fd), (b - h/2, fe)]:
                    if punto not in self.puntos_evaluados:
                        self.puntos_evaluados.append(punto)
            
            self.llamadas_funcion += 2
            
            S_ab = h/3 * (fa + 4*fc + fb)  
            S_ac = h/6 * (fa + 4*fd + fc)  
            S_cb = h/6 * (fc + 4*fe + fb) 
            S_total = S_ac + S_cb
            
            error = abs(S_total - S_ab) / 15
            
            if error < tol or depth >= max_depth:
                
                return S_total + (S_total - S_ab) / 15  
            else:
               
                left = _simpson_rec(a, m, fa, fc, fd, tol/2, depth+1)
                right = _simpson_rec(m, b, fc, fb, fe, tol/2, depth+1)
                return left + right
        
        fa, fb, fc = f(a), f(b), f((a+b)/2)
        
        if self.registrar_puntos:
            self.puntos_evaluados.extend([(a, fa), (b, fb), ((a+b)/2, fc)])
        
        self.llamadas_funcion += 3
        
        integral = _simpson_rec(a, b, fa, fb, fc, tol, 0)
        
        info = {
            'llamadas_funcion': self.llamadas_funcion,
            'profundidad_maxima': self.profundidad_maxima,
            'num_puntos': len(self.puntos_evaluados),
            'puntos_evaluados': sorted(self.puntos_evaluados, key=lambda x: x[0])
        }
        
        return integral, info

def simpson_adaptativo(f: Callable, a: float, b: float, 
                       tol: float = 1e-8, max_depth: int = 20) -> Tuple[complex, Dict]:
    """Wrapper para compatibilidad con código existente."""
    return simpson_global.simpson(f, a, b, tol, max_depth)

simpson_global = SimpsonAdaptativo(registrar_puntos=False)

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
        self.k = 2 * np.pi / lambda_  
        self.z = None
        self.NF = None
        
        self.info_integracion = {}
    
    def set_numero_fresnel(self, NF: float) -> None:
        """Establece la distancia z a partir del número de Fresnel."""
        
        a_prom = (self.ancho_x + self.ancho_y) / 2
        self.z = a_prom**2 / (self.lambda_ * NF)
        self.NF = NF
    
    def set_distancia(self, z: float) -> None:
        """Establece la distancia z directamente."""
        self.z = z
        
        a_prom = (self.ancho_x + self.ancho_y) / 2
        self.NF = a_prom**2 / (self.lambda_ * z)
    
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
            def integral_en_y(yp):
                if abs(yp) <= self.ancho_y/2:
                    return np.exp(-1j * self.k * (y_fijo - yp)**2 / (2 * self.z))
                return 0.0
            
            def integral_en_x(xp):
                if abs(xp) <= self.ancho_x/2:
                    
                    int_y, _ = simpson_adaptativo(
                        lambda yp: integral_en_y(yp),
                        -self.ancho_y/2, self.ancho_y/2,
                        tol=1e-8
                    )
                    return np.exp(-1j * self.k * (x - xp)**2 / (2 * self.z)) * int_y
                return 0.0
            
            if registrar_puntos and abs(x) < 1e-10: 
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
                    
                    int_y, _ = simpson_adaptativo(
                        lambda yp: integral_en_y(yp),
                        -self.ancho_y/2, self.ancho_y/2,
                        tol=1e-8
                    )
                    r = np.sqrt(self.z**2 + (x - xp)**2 + (y_fijo - 0)**2)
                    return (self.z/r) * np.exp(-1j * self.k * r) / r * int_y
                return 0.0
            
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
            def integral_en_x(xp):
                if abs(xp) <= self.ancho_x/2:
                    return np.exp(-1j * self.k * (x_fijo - xp)**2 / (2 * self.z))
                return 0.0
            
            def integral_en_y(yp):
                if abs(yp) <= self.ancho_y/2:
                    
                    int_x, _ = simpson_adaptativo(
                        lambda xp: integral_en_x(xp),
                        -self.ancho_x/2, self.ancho_x/2,
                        tol=1e-8
                    )
                    return np.exp(-1j * self.k * (y - yp)**2 / (2 * self.z)) * int_x
                return 0.0
            
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
                    
                    int_x, _ = simpson_adaptativo(
                        lambda xp: integral_en_x(xp),
                        -self.ancho_x/2, self.ancho_x/2,
                        tol=1e-8
                    )
                    r = np.sqrt(self.z**2 + (x_fijo - 0)**2 + (y - yp)**2)
                    return (self.z/r) * np.exp(-1j * self.k * r) / r * int_x
                return 0.0
            
            int_total, _ = simpson_adaptativo(
                lambda yp: integral_en_y(yp),
                -self.ancho_y/2, self.ancho_y/2,
                tol=1e-8
            )
            
            campo = factor * int_total
            intensidad[i] = np.abs(campo)**2
        
        return intensidad
    
    def calcular_mapa_fresnel_2d(self, x_vals: np.ndarray, y_vals: np.ndarray) -> np.ndarray:
        """Calcula el mapa de intensidad 2D usando aproximación de Fresnel."""
        print(f"Calculando mapa Fresnel 2D (NF={self.NF:.2f})...")
        inicio = time.time()
        
        intensidad = np.zeros((len(y_vals), len(x_vals)))
        factor = (1j/(self.lambda_ * self.z)) * np.exp(-1j * self.k * self.z)
        
        campos_x = np.zeros(len(x_vals), dtype=complex)
        campos_y = np.zeros(len(y_vals), dtype=complex)
        
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
        
        xp_vals = np.linspace(-self.ancho_x/2, self.ancho_x/2, resolucion_integracion)
        yp_vals = np.linspace(-self.ancho_y/2, self.ancho_y/2, resolucion_integracion)
        
        def pesos_simpson(n):
            """Genera pesos para la regla compuesta de Simpson."""
            h = 1.0 / (n - 1)
            weights = np.ones(n)
            weights[1:-1:2] = 4
            weights[2:-1:2] = 2
            return h * weights / 3
        
        wx = pesos_simpson(resolucion_integracion) * self.ancho_x
        wy = pesos_simpson(resolucion_integracion) * self.ancho_y
        
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                Xp, Yp = np.meshgrid(xp_vals, yp_vals)
                r = np.sqrt(self.z**2 + (x - Xp)**2 + (y - Yp)**2)
                integrando = (self.z/r) * np.exp(-1j * self.k * r) / r
                integral = np.sum(wx[None, :] * wy[:, None] * integrando)
                
                campo = factor * integral
                intensidad[i, j] = np.abs(campo)**2
        
        tiempo = time.time() - inicio
        print(f"  Tiempo de cálculo: {tiempo:.2f} segundos")
        
        return intensidad
    
def visualizar_comparacion_completa(dif: DifraccionRectangular2DCompleta, 
                                    x_range_mm: Tuple[float, float] = (-5, 5),
                                    y_range_mm: Tuple[float, float] = (-5, 5),
                                    resolucion_2d: int = 100,
                                    guardar_imagen: bool = False,
                                    nombre_archivo: str = None):
    """
    Visualización completa con mapas 2D y perfiles 1D comparando ambos métodos.
    """
    x_min, x_max = x_range_mm[0] * 1e-3, x_range_mm[1] * 1e-3
    y_min, y_max = y_range_mm[0] * 1e-3, y_range_mm[1] * 1e-3
    
    x_vals = np.linspace(x_min, x_max, resolucion_2d)
    y_vals = np.linspace(y_min, y_max, resolucion_2d)
    x_vals_mm = x_vals * 1e3
    y_vals_mm = y_vals * 1e3
    
    I_fresnel_2d = dif.calcular_mapa_fresnel_2d(x_vals, y_vals)
    I_hf_2d = dif.calcular_mapa_huygens_fresnel_2d(x_vals, y_vals, resolucion_integracion=40)
    print("\nCalculando perfiles 1D...")
    inicio_perfiles = time.time()
    
    perfil_x_f, _ = dif.perfil_x_fresnel(x_vals, y_fijo=0.0, registrar_puntos=False)
    perfil_x_hf = dif.perfil_x_huygens_fresnel(x_vals, y_fijo=0.0)
    
    perfil_y_f = dif.perfil_y_fresnel(y_vals, x_fijo=0.0)
    perfil_y_hf = dif.perfil_y_huygens_fresnel(y_vals, x_fijo=0.0)
    
    tiempo_perfiles = time.time() - inicio_perfiles
    print(f"  Tiempo cálculo perfiles: {tiempo_perfiles:.2f} segundos")
    
    max_2d = max(np.max(I_fresnel_2d), np.max(I_hf_2d))
    I_fresnel_2d_norm = I_fresnel_2d / max_2d
    I_hf_2d_norm = I_hf_2d / max_2d
    
    max_perfil_x = max(np.max(perfil_x_f), np.max(perfil_x_hf))
    perfil_x_f_norm = perfil_x_f / max_perfil_x
    perfil_x_hf_norm = perfil_x_hf / max_perfil_x
    
    max_perfil_y = max(np.max(perfil_y_f), np.max(perfil_y_hf))
    perfil_y_f_norm = perfil_y_f / max_perfil_y
    perfil_y_hf_norm = perfil_y_hf / max_perfil_y
    
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'Difracción Rectangular 2D - Comparación Fresnel vs Huygens-Fresnel\n'
                 f'NF = {dif.NF:.2f}, z = {dif.z*100:.2f} cm, '
                 f'λ = {dif.lambda_*1e9:.0f} nm, '
                 f'Ancho X = {dif.ancho_x*1000:.1f} mm, Ancho Y = {dif.ancho_y*1000:.1f} mm',
                 fontsize=14, fontweight='bold')
    
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
    
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(x_vals_mm, perfil_x_f_norm, 'b-', linewidth=2, label='Fresnel')
    ax3.plot(x_vals_mm, perfil_x_hf_norm, 'r--', linewidth=2, label='Huygens-Fresnel')
    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('Intensidad (normalizada)')
    ax3.set_title(f'Perfil en X (y=0)\nNF = {dif.NF:.2f}')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)
    
    diff_x = np.max(np.abs(perfil_x_f_norm - perfil_x_hf_norm))
    ax3.text(0.02, 0.98, f'Diff máx: {diff_x:.4f}', 
             transform=ax3.transAxes, verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(y_vals_mm, perfil_y_f_norm, 'b-', linewidth=2, label='Fresnel')
    ax4.plot(y_vals_mm, perfil_y_hf_norm, 'r--', linewidth=2, label='Huygens-Fresnel')
    ax4.set_xlabel('y (mm)')
    ax4.set_ylabel('Intensidad (normalizada)')
    ax4.set_title(f'Perfil en Y (x=0)\nNF = {dif.NF:.2f}')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.1)
    
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
    
    print("\n" + "="*70)
    print("RESUMEN ESTADÍSTICO")
    print("="*70)
    print(f"Número de Fresnel: {dif.NF:.4f}")
    print(f"Distancia z: {dif.z:.4f} m ({dif.z*100:.2f} cm)")
    print(f"Intensidad máxima Fresnel 2D: {np.max(I_fresnel_2d):.4e}")
    print(f"Intensidad máxima Huygens-Fresnel 2D: {np.max(I_hf_2d):.4e}")
    print(f"Diferencia máxima en perfil X: {diff_x:.6f}")
    print(f"Diferencia máxima en perfil Y: {diff_y:.6f}")
    
def main(parametros = []):
    
    if  parametros[3]:
        lambda_, ancho_x, ancho_y = parametros[0], parametros[1], parametros[2]
        
        dif = DifraccionRectangular2DCompleta(lambda_=lambda_, 
                                                  ancho_x=ancho_x, 
                                                  ancho_y=ancho_y)
        dif.set_numero_fresnel(parametros[4])
            
        visualizar_comparacion_completa(
            dif,
            x_range_mm=(-5, 5),
            y_range_mm=(-5, 5),
            resolucion_2d=parametros[5],
            guardar_imagen=False
            )
    else:
        lambda_, ancho_x, ancho_y , z = parametros[0], parametros[1], parametros[2], parametros[4]
        
        dif = DifraccionRectangular2DCompleta(lambda_=lambda_, 
                                                  ancho_x=ancho_x, 
                                                  ancho_y=ancho_y)
        dif.set_distancia(z=z)
            
        visualizar_comparacion_completa(
            dif,
            x_range_mm=(-5, 5),
            y_range_mm=(-5, 5),
            resolucion_2d=parametros[5],
            guardar_imagen=False
            )
    
lambda_ = 632.8e-9
ancho_x = 1.0e-3
ancho_y = 0.5e-3
z = 8.89e-1
resolucion = 160
parametros = [lambda_,ancho_x,ancho_y,False,z, resolucion]

main(parametros=parametros)
import numpy as np
from scipy.optimize import differential_evolution, minimize
import random

class ParameterOptimizer:
    """
    Clase para optimizar parámetros del sistema de difracción
    para maximizar la diferencia entre métodos.
    """
    
    def __init__(self, simulator):
        """
        Inicializa el optimizador.
        
        Parámetros:
        -----------
        simulator : FresnelDiffractionSimulator
            Instancia del simulador
        """
        self.simulator = simulator
        self.best_params_history = []
        self.best_diff_history = []
        
    def define_parameter_bounds(self):
        """
        Define los límites razonables para cada parámetro.
        
        Estos límites evitan configuraciones físicamente imposibles
        o numéricamente inestables.
        """
        bounds = {
            'p': (0.05e-3, 1.0e-3),     # Ancho primera rendija (50μm - 1mm)
            'q': (0.05e-3, 1.0e-3),     # Alto primera rendija
            'p2': (0.05e-3, 1.0e-3),    # Ancho segunda rendija
            'q2': (0.05e-3, 1.0e-3),    # Alto segunda rendija
            'a': (-2.0e-3, 2.0e-3),     # Desplazamiento x segunda rendija
            'b': (-2.0e-3, 2.0e-3),     # Desplazamiento y segunda rendija
            'n': (5e-3, 50e-3),         # Distancia a primera rendija (5-50mm)
            'c': (1e-3, 20e-3),         # Separación entre rendijas (1-20mm)
            'z0': (30e-3, 200e-3)       # Distancia observación (30-200mm)
        }
        return bounds
    
    def calculate_difference(self, x, fast_mode=True):
        """
        Función objetivo: calcula la diferencia entre métodos.
        
        Parámetros:
        -----------
        x : array
            Vector con los 9 parámetros en orden
        fast_mode : bool
            Si True, usa baja resolución para evaluación rápida
            
        Retorna:
        --------
        float : Diferencia promedio (negativa para maximizar)
        """
        # Convertir array a diccionario de parámetros
        params = self._array_to_params(x)
        
        # Asegurar restricciones físicas
        if not self._validate_params(params):
            return 0  # Penalizar configuraciones inválidas
        
        try:
            # Ejecutar simulación (modo rápido para optimización)
            if fast_mode:
                Nx, Ny = 40, 40
                x_range, y_range = 4e-3, 4e-3
            else:
                Nx, Ny = 80, 80
                x_range, y_range = 6e-3, 6e-3
            
            results = self.simulator.simulate_2d_diffraction_complete(
                params,
                Nx=Nx,
                Ny=Ny,
                x_range=x_range,
                y_range=y_range
            )
            
            # Calcular diferencia promedio
            diff = np.mean(np.abs(results['I_fresnel'] - results['I_hf']))
            
            # Almacenar histórico
            self.best_params_history.append(params.copy())
            self.best_diff_history.append(diff)
            
            # Devolver negativo para maximización
            return -diff
            
        except Exception as e:
            print(f"Error en evaluación: {e}")
            return 0  # Penalizar errores
    
    def _array_to_params(self, x):
        """Convierte array numpy a diccionario de parámetros."""
        bounds = self.define_parameter_bounds()
        param_names = list(bounds.keys())
        
        params = {}
        for i, name in enumerate(param_names):
            params[name] = x[i]
        
        # Parámetros fijos
        params['wavelength'] = self.simulator.wavelength
        
        return params
    
    def _validate_params(self, params):
        """Valida que los parámetros sean físicamente razonables."""
        # z0 debe ser mayor que n + c (plano de observación después de segunda rendija)
        if params['z0'] <= params['n'] + params['c']:
            return False
        
        # Las dimensiones deben ser positivas
        if any(v <= 0 for v in [params['p'], params['q'], params['p2'], params['q2'], params['c']]):
            return False
        
        # Distancias deben ser positivas
        if any(v <= 0 for v in [params['n'], params['z0']]):
            return False
        
        return True
    
    def optimize_with_differential_evolution(self, max_iter=50, pop_size=15):
        """
        Optimización usando evolución diferencial.
        
        Parámetros:
        -----------
        max_iter : int
            Máximo de iteraciones
        pop_size : int
            Tamaño de la población
            
        Retorna:
        --------
        dict : Mejores parámetros encontrados
        float : Mejor diferencia (positiva)
        """
        bounds = self.define_parameter_bounds()
        bounds_list = list(bounds.values())
        
        print("=" * 70)
        print("INICIANDO OPTIMIZACIÓN CON EVOLUCIÓN DIFERENCIAL")
        print("=" * 70)
        print(f"Población: {pop_size}, Iteraciones máximas: {max_iter}")
        print("Usando evaluación rápida (40x40) para búsqueda inicial")
        
        # Función objetivo para scipy
        def objective(x):
            return self.calculate_difference(x, fast_mode=True)
        
        # Ejecutar optimización
        result = differential_evolution(
            objective,
            bounds=bounds_list,
            maxiter=max_iter,
            popsize=pop_size,
            disp=True,
            polish=True,  # Refinar con método local al final
            seed=42  # Para reproducibilidad
        )
        
        # Convertir resultado
        best_params = self._array_to_params(result.x)
        best_diff = -result.fun  # Convertir a positivo
        
        print("\n" + "=" * 70)
        print("OPTIMIZACIÓN COMPLETADA")
        print("=" * 70)
        print(f"Mejor diferencia encontrada: {best_diff:.6f}")
        
        return best_params, best_diff
    
    def refine_optimization(self, initial_params, method='Nelder-Mead'):
        """
        Refina la optimización con un método local.
        
        Parámetros:
        -----------
        initial_params : dict
            Parámetros iniciales (de la búsqueda global)
        method : str
            Método de optimización local
            
        Retorna:
        --------
        dict : Parámetros refinados
        float : Diferencia refinada
        """
        print("\n" + "=" * 70)
        print("REFINANDO CON MÉTODO LOCAL")
        print("=" * 70)
        print(f"Método: {method}")
        print("Usando evaluación detallada (80x80) para refinamiento")
        
        # Convertir a array
        bounds = self.define_parameter_bounds()
        param_names = list(bounds.keys())
        x0 = [initial_params[name] for name in param_names]
        
        # Función objetivo con evaluación más precisa
        def objective(x):
            return self.calculate_difference(x, fast_mode=False)
        
        # Ejecutar optimización local
        result = minimize(
            objective,
            x0=x0,
            method=method,
            options={
                'maxiter': 20,
                'disp': True,
                'xatol': 1e-6,
                'fatol': 1e-6
            }
        )
        
        # Convertir resultado
        refined_params = self._array_to_params(result.x)
        refined_diff = -result.fun
        
        print(f"\nDiferencia refinada: {refined_diff:.6f}")
        print(f"Mejora: {(refined_diff - (-objective(x0))) / (-objective(x0)) * 100:.2f}%")
        
        return refined_params, refined_diff
    
    def print_optimization_results(self, best_params, best_diff):
        """Imprime resultados de forma legible."""
        print("\n" + "=" * 70)
        print("MEJORES PARÁMETROS ENCONTRADOS")
        print("=" * 70)
        
        for key, value in best_params.items():
            if key == 'wavelength':
                print(f"{key}: {value*1e9:.1f} nm")
            elif key in ['p', 'q', 'p2', 'q2', 'a', 'b']:
                print(f"{key}: {value*1e6:.2f} μm")
            elif key in ['n', 'c', 'z0']:
                print(f"{key}: {value*1e3:.2f} mm")
            else:
                print(f"{key}: {value}")
        
        print(f"\nDiferencia máxima alcanzada: {best_diff:.6f}")
        print(f"Número de Fresnel: {(best_params['p']**2) / (best_params['wavelength'] * (best_params['z0'] - best_params['n'])):.3f}")
    
    def plot_optimization_history(self):
        """Grafica el historial de la optimización."""
        if not self.best_diff_history:
            print("No hay historial para graficar")
            return
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Evolución de la diferencia
        ax1.plot(-np.array(self.best_diff_history), 'b-', linewidth=2)
        ax1.set_xlabel('Evaluación')
        ax1.set_ylabel('Diferencia promedio')
        ax1.set_title('Evolución de la optimización')
        ax1.grid(True, alpha=0.3)
        
        # Histograma de diferencias
        ax2.hist(-np.array(self.best_diff_history), bins=20, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Diferencia promedio')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de diferencias evaluadas')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
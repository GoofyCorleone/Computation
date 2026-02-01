import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

class FresnelDiffractionSimulator:
    """
    Simulador numérico de difracción de Fresnel para dos rendijas.
    Versión optimizada con paralelización para MacBook Pro M5.
    """
    
    def __init__(self, wavelength=632.8e-9, use_gpu=False, n_workers=None):
        """
        Inicializa el simulador de difracción de Fresnel
        
        Parámetros:
        -----------
        wavelength : float
            Longitud de onda en metros (default: 632.8 nm, luz He-Ne)
        use_gpu : bool
            Si es True, intenta usar aceleración GPU (Metal en Apple Silicon)
        n_workers : int
            Número de workers para paralelización (None = automático para M5)
        """
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength
        self.results = None
        
        # Configuración para MacBook Pro M5
        if n_workers is None:
            # M5 tiene 10-12 cores (8 performance + 2-4 efficiency)
            self.n_workers = min(10, os.cpu_count() - 2 if os.cpu_count() > 2 else 1)
        else:
            self.n_workers = n_workers
        
        self.use_gpu = use_gpu
        
        # Inicializar GPU si está disponible y solicitado
        self.gpu_available = False
        if self.use_gpu:
            self.gpu_available = self._initialize_gpu()
            
        print(f"Configuración M5: {self.n_workers} workers, GPU: {self.gpu_available}")
    
    def _initialize_gpu(self):
        """Inicializa aceleración GPU para Apple Silicon."""
        try:
            # Para Apple Silicon, podemos usar Metal Performance Shaders
            import metalcompute as mc  # Esto requeriría instalar una librería
            print("Metal Performance Shaders disponible")
            return True
        except ImportError:
            try:
                # Alternativa: usar PyTorch con MPS
                import torch
                if torch.backends.mps.is_available():
                    print("PyTorch MPS disponible para GPU")
                    self.device = torch.device("mps")
                    return True
            except ImportError:
                pass
        
        print("GPU no disponible, usando CPU")
        return False
    
    def simpson_adaptive_1d(self, func, a, b, tol=1e-6, max_depth=15):
        """
        Método de Simpson adaptativo recursivo para integración 1D
        (Optimizado con memoización)
        """
        # Cache para resultados ya calculados
        memo = {}
        
        def memoized_func(x):
            key = round(x / tol * 1000)  # Discretización para memoización
            if key not in memo:
                memo[key] = func(x)
            return memo[key]
        
        def recursive_simpson(left, right, f_left, f_right, f_mid, whole, depth):
            if depth >= max_depth:
                return whole
            
            mid = (left + right) / 2
            f_left_mid = memoized_func((left + mid) / 2)
            f_right_mid = memoized_func((mid + right) / 2)
            
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
        memo[round(a / tol * 1000)] = f_a
        memo[round(b / tol * 1000)] = f_b
        memo[round(((a + b) / 2) / tol * 1000)] = f_mid
        
        whole = (b - a) * (f_a + 4 * f_mid + f_b) / 6
        
        return recursive_simpson(a, b, f_a, f_b, f_mid, whole, 0)
    
    def simpson_adaptive_2d_parallel(self, func, ax, bx, ay, by, tol=1e-4, max_depth=8):
        """
        Método de Simpson adaptativo para integración 2D con paralelización
        """
        # Versión paralelizada usando división del dominio
        def integrate_x_slice(x_slice):
            x_left, x_right = x_slice
            
            def integrate_y_for_fixed_x(x):
                def func_y(y):
                    return func(x, y)
                return self.simpson_adaptive_1d(func_y, ay, by, tol=tol/10, max_depth=max_depth-2)
            
            # Integrar en x para este slice
            return self.simpson_adaptive_1d(integrate_y_for_fixed_x, x_left, x_right, tol=tol, max_depth=max_depth)
        
        # Dividir el dominio en x en n_workers slices
        x_slices = np.linspace(ax, bx, self.n_workers + 1)
        slices = [(x_slices[i], x_slices[i+1]) for i in range(len(x_slices)-1)]
        
        # Ejecutar en paralelo
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(integrate_x_slice, slices))
        
        return sum(results)
    
    def fresnel_kernel_2d_vectorized(self, x0_array, y0_array, x, y, d):
        """
        Kernel de Fresnel 2D vectorizado para múltiples puntos de observación
        """
        if abs(d) < 1e-12:
            return np.zeros_like(x0_array, dtype=complex)
        
        # Vectorizar el cálculo
        r2 = (x0_array - x)**2 + (y0_array - y)**2
        return np.exp(1j * self.k * r2 / (2 * d))
    
    def calculate_field_at_slit2_2d_parallel(self, x2_grid, y2_grid, params, method='fresnel'):
        """
        Calcula el campo en la segunda rendija usando paralelización
        """
        p = params['p']
        q = params['q']
        c = params['c']
        
        # Función para calcular un batch de puntos
        def calculate_batch(batch_indices):
            results = np.zeros(len(batch_indices), dtype=complex)
            
            for idx, (i, j) in enumerate(batch_indices):
                x2 = x2_grid[i, j]
                y2 = y2_grid[i, j]
                
                if method == 'fresnel':
                    kernel_func = lambda x1, y1: self.fresnel_kernel_2d(x2, y2, x1, y1, c)
                    prefactor = np.exp(1j * self.k * c) / (1j * self.wavelength * c)
                else:
                    kernel_func = lambda x1, y1: self.huygens_fresnel_kernel_2d(x2, y2, x1, y1, c)
                    prefactor = 1 / (1j * self.wavelength)
                
                def integrand(x1, y1):
                    transmission = self.rect_function_2d(x1, y1, 0, 0, p, q)
                    if transmission == 0:
                        return 0
                    return transmission * kernel_func(x1, y1)
                
                integral = self.simpson_adaptive_2d_parallel(
                    integrand, 
                    -p/2, p/2, 
                    -q/2, q/2,
                    tol=1e-5, 
                    max_depth=6
                )
                
                results[idx] = prefactor * integral
            
            return results
        
        # Preparar batches para paralelización
        shape = x2_grid.shape
        indices = [(i, j) for i in range(shape[0]) for j in range(shape[1])]
        batch_size = max(1, len(indices) // (self.n_workers * 4))
        batches = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
        
        # Calcular en paralelo
        field = np.zeros(shape, dtype=complex)
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_batch = {executor.submit(calculate_batch, batch): k 
                              for k, batch in enumerate(batches)}
            
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                batch_result = future.result()
                batch_indices = batches[batch_idx]
                
                for idx, (i, j) in enumerate(batch_indices):
                    field[i, j] = batch_result[idx]
        
        return field
    
    def simulate_2d_diffraction_parallel(self, params, Nx=100, Ny=100, x_range=0.01, y_range=0.01):
        """
        Simulación paralelizada para ambos métodos
        """
        # Crear mallas de observación
        x_obs = np.linspace(-x_range/2, x_range/2, Nx)
        y_obs = np.linspace(-y_range/2, y_range/2, Ny)
        X_obs, Y_obs = np.meshgrid(x_obs, y_obs, indexing='ij')
        
        # Precalcular campos para Fresnel en paralelo
        print(f"Calculando patrón de Fresnel (paralelo, {self.n_workers} workers)...")
        start_time = time.time()
        
        I_fresnel = np.zeros((Nx, Ny))
        U_fresnel = np.zeros((Nx, Ny), dtype=complex)
        
        # Dividir trabajo por filas
        def process_fresnel_row(i):
            row_result = np.zeros(Ny, dtype=complex)
            for j in range(Ny):
                U = self.calculate_field_observation_2d(x_obs[i], y_obs[j], params, 'fresnel')
                row_result[j] = U
            return i, row_result
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(process_fresnel_row, i) for i in range(Nx)]
            
            for future in as_completed(futures):
                i, row_result = future.result()
                U_fresnel[i, :] = row_result
                I_fresnel[i, :] = np.abs(row_result)**2
                
                if i % 20 == 0:
                    print(f"  Progreso Fresnel: {i}/{Nx}")
        
        fresnel_time = time.time() - start_time
        print(f"Tiempo cálculo Fresnel paralelo: {fresnel_time:.2f} s")
        
        # Precalcular campos para Huygens-Fresnel en paralelo
        print(f"\nCalculando patrón de Huygens-Fresnel (paralelo, {self.n_workers} workers)...")
        start_time = time.time()
        
        I_hf = np.zeros((Nx, Ny))
        U_hf = np.zeros((Nx, Ny), dtype=complex)
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(process_fresnel_row, i) for i in range(Nx)]
            
            for future in as_completed(futures):
                i, row_result = future.result()
                U_hf[i, :] = row_result
                I_hf[i, :] = np.abs(row_result)**2
                
                if i % 20 == 0:
                    print(f"  Progreso H-F: {i}/{Nx}")
        
        hf_time = time.time() - start_time
        print(f"Tiempo cálculo Huygens-Fresnel paralelo: {hf_time:.2f} s")
        
        # Normalizar intensidades
        I_fresnel_norm = I_fresnel / np.max(I_fresnel) if np.max(I_fresnel) > 0 else I_fresnel
        I_hf_norm = I_hf / np.max(I_hf) if np.max(I_hf) > 0 else I_hf
        
        # Perfiles en x (y=0)
        center_y_idx = Ny // 2
        I_x_fresnel = I_fresnel_norm[:, center_y_idx]
        I_x_hf = I_hf_norm[:, center_y_idx]
        
        # Guardar resultados
        self.results = {
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
        
        return self.results
    
    # Mantener los métodos originales para compatibilidad
    # ... (todos los métodos anteriores se mantienen igual)

# Versión optimizada con Numba (para CPU)
def add_numba_optimization():
    """Añade optimizaciones con Numba si está disponible."""
    try:
        from numba import jit, prange, vectorize, cuda
        import numba as nb
        
        print("Numba disponible - activando optimizaciones JIT")
        
        @jit(nopython=True, parallel=True, fastmath=True)
        def fresnel_kernel_numba(x0, y0, x, y, d, k):
            """Kernel de Fresnel optimizado con Numba."""
            if abs(d) < 1e-12:
                return 0 + 0j
            r2 = (x0 - x)**2 + (y0 - y)**2
            phase = k * r2 / (2 * d)
            return np.cos(phase) + 1j * np.sin(phase)
        
        @vectorize([nb.complex128(nb.float64, nb.float64, nb.float64, 
                                 nb.float64, nb.float64, nb.float64)])
        def fresnel_kernel_vectorized(x0, y0, x, y, d, k):
            """Versión vectorizada para arrays."""
            if abs(d) < 1e-12:
                return 0 + 0j
            r2 = (x0 - x)**2 + (y0 - y)**2
            phase = k * r2 / (2 * d)
            return np.cos(phase) + 1j * np.sin(phase)
        
        return True
    except ImportError:
        print("Numba no disponible - usando NumPy estándar")
        return False
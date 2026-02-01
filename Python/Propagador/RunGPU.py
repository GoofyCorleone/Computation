# test_m5_optimization.py
import time
import numpy as np
from DobleDifraccion_optimized import FresnelDiffractionSimulator, FresnelDiffractionFFT

def benchmark_m5():
    """Benchmark para MacBook Pro M5."""
    print("=" * 70)
    print("BENCHMARK PARA MacBook Pro M5")
    print("=" * 70)
    
    # Parámetros de prueba
    params = {
        'p': 0.2e-3,
        'q': 0.2e-3,
        'p2': 0.15e-3,
        'q2': 0.1e-3,
        'a': 0.5e-3,
        'b': 0.3e-3,
        'n': 20e-3,
        'c': 5e-3,
        'z0': 100e-3,
        'wavelength': 632.8e-9
    }
    
    # 1. Versión original (sin paralelizar)
    print("\n1. VERSIÓN ORIGINAL (sin optimizar):")
    simulator_orig = FresnelDiffractionSimulator(wavelength=632.8e-9, n_workers=1)
    
    start = time.time()
    results_orig = simulator_orig.simulate_2d_diffraction_complete(
        params, Nx=80, Ny=80, x_range=6e-3, y_range=6e-3
    )
    time_orig = time.time() - start
    print(f"Tiempo: {time_orig:.2f} s")
    
    # 2. Versión paralelizada
    print("\n2. VERSIÓN PARALELIZADA (CPU multi-core):")
    simulator_par = FresnelDiffractionSimulator(wavelength=632.8e-9, n_workers=8)
    
    start = time.time()
    results_par = simulator_par.simulate_2d_diffraction_parallel(
        params, Nx=80, Ny=80, x_range=6e-3, y_range=6e-3
    )
    time_par = time.time() - start
    print(f"Tiempo: {time_par:.2f} s")
    print(f"Speedup: {time_orig/time_par:.2f}x")
    
    # 3. Versión FFT (mucho más rápida)
    print("\n3. VERSIÓN FFT (aproximación rápida):")
    simulator_fft = FresnelDiffractionFFT(wavelength=632.8e-9)
    
    start = time.time()
    results_fft, fft_time = simulator_fft.compare_methods_fft_vs_integral(params, N=256)
    print(f"Tiempo: {fft_time:.2f} s")
    print(f"Speedup vs original: {time_orig/fft_time:.2f}x")
    print(f"Resolución: 256x256 (vs 80x80)")
    
    # 4. Recomendaciones para el M5
    print("\n" + "=" * 70)
    print("RECOMENDACIONES PARA MacBook Pro M5:")
    print("=" * 70)
    print("1. Usa la versión paralelizada para cálculos precisos")
    print("2. Usa FFT para exploraciones rápidas y optimizaciones")
    print("3. Ajusta n_workers a 8-10 para máximo rendimiento")
    print("4. Para la GPU, considera usar PyTorch con MPS (más complejo)")
    
    return {
        'original': time_orig,
        'parallel': time_par,
        'fft': fft_time,
        'speedup_parallel': time_orig/time_par,
        'speedup_fft': time_orig/fft_time
    }

def optimize_for_m5():
    """Configuración óptima para M5."""
    import multiprocessing as mp
    
    print("\n" + "=" * 70)
    print("CONFIGURACIÓN ÓPTIMA M5")
    print("=" * 70)
    
    # Información del sistema
    cpu_count = mp.cpu_count()
    print(f"Núcleos disponibles: {cpu_count}")
    print(f"Arquitectura: ARM Apple Silicon")
    
    # Configuración recomendada
    recommended_workers = min(10, cpu_count - 2)
    print(f"\nConfiguración recomendada:")
    print(f"- Workers paralelos: {recommended_workers}")
    print(f"- Resolución inicial: 80x80")
    print(f"- Usar FFT para optimizaciones: Sí")
    print(f"- Batch size: 4-8 por worker")
    
    # Crear simulador optimizado
    simulator = FresnelDiffractionSimulator(
        wavelength=632.8e-9,
        n_workers=recommended_workers,
        use_gpu=False  # Para MPS necesitarías PyTorch
    )
    
    return simulator

if __name__ == "__main__":
    # Ejecutar benchmark
    results = benchmark_m5()
    
    # Crear simulador optimizado
    simulator = optimize_for_m5()
    
    # Ejemplo de uso
    params = {
        'p': 0.2e-3,
        'q': 0.2e-3,
        'p2': 0.15e-3,
        'q2': 0.1e-3,
        'a': 0.5e-3,
        'b': 0.3e-3,
        'n': 20e-3,
        'c': 5e-3,
        'z0': 100e-3,
        'wavelength': 632.8e-9
    }
    
    print("\nEjecutando simulación optimizada...")
    results = simulator.run_simulation(
        params=params,
        show_patterns=True,
        show_comparison=True,
        save_plots=False,
        Nx=80,
        Ny=80,
        x_range=6e-3,
        y_range=6e-3
    )
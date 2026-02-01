# optimizacion_difraccion.py
from DobleDifraccion import FresnelDiffractionSimulator
from ParameterOptimizer import ParameterOptimizer
import numpy as np
import time

def main():
    simulator = FresnelDiffractionSimulator(wavelength=632.8e-9)
    optimizer = ParameterOptimizer(simulator)
    
    print("CONFIGURACIÓN DE OPTIMIZACIÓN")
    print("=" * 70)
    print("Objetivo: Maximizar diferencia entre métodos Fresnel y Huygens-Fresnel")
    print("Parámetros a optimizar: 9 (p, q, p2, q2, a, b, n, c, z0)")
    print("Método: Evolución diferencial (búsqueda global) + refinamiento local")
    print("=" * 70)
    
    print("\nFASE 1: BÚSQUEDA GLOBAL")
    print("-" * 50)
    
    start_time = time.time()
    best_params_global, best_diff_global = optimizer.optimize_with_differential_evolution(
        max_iter=30,      # Iteraciones máximas
        pop_size=12       # Tamaño de población
    )
    global_time = time.time() - start_time
    print(f"Tiempo búsqueda global: {global_time:.2f} segundos")
    print("\nFASE 2: REFINAMIENTO LOCAL")
    print("-" * 50)
    
    start_time = time.time()
    best_params_refined, best_diff_refined = optimizer.refine_optimization(
        initial_params=best_params_global,
        method='Nelder-Mead'
    )
    refine_time = time.time() - start_time
    print(f"Tiempo refinamiento: {refine_time:.2f} segundos")
    
    total_time = global_time + refine_time
    print(f"\nTiempo total de optimización: {total_time:.2f} segundos")
    
    optimizer.print_optimization_results(best_params_refined, best_diff_refined)
    optimizer.plot_optimization_history()
    
    print("\n" + "=" * 70)
    print("SIMULACIÓN COMPLETA CON PARÁMETROS OPTIMIZADOS")
    print("=" * 70)
    
    results = simulator.run_simulation(
        params=best_params_refined,
        show_patterns=True,
        show_comparison=True,
        save_plots=False,
        Nx=100,
        Ny=100,
        x_range=8e-3,
        y_range=8e-3
    )
    
    import json
    with open('parametros_optimizados.json', 'w') as f:
        json.dump(best_params_refined, f, indent=4, default=str)
    
    print("\nParámetros optimizados guardados en 'parametros_optimizados.json'")
    
    return best_params_refined, best_diff_refined

if __name__ == "__main__":
    best_params, best_diff = main()
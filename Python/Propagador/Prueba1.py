from DobleDifraccion import FresnelDiffractionSimulator
simulator = FresnelDiffractionSimulator(wavelength=632.8e-9)

params = {
    'p': 0.2e-3,      # Ancho de rendija en x (m)
    'q': 0.2e-3,      # Alto de rendija en y (m)
    'p2': 0.15e-3,    # Ancho de rendija en x (m)
    'q2': 0.1e-3,     # Ancho de rendija en y (m)
    'a': 0.5e-3,      # Desplazamiento en x de la segunda rendija (m)
    'b': 0.3e-3,      # Desplazamiento en y de la segunda rendija (m)
    'n': 20e-3,       # Distancia a la primera rendija (m)
    'c': 5e-3,        # Separación entre rendijas (m)
    'z0': 100e-3,     # Distancia al plano de observación (m)
    'wavelength': 632.8e-9  # Longitud de onda (m)
}

results = simulator.run_simulation(
    params=params,
    show_patterns=True,    # Mostrar gráfico de 4 paneles
    show_comparison=True,  # Mostrar gráfico comparativo
    save_plots=False,       # Guardar gráficos como PNG
    Nx=80, Ny=80,          # Resolución de la simulación (más rápida)
    x_range=6e-3,          # Rango en x (m)
    y_range=6e-3           # Rango en y (m)
)

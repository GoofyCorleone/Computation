import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema
L = 10  # Longitud del dominio
T = 10  # Duración de la simulación
Nx = 200  # Número de puntos en la malla espacial
Nt = 200  # Número de pasos de tiempo
c = 1  # Velocidad de la luz en el medio
epsilon = 1  # Permitividad del medio
mu = 1  # Permeabilidad del medio

dx = L / Nx  # Tamaño de paso espacial
dt = T / Nt  # Tamaño de paso temporal

# Asegurar estabilidad numérica (CFL condition)
if c * dt / dx > 1:
    raise ValueError("El esquema numérico es inestable. Ajuste los parámetros.")

# Condiciones iniciales (pulso gaussiano)
x = np.linspace(0, L, Nx)
E0 = np.exp(-(x - L / 4) ** 2)
H0 = np.zeros_like(x)

E = E0.copy()
H = H0.copy()

# Evolución temporal
E_history = [E0]
H_history = [H0]

for _ in range(Nt):
    # Actualizar campo eléctrico
    E[1:-1] += -dt * c**2 * mu / (2 * dx) * (H[2:] - H[:-2])
    
    # Condiciones de frontera absorbentes para el campo eléctrico
    E[0] = E[1] - (c * dt - dx) / (c * dt + dx) * (E[0] - E[1])
    E[-1] = E[-2] - (c * dt - dx) / (c * dt + dx) * (E[-1] - E[-2])
    
    # Actualizar campo magnético
    H[1:-1] += -dt / (2 * dx * epsilon) * (E[2:] - E[:-2])
    
    # Condiciones de frontera absorbentes para el campo magnético
    H[0] = H[1] - (c * dt - dx) / (c * dt + dx) * (H[0] - H[1])
    H[-1] = H[-2] - (c * dt - dx) / (c * dt + dx) * (H[-1] - H[-2])

    E_history.append(E.copy())
    H_history.append(H.copy())

# Gráfica de resultados
plt.figure()
plt.plot(x, E0, label='E0')
plt.plot(x, E, label='E final')
plt.xlabel('Posición (x)')
plt.ylabel('Campo eléctrico (E)')
plt.legend()
plt.show()

plt.figure()
plt.plot(x, H0, label='H0')
plt.plot(x, H, label='H final')
plt.xlabel('Posición (x)')
plt.ylabel('Campo magnético (H)')
plt.legend()
plt.show()

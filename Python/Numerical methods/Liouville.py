import numpy as np
import matplotlib.pyplot as plt

def solve_liouville_equation(a, b, N, T):
    h = (b - a) / N
    k = T / N

    x = np.linspace(a, b, N+1)
    t = np.linspace(0, T, N+1)

    # Matriz de solución
    u = np.zeros((N+1, N+1))

    # Condiciones iniciales
    u[:, 0] = np.sin(np.pi * x)*np.exp(-x)

    # Condiciones de frontera
    # u[0, :] = 0
    # u[N, :] = 0
    u[0,:] = u[N-1,:]
    u[N,:] = u[1,:]

    # Método de diferencias finitas
    for j in range(0, N):
        for i in range(1, N):
            u[i, j+1] = u[i, j] + k * (u[i+1, j] - 2 * u[i, j] + u[i-1, j]) / (h**2)

    return x, t, u

# Parámetros del problema
a = 0
b = 1
N = 100
T = 1

# Resolución numérica
x, t, u = solve_liouville_equation(a, b, N, T)

# Visualización de la solución
X, T = np.meshgrid(x, t)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, u, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
ax.set_title('Solución de la ecuación de Liouville')
plt.show()

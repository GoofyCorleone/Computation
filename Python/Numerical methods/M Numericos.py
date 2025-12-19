import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema
Lx, Ly = 6, 6  # Dimensiones del dominio en cm
Nx, Ny = 50, 50  # Número de puntos en la malla en x e y
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)  # Tamaño del paso en x e y
tol = 1e-4  # Tolerancia para el criterio de convergencia

# Condiciones de contorno
T_left = 50  # u(0, y) = 50°C
T_right = 50  # u(6, y) = 50°C
T_bottom = 20  # u(x, 0) = 20°C
T_top = 200  # u(x, 6) = 200°C

# Inicialización de la matriz de temperaturas
u = np.zeros((Ny, Nx))

# Aplicar las condiciones de contorno
u[:, 0] = T_left  # Borde izquierdo
u[:, -1] = T_right  # Borde derecho
u[0, :] = T_bottom  # Borde inferior
u[-1, :] = T_top  # Borde superior


# Método iterativo de Gauss-Seidel para resolver la ecuación de Laplace
def solve_laplace(u, tol):
    error = tol + 1  # Inicializar el error para entrar en el bucle
    while error > tol:
        u_old = u.copy()
        # Actualización de u utilizando el esquema de diferencias finitas
        for i in range(1, Ny - 1):
            for j in range(1, Nx - 1):
                u[i, j] = 0.25 * (u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1])

        # Calcular el error como la norma máxima de la diferencia entre iteraciones
        error = np.max(np.abs(u - u_old))
    return u


# Resolver la ecuación de Laplace
u = solve_laplace(u, tol)

# Graficar el resultado
X, Y = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, u, 50, cmap="hot")
plt.colorbar(contour, label="Temperatura (°C)")
plt.title("Solución de la Ecuación de Laplace en 2D")
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")
plt.show()

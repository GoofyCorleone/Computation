import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Crear los datos para el plot 3D
X, Y = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = np.sin(np.sqrt(X**2 + Y**2))

# Crear la figura y los ejes 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Función que actualiza el plot 3D en cada cuadro
def update(frame):
    # Hacer los cálculos para el cuadro actual
    Z_new = np.sin(np.sqrt(X**2 + Y**2 + frame/10))

    # Actualizar el plot 3D
    ax.clear()
    ax.plot_surface(X, Y, Z_new, cmap='viridis')

# Crear la animación
ani = FuncAnimation(fig, update, frames=range(100), interval=50)

# Mostrar la animación
plt.show()

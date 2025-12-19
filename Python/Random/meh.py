import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros de la trayectoria
a = 1.0  # amplitud
b = 0.5  # frecuencia
c = 2.0  # velocidad angular

# Crear la figura y el eje
fig, ax = plt.subplots()

# Crear el objeto de la línea
line, = ax.plot([], [], lw=2)

# Función de inicialización de la animación
def init():
    ax.set_xlim(-2*a, 2*a)
    ax.set_ylim(-2*a, 2*a)
    return line,

# Función de actualización de la animación
def update(frame):
    t = frame/100.0
    x = a*np.cos(c*t)
    y = a*np.sin(c*t)
    line.set_data([0, x], [0, y])
    return line,

# Crear la animación
ani = FuncAnimation(fig, update, frames=range(800), init_func=init, blit=True)

# Mostrar la animación
plt.show()

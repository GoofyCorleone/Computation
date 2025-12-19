import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

Z = np.arctan2(X, Y)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
contour = plt.contour(X, Y, Z, cmap='viridis')
plt.colorbar(contour, label='Valor de arctan(x/y)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gráfica de contorno de arctan(x/y)')
plt.grid(True)

ax = plt.subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Valor de arctan(x/y)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('arctan(x/y)')
ax.set_title('Gráfica de superficie de arctan(x/y)')
plt.tight_layout()
plt.show()


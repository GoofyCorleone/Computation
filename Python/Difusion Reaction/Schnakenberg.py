import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift

# ================================================
# Parámetros y configuraciones
# ================================================
L = 128                   # Tamaño del dominio (LxL)
T = 5000                  # Tiempo total
dt = 0.01                 # Paso temporal
N = 128                   # Puntos de la malla (NxN)
dx = L / N                # Resolución espacial

# Parámetros del modelo (Fig 2a: franjas)
a = 0.025
b = 1.55
d_y = 20.0
d_x_ratio = 0.5           # Probar 0.5, 0.93, 1.0, 1.07, 2.0
d_x = d_x_ratio * d_y

# Malla en el espacio físico
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# Malla en el espacio espectral (frecuencias)
k_x = 2 * np.pi * np.fft.fftfreq(N, d=dx)
k_y = 2 * np.pi * np.fft.fftfreq(N, d=dx)
Kx, Ky = np.meshgrid(k_x, k_y)
K_sq = Kx**2 + Ky**2      # k^2 para términos de difusión

# Operador de difusión anisotrópica para v
D_v = -d_x * Kx**2 - d_y * Ky**2

# Condiciones iniciales
u = np.ones((N, N)) * (a + b) + 0.01 * np.random.randn(N, N)
v = np.ones((N, N)) * (b / (a + b)**2) + 0.01 * np.random.randn(N, N)

# Transformadas iniciales
u_hat = fft2(u)
v_hat = fft2(v)

# ================================================
# Integración temporal pseudoespectral
# ================================================
for _ in range(T):
    # Paso 1: Calcular términos no lineales en espacio físico
    u = np.real(ifft2(u_hat))
    v = np.real(ifft2(v_hat))
    
    f = a + u**2 * v - u
    g = b - u**2 * v
    
    # Paso 2: Transformar términos no lineales al espacio espectral
    f_hat = fft2(f)
    g_hat = fft2(g)
    
    # Paso 3: Actualizar coeficientes espectrales (Euler explícito)
    u_hat = u_hat + dt * (-K_sq * u_hat + f_hat)
    v_hat = v_hat + dt * (D_v * v_hat + g_hat)

# ================================================
# Visualización
# ================================================
u_final = np.real(ifft2(u_hat))
plt.figure(figsize=(8, 6))
plt.imshow(u_final, cmap='viridis', extent=[0, L, 0, L])
plt.title(f'Patrón para $d_x/d_y = {d_x_ratio}$')
plt.colorbar()
plt.show()
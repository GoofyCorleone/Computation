import numpy as np
import matplotlib.pyplot as plt

# Parámetros del modelo
a = 0.025
b = 1.55
dy = 20.0
dx = 4 * dy

# Parámetros de simulación mejorados
Nx, Ny = 128, 128
Lx, Ly = 100.0, 100.0
T = 100.0
dt = 0.001  # Paso temporal reducido para estabilidad

# Estado estacionario con protección contra divisiones por cero
u_steady = (a + b)/2
v_steady = b/(u_steady**2 + 1e-12)  # Evita división por cero

# Inicialización con perturbación controlada
np.random.seed(42)
u = u_steady + 0.001*np.random.normal(size=(Nx, Ny))  # Perturbación más pequeña
v = v_steady + 0.001*np.random.normal(size=(Nx, Ny))

# Espaciamiento de malla
hx = Lx/(Nx-1)
hy = Ly/(Ny-1)

def laplacian_anisotropic(u, Dx, Dy, hx, hy):
    """Laplaciano anisotrópico con protección contra overflow"""
    u_xx = (np.roll(u, 1, axis=0) - 2*u + np.roll(u, -1, axis=0))/hx**2
    u_yy = (np.roll(u, 1, axis=1) - 2*u + np.roll(u, -1, axis=1))/hy**2
    return np.nan_to_num(Dx*u_xx + Dy*u_yy)  # Convierte NaN a cero

# Simulación con protección numérica
n_steps = int(T/dt)
for step in range(n_steps):
    # Términos de difusión con límites numéricos
    lap_u = laplacian_anisotropic(np.clip(u, -1e3, 1e3), 1.0, 1.0, hx, hy)
    lap_v = laplacian_anisotropic(np.clip(v, -1e3, 1e3), dx, dy, hx, hy)
    
    # Términos de reacción con protección
    u_safe = np.clip(u, -10, 10)  # Limita valores extremos
    v_safe = np.clip(v, -10, 10)
    f = a + u_safe**2*v_safe - u_safe
    g = b - u_safe**2*v_safe
    
    # Actualización con protección contra NaN
    u_new = u + dt*(lap_u + f)
    v_new = v + dt*(lap_v + g)
    
    # Detección de inestabilidades
    if np.any(np.isnan(u_new)) or np.any(np.isnan(v_new)):
        print(f"Simulación inestable en paso {step}")
        break
    
    u, v = u_new.copy(), v_new.copy()

# Visualización final
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(u.T, cmap='inferno', extent=[0, Lx, 0, Ly], vmin=0, vmax=3)  # Límites de color
plt.title('Concentración final de u')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(v.T, cmap='inferno', extent=[0, Lx, 0, Ly], vmin=0, vmax=3)
plt.title('Concentración final de v')
plt.colorbar()

plt.tight_layout()
plt.show()
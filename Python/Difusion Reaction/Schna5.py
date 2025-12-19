import numpy as np
import matplotlib.pyplot as plt

# Parámetros del modelo
a = 0.025
b = 1.55
dy = 20.0
dx = 2.0 * dy

# Parámetros de simulación
Nx, Ny = 128, 128
Lx, Ly = 200.0, 200.0
T = 500.0
dt = 0.005  # Paso temporal mayor que Euler pero menor que RK4

# Estado estacionario
u_steady = (a + b)/2
v_steady = b/(u_steady**2 + 1e-12)

# Inicialización con perturbación controlada
np.random.seed(42)
u = u_steady + 0.001*np.random.normal(size=(Nx, Ny))
v = v_steady + 0.001*np.random.normal(size=(Nx, Ny))

# Espaciamiento de malla
hx = Lx/(Nx-1)
hy = Ly/(Ny-1)

def laplacian_anisotropic(u, Dx, Dy, hx, hy):
    """Laplaciano anisotrópico con protección numérica"""
    u_xx = (np.roll(u, 1, axis=0) - 2*u + np.roll(u, -1, axis=0))/hx**2
    u_yy = (np.roll(u, 1, axis=1) - 2*u + np.roll(u, -1, axis=1))/hy**2
    return np.nan_to_num(Dx*u_xx + Dy*u_yy)

def compute_rhs(u, v):
    """Calcula los términos de difusión y reacción"""
    # Términos de difusión
    lap_u = laplacian_anisotropic(u, 1.0, 1.0, hx, hy)
    lap_v = laplacian_anisotropic(v, dx, dy, hx, hy)
    
    # Términos de reacción con protección
    u_safe = np.clip(u, 0, 10)
    v_safe = np.clip(v, 0, 10)
    f = a + u_safe**2*v_safe - u_safe
    g = b - u_safe**2*v_safe
    
    return lap_u + f, lap_v + g

# Simulación con método de Heun (2do orden)
n_steps = int(T/dt)
for step in range(n_steps):
    # Paso predictor (Euler)
    rhs_u1, rhs_v1 = compute_rhs(u, v)
    u_pred = u + dt * rhs_u1
    v_pred = v + dt * rhs_v1
    
    # Paso corrector (Promedio de términos)
    rhs_u2, rhs_v2 = compute_rhs(u_pred, v_pred)
    
    # Actualización de segundo orden
    u = u + 0.5*dt*(rhs_u1 + rhs_u2)
    v = v + 0.5*dt*(rhs_v1 + rhs_v2)
    
    # Control de estabilidad y positividad
    u = np.clip(u, 0, 10)
    v = np.clip(v, 0, 10)
    
    # Detección de errores
    if np.any(np.isnan(u)) or np.any(np.isnan(v)):
        print(f"Simulación inestable en paso {step}")
        break

# Visualización final
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(u.T, cmap='viridis', extent=[0, Lx, 0, Ly], vmin=0, vmax=3)
plt.title('Concentración final de u')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(v.T, cmap='viridis', extent=[0, Lx, 0, Ly], vmin=0, vmax=3)
plt.title('Concentración final de v')
plt.colorbar()

plt.tight_layout()
plt.show()
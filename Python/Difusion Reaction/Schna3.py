import numpy as np
import matplotlib.pyplot as plt

# Parámetros del modelo
a = 0.025
b = 1.55
dy = 20.0
dx = 2 * dy

# Parámetros de simulación
Nx, Ny = 128, 128
Lx, Ly = 100.0, 100.0
T = 100.0
dt = 0.001  # Paso temporal más grande que con Euler

# Estado estacionario protegido
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

def reaction_terms(u, v):
    """Términos de reacción con protección"""
    u_safe = np.clip(u, 0, 10)  # Solo valores positivos
    v_safe = np.clip(v, 0, 10)
    f = a + u_safe**2*v_safe - u_safe
    g = b - u_safe**2*v_safe
    return f, g

def system_rhs(u, v):
    """Función para RK4 que calcula el lado derecho de las EDOs"""
    lap_u = laplacian_anisotropic(u, 1.0, 1.0, hx, hy)
    lap_v = laplacian_anisotropic(v, dx, dy, hx, hy)
    f, g = reaction_terms(u, v)
    return lap_u + f, lap_v + g

# Simulación con RK4
n_steps = int(T/dt)
for step in range(n_steps):
    # Etapas RK4
    k1u, k1v = system_rhs(u, v)
    k2u, k2v = system_rhs(u + 0.5*dt*k1u, v + 0.5*dt*k1v)
    k3u, k3v = system_rhs(u + 0.5*dt*k2u, v + 0.5*dt*k2v)
    k4u, k4v = system_rhs(u + dt*k3u, v + dt*k3v)
    
    # Actualización RK4
    u = u + (dt/6.0)*(k1u + 2*k2u + 2*k3u + k4u)
    v = v + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
    
    # Control de estabilidad
    u = np.clip(u, 0, 10)  # Mantener concentraciones positivas
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
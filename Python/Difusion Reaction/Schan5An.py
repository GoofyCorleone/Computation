import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Parámetros del modelo
a = 0.025
b = 1.55
dy = 20.0
dx = 2.0 * dy

# Parámetros de simulación
Nx, Ny = 128, 128
Lx, Ly = 200.0, 200.0
T = 500.0
dt = 0.005

# Estado estacionario
u_steady = (a + b)/2
v_steady = b/(u_steady**2 + 1e-12)

# Inicialización con perturbación
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
    lap_u = laplacian_anisotropic(u, 1.0, 1.0, hx, hy)
    lap_v = laplacian_anisotropic(v, dx, dy, hx, hy)
    
    u_safe = np.clip(u, 0, 10)
    v_safe = np.clip(v, 0, 10)
    f = a + u_safe**2*v_safe - u_safe
    g = b - u_safe**2*v_safe
    
    return lap_u + f, lap_v + g

# Configuración de la animación
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
img1 = ax1.imshow(u.T, cmap='viridis', extent=[0, Lx, 0, Ly], vmin=0, vmax=3)
img2 = ax2.imshow(v.T, cmap='viridis', extent=[0, Lx, 0, Ly], vmin=0, vmax=3)

ax1.set_title('Concentración de u')
ax2.set_title('Concentración de v')
fig.colorbar(img1, ax=ax1)
fig.colorbar(img2, ax=ax2)
plt.tight_layout()

# Parámetros de animación
steps_per_frame = 100  # Pasos de simulación por frame
total_frames = int(T/(dt*steps_per_frame))  # Total de frames

def init():
    img1.set_data(u.T)
    img2.set_data(v.T)
    return [img1, img2]

def update(frame):
    global u, v
    
    for _ in range(steps_per_frame):
        # Paso predictor
        rhs_u1, rhs_v1 = compute_rhs(u, v)
        u_pred = u + dt * rhs_u1
        v_pred = v + dt * rhs_v1
        
        # Paso corrector
        rhs_u2, rhs_v2 = compute_rhs(u_pred, v_pred)
        u = u + 0.5*dt*(rhs_u1 + rhs_u2)
        v = v + 0.5*dt*(rhs_v1 + rhs_v2)
        
        # Control de estabilidad
        u = np.clip(u, 0, 10)
        v = np.clip(v, 0, 10)
        
        # Detener animación si hay NaN
        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            print("Simulación inestable. Deteniendo animación.")
            return [img1, img2]
    
    # Actualizar imágenes
    img1.set_data(u.T)
    img2.set_data(v.T)
    fig.suptitle(f'Tiempo de simulación: {frame*steps_per_frame*dt:.2f} s')
    
    return [img1, img2]

# Crear y guardar la animación
ani = FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True)

# Guardar como GIF (requiere Pillow)
print("Guardando animación como GIF...")
ani.save("reaccion_diffusion.gif", writer=PillowWriter(fps=20))

# Para video MP4 (requiere ffmpeg):
# ani.save("reaccion_diffusion.mp4", writer="ffmpeg", fps=20, bitrate=1800)

plt.show()
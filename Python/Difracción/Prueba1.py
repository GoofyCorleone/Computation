import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from diffractsim import MonochromaticField, CircularAperture, Lens

# Configuración de la simulación
wavelength = 632.8e-9  # Longitud de onda (luz roja He-Ne)
sim_size = 4e-3         # Tamaño de la simulación: 4 mm
N = 512                 # Resolución de la simulación

# Parámetros ópticos
focal_length = 0.2      # Distancia focal de la lente: 20 cm
lens_diameter = 0.01    # Diámetro de la lente: 1 cm
source_distance = 0.3   # Distancia fuente-lente: 30 cm
propagation_distances = np.linspace(0.1, 0.3, 30)  # Distancias a simular

# Crear campo óptico
field = MonochromaticField(
    wavelength=wavelength,
    extent_x=sim_size,
    extent_y=sim_size,
    Nx=N,
    Ny=N
)

# Fuente puntual (abertura circular muy pequeña)
pinhole = CircularAperture(radius=0.5e-6)
field.add(pinhole)

# Propagación hasta la lente
field.propagate(source_distance)

# Añadir lente convergente
lens = Lens(f=focal_length,radius=lens_diameter/2)
field.add(lens)

# Precalcular los patrones de difracción
print("Calculando patrones de difracción...")
frames = []
max_intensity = 0

for z in propagation_distances:
    # Crear copia del campo para no alterar el original
    field_copy = field
    
    # Propagación a distancia z después de la lente
    field_copy.propagate(z)
    
    # Obtener intensidad y actualizar máximo
    I = field_copy.get_intensity()
    frames.append(I)
    max_intensity = max(max_intensity, np.max(I))

# Configurar animación
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("Evolución del haz después de la lente convergente")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")

# Primer frame
im = ax.imshow(frames[0], cmap='hot', 
               extent=[-sim_size/2e-3, sim_size/2e-3, -sim_size/2e-3, sim_size/2e-3],
               vmin=0, vmax=max_intensity)
fig.colorbar(im, label='Intensidad')

# Función de actualización
def update(frame):
    im.set_data(frame)
    z = propagation_distances[np.where(frames == frame)[0][0]]
    ax.set_title(f"Distancia después de la lente: {z*100:.1f} cm")
    return [im]

# Crear animación
ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=frames,
    interval=100,
    blit=True
)

plt.tight_layout()
plt.show()

# Para guardar la animación (opcional)
# ani.save('evolucion_haz.mp4', writer='ffmpeg', fps=10, dpi=200)
print("¡Simulación completada!")
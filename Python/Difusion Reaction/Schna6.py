import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter  # Agregar PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import laplace

# Parámetros para el modelo de Gray-Scott
def gray_scott_2d(u, v, Du, Dv, f, k, dt):
    # Aplicar laplaciano a ambas variables
    laplacian_u = laplace(u)
    laplacian_v = laplace(v)

    # Términos de reacción
    reaction_u = -u * v**2 + f * (1 - u)
    reaction_v = u * v**2 - (f + k) * v

    # Actualizar variables
    u_new = u + dt * (Du * laplacian_u + reaction_u)
    v_new = v + dt * (Dv * laplacian_v + reaction_v)

    return u_new, v_new

# Configuración del espacio y tiempo
size = 100  # Tamaño de la rejilla
dt = 1.0  # Paso de tiempo

# Conjunto de parámetros para simular manchas similares a las de leopardo
Du = 0.16  # Coeficiente de difusión de u
Dv = 0.08  # Coeficiente de difusión de v
f = 0.055  # Tasa de alimentación
k = 0.062  # Tasa de muerte

# Inicialización de las matrices u y v con valores base
u = np.ones((size, size))
v = np.zeros((size, size))

# Creación de perturbaciones iniciales más visibles
r = 20
center = size // 2

# Perturbación central
for i in range(size):
    for j in range(size):
        if ((i - center)**2 + (j - center)**2) < r**2:
            u[i, j] = 0.5
            v[i, j] = 0.25

# Perturbaciones aleatorias para crear más puntos de inicio
np.random.seed(42)  # Para reproducibilidad
for _ in range(15):  # Aumentamos a 15 perturbaciones para ver más patrones
    x, y = np.random.randint(0, size, 2)
    radius = np.random.randint(2, 5)  # Radio variable
    for i in range(x-radius, x+radius):
        for j in range(y-radius, y+radius):
            if 0 <= i < size and 0 <= j < size:
                if ((i - x)**2 + (j - y)**2) < radius**2:
                    u[i, j] = 0.5
                    v[i, j] = 0.25

# Definir colormap para las manchas de animal
# Un mapa de colores más natural y contrastante
colors = [(0.98, 0.94, 0.85), (0.35, 0.25, 0.15)]  # Beige claro a marrón oscuro
animal_spots_cmap = LinearSegmentedColormap.from_list("animal_spots", colors)

# Función para crear la animación
def create_animation():
    # Crear la figura
    fig, ax = plt.subplots(figsize=(10, 8))

    # Configuración inicial de la imagen
    img = ax.imshow(u, cmap=animal_spots_cmap, vmin=0, vmax=1, animated=True)

    # Añadir barra de color y título
    cbar = fig.colorbar(img, ax=ax, label='Concentración (u)')
    title = ax.set_title('Evolución de Patrones de Piel Animal - Paso: 0', fontsize=14)

    # Variables para la simulación
    frames = 250  # Total de frames para la animación
    steps_per_frame = 8  # Pasos de simulación por cada frame

    # Variables locales para la simulación
    u_current = u.copy()
    v_current = v.copy()

    # Función de actualización para cada frame
    def update(frame):
        nonlocal u_current, v_current

        # Ejecutar varios pasos del modelo por cada frame
        for _ in range(steps_per_frame):
            u_current, v_current = gray_scott_2d(u_current, v_current, Du, Dv, f, k, dt)

        img.set_array(u_current)
        paso_actual = (frame + 1) * steps_per_frame
        title.set_text(f'Evolución de Patrones de Piel Animal - Paso: {paso_actual}')

        if frame % 10 == 0:
            print(f"Progreso: {frame}/{frames} frames")

        return [img, title]

    # Crear la animación
    ani = FuncAnimation(fig, update, frames=frames, interval=40, blit=True)
    
    # Guardar como GIF
    print("Guardando animación como GIF...")
    ani.save("animacion_patrones.gif", writer=PillowWriter(fps=25))  # FPS ajustados
    
    # Guardar como MP4 (requiere ffmpeg)
    # ani.save("animacion_patrones.mp4", writer="ffmpeg", fps=25)

    print("Animación creada. Mostrando evolución...")
    plt.show()

    return ani

# Ejecutar la simulación y mostrar la animación
print("Iniciando simulación de patrones de piel animal...")
animation = create_animation()
print("Simulación completada.")

# Función adicional para experimentar con diferentes patrones
def mostrar_otro_patron(tipo="rayas"):
    # Reinicializar matrices
    u_new = np.ones((size, size))
    v_new = np.zeros((size, size))

    # Crear perturbaciones iniciales
    for i in range(size):
        for j in range(size):
            if ((i - center)**2 + (j - center)**2) < r**2:
                u_new[i, j] = 0.5
                v_new[i, j] = 0.25

    # Añadir perturbaciones aleatorias
    for _ in range(15):
        x, y = np.random.randint(0, size, 2)
        radius = np.random.randint(2, 5)
        for i in range(x-radius, x+radius):
            for j in range(y-radius, y+radius):
                if 0 <= i < size and 0 <= j < size:
                    if ((i - x)**2 + (j - y)**2) < radius**2:
                        u_new[i, j] = 0.5
                        v_new[i, j] = 0.25

    # Establecer parámetros según el tipo de patrón
    if tipo == "rayas":
        Du_new = 0.16
        Dv_new = 0.08
        f_new = 0.026
        k_new = 0.059
        descripcion = "Rayas (similar a cebra)"
    elif tipo == "puntos_pequenos":
        Du_new = 0.16
        Dv_new = 0.08
        f_new = 0.082
        k_new = 0.059
        descripcion = "Puntos pequeños (similar a guepardo)"
    elif tipo == "moteado":
        Du_new = 0.16
        Dv_new = 0.08
        f_new = 0.035
        k_new = 0.065
        descripcion = "Patrón moteado (similar a jirafa)"
    else:
        # Parámetros por defecto (manchas de leopardo)
        Du_new = 0.16
        Dv_new = 0.08
        f_new = 0.055
        k_new = 0.062
        descripcion = "Manchas (similar a leopardo)"

    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(u_new, cmap=animal_spots_cmap, vmin=0, vmax=1, animated=True)
    cbar = fig.colorbar(img, ax=ax, label='Concentración (u)')
    title = ax.set_title(f'Evolución de {descripcion} - Paso: 0', fontsize=14)

    # Parámetros de animación
    frames = 250
    steps_per_frame = 8

    # Función de actualización
    def update(frame):
        nonlocal u_new, v_new

        for _ in range(steps_per_frame):
            u_new, v_new = gray_scott_2d(u_new, v_new, Du_new, Dv_new, f_new, k_new, dt)

        img.set_array(u_new)
        paso_actual = (frame + 1) * steps_per_frame
        title.set_text(f'Evolución de {descripcion} - Paso: {paso_actual}')

        return [img, title]

    # Crear animación
    ani = FuncAnimation(fig, update, frames=frames, interval=40, blit=True)
    plt.tight_layout()
    plt.show()

    return ani

# Para ver otros patrones, descomentar las siguientes líneas:
# rayas_animation = mostrar_otro_patron("rayas")
# puntos_animation = mostrar_otro_patron("puntos_pequenos")
# moteado_animation = mostrar_otro_patron("moteado")

# Configuración de la animación
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames, interval=30, blit=True)

# Guardar como GIF (requiere Pillow)
print("Guardando animación como GIF...")
anim.save('schrodinger_simulation.gif', writer='pillow', fps=15, dpi=100)

# Guardar como MP4 (requiere ffmpeg)
# print("Guardando animación como MP4...")
# anim.save('schrodinger_simulation.mp4', writer='ffmpeg', fps=15, bitrate=1000, 
#           extra_args=['-vcodec', 'libx264'])

pl.show()
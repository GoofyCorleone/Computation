import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import quad

# Parámetros de la simulación
L = 10.0          # Longitud del dominio
alpha = 0.1       # Coeficiente de difusión térmica
N_terms = 100     # Número de términos en la serie de Fourier
N_points = 1000   # Número de puntos espaciales
t_max = 20.0      # Tiempo máximo de simulación
fps = 20          # Cuadros por segundo para la animación

# Parámetros de la función de Weierstrass
a = 0.5           # Parámetro a (debe estar entre 0 y 1)
b = 7             # Parámetro b (debe ser un entero impar)
N_w = 50          # Número de términos para aproximar la función de Weierstrass

# Función de Weierstrass (condición inicial)
def weierstrass(x, a=a, b=b, N=N_w):
    """
    Función de Weierstrass: una función continua en todas partes 
    pero diferenciable en ninguna.
    """
    result = 0
    for n in range(N):
        result += a**n * np.cos(b**n * np.pi * x / L)
    return result

# Cálculo de coeficientes de Fourier
print("Calculando coeficientes de Fourier...")

# Coeficiente constante
A0, _ = quad(weierstrass, 0, L)
A0 /= L

# Arrays para almacenar coeficientes
A_n = np.zeros(N_terms)
B_n = np.zeros(N_terms)

# Calcular coeficientes para cada modo
for n in range(1, N_terms + 1):
    k_n = 2 * np.pi * n / L
    
    # Función integrando para coeficiente A_n
    def integrand_cos(x):
        return weierstrass(x) * np.cos(k_n * x)
    
    # Función integrando para coeficiente B_n
    def integrand_sin(x):
        return weierstrass(x) * np.sin(k_n * x)
    
    A_n[n-1], _ = quad(integrand_cos, 0, L)
    B_n[n-1], _ = quad(integrand_sin, 0, L)
    
    A_n[n-1] *= 2 / L
    B_n[n-1] *= 2 / L

print("Coeficientes calculados. Preparando simulación...")

# Función que calcula la solución en un tiempo t
def solution(x, t):
    result = A0 * np.ones_like(x)  # Término constante
    
    for n in range(1, N_terms + 1):
        k_n = 2 * np.pi * n / L
        decay = np.exp(-alpha * k_n**2 * t)
        result += (A_n[n-1] * np.cos(k_n * x) + B_n[n-1] * np.sin(k_n * x)) * decay
    
    return result

# Configuración de la visualización
x = np.linspace(0, L, N_points)
t_values = np.linspace(0, t_max, int(t_max * fps))

# Calcular el rango y para la visualización
y_min = min(np.min(weierstrass(x)), np.min(solution(x, t_max)))
y_max = max(np.max(weierstrass(x)), np.max(solution(x, 0)))

# Crear figura y ejes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Primer subplot: evolución temporal
ax1.set_xlim(0, L)
ax1.set_ylim(y_min, y_max)
ax1.set_xlabel('Posición (x)')
ax1.set_ylabel('Temperatura (u)')
ax1.set_title('Evolución de la Ecuación de Calor con Función de Weierstrass')
ax1.grid(True)

# Línea para la solución
line, = ax1.plot(x, solution(x, 0), 'b-', linewidth=2)

# Texto para mostrar el tiempo
time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

# Segundo subplot: condición inicial (función de Weierstrass)
ax2.plot(x, weierstrass(x), 'r-', linewidth=2)
ax2.set_xlim(0, L)
ax2.set_ylim(y_min, y_max)
ax2.set_xlabel('Posición (x)')
ax2.set_ylabel('f(x)')
ax2.set_title('Condición Inicial: Función de Weierstrass')
ax2.grid(True)

# Función de inicialización de la animación
def init():
    line.set_ydata(solution(x, 0))
    time_text.set_text('Tiempo = 0.00 s')
    return line, time_text

# Función de animación
def animate(i):
    t = t_values[i]
    u = solution(x, t)
    line.set_ydata(u)
    time_text.set_text(f'Tiempo = {t:.2f} s')
    return line, time_text

# Crear la animación
ani = FuncAnimation(fig, animate, frames=len(t_values),
                    init_func=init, blit=True, interval=1000/fps)

plt.tight_layout()
plt.show()

# Para guardar la animación (descomenta la siguiente línea)
# ani.save('ecuacion_calor_weierstrass.gif', writer='pillow', fps=fps)
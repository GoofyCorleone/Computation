import numpy as np
import plotly.graph_objects as go
# Definir constantes
mu = 2.6
alpha = 3
kappa = 300
gamma_p = 30
gamma_alpha = 0.5
gamma = 1
gamma_s = 0.5

# Definir condiciones iniciales
E_x0 = 1.5 + 0.0j  # Valor inicial de E_x
E_y0 = 1 + 3.5j  # Valor inicial de E_y
N0 = 10  # Valor inicial de N
n0 = 0.7 # Valor inicial de n

# Paso de tiempo
h = 0.0005
t_final = 10
num_steps = int(t_final / h)


# Definir las ecuaciones diferenciales
def system(t, state):
    E_x, E_y, N, n = state

    dE_x_dt = kappa * (1 + 1j * alpha) * ((N - 1) * E_x + 1j * n * E_y) - (gamma_alpha + 1j * gamma_p) * E_x
    dE_y_dt = kappa * (1 + 1j * alpha) * ((N - 1) * E_y + 1j * n * E_x) - (gamma_alpha + 1j * gamma_p) * E_y
    dN_dt = -gamma * (N * (1 + abs(E_x) ** 2 + abs(E_y) ** 2) - mu + 1j * n * (E_y * np.conj(E_x) - E_x * np.conj(E_y)))
    dn_dt = -gamma_s * n - gamma * (
                n * (abs(E_x) ** 2 + abs(E_y) ** 2) + 1j * N * (E_y * np.conj(E_x) - E_x * np.conj(E_y)))

    return np.array([dE_x_dt, dE_y_dt, dN_dt, dn_dt])


# Método RK4
def rk4_step(t, state, h):
    k1 = h * system(t, state)
    k2 = h * system(t + 0.5 * h, state + 0.5 * k1)
    k3 = h * system(t + 0.5 * h, state + 0.5 * k2)
    k4 = h * system(t + h, state + k3)

    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6


# Inicializar variables
t = 0
state = np.array([E_x0, E_y0, N0, n0], dtype=complex)

# Almacenar resultados
results = np.zeros((num_steps, 5), dtype=complex)
results[0] = [t, *state]

# Iterar con RK4
for step in range(1, num_steps):
    state = rk4_step(t, state, h)
    t += h
    results[step] = [t, *state]

# Extraer resultados
times = results[:, 0]
E_x_values = results[:, 1]
E_y_values = results[:, 2]
N_values = results[:, 3]
n_values = results[:, 4]

# Ahora se pueden usar estos resultados para análisis o visualización
S0 =  E_x_values.conjugate()*E_x_values + E_y_values.conjugate()*E_y_values
S1 =  E_x_values.conjugate()*E_x_values - E_y_values.conjugate()*+E_y_values
S2 =  E_x_values.conjugate()*E_y_values + E_y_values.conjugate()*+E_x_values
S3 =  1j*(E_x_values.conjugate()*E_y_values - E_y_values.conjugate()*E_x_values)

s1 , s2 , s3 = S1/S0 , S2/S0 , S3/S0
s1 , s2 , s3 = np.real(s1) , np.real(s2) , np.real(s3)

# Generar puntos aleatorios sobre una esfera
num_puntos = len(s1)
print(num_puntos)
x,y,z = s1 , s2 , s3

# Crear la esfera
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

# Crear la figura
fig = go.Figure()

# Agregar la superficie de la esfera
fig.add_trace(go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, opacity=0.5, colorscale='Blues', showscale=False))

# Agregar los puntos sobre la esfera
fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', marker=dict(size=5, color='red'), name='Distribución de estados de polarización'))

# Agregar proyecciones en el plano xy (desplazadas en z)
fig.add_trace(go.Scatter3d(x=x, y=y, z=np.ones(num_puntos)*2, mode='lines', marker=dict(size=5, color='green'), name='Proyección en el plano S1-S2'))

# Agregar proyecciones en el plano xz (desplazadas en y)
fig.add_trace(go.Scatter3d(x=x, y=np.ones(num_puntos)*2, z=z, mode='lines', marker=dict(size=5, color='blue'), name='Proyección en el plano S1-S3'))

# Agregar proyecciones en el plano yz (desplazadas en x)
fig.add_trace(go.Scatter3d(x=np.ones(num_puntos)*2, y=y, z=z, mode='lines', marker=dict(size=5, color='purple'), name='Proyección en el plano S2-S3'))

# Ajustar la apariencia del gráfico
fig.update_layout(scene=dict(
    xaxis=dict(title='S1'),
    yaxis=dict(title='S2'),
    zaxis=dict(title='S3')
))

# Mostrar la figura
fig.show()

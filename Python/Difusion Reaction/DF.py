import numpy as np
import matplotlib.pyplot as plt

# Parámetros del modelo
a = 0.025
b = 1.55
u_s = a + b  # Estado estacionario de u
v_s = b / (u_s**2)  # Estado estacionario de v

# Derivadas parciales (modelo de Schnakenberg)
f_u = 2 * u_s * v_s - 1
f_v = u_s**2
g_u = -2 * u_s * v_s
g_v = -u_s**2

# Función para calcular lambda(k)
def growth_rate(k, d):
    A = (1 + d) * k**2 - (f_u + g_v)
    b_term = d * k**4 - k**2 * (g_v + d * f_u) + (f_u * g_v - f_v * g_u)
    discriminant = A**2 - 4 * b_term
    sigma = (-A + np.sqrt(discriminant)) / 2  # Tomamos la raíz con Re(sigma) > 0
    return sigma.real

def growth_rate_max(k,d):
    L1 = f_u + g_v + (d + 1)/ (d- 1)* (f_u - g_v)
    L2 = 4*d / (d-1) * np.sqrt(- f_v*g_u/ d)
    return 0.5 * (L1 - L2)

# Función para calcular k_s (frecuencia dominante)
def k_s(d):
    term1 = -(f_u - g_v) / (d - 1)
    term2 = ((d + 1) / (d - 1)) * np.sqrt(-(f_v * g_u) / d)
    return np.sqrt(term1 + term2)

# Función para calcular dλ/dd
def dlambda_dd(d):
    return ( -(f_u - g_v) + (d + 1) * np.sqrt(-(f_v * g_u)/d) ) / (2 * (d - 1)**2)

# ================================================
# Figura 1b: λ(k) vs k para diferentes d
# ================================================
k_values = np.linspace(0, 2, 100)  # Rango de k
d_values = [10, 30, 100, 300 , 1000]  # Valores de d

plt.figure(figsize=(8, 5))
for d in d_values:
    lambda_values = [growth_rate(k, d) for k in k_values]
    plt.plot(k_values, lambda_values, label=f'd = {d}')

plt.xlabel('Frecuencia $k$')
plt.ylabel('Tasa de crecimiento $\\lambda(k)$')
plt.title('Tasa de crecimiento vs frecuencia')
plt.legend()
plt.grid(True)
plt.show()

# ================================================
# Figura 1c: λ_max vs d
# ================================================
d_range = np.linspace(10, 1000, 200)  # Rango de d
lambda_max = [np.max([growth_rate_max(k, d) for k in k_values]) for d in d_range]

plt.figure(figsize=(8, 5))
plt.plot(d_range, lambda_max, 'r-')
plt.xlabel('Coeficiente de difusión $d$')
plt.ylabel('$\\lambda_{\\text{max}}$')
plt.title(' Máxima tasa de crecimiento vs $d$')
plt.grid(True)
plt.show()

# ================================================
# Figura 1d: k_s vs d
# ================================================
k_s_values = [k_s(d) for d in d_range]

plt.figure(figsize=(8, 5))
plt.plot(d_range, k_s_values, 'g-')
plt.xlabel('Coeficiente de difusión $d$')
plt.ylabel('Frecuencia dominante $k_s$')
plt.title('Frecuencia dominante vs $d$')
plt.grid(True)
plt.show()
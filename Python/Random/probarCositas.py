import numpy as np
import matplotlib.pyplot as plt

# Definir constantes
q = 10E-4 # Reemplaza con la carga de la partícula
m = 5E-3 # Reemplaza con la masa de la partícula

# Definir campos eléctricos y magnéticos
def E(t, r):
    # Reemplaza con la función que define el campo eléctrico
    return np.array([0,0,0])

def B(t, r):
    # Reemplaza con la función que define el campo magnético
    return np.array([0,100,0])

# Función vectorial f(t, X) para el sistema de EDOs de primer orden
def f(t, X):
    x, y, z, vx, vy, vz = X
    r = np.array([x, y, z])
    v = np.array([vx, vy, vz])

    a = (q/m) * (E(t, r) + np.cross(v, B(t, r)))
    return np.array([vx, vy, vz, a[0], a[1], a[2]])

# Método de Runge-Kutta de cuarto orden
def runge_kutta_4th_order(f, X0, t0, h, N):
    X = np.zeros((N + 1, 6))
    t = np.zeros(N + 1)
    
    X[0] = X0
    t[0] = t0

    for i in range(N):
        k1 = f(t[i], X[i])
        k2 = f(t[i] + h/2, X[i] + h*k1/2)
        k3 = f(t[i] + h/2, X[i] + h*k2/2)
        k4 = f(t[i] + h, X[i] + h*k3)

        X[i+1] = X[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        t[i+1] = t[i] + h

    return t, X

# Condiciones iniciales
X0 = np.array([0, 0, 0, 1, 0, 0])  # Reemplaza con las condiciones iniciales
t0 = 0
h = 0.01
N = 100000

t, X = runge_kutta_4th_order(f, X0, t0, h, N)

# X contiene la solución aproximada para x, y, z, vx, vy y vz en cada paso de tiempo t
fig , ax = plt.figure() , plt.axes(projection = '3d')
ax.plot3D(X[:,0] , X[:,1] , X[:,2])
ax.set_ylabel('y')
ax.set_xlabel('x')
plt.show()
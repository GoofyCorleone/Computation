import numpy as np
import matplotlib.pyplot as plt
from pde import CartesianGrid, ScalarField, FieldCollection, PDE, MemoryStorage

# ===== PARÁMETROS DEL MODELO =====
a = 0.025
b = 1.55
d_y = 20
d_x_values = [10, 20, 40]

# Definir u_steady (FALTA PREVIA)
u_steady = a + b  # <----- ¡CORRECCIÓN CLAVE!

# Configuración de simulación
L = 80
N = 64
T = 100

# Estado estacionario
denominator = (a + b)**2
v_steady = b / denominator if denominator != 0 else 0

def simulate_schnakenberg(d_x):
    grid = CartesianGrid([[0, L], [0, L]], [N, N], periodic=True)
    
    # Condiciones iniciales
    np.random.seed(42)
    u_data = np.clip(u_steady + 0.005 * np.random.uniform(-1, 1, (N, N)), 0, 3)
    v_data = np.clip(v_steady + 0.005 * np.random.uniform(-1, 1, (N, N)), 0, 3)
    
    u = ScalarField(grid, u_data, label="u")
    v = ScalarField(grid, v_data, label="v")
    fields = FieldCollection([u, v], labels=["u", "v"])
    
    # Ecuaciones
    eq = PDE(
        {
            "u": "0.5*laplace(u) + a - u + u**2 * v",
            "v": f"{d_x}*d2_dx2(v) + {d_y}*d2_dy2(v) + b - u**2 * v - 0.1*v"
        },
        consts={"a": a, "b": b},
    )
    
    # Simulación
    storage = MemoryStorage()
    eq.solve(
        fields,
        t_range=T,
        dt=0.001,
        solver="scipy",
        method="LSODA",
        tracker=storage.tracker(5)
    )  # <----- Paréntesis cerrado correctamente
    
    return storage[-1]

# Visualización
plt.figure(figsize=(15, 5))
for i, d_x in enumerate(d_x_values):
    try:
        solution = simulate_schnakenberg(d_x)
        u_final = solution["u"].data
        u_final = np.nan_to_num(u_final, nan=0, posinf=3, neginf=0)
        u_final = np.clip(u_final, 0, 3)
        
        plt.subplot(1, 3, i+1)
        plt.imshow(u_final.T, cmap="viridis", origin="lower", vmin=0, vmax=3, aspect="auto")
        plt.title(f"$d_x/d_y$ = {d_x/d_y:.1f}")
        plt.axis("off")
        
    except Exception as e:
        print(f"Error con d_x={d_x}: {str(e)}")
        plt.subplot(1, 3, i+1)
        plt.text(0.5, 0.5, "Error", ha="center")
        plt.axis("off")

plt.tight_layout()
plt.savefig("resultados.png", dpi=150)
plt.show()
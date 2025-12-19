import numpy as np
import matplotlib.pyplot as plt
from fenics import *

# Parámetros geométricos
W = 3.0
H = 2.0
gap = 0.3

x0 = gap
x1 = W - gap
y0 = gap
y1 = H - gap

# Definición de las dos aperturas en el costado izquierdo
lenSpecial = H / 5.0
centerSpecial1 = 0.65 * H
centerSpecial2 = 0.35 * H

yStart1 = centerSpecial1 - lenSpecial / 2.0
yEnd1 = centerSpecial1 + lenSpecial / 2.0

yStart2 = centerSpecial2 - lenSpecial / 2.0
yEnd2 = centerSpecial2 + lenSpecial / 2.0

# Definición de las dos salidas en el costado derecho
lenOut = H / 5.0
centerOut1 = 0.65 * H
centerOut2 = 0.35 * H

yStartOut1 = centerOut1 - lenOut / 2.0
yEndOut1 = centerOut1 + lenOut / 2.0

yStartOut2 = centerOut2 - lenOut / 2.0
yEndOut2 = centerOut2 + lenOut / 2.0

# Crear la malla usando mshr
from mshr import *

# Dominio exterior (tubería)
outer_domain = Rectangle(Point(0, 0), Point(W, H))

# Dominio interior (tablero)
inner_domain = Rectangle(Point(x0, y0), Point(x1, y1))

# Crear malla del dominio combinado
domain = outer_domain
resolution = 60

mesh = generate_mesh(domain, resolution)

# Espacios de elementos finitos
# P2-P1 Taylor-Hood para Navier-Stokes
V_element = VectorElement("P", mesh.ufl_cell(), 2)
Q_element = FiniteElement("P", mesh.ufl_cell(), 1)
W_element = MixedElement([V_element, Q_element])

W = FunctionSpace(mesh, W_element)
V = FunctionSpace(mesh, V_element)
Q = FunctionSpace(mesh, Q_element)
T_space = FunctionSpace(mesh, "P", 1)

# Parámetros
dt = 0.01
nSteps = 200
Re = 100.0
nPicard = 5
Uin = 1.0
Tin = -1.0
alpha = 10.0


# Funciones características para distinguir regiones
class InnerDomain(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > x0 and x[0] < x1 and x[1] > y0 and x[1] < y1


class FrameDomain(SubDomain):
    def inside(self, x, on_boundary):
        return not (x[0] > x0 and x[0] < x1 and x[1] > y0 and x[1] < y1)


# Marcar subdominios
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
subdomains.set_all(0)
inner_domain_marker = InnerDomain()
inner_domain_marker.mark(subdomains, 1)


# Definir fronteras
class LeftInlet1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0) and x[1] >= yStart1 and x[1] <= yEnd1


class LeftInlet2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0) and x[1] >= yStart2 and x[1] <= yEnd2


class RightOutlet1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], W) and x[1] >= yStartOut1 and x[1] <= yEndOut1


class RightOutlet2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], W) and x[1] >= yStartOut2 and x[1] <= yEndOut2


class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class InnerBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-3
        return (near(x[0], x0, tol) or near(x[0], x1, tol) or
                near(x[1], y0, tol) or near(x[1], y1, tol))


# Marcar fronteras
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)

walls = Walls()
walls.mark(boundaries, 1)

inlet1 = LeftInlet1()
inlet1.mark(boundaries, 10)

inlet2 = LeftInlet2()
inlet2.mark(boundaries, 11)

outlet1 = RightOutlet1()
outlet1.mark(boundaries, 20)

outlet2 = RightOutlet2()
outlet2.mark(boundaries, 21)

inner_boundary = InnerBoundary()
inner_boundary.mark(boundaries, 100)

# Condiciones de frontera para Navier-Stokes
u_inlet = Constant((Uin, 0.0))
u_wall = Constant((0.0, 0.0))

bc_inlet1 = DirichletBC(W.sub(0), u_inlet, boundaries, 10)
bc_inlet2 = DirichletBC(W.sub(0), u_inlet, boundaries, 11)
bc_walls = DirichletBC(W.sub(0), u_wall, boundaries, 1)

bcs_ns = [bc_inlet1, bc_inlet2, bc_walls]

# Condiciones de frontera para temperatura
T_inlet = Constant(Tin)
bc_T_inlet1 = DirichletBC(T_space, T_inlet, boundaries, 10)
bc_T_inlet2 = DirichletBC(T_space, T_inlet, boundaries, 11)
bcs_T = [bc_T_inlet1, bc_T_inlet2]

# Funciones para almacenar soluciones
w = Function(W)
u, p = split(w)
w_test = TestFunction(W)
v, q = split(w_test)

u_old = Function(V)
u_old.assign(Constant((0.0, 0.0)))

Tf = Function(T_space)
Tf_old = Function(T_space)
Tf_old.assign(Constant(0.0))

Ti = Function(T_space)
Ti_old = Function(T_space)
Ti_old.assign(Constant(1.0))

phi = TestFunction(T_space)

# Funciones características
chi_inner = Expression('(x[0] > x0 && x[0] < x1 && x[1] > y0 && x[1] < y1) ? 1.0 : 0.0',
                       x0=x0, x1=x1, y0=y0, y1=y1, degree=0)
chi_frame = Expression('(x[0] > x0 && x[0] < x1 && x[1] > y0 && x[1] < y1) ? 0.0 : 1.0',
                       x0=x0, x1=x1, y0=y0, y1=y1, degree=0)

# Medidas de integración
dx = Measure('dx', domain=mesh, subdomain_data=subdomains)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

print("Iniciando simulación...")
print(f"Pasos temporales: {nSteps}, dt: {dt}")

# Bucle temporal
for it in range(nSteps):

    # Resolver Navier-Stokes con iteración de Picard
    for k in range(nPicard):
        # Formulación débil de Navier-Stokes
        F_ns = (1.0 / dt) * inner(u - u_old, v) * dx \
               + (1.0 / Re) * inner(grad(u), grad(v)) * dx \
               + inner(grad(u) * u_old, v) * dx \
               - Tf_old * v[0] * dx \
               - p * div(v) * dx \
               - q * div(u) * dx

        solve(F_ns == 0, w, bcs_ns, solver_parameters={'linear_solver': 'mumps'})

        u_func, p_func = w.split(True)
        u_old.assign(u_func)

    # Resolver problema térmico
    # Para el dominio exterior (frame)
    F_T_frame = chi_frame * (1.0 / dt) * (Tf - Tf_old) * phi * dx \
                + chi_frame * inner(grad(Tf), grad(phi)) * dx \
                + chi_frame * inner(u_old, grad(Tf)) * phi * dx

    # Para el dominio interior
    F_T_inner = chi_inner * (1.0 / dt) * (Ti - Ti_old) * phi * dx \
                + chi_inner * 0.5 * inner(grad(Ti), grad(phi)) * dx

    # Acoplamiento en la interfaz
    F_T_coupling = alpha * (Tf - Ti) * phi * ds(100)

    # Sistema completo de temperatura
    F_T = F_T_frame + F_T_inner + F_T_coupling

    # Resolver por separado (simplificación)
    solve((1.0 / dt) * (Tf - Tf_old) * phi * dx +
          inner(grad(Tf), grad(phi)) * dx +
          inner(u_old, grad(Tf)) * phi * dx == 0,
          Tf, bcs_T, solver_parameters={'linear_solver': 'cg'})

    solve((1.0 / dt) * (Ti - Ti_old) * phi * dx +
          0.5 * inner(grad(Ti), grad(phi)) * dx == 0,
          Ti, solver_parameters={'linear_solver': 'cg'})

    # Actualizar soluciones antiguas
    Tf_old.assign(Tf)
    Ti_old.assign(Ti)

    if it % 20 == 0:
        print(f"Paso temporal {it}/{nSteps}")

print("Simulación completada!")

# Visualización final
plt.figure(figsize=(15, 5))

# Velocidad
plt.subplot(1, 3, 1)
u_plot, p_plot = w.split(True)
c = plot(u_plot, title='Campo de velocidad final')
plt.colorbar(c)

# Presión
plt.subplot(1, 3, 2)
c = plot(p_plot, title='Presión final')
plt.colorbar(c)

# Temperatura combinada
plt.subplot(1, 3, 3)
T_combined = project(chi_frame * Tf + chi_inner * Ti, T_space)
c = plot(T_combined, title='Temperatura combinada final')
plt.colorbar(c)

plt.tight_layout()
plt.savefig('resultados_simulacion.png', dpi=150)
plt.show()

print("Resultados guardados en 'resultados_simulacion.png'")
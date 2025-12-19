import numpy as np
import pymeshlab as ml
from fipy import CellVariable, TransientTerm, DiffusionTerm, ImplicitSourceTerm, explicitSourceTerm, Viewer
import fipy
from fipy.meshes import Gmsh3D
from fipy.meshes import Grid3D  # Importación explícita de Grid3D
import os

# 1. Cargar y preparar la malla 3D
print("Preparando la malla 3D...")
ms = ml.MeshSet()

# Verificar si el archivo existe
mesh_file = 'modelo_3d.obj'
if not os.path.exists(mesh_file):
    print(f"El archivo de malla '{mesh_file}' no existe. Creando una malla simple...")
    # Crear una esfera como malla simple
    ms.create_sphere(radius=10, subdiv=3)
    print("Malla de esfera creada correctamente")
    
    # Guardar la esfera creada como modelo_3d.obj para futuras ejecuciones
    ms.save_current_mesh(mesh_file)
    print(f"Malla guardada como {mesh_file}")
else:
    # Si el archivo existe, cargarlo
    ms.load_new_mesh(mesh_file)
    print(f"Malla cargada desde {mesh_file}")

# Procesar la malla para generar elementos 3D y corregir problemas comunes
ms.meshing_remove_duplicate_vertices()
ms.meshing_remove_duplicate_faces()
ms.meshing_repair_non_manifold_edges()
ms.meshing_repair_non_manifold_vertices()

# Generar una reconstrucción superficial para asegurar calidad
try:
    ms.generate_surface_reconstruction_screened_poisson(depth=8, fulldepth=5)
except:
    print("Advertencia: No se pudo realizar la reconstrucción de Poisson, continuando con la malla original")

# Limpiar la malla
ms.meshing_remove_folded_faces()
try:
    # En versiones más nuevas de PyMeshLab
    ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=ml.AbsoluteValue(10.0))
except (AttributeError, TypeError) as e:
    try:
        # Alternativa con otro nombre de filtro
        ms.meshing_remove_small_connected_components_diameter(mincomponentdiag=ml.AbsoluteValue(10.0))
    except:
        # Si todo falla, intentamos otros métodos
        print("No se pudo eliminar componentes pequeños, continuando sin esta operación")

try:
    ms.meshing_close_holes(maxholesize=30)
except:
    print("No se pudo cerrar agujeros, continuando sin esta operación")

# Guardar malla procesada en formato compatible con FiPy
processed_mesh_file = 'modelo_procesado.stl'  # Cambiamos a STL que es más estándar
try:
    ms.save_current_mesh(processed_mesh_file)
    print(f"Malla procesada guardada como {processed_mesh_file}")
except Exception as e:
    print(f"Error al guardar la malla: {e}")
    # Intentar con otro formato
    processed_mesh_file = 'modelo_procesado.obj'
    try:
        ms.save_current_mesh(processed_mesh_file)
        print(f"Malla procesada guardada como {processed_mesh_file}")
    except Exception as e:
        print(f"Error al guardar la malla: {e}")
        raise RuntimeError("No se pudo guardar la malla en ningún formato compatible")

# 2. Calcular curvaturas y direcciones principales
print("Calculando curvaturas y direcciones principales...")
try:
    # Nuevo nombre del método en versiones recientes
    ms.compute_curvature_principal_directions_per_vertex(method=0)  # 0 = Normal Cycles
except AttributeError:
    try:
        # Intentar con el nombre antiguo
        ms.compute_curvature_principal_directions(method=0)
    except:
        print("No se pudo calcular las curvaturas, se usarán valores aleatorios.")

# Obtener información de curvatura y direcciones principales
try:
    # Intentar obtener nombres de atributos (versión nueva)
    mesh_attributes = ms.current_mesh().vertex_scalar_attributes_names()
except AttributeError:
    try:
        # Intentar versión alternativa del nombre
        mesh_attributes = ms.current_mesh().vertex_scalar_attribute_names()
    except AttributeError:
        # Si no podemos obtener los nombres de atributos, usar valores predeterminados
        mesh_attributes = []

print(f"Atributos disponibles: {mesh_attributes}")

# Obtener curvaturas principales
kappa1 = None
kappa2 = None
t1 = None
t2 = None

try:
    # Función genérica para obtener atributos de vértice, maneja diferentes versiones de API
    def get_vertex_attribute(mesh, attribute_name):
        try:
            # Intentar con la versión nueva
            return mesh.vertex_scalar_attribute(attribute_name)
        except AttributeError:
            try:
                # Intentar con pluralización
                return mesh.vertex_scalar_attributes(attribute_name)
            except:
                return None
    
    # Intentar obtener curvaturas
    for curvature_name in ['Principal_Curvature_1', 'Curv1', 'k1']:
        kappa1 = get_vertex_attribute(ms.current_mesh(), curvature_name)
        if kappa1 is not None:
            print(f"Encontrado atributo de curvatura principal 1: {curvature_name}")
            break
    
    for curvature_name in ['Principal_Curvature_2', 'Curv2', 'k2']:
        kappa2 = get_vertex_attribute(ms.current_mesh(), curvature_name)
        if kappa2 is not None:
            print(f"Encontrado atributo de curvatura principal 2: {curvature_name}")
            break
    
    # Intentar obtener direcciones
    # Componentes X
    for dir_name in ['Principal_Direction_1_X', 'PD1X', 't1_x']:
        t1_x = get_vertex_attribute(ms.current_mesh(), dir_name)
        if t1_x is not None:
            print(f"Encontrado componente X de dirección principal 1: {dir_name}")
            break
    
    # Componentes Y
    for dir_name in ['Principal_Direction_1_Y', 'PD1Y', 't1_y']:
        t1_y = get_vertex_attribute(ms.current_mesh(), dir_name)
        if t1_y is not None:
            print(f"Encontrado componente Y de dirección principal 1: {dir_name}")
            break
    
    # Componentes Z
    for dir_name in ['Principal_Direction_1_Z', 'PD1Z', 't1_z']:
        t1_z = get_vertex_attribute(ms.current_mesh(), dir_name)
        if t1_z is not None:
            print(f"Encontrado componente Z de dirección principal 1: {dir_name}")
            break
    
    # Segunda dirección principal
    for dir_name in ['Principal_Direction_2_X', 'PD2X', 't2_x']:
        t2_x = get_vertex_attribute(ms.current_mesh(), dir_name)
        if t2_x is not None:
            print(f"Encontrado componente X de dirección principal 2: {dir_name}")
            break
    
    for dir_name in ['Principal_Direction_2_Y', 'PD2Y', 't2_y']:
        t2_y = get_vertex_attribute(ms.current_mesh(), dir_name)
        if t2_y is not None:
            print(f"Encontrado componente Y de dirección principal 2: {dir_name}")
            break
    
    for dir_name in ['Principal_Direction_2_Z', 'PD2Z', 't2_z']:
        t2_z = get_vertex_attribute(ms.current_mesh(), dir_name)
        if t2_z is not None:
            print(f"Encontrado componente Z de dirección principal 2: {dir_name}")
            break
    
    # Verificar si obtuvimos todo lo necesario y crear vectores
    if (t1_x is not None and t1_y is not None and t1_z is not None and 
        t2_x is not None and t2_y is not None and t2_z is not None):
        # Convertir componentes a vectores completos
        t1 = np.array([t1_x, t1_y, t1_z]).T
        t2 = np.array([t2_x, t2_y, t2_z]).T
    else:
        raise ValueError("No se pudieron obtener todas las componentes de las direcciones principales")
    
except Exception as e:
    print(f"Error al obtener curvaturas: {e}")
    print("Usando valores aleatorios para pruebas.")
    # Si falla, usar valores aleatorios para pruebas
    num_vertices = ms.current_mesh().vertex_number()
    kappa1 = np.random.uniform(-0.5, 0.5, num_vertices)
    kappa2 = np.random.uniform(-0.5, 0.5, num_vertices)
    
    # Generar vectores unitarios aleatorios ortogonales
    t1 = np.random.randn(num_vertices, 3)
    t1 = t1 / np.linalg.norm(t1, axis=1)[:, np.newaxis]
    
    # Generar segundo conjunto de vectores ortogonales al primero
    random_vectors = np.random.randn(num_vertices, 3)
    t2 = np.cross(t1, random_vectors)
    t2 = t2 / np.linalg.norm(t2, axis=1)[:, np.newaxis]

print(f"Curvaturas calculadas: {len(kappa1)} valores")

# 3. Cargar malla en FiPy
print("Cargando malla en FiPy...")
try:
    # Intentar cargar directamente
    mesh = None
    
    # Si se guardó en STL o OBJ, necesitamos convertirlo o usar una malla alternativa
    if processed_mesh_file.endswith('.stl') or processed_mesh_file.endswith('.obj'):
        print("Usando una malla simple para FiPy debido a limitaciones de formato")
        
        # Crear una malla cuboide simple como alternativa
        # Esto es temporal - en un caso real necesitarías convertir la malla a formato .msh
        nx, ny, nz = 50, 50, 20
        mesh = Grid3D(nx=nx, ny=ny, nz=nz, dx=1.0, dy=1.0, dz=1.0)
        print(f"Malla Grid3D creada: {mesh.numberOfCells} celdas")
    else:
        # Si tenemos un archivo .msh, usarlo directamente
        mesh = Gmsh3D(processed_mesh_file)
        print(f"Malla cargada desde archivo: {mesh.numberOfCells} celdas, {mesh.numberOfFaces} caras")
    
except Exception as e:
    print(f"Error al cargar la malla en FiPy: {e}")
    # Crear una malla simple como alternativa
    nx, ny, nz = 50, 50, 20
    mesh = Grid3D(nx=nx, ny=ny, nz=nz, dx=1.0, dy=1.0, dz=1.0)
    print(f"Usando malla alternativa Grid3D: {mesh.numberOfCells} celdas")

# 4. Mapear curvaturas y direcciones a la malla de FiPy (de vértices a celdas)
print("Mapeando datos de curvatura a la malla...")

# Corregir la verificación del tipo usando el nombre de la clase en lugar de la clase misma
# Cambiamos:
# if isinstance(mesh, fipy.meshes.Grid3D):
# Por:
if mesh.__class__.__name__ == 'Grid3D':
    print("Generando datos sintéticos de curvatura para la malla Grid3D...")
    # Crear curvaturas y direcciones principales sintéticas
    cell_kappa1 = np.random.uniform(-0.5, 0.5, mesh.numberOfCells)
    cell_kappa2 = np.random.uniform(-0.5, 0.5, mesh.numberOfCells)
    
    # Generar vectores unitarios aleatorios ortogonales
    cell_t1 = np.random.randn(mesh.numberOfCells, 3)
    cell_t1 = cell_t1 / np.linalg.norm(cell_t1, axis=1)[:, np.newaxis]
    
    # Generar segundo conjunto de vectores ortogonales al primero
    random_vectors = np.random.randn(mesh.numberOfCells, 3)
    cell_t2 = np.cross(cell_t1, random_vectors)
    cell_t2 = cell_t2 / np.linalg.norm(cell_t2, axis=1)[:, np.newaxis]
else:
    # Si tenemos una malla real, intentamos mapear los datos
    def map_vertex_to_cell_data(vertex_data, mesh):
        """Mapea datos de vértices a celdas usando el vértice más cercano a cada centro de celda"""
        cell_centers = mesh.cellCenters.value.T
        vertex_coords = ms.current_mesh().vertex_matrix()
        
        cell_data = np.zeros(mesh.numberOfCells)
        
        for i in range(mesh.numberOfCells):
            # Encontrar el vértice más cercano al centro de la celda
            distances = np.linalg.norm(vertex_coords - cell_centers[i], axis=1)
            nearest_vertex = np.argmin(distances)
            cell_data[i] = vertex_data[nearest_vertex]
        
        return cell_data

    def map_vertex_vectors_to_cell(vertex_vectors, mesh):
        """Mapea vectores de vértices a celdas"""
        cell_centers = mesh.cellCenters.value.T
        vertex_coords = ms.current_mesh().vertex_matrix()
        
        cell_vectors = np.zeros((mesh.numberOfCells, 3))
        
        for i in range(mesh.numberOfCells):
            distances = np.linalg.norm(vertex_coords - cell_centers[i], axis=1)
            nearest_vertex = np.argmin(distances)
            cell_vectors[i] = vertex_vectors[nearest_vertex]
        
        return cell_vectors

    # Mapear datos a celdas
    cell_kappa1 = map_vertex_to_cell_data(kappa1, mesh)
    cell_kappa2 = map_vertex_to_cell_data(kappa2, mesh)
    cell_t1 = map_vertex_vectors_to_cell(t1, mesh)
    cell_t2 = map_vertex_vectors_to_cell(t2, mesh)

# 5. Parámetros del modelo
print("Configurando parámetros del modelo...")
# Parámetros de la cinética de reacción (f y g)
a = 0.025
b = 1.55

# Parámetros de difusión
d0 = 20.0  # Difusión base
rd = 10.0  # Acoplamiento curvatura-difusión

# 6. Funciones de difusión dependiente de curvatura
def d_kappa(kappa):
    """Función de difusión dependiente de la curvatura"""
    return d0 * (0.5 + 1 / (1 + np.exp(-kappa * rd)))

# 7. Calcular los tensores de difusión anisotrópica para cada celda
print("Calculando tensores de difusión anisotrópica...")
D_v = np.zeros((mesh.numberOfCells, 3, 3))

for i in range(mesh.numberOfCells):
    # Calcular los componentes del tensor de difusión
    t1_vec = cell_t1[i].reshape(3, 1)  # Convertir a columna
    t2_vec = cell_t2[i].reshape(3, 1)  # Convertir a columna
    
    # Producto tensor (dyadic product) t⊗t = t·tᵀ
    t1_tensor = np.dot(t1_vec, t1_vec.T)
    t2_tensor = np.dot(t2_vec, t2_vec.T)
    
    # Tensor de difusión completo para v
    D_v[i] = d_kappa(cell_kappa1[i]) * t1_tensor + d_kappa(cell_kappa2[i]) * t2_tensor

# Simplificar a un solo tensor para toda la malla (promedio)
# Nota: Esta es una simplificación, idealmente cada celda debería tener su propio tensor
D_v_avg = np.mean(D_v, axis=0)
print(f"Tensor de difusión promedio:\n{D_v_avg}")

# 8. Crear variables y condiciones iniciales
print("Inicializando variables...")
u = CellVariable(name="u", mesh=mesh, hasOld=True)
v = CellVariable(name="v", mesh=mesh, hasOld=True)

# Condiciones iniciales con pequeñas perturbaciones aleatorias
u.value = (a + b)/2 + 0.01 * np.random.randn(mesh.numberOfCells)
v.value = b/(u.value**2 + 1e-6)

# 9. Definir las funciones f(u,v) y g(u,v) del sistema
def f(u, v):
    """Función de reacción f(u,v) para la primera ecuación"""
    return a + u**2 * v - u

def g(u, v):
    """Función de reacción g(u,v) para la segunda ecuación"""
    return b - u**2 * v

# 10. Ecuaciones diferenciales
print("Configurando ecuaciones diferenciales...")

# Ecuación para u: u̇ = ∇²ₛu + f(u, v)
eq_u = (TransientTerm(var=u) == 
        DiffusionTerm(coeff=1.0, var=u) + 
        explicitSourceTerm(f(u, v)))

# Ecuación para v: v̇ = ∇ₛ · [(d(κ₁)t₁ ⊗ t₁ + d(κ₂)t₂ ⊗ t₂) · ∇ₛv] + g(u, v)
# Usamos el tensor de difusión promedio como aproximación
eq_v = (TransientTerm(var=v) == 
        DiffusionTerm(coeff=D_v_avg, var=v) + 
        explicitSourceTerm(g(u, v)))

# 11. Configuración de la simulación
print("Iniciando simulación...")
time_step = 0.1
total_time = 50.0  # Tiempo total de simulación
steps = int(total_time / time_step)
save_interval = 10  # Guardar cada 10 pasos

# Parámetros para el solver
tolerance = 1e-6
sweeps = 3

# Intentar crear visor si está disponible
try:
    viewer = Viewer(vars=(u, v), title="Reacción-Difusión en Superficie 3D")
    has_viewer = True
except:
    print("No se pudo inicializar el visor. Continuando sin visualización.")
    has_viewer = False

# 12. Bucle de simulación
print(f"Ejecutando {steps} pasos de simulación...")
for step in range(steps):
    # Actualizar valores antiguos
    u.updateOld()
    v.updateOld()
    
    # Resolver ecuación para u
    residual_u = 1.0
    for sweep in range(sweeps):
        residual_u = eq_u.sweep(dt=time_step)
        if residual_u < tolerance:
            break
    
    # Actualizar término fuente explícito para la ecuación v
    source_term_v = g(u, v)
    
    # Resolver ecuación para v
    residual_v = 1.0
    for sweep in range(sweeps):
        residual_v = eq_v.sweep(dt=time_step)
        if residual_v < tolerance:
            break
    
    # Imprimir progreso
    current_time = (step + 1) * time_step
    if step % save_interval == 0:
        print(f"Paso {step+1}/{steps}, Tiempo: {current_time:.2f}, Residuales: u={residual_u:.6e}, v={residual_v:.6e}")
        
        # Visualizar si el visor está disponible
        if has_viewer:
            viewer.plot()

# 13. Visualización final
if has_viewer:
    viewer.plot(filename="resultado_final.png")
    print("Visualización final guardada como resultado_final.png")

print("Simulación completada.")

# 14. Guardar resultados
try:
    np.savetxt("resultado_u.txt", u.value)
    np.savetxt("resultado_v.txt", v.value)
    print("Resultados guardados como resultado_u.txt y resultado_v.txt")
except Exception as e:
    print(f"Error al guardar resultados: {e}")
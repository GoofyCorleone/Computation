import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
import time
from matplotlib import cm

class SchnakenbergModel:
    """
    Implementación del modelo de Schnakenberg con difusión dependiente de curvatura
    basado en el artículo "How the zebra got its stripes: Curvature-dependent diffusion
    orients Turing patterns on three-dimensional surfaces" por Michael F. Staddon.

    Las ecuaciones del modelo son:
    u̇ = ∇²u + a + u²v - u
    v̇ = d∇²v + b - u²v

    Donde el coeficiente de difusión tiene la forma:
    d(κ) = d₀(1/2 + 1/(1 + e^(-κrd)))
    """

    def __init__(self, a=0.025, b=1.55, d0=20, nx=200, ny=200, dt=0.1, dx=1.0, dy=1.0):
        # Parámetros de reacción
        self.a = a
        self.b = b
        self.d0 = d0  # Coeficiente de difusión base

        # Parámetros de la simulación
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.dx = dx
        self.dy = dy

        # Inicializar campos u y v con el estado estacionario
        self.u_steady = a + b
        self.v_steady = b / ((a + b)**2)

        self.u = np.ones((ny, nx)) * self.u_steady
        self.v = np.ones((ny, nx)) * self.v_steady

        # Añadir una pequeña perturbación aleatoria
        np.random.seed(42)
        self.u += 0.01 * (np.random.random((ny, nx)) - 0.5)
        self.v += 0.01 * (np.random.random((ny, nx)) - 0.5)

        # Inicializar matrices de curvatura
        self.kappa_x = np.zeros((ny, nx))
        self.kappa_y = np.zeros((ny, nx))

        # Inicializar matrices de difusión
        self.D_u = np.ones((ny, nx))  # Difusión constante para u
        self.D_v_x = np.ones((ny, nx)) * d0  # Difusión para v en dirección x
        self.D_v_y = np.ones((ny, nx)) * d0  # Difusión para v en dirección y

    def set_curvature(self, type='cylinder', radius=20):
        """
        Define la curvatura del dominio.

        Parámetros:
        - type: Tipo de curvatura ('cylinder', 'torus', etc.)
        - radius: Radio de curvatura
        """
        if type == 'cylinder':
            # En un cilindro, la curvatura en dirección x es 1/R y en y es 0
            self.kappa_x = np.ones((self.ny, self.nx)) * (1.0/radius)
            self.kappa_y = np.zeros((self.ny, self.nx))
        elif type == 'torus':
            # Simulación simplificada de un toro
            y_center = self.ny // 2
            x_center = self.nx // 2

            # Crear coordenadas
            y, x = np.ogrid[:self.ny, :self.nx]

            # Calcular distancia al centro
            r = np.sqrt((x - x_center)**2 + (y - y_center)**2)

            # Curvatura en dirección radial y azimutal
            self.kappa_x = 1.0 / (radius + r * np.cos(np.arctan2(y - y_center, x - x_center)))
            self.kappa_y = 1.0 / r
        elif type == 'saddle':
            # Punto de silla: curvatura positiva en una dirección, negativa en otra
            y, x = np.ogrid[:self.ny, :self.nx]
            y_center = self.ny // 2
            x_center = self.nx // 2

            # Curvatura varía con la posición
            self.kappa_x = (x - x_center) / (radius**2)
            self.kappa_y = -(y - y_center) / (radius**2)
        else:
            # Plano (curvatura cero)
            self.kappa_x = np.zeros((self.ny, self.nx))
            self.kappa_y = np.zeros((self.ny, self.nx))

    def update_diffusion_coefficients(self, rd):
        """
        Actualiza los coeficientes de difusión usando la curvatura.

        La función de difusión dependiente de curvatura es:
        d(κ) = d₀(1/2 + 1/(1 + e^(-κrd)))

        Parámetros:
        - rd: Fuerza de acoplamiento curvatura-difusión
        """
        # Implementar la ecuación (13) del artículo
        self.D_v_x = self.d0 * (0.5 + 1.0 / (1.0 + np.exp(-self.kappa_x * rd)))
        self.D_v_y = self.d0 * (0.5 + 1.0 / (1.0 + np.exp(-self.kappa_y * rd)))

    def laplacian(self, field, D_x, D_y):
        """
        Calcula el laplaciano anisotrópico: ∇·(D∇field)
        usando diferencias finitas centrales con condiciones de contorno periódicas.
        """
        # Calcular derivadas segundas con condiciones periódicas
        dx2 = self.dx * self.dx
        dy2 = self.dy * self.dy

        # Calcular derivadas en x
        d2u_dx2 = np.zeros_like(field)
        d2u_dx2[:, 1:-1] = (field[:, 2:] - 2*field[:, 1:-1] + field[:, :-2]) / dx2
        d2u_dx2[:, 0] = (field[:, 1] - 2*field[:, 0] + field[:, -1]) / dx2
        d2u_dx2[:, -1] = (field[:, 0] - 2*field[:, -1] + field[:, -2]) / dx2

        # Calcular derivadas en y
        d2u_dy2 = np.zeros_like(field)
        d2u_dy2[1:-1, :] = (field[2:, :] - 2*field[1:-1, :] + field[:-2, :]) / dy2
        d2u_dy2[0, :] = (field[1, :] - 2*field[0, :] + field[-1, :]) / dy2
        d2u_dy2[-1, :] = (field[0, :] - 2*field[-1, :] + field[-2, :]) / dy2

        # Laplaciano anisotrópico
        return D_x * d2u_dx2 + D_y * d2u_dy2

    def step(self):
        """
        Avanza la simulación un paso de tiempo usando el método de Euler.
        """
        # Calcular términos de reacción
        reaction_u = self.a + self.u**2 * self.v - self.u
        reaction_v = self.b - self.u**2 * self.v

        # Calcular términos de difusión
        diffusion_u = self.laplacian(self.u, self.D_u, self.D_u)
        diffusion_v = self.laplacian(self.v, self.D_v_x, self.D_v_y)

        # Actualizar u y v usando el método de Euler explícito
        self.u += self.dt * (diffusion_u + reaction_u)
        self.v += self.dt * (diffusion_v + reaction_v)

    def run_simulation(self, n_steps=5000, rd=0, plot_interval=500, save_images=False):
        """
        Ejecuta la simulación por n_steps pasos de tiempo.

        Parámetros:
        - n_steps: Número de pasos de tiempo
        - rd: Fuerza de acoplamiento curvatura-difusión
        - plot_interval: Intervalo para mostrar y guardar resultados
        - save_images: Si es True, guarda imágenes de la simulación
        """
        # Actualizar coeficientes de difusión
        self.update_diffusion_coefficients(rd)

        # Configurar visualización
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Mostrar campos iniciales
        im1 = ax1.imshow(self.u, cmap='viridis', vmin=0, vmax=2)
        im2 = ax2.imshow(self.v, cmap='plasma', vmin=0, vmax=1)

        ax1.set_title('Activador (u)')
        ax2.set_title('Inhibidor (v)')

        plt.colorbar(im1, ax=ax1)
        plt.colorbar(im2, ax=ax2)

        # Añadir información sobre parámetros
        info_text = f'a={self.a}, b={self.b}, d0={self.d0}, rd={rd}'
        fig.suptitle(info_text)

        # Lista para guardar imágenes si se solicita
        images = []

        # Bucle principal de simulación
        start_time = time.time()
        for step in range(1, n_steps + 1):
            # Avanzar un paso de tiempo
            self.step()

            # Actualizar visualización en intervalos
            if step % plot_interval == 0:
                # Actualizar campos
                im1.set_array(self.u)
                im2.set_array(self.v)

                # Calcular tiempo restante estimado
                elapsed = time.time() - start_time
                remaining = (elapsed / step) * (n_steps - step)

                # Actualizar título
                fig.suptitle(f'{info_text} - Paso {step}/{n_steps} - Tiempo restante: {remaining:.1f}s')

                # Mostrar actualización
                plt.pause(0.01)

                # Guardar imagen si se solicita
                if save_images:
                    plt.savefig(f'schnakenberg_rd{rd}_step{step}.png')
                    images.append([im1.get_array().copy(), im2.get_array().copy()])

                print(f'Paso {step}/{n_steps} completado')

        plt.show()
        return self.u, self.v, images

    def gradient_fraction(self):
        """
        Calcula la fracción de gradiente en direcciones x e y.

        G_x = ∫|∂u/∂x|² dA / ∫(|∂u/∂x|² + |∂u/∂y|²) dA
        G_y = ∫|∂u/∂y|² dA / ∫(|∂u/∂x|² + |∂u/∂y|²) dA
        """
        # Calcular gradientes con condiciones periódicas
        grad_x = np.zeros_like(self.u)
        grad_x[:, :-1] = (self.u[:, 1:] - self.u[:, :-1]) / self.dx
        grad_x[:, -1] = (self.u[:, 0] - self.u[:, -1]) / self.dx

        grad_y = np.zeros_like(self.u)
        grad_y[:-1, :] = (self.u[1:, :] - self.u[:-1, :]) / self.dy
        grad_y[-1, :] = (self.u[0, :] - self.u[-1, :]) / self.dy

        # Calcular fracciones
        grad_x_squared = np.sum(grad_x**2)
        grad_y_squared = np.sum(grad_y**2)
        total_gradient = grad_x_squared + grad_y_squared

        if total_gradient > 0:
            return grad_x_squared / total_gradient, grad_y_squared / total_gradient
        else:
            return 0.5, 0.5

# Función para simular patrones de rayas (Figura 3)
def simulate_stripes():
    """
    Simula patrones de rayas usando el modelo de Schnakenberg.
    Parámetros: a = 0.025, b = 1.55, d0 = 20
    """
    print("Simulando patrones de rayas...")
    model = SchnakenbergModel(a=0.025, b=1.55, d0=20, nx=200, ny=200)

    # Definir curvatura de tipo cilindro
    model.set_curvature(type='cylinder', radius=20)

    # Simular con diferentes acoplamientos curvatura-difusión
    rd_values = [-10, 0, 10]  # negativo, cero, positivo

    results = []
    gradient_fractions = []

    for rd in rd_values:
        print(f"Simulando con acoplamiento rd = {rd}")
        u, v, _ = model.run_simulation(n_steps=5000, rd=rd, plot_interval=500)
        results.append((u.copy(), v.copy()))

        # Calcular fracciones de gradiente
        G_x, G_y = model.gradient_fraction()
        gradient_fractions.append((G_x, G_y))
        print(f"Fracción de gradiente: G_x={G_x:.3f}, G_y={G_y:.3f}")

    # Visualizar resultados comparativos
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (rd, (u, v)) in enumerate(zip(rd_values, results)):
        axes[i].imshow(u, cmap='viridis')
        axes[i].set_title(f'rd = {rd}\nG_x={gradient_fractions[i][0]:.2f}, G_y={gradient_fractions[i][1]:.2f}')

    fig.suptitle('Efecto del acoplamiento curvatura-difusión en patrones de rayas')
    plt.tight_layout()
    plt.show()

# Función para simular patrones de puntos (Figura 4)
def simulate_spots():
    """
    Simula patrones de puntos usando el modelo de Schnakenberg.
    Parámetros: a = 0.025, b = 1.24, d0 = 20
    """
    print("Simulando patrones de puntos...")
    model = SchnakenbergModel(a=0.025, b=1.24, d0=20, nx=200, ny=200)

    # Definir curvatura de tipo cilindro
    model.set_curvature(type='cylinder', radius=20)

    # Simular con diferentes acoplamientos curvatura-difusión
    rd_values = [-10, -5, 0]  # fuerte, débil, ninguno

    results = []
    gradient_fractions = []

    for rd in rd_values:
        print(f"Simulando con acoplamiento rd = {rd}")
        u, v, _ = model.run_simulation(n_steps=5000, rd=rd, plot_interval=500)
        results.append((u.copy(), v.copy()))

        # Calcular fracciones de gradiente
        G_x, G_y = model.gradient_fraction()
        gradient_fractions.append((G_x, G_y))
        print(f"Fracción de gradiente: G_x={G_x:.3f}, G_y={G_y:.3f}")

    # Visualizar resultados comparativos
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (rd, (u, v)) in enumerate(zip(rd_values, results)):
        axes[i].imshow(u, cmap='viridis')
        axes[i].set_title(f'rd = {rd}\nG_x={gradient_fractions[i][0]:.2f}, G_y={gradient_fractions[i][1]:.2f}')

    fig.suptitle('Transición de puntos a rayas con acoplamiento curvatura-difusión')
    plt.tight_layout()
    plt.show()

# Función para probar difusión anisotrópica (Figura 2)
def test_anisotropic_diffusion():
    """
    Prueba el efecto de la difusión anisotrópica en la orientación de patrones.
    """
    print("Probando difusión anisotrópica...")

    # Ratios de difusión d_x/d_y para probar
    ratios = [0.5, 0.93, 1.0, 1.07, 2.0]

    results_stripes = []  # Para a=0.025, b=1.55 (rayas)
    results_spots = []    # Para a=0.025, b=1.24 (puntos)

    # Simulaciones para patrones de rayas
    for ratio in ratios:
        print(f"Ratio d_x/d_y = {ratio} (rayas)")
        model = SchnakenbergModel(a=0.025, b=1.55, d0=20, nx=100, ny=100)

        # Configurar difusión anisotrópica manualmente
        model.D_v_x = np.ones((model.ny, model.nx)) * (model.d0 * ratio)
        model.D_v_y = np.ones((model.ny, model.nx)) * model.d0

        # Ejecutar simulación
        for step in range(3000):
            model.step()

        results_stripes.append(model.u.copy())

    # Simulaciones para patrones de puntos
    for ratio in ratios:
        print(f"Ratio d_x/d_y = {ratio} (puntos)")
        model = SchnakenbergModel(a=0.025, b=1.24, d0=20, nx=100, ny=100)

        # Configurar difusión anisotrópica manualmente
        model.D_v_x = np.ones((model.ny, model.nx)) * (model.d0 * ratio)
        model.D_v_y = np.ones((model.ny, model.nx)) * model.d0

        # Ejecutar simulación
        for step in range(3000):
            model.step()

        results_spots.append(model.u.copy())

    # Visualizar resultados
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for i, ratio in enumerate(ratios):
        # Rayas
        axes[0, i].imshow(results_stripes[i], cmap='viridis')
        axes[0, i].set_title(f'd_x/d_y = {ratio} (rayas)')

        # Puntos
        axes[1, i].imshow(results_spots[i], cmap='viridis')
        axes[1, i].set_title(f'd_x/d_y = {ratio} (puntos)')

    plt.tight_layout()
    plt.show()

# Función principal
if __name__ == "__main__":
    # Elegir qué simulación ejecutar
    print("Seleccione la simulación a ejecutar:")
    print("1. Patrones de rayas (Figura 3)")
    print("2. Patrones de puntos y transición a rayas (Figura 4)")
    print("3. Difusión anisotrópica (Figura 2)")

    choice = input("Ingrese su elección (1/2/3): ")

    if choice == '1':
        simulate_stripes()
    elif choice == '2':
        simulate_spots()
    elif choice == '3':
        test_anisotropic_diffusion()
    else:
        print("Opción no válida. Ejecutando simulación de rayas por defecto.")
        simulate_stripes()
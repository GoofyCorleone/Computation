import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp, cumulative_trapezoid

class GeneralizedSagnacWorldLines:
    def __init__(self, R=1.0, c=1.0):
        """
        Inicializa los parámetros del interferómetro de Sagnac
        para aceleraciones no constantes
        
        Parámetros:
        R: radio del interferómetro
        c: velocidad de la luz (normalizada a 1)
        """
        self.R = R
        self.c = c
        
    def time_dependent_angular_velocity(self, t):
        """
        Define una velocidad angular que varía con el tiempo
        Puedes modificar esta función para diferentes perfiles de aceleración
        """
        # Ejemplo 1: Aceleración constante
        # return 0.1 * t
        
        # Ejemplo 2: Oscilación armónica
        # return 0.3 * (1 + 0.5 * np.sin(0.5 * t))
        
        # Ejemplo 3: Aceleración que satura
        return 0.4 * (1 - np.exp(-0.3 * t))
        
        # Ejemplo 4: Pulso de aceleración
        # return 0.3 * np.exp(-0.1 * (t-5)**2)
    
    def time_dependent_angular_acceleration(self, t):
        """
        Calcula la aceleración angular derivando numéricamente Ω(t)
        """
        dt = 1e-6
        omega_plus = self.time_dependent_angular_velocity(t + dt)
        omega_minus = self.time_dependent_angular_velocity(t - dt)
        return (omega_plus - omega_minus) / (2 * dt)
    
    def metric_components(self, t):
        """
        Componentes de la métrica en coordenadas rotantes con Ω(t)
        ds² = -(c² - Ω(t)²R²)dt² + 2Ω(t)R² dφ dt + R² dφ²
        """
        Omega_t = self.time_dependent_angular_velocity(t)
        
        g00 = -(self.c**2 - Omega_t**2 * self.R**2)
        g0phi = 2 * Omega_t * self.R**2
        gphiphi = self.R**2
        
        return g00, g0phi, gphiphi
    
    def compute_geodesic_equations(self, t, y, direction):
        """
        Ecuaciones geodésicas para la luz en el marco rotante no-inercial
        y = [φ, dφ/dt]
        """
        phi, dphidt = y
        
        Omega_t = self.time_dependent_angular_velocity(t)
        alpha_t = self.time_dependent_angular_acceleration(t)
        
        g00, g0phi, gphiphi = self.metric_components(t)
        
        # Para luz: ds² = 0
        # Resolvemos la ecuación cuadrática para dφ/dt
        a = gphiphi
        b = 2 * g0phi
        c = g00
        
        discriminant = b**2 - 4 * a * c
        
        if discriminant < 0:
            raise ValueError("Discriminante negativo - métrica no válida")
        
        dphidt1 = (-b + np.sqrt(discriminant)) / (2 * a)
        dphidt2 = (-b - np.sqrt(discriminant)) / (2 * a)
        
        if direction == 1:  # Co-rotación
            dphidt_sol = max(dphidt1, dphidt2)
        else:  # Contra-rotación
            dphidt_sol = min(dphidt1, dphidt2)
        
        return [dphidt_sol, 0]  # d²φ/dt² = 0 para métrica estacionaria aproximada
    
    def solve_worldline(self, t_span, initial_phi=0, direction=1):
        """
        Resuelve la línea de mundo para una dirección dada
        """
        def geodesic_eq(t, y):
            return self.compute_geodesic_equations(t, y, direction)
        
        y0 = [initial_phi, 0]
        
        sol = solve_ivp(geodesic_eq, t_span, y0, t_eval=np.linspace(t_span[0], t_span[1], 1000),
                       method='RK45', rtol=1e-8)
        
        phi_values = sol.y[0]
        t_values = sol.t
        
        x_values = self.R * np.cos(phi_values)
        y_values = self.R * np.sin(phi_values)
        z_values = t_values
        
        return x_values, y_values, z_values, phi_values, t_values
    
    def compute_instantaneous_proper_time_difference(self, t):
        """
        Calcula la diferencia de tiempo propio instantánea usando la fórmula
        diferencial: d(Δτ)/dt ≈ (4A dΩ/dt)/c² + (4Ω dA/dt)/c²
        Para área constante: d(Δτ)/dt ≈ (4A α(t))/c²
        """
        A = np.pi * self.R**2
        alpha_t = self.time_dependent_angular_acceleration(t)
        return (4 * A * alpha_t) / self.c**2
    
    def compute_accumulated_phase_difference(self, t_values):
        """
        Calcula la diferencia de fase acumulada integrando en el tiempo
        """
        omega_values = [self.time_dependent_angular_velocity(t) for t in t_values]
        A = np.pi * self.R**2
        
        # ΔΦ ≈ (8πA/(λc)) * ∫Ω(t) dt
        # Para simplificar, tomamos λ=1
        integral_omega = cumulative_trapezoid(omega_values, t_values, initial=0)
        phase_diff = (8 * np.pi * A / self.c) * integral_omega
        
        return phase_diff

def create_generalized_animation():
    
    sagnac = GeneralizedSagnacWorldLines(R=1.0, c=1.0)
    t_max = 15
    t_span = (0, t_max)
    
    print("Resolviendo geodésicas para haz co-rotante...")
    x_co, y_co, z_co, phi_co, t_co = sagnac.solve_worldline(t_span, direction=1)
    
    print("Resolviendo geodésicas para haz contra-rotante...")
    x_counter, y_counter, z_counter, phi_counter, t_counter = sagnac.solve_worldline(t_span, direction=-1)
    
    fig = plt.figure(figsize=(18, 12))
    ax1 = fig.add_subplot(231, projection='3d')
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    limit = sagnac.R * 1.2
    
    line_co_3d, = ax1.plot(x_co, y_co, z_co, 'r-', linewidth=2, label='Haz co-rotante')
    line_counter_3d, = ax1.plot(x_counter, y_counter, z_counter, 'b-', linewidth=2, label='Haz contra-rotante')
    
    line_co_2d, = ax2.plot(x_co, y_co, 'r-', linewidth=2, alpha=0.7)
    line_counter_2d, = ax2.plot(x_counter, y_counter, 'b-', linewidth=2, alpha=0.7)
    
    point_co_3d, = ax1.plot([], [], [], 'ro', markersize=8)
    point_counter_3d, = ax1.plot([], [], [], 'bo', markersize=8)
    
    point_co_2d, = ax2.plot([], [], 'ro', markersize=8)
    point_counter_2d, = ax2.plot([], [], 'bo', markersize=8)
    
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = sagnac.R * np.cos(theta)
    y_circle = sagnac.R * np.sin(theta)
    ax2.plot(x_circle, y_circle, 'k--', alpha=0.3, label='Interferómetro')
    
    t_plot = np.linspace(0, t_max, 300)
    omega_plot = [sagnac.time_dependent_angular_velocity(t) for t in t_plot]
    alpha_plot = [sagnac.time_dependent_angular_acceleration(t) for t in t_plot]
    
    ax3.plot(t_plot, omega_plot, 'g-', linewidth=2)
    ax3.set_xlabel('Tiempo')
    ax3.set_ylabel('Ω(t)')
    ax3.set_title('Velocidad Angular vs Tiempo')
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(t_plot, alpha_plot, 'm-', linewidth=2)
    ax4.set_xlabel('Tiempo')
    ax4.set_ylabel('α(t)')
    ax4.set_title('Aceleración Angular vs Tiempo')
    ax4.grid(True, alpha=0.3)
    
    phase_diff = sagnac.compute_accumulated_phase_difference(t_plot)
    ax5.plot(t_plot, phase_diff, 'c-', linewidth=2)
    ax5.set_xlabel('Tiempo')
    ax5.set_ylabel('ΔΦ(t)')
    ax5.set_title('Diferencia de Fase Acumulada')
    ax5.grid(True, alpha=0.3)

    delta_tau_inst = [sagnac.compute_instantaneous_proper_time_difference(t) for t in t_plot]
    ax6.plot(t_plot, delta_tau_inst, 'y-', linewidth=2)
    ax6.set_xlabel('Tiempo')
    ax6.set_ylabel('d(Δτ)/dt')
    ax6.set_title('Tasa de Cambio de Diferencia\nde Tiempo Propio')
    ax6.grid(True, alpha=0.3)
    
    def init():
        ax1.set_xlim(-limit, limit)
        ax1.set_ylim(-limit, limit)
        ax1.set_zlim(0, t_max)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Tiempo Coordenado')
        ax1.set_title('Líneas de Mundo - Aceleración No Constante\n(Vista 3D)')
        ax1.legend()
        
        ax2.set_xlim(-limit, limit)
        ax2.set_ylim(-limit, limit)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Trayectorias Espaciales\n(Vista desde arriba)')
        ax2.legend()
        ax2.set_aspect('equal')
        
        return (line_co_3d, line_counter_3d, point_co_3d, point_counter_3d,
                line_co_2d, line_counter_2d, point_co_2d, point_counter_2d)
    
    def animate(i):
        
        t_current = i * t_max / len(t_co)
        
        idx_co = np.argmin(np.abs(t_co - t_current))
        idx_counter = np.argmin(np.abs(t_counter - t_current))
        
        point_co_3d.set_data([x_co[idx_co]], [y_co[idx_co]])
        point_co_3d.set_3d_properties([z_co[idx_co]])
        
        point_counter_3d.set_data([x_counter[idx_counter]], [y_counter[idx_counter]])
        point_counter_3d.set_3d_properties([z_counter[idx_counter]])
        
        point_co_2d.set_data([x_co[idx_co]], [y_co[idx_co]])
        point_counter_2d.set_data([x_counter[idx_counter]], [y_counter[idx_counter]])
        
        return (line_co_3d, line_counter_3d, point_co_3d, point_counter_3d,
                line_co_2d, line_counter_2d, point_co_2d, point_counter_2d)
    
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(t_co), interval=30, blit=True)
    
    plt.tight_layout()
    return anim, sagnac

def create_comparison_plot():
    """Crear un gráfico comparando diferentes perfiles de aceleración"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    profiles = [
        lambda t: 0.1 * t,                           
        lambda t: 0.3 * (1 + 0.5 * np.sin(0.5 * t)),
        lambda t: 0.4 * (1 - np.exp(-0.3 * t)),      
        lambda t: 0.3 * np.exp(-0.1 * (t-5)**2)     
    ]
    
    profile_names = [
        "Aceleración Constante",
        "Oscilación Armónica", 
        "Saturación Exponencial",
        "Pulso Gaussiano"
    ]
    
    t_plot = np.linspace(0, 15, 300)
    
    for i, (profile, name) in enumerate(zip(profiles, profile_names)):
        ax = axes[i//2, i%2]
        
        sagnac = GeneralizedSagnacWorldLines(R=1.0, c=1.0)
        sagnac.time_dependent_angular_velocity = profile
        
        omega_vals = [sagnac.time_dependent_angular_velocity(t) for t in t_plot]
        alpha_vals = [sagnac.time_dependent_angular_acceleration(t) for t in t_plot]
        phase_diff = sagnac.compute_accumulated_phase_difference(t_plot)
        
        ax.plot(t_plot, omega_vals, 'g-', label='Ω(t)')
        ax.plot(t_plot, alpha_vals, 'm-', label='α(t)')
        ax.plot(t_plot, phase_diff, 'c-', label='ΔΦ(t)')
        
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Magnitud')
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("Generando animación del efecto Sagnac con aceleración no constante...")
    print("Basado en el formalismo de Eric Gourgoulhon para marcos no-inerciales generalizados")
    
    anim, sagnac_system = create_generalized_animation()
    comparison_fig = create_comparison_plot()
    
    plt.show()
    
    # Para guardar la animación (descomentar si se desea)
    # print("Guardando animación...")
    # anim.save('sagnac_generalized.mp4', writer='ffmpeg', fps=30)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class SagnacWorldLines:
    def __init__(self, R=1.0, Omega=0.5, c=1.0):
        """
        Inicializa los parámetros del interferómetro de Sagnac
        
        Parámetros:
        R: radio del interferómetro
        Omega: velocidad angular de rotación
        c: velocidad de la luz (normalizada a 1)
        """
        self.R = R
        self.Omega = Omega
        self.c = c
        self.T = 2 * np.pi * R / c
    
        # ds² = -(c² - Ω²R²)dt² + 2ΩR² dφ dt + R² dφ²
        self.g00 = -(c**2 - Omega**2 * R**2)
        self.g0phi = 2 * Omega * R**2
        self.gphiphi = R**2
    
    def compute_proper_time_difference(self):
        """Calcula la diferencia de tiempo propio entre los dos haces"""
        # Δτ ≈ (4AΩ)/c² donde A = πR²
        A = np.pi * self.R**2
        delta_tau = (4 * A * self.Omega) / self.c**2
        return delta_tau
    
    def worldline_co_rotating(self, t_coord):
        """
        Línea de mundo para el haz que co-rota (en dirección de la rotación)
        En el marco rotante
        """
        omega_eff_co = self.c/self.R - self.Omega
        
        phi_co = omega_eff_co * t_coord
        x_co = self.R * np.cos(phi_co)
        y_co = self.R * np.sin(phi_co)
        z_co = t_coord  
        
        return x_co, y_co, z_co, phi_co
    
    def worldline_counter_rotating(self, t_coord):
        """
        Línea de mundo para el haz que contra-rota (en dirección opuesta a la rotación)
        En el marco rotante
        """
        omega_eff_counter = -self.c/self.R - self.Omega
        
        phi_counter = omega_eff_counter * t_coord
        x_counter = self.R * np.cos(phi_counter)
        y_counter = self.R * np.sin(phi_counter)
        z_counter = t_coord  
        
        return x_counter, y_counter, z_counter, phi_counter
    
    def worldline_inertial_coordinates(self, t_coord, direction=1):
        """
        Línea de mundo en coordenadas del sistema inercial
        direction = +1 para co-rotación, -1 para contra-rotación
        """
        # En el sistema inercial
        phi_inertial = direction * self.c/self.R * t_coord
        phi_rotating = phi_inertial - self.Omega * t_coord
        
        x = self.R * np.cos(phi_rotating)
        y = self.R * np.sin(phi_rotating)
        z = t_coord
        
        return x, y, z, phi_rotating

def create_animation():
    
    sagnac = SagnacWorldLines(R=1.0, Omega=0.3, c=1.0)
    
    t_max = 3 * sagnac.T
    t_values = np.linspace(0, t_max, 500)
    
    fig = plt.figure(figsize=(16, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    
    ax2 = fig.add_subplot(122)
    
    limit = sagnac.R * 1.2
    
    line_co, = ax1.plot([], [], [], 'r-', linewidth=2, label='Haz co-rotante')
    line_counter, = ax1.plot([], [], [], 'b-', linewidth=2, label='Haz contra-rotante')
    point_co, = ax1.plot([], [], [], 'ro', markersize=8)
    point_counter, = ax1.plot([], [], [], 'bo', markersize=8)
    
    line_co_proj, = ax2.plot([], [], 'r-', linewidth=2, alpha=0.7)
    line_counter_proj, = ax2.plot([], [], 'b-', linewidth=2, alpha=0.7)
    point_co_proj, = ax2.plot([], [], 'ro', markersize=8)
    point_counter_proj, = ax2.plot([], [], 'bo', markersize=8)
    
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = sagnac.R * np.cos(theta)
    y_circle = sagnac.R * np.sin(theta)
    ax2.plot(x_circle, y_circle, 'k--', alpha=0.3, label='Interferómetro')
    
    def init():
        ax1.set_xlim(-limit, limit)
        ax1.set_ylim(-limit, limit)
        ax1.set_zlim(0, t_max)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Tiempo Coordenado')
        ax1.set_title('Líneas de Mundo en el Efecto Sagnac\n(Vista 3D)')
        ax1.legend()
        
        ax2.set_xlim(-limit, limit)
        ax2.set_ylim(-limit, limit)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Trayectorias Espaciales\n(Vista desde arriba)')
        ax2.legend()
        ax2.set_aspect('equal')
        
        return line_co, line_counter, point_co, point_counter, line_co_proj, line_counter_proj, point_co_proj, point_counter_proj
    
    def animate(i):

        t_current = t_values[i]
        t_segment = t_values[:i+1]
        
        x_co, y_co, z_co, phi_co = sagnac.worldline_co_rotating(t_segment)
        x_counter, y_counter, z_counter, phi_counter = sagnac.worldline_counter_rotating(t_segment)
        
        line_co.set_data(x_co, y_co)
        line_co.set_3d_properties(z_co)
        
        line_counter.set_data(x_counter, y_counter)
        line_counter.set_3d_properties(z_counter)
        
        x_co_curr, y_co_curr, z_co_curr, _ = sagnac.worldline_co_rotating(t_current)
        x_counter_curr, y_counter_curr, z_counter_curr, _ = sagnac.worldline_counter_rotating(t_current)
        
        point_co.set_data([x_co_curr], [y_co_curr])
        point_co.set_3d_properties([z_co_curr])
        
        point_counter.set_data([x_counter_curr], [y_counter_curr])
        point_counter.set_3d_properties([z_counter_curr])
        
        line_co_proj.set_data(x_co, y_co)
        line_counter_proj.set_data(x_counter, y_counter)
        
        point_co_proj.set_data([x_co_curr], [y_co_curr])
        point_counter_proj.set_data([x_counter_curr], [y_counter_curr])
        
        return line_co, line_counter, point_co, point_counter, line_co_proj, line_counter_proj, point_co_proj, point_counter_proj
    
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(t_values), interval=20, blit=True)
    
    delta_tau = sagnac.compute_proper_time_difference()
    info_text = f'Parámetros:\nR = {sagnac.R}, Ω = {sagnac.Omega}, c = {sagnac.c}\n'
    info_text += f'Diferencia de tiempo propio: Δτ ≈ {delta_tau:.4f}'
    
    ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return anim, sagnac

def create_phase_difference_plot(sagnac):
    """Crear un gráfico adicional mostrando la diferencia de fase acumulada"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    t_values = np.linspace(0, 3*sagnac.T, 300)
    
    _, _, _, phi_co = sagnac.worldline_co_rotating(t_values)
    _, _, _, phi_counter = sagnac.worldline_counter_rotating(t_values)
    
    phase_diff = (phi_co - phi_counter) / (2 * np.pi)
    
    ax.plot(t_values, phase_diff, 'g-', linewidth=2)
    ax.set_xlabel('Tiempo Coordenado')
    ax.set_ylabel('Diferencia de Fase (revoluciones)')
    ax.set_title('Acumulación de Diferencia de Fase en el Efecto Sagnac')
    ax.grid(True, alpha=0.3)
    
    delta_tau = sagnac.compute_proper_time_difference()
    ax.axhline(y=delta_tau/sagnac.T, color='r', linestyle='--', 
               label=f'Δτ/T = {delta_tau/sagnac.T:.4f}')
    ax.legend()
    
    return fig

if __name__ == "__main__":
    print("Generando animación del efecto Sagnac...")
    print("Basado en el formalismo de Eric Gourgoulhon para relatividad especial en marcos rotantes")
    
    anim, sagnac_system = create_animation()
    
    phase_plot = create_phase_difference_plot(sagnac_system)
    
    plt.show()
    
    # print("Guardando animación...")
    # anim.save('sagnac_effect.mp4', writer='ffmpeg', fps=30)
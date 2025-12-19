import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import matplotlib.colors as colors

class SagnacWaveInterferometer:
    def __init__(self, radius=1.0, wavelength=0.05, c=1.0):
        """
        Interferómetro Sagnac con ondas planas - Parámetros optimizados
        
        Parámetros:
        radius: radio del interferómetro
        wavelength: longitud de onda de la luz (reducida para más franjas)
        c: velocidad de la luz
        """
        self.R = radius
        self.lambda0 = wavelength
        self.c = c
        self.k = 2 * np.pi / wavelength  # número de onda
        
        # Parámetros de aceleración optimizados para observar interferencia
        self.alpha = 0.1  # mayor aceleración angular inicial
        self.alpha_growth = 0.02  # mayor tasa de crecimiento
    
    def time_dependent_angular_velocity(self, t):
        """Velocidad angular con aceleración lineal - optimizada"""
        return self.alpha * t + 0.5 * self.alpha_growth * t**2
    
    def sagnac_phase_difference(self, t):
        """
        Diferencia de fase Sagnac - optimizada para mayor sensibilidad
        """
        A = np.pi * self.R**2  # área
        Omega_t = self.time_dependent_angular_velocity(t)
        
        # Término principal amplificado
        phase_main = (16 * np.pi * A / (self.lambda0 * self.c)) * Omega_t
        
        return phase_main
    
    def wave_amplitude_co_rotating(self, x, y, t):
        """Onda co-rotante con parámetros optimizados"""
        Omega_t = self.time_dependent_angular_velocity(t)
        
        # Fase con modulación espacial para crear patrón de interferencia visible
        phase = (self.k * x * np.cos(0.5 * Omega_t * t) + 
                self.k * y * np.sin(0.5 * Omega_t * t) - 
                self.c * self.k * t +
                0.1 * self.k * x)  # Término adicional para crear franjas
        
        return np.exp(1j * phase)
    
    def wave_amplitude_counter_rotating(self, x, y, t):
        """Onda contra-rotante con parámetros optimizados"""
        Omega_t = self.time_dependent_angular_velocity(t)
        
        # Fase con modulación opuesta
        phase = (self.k * x * np.cos(0.5 * Omega_t * t + np.pi) + 
                self.k * y * np.sin(0.5 * Omega_t * t + np.pi) - 
                self.c * self.k * t -
                0.1 * self.k * x)  # Término opuesto para crear franjas
        
        return np.exp(1j * phase)
    
    def interference_pattern(self, x, y, t):
        """Patrón de interferencia optimizado para visualización"""
        # Amplitudes de las ondas
        A_co = self.wave_amplitude_co_rotating(x, y, t)
        A_counter = self.wave_amplitude_counter_rotating(x, y, t)
        
        # Intensidad del patrón de interferencia
        total_amplitude = A_co + A_counter
        intensity = np.abs(total_amplitude)**2
        
        # Normalizar para mejor visualización
        return 4 * intensity / np.max(intensity) if np.max(intensity) > 0 else intensity

def create_optimized_interference_animation():
    # Crear el interferómetro con parámetros optimizados
    interferometer = SagnacWaveInterferometer(radius=1.5, wavelength=0.08, c=1.0)
    
    # Crear malla espacial más densa para mejor resolución de franjas
    x = np.linspace(-2.0, 2.0, 300)
    y = np.linspace(-2.0, 2.0, 300)
    X, Y = np.meshgrid(x, y)
    
    # Máscara circular
    R_mask = np.sqrt(X**2 + Y**2)
    mask = R_mask <= interferometer.R
    
    # Configurar la figura
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Subplots
    ax1 = fig.add_subplot(gs[0:2, 0])  # Patrón de interferencia
    ax2 = fig.add_subplot(gs[0, 1])    # Velocidad angular
    ax3 = fig.add_subplot(gs[1, 1])    # Diferencia de fase
    ax4 = fig.add_subplot(gs[2, 0])    # Corte horizontal
    ax5 = fig.add_subplot(gs[2, 1])    # Corte vertical
    
    # Tiempos para la animación
    t_max = 12
    t_values = np.linspace(0, t_max, 80)
    
    # Precalcular algunos patrones para ajustar la escala de colores
    sample_patterns = []
    sample_times = [0, t_max//3, 2*t_max//3, t_max-1]
    for t_sample in sample_times:
        pattern = np.zeros_like(X)
        pattern[mask] = interferometer.interference_pattern(X[mask], Y[mask], t_sample)
        sample_patterns.append(pattern)
    
    vmax = max(np.max(p) for p in sample_patterns) if sample_patterns else 4
    
    # Inicializar gráficos
    interference_img = ax1.imshow(np.zeros_like(X), extent=[-2.0, 2.0, -2.0, 2.0], 
                                 cmap='hot', origin='lower', animated=True,
                                 vmin=0, vmax=vmax)
    
    # Círculo del interferómetro
    circle = plt.Circle((0, 0), interferometer.R, fill=False, color='white', linewidth=2)
    ax1.add_patch(circle)
    
    # Configurar ejes
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Patrón de Interferencia Sagnac - Optimizado')
    ax1.set_aspect('equal')
    
    # Gráficos de tiempo
    time_line_omega, = ax2.plot([], [], 'r-', linewidth=2, label='Ω(t)')
    time_line_phase, = ax3.plot([], [], 'b-', linewidth=2, label='ΔΦ(t)')
    current_time_omega = ax2.axvline(0, color='k', linestyle='--', alpha=0.5)
    current_time_phase = ax3.axvline(0, color='k', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Velocidad Angular (rad/s)')
    ax2.set_title('Evolución Temporal de Ω(t)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3.set_xlabel('Tiempo (s)')
    ax3.set_ylabel('Diferencia de Fase (rad)')
    ax3.set_title('Diferencia de Fase Sagnac')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Cortes espaciales
    cut_horizontal, = ax4.plot([], [], 'r-', linewidth=2, label='Corte en y=0')
    cut_vertical, = ax5.plot([], [], 'b-', linewidth=2, label='Corte en x=0')
    
    ax4.set_xlabel('Posición X')
    ax4.set_ylabel('Intensidad')
    ax4.set_title('Perfil de Intensidad - Corte Horizontal')
    ax4.set_ylim(0, vmax)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    ax5.set_xlabel('Posición Y')
    ax5.set_ylabel('Intensidad')
    ax5.set_title('Perfil de Intensidad - Corte Vertical')
    ax5.set_ylim(0, vmax)
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Información textual
    info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def init():
        interference_img.set_array(np.zeros_like(X))
        
        time_line_omega.set_data([], [])
        time_line_phase.set_data([], [])
        
        cut_horizontal.set_data([], [])
        cut_vertical.set_data([], [])
        
        info_text.set_text('')
        
        return (interference_img, time_line_omega, time_line_phase, 
                current_time_omega, current_time_phase, cut_horizontal, 
                cut_vertical, info_text)
    
    def animate(frame):
        t_current = t_values[frame]
        
        # Calcular patrón de interferencia
        pattern = np.zeros_like(X)
        pattern[mask] = interferometer.interference_pattern(X[mask], Y[mask], t_current)
        
        # Actualizar imagen principal
        interference_img.set_array(pattern)
        
        # Calcular datos temporales
        t_segment = t_values[:frame+1]
        omega_segment = [interferometer.time_dependent_angular_velocity(t) for t in t_segment]
        phase_segment = [interferometer.sagnac_phase_difference(t) for t in t_segment]
        
        time_line_omega.set_data(t_segment, omega_segment)
        time_line_phase.set_data(t_segment, phase_segment)
        
        current_time_omega.set_xdata([t_current, t_current])
        current_time_phase.set_xdata([t_current, t_current])
        
        # Actualizar límites
        ax2.set_xlim(0, t_max)
        ax2.set_ylim(0, max(omega_segment) * 1.2 if omega_segment else 1)
        
        ax3.set_xlim(0, t_max)
        ax3.set_ylim(0, max(phase_segment) * 1.2 if phase_segment else 1)
        
        # Cortes espaciales
        y0_index = len(y) // 2  # índice para y=0
        x0_index = len(x) // 2  # índice para x=0
        
        cut_horizontal.set_data(x, pattern[y0_index, :])
        cut_vertical.set_data(y, pattern[:, x0_index])
        
        # Información en tiempo real
        info_text.set_text(f't = {t_current:.1f} s\nΩ = {omega_segment[-1]:.3f} rad/s\nΔΦ = {phase_segment[-1]:.1f} rad\nFranjas visibles: {np.max(pattern):.1f}')
        
        return (interference_img, time_line_omega, time_line_phase, 
                current_time_omega, current_time_phase, cut_horizontal, 
                cut_vertical, info_text)
    
    # Crear animación
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(t_values), interval=150, blit=False)
    
    plt.tight_layout()
    return anim, interferometer

def create_interference_analysis():
    """Análisis detallado del patrón de interferencia"""
    interferometer = SagnacWaveInterferometer(radius=1.5, wavelength=0.08, c=1.0)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Tiempos específicos para análisis
    analysis_times = [2, 6, 10]
    
    # Crear malla
    x = np.linspace(-2.0, 2.0, 400)
    y = np.linspace(-2.0, 2.0, 400)
    X, Y = np.meshgrid(x, y)
    R_mask = np.sqrt(X**2 + Y**2)
    mask = R_mask <= interferometer.R
    
    for i, t in enumerate(analysis_times):
        # Calcular patrón
        pattern = np.zeros_like(X)
        pattern[mask] = interferometer.interference_pattern(X[mask], Y[mask], t)
        
        # Patrón completo
        im = axes[0, i].imshow(pattern, extent=[-2.0, 2.0, -2.0, 2.0], 
                              cmap='hot', origin='lower')
        axes[0, i].set_title(f't = {t} s\nΩ = {interferometer.time_dependent_angular_velocity(t):.3f} rad/s')
        axes[0, i].set_xlabel('X')
        axes[0, i].set_ylabel('Y')
        plt.colorbar(im, ax=axes[0, i])
        
        # Cortes
        y0_index = len(y) // 2
        x0_index = len(x) // 2
        
        axes[1, i].plot(x, pattern[y0_index, :], 'r-', linewidth=2, label='Corte Y=0')
        axes[1, i].plot(y, pattern[:, x0_index], 'b-', linewidth=2, label='Corte X=0')
        axes[1, i].set_xlabel('Posición')
        axes[1, i].set_ylabel('Intensidad')
        axes[1, i].set_title(f'Perfiles de Intensidad - t = {t} s')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
        
        # Contar franjas visibles
        cut_horizontal = pattern[y0_index, :]
        peaks = np.where((cut_horizontal[1:-1] > cut_horizontal[:-2]) & 
                        (cut_horizontal[1:-1] > cut_horizontal[2:]))[0] + 1
        axes[1, i].plot(x[peaks], cut_horizontal[peaks], 'go', markersize=4, label=f'Franjas: {len(peaks)}')
        axes[1, i].legend()
    
    plt.tight_layout()
    return fig

# Ejecutar la animación optimizada
if __name__ == "__main__":
    print("Generando animación OPTIMIZADA del patrón de interferencia Sagnac...")
    print("Parámetros ajustados para visualización clara de franjas de interferencia")
    
    # Crear animación principal optimizada
    anim, interferometer = create_optimized_interference_animation()
    
    # Crear análisis detallado
    analysis_fig = create_interference_analysis()
    
    plt.show()
    
    # Para guardar la animación
    # print("Guardando animación...")
    # anim.save('sagnac_interference_optimized.mp4', writer='ffmpeg', fps=8, dpi=150)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

N = 8
radio_rueda = 1.0
centro_rueda = (2, 0)
punto_paso = (centro_rueda[0] + radio_rueda, centro_rueda[1])
espejo_lejano_x = 6.0
ojo_pos = (0.5, -0.5)
semi_espejo_pos = (1.0, 0)
delta_t_viaje = 0.3
delta_t_retorno = 0.2
d_real_km = 10.0
omega_inicial = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(bottom=0.2, left=0.05, right=0.95)
ax.set_xlim(-1, 8)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_title('Experimento de Fizeau - Trayectoria real hacia el ojo')

ax.plot(-0.5, 1, 'yo', markersize=10, label='Fuente')
ax.plot([-0.5, semi_espejo_pos[0]], [1, semi_espejo_pos[1]], 'k:', alpha=0.3)
ax.plot([semi_espejo_pos[0], punto_paso[0]], [semi_espejo_pos[1], punto_paso[1]], 'k:', alpha=0.3)
ax.plot([punto_paso[0], espejo_lejano_x], [punto_paso[1], punto_paso[1]], 'k:', alpha=0.3)
ax.plot([espejo_lejano_x, punto_paso[0]], [punto_paso[1], punto_paso[1]], 'k:', alpha=0.3)
ax.plot([punto_paso[0], semi_espejo_pos[0]], [punto_paso[1], semi_espejo_pos[1]], 'k:', alpha=0.3)
ax.plot([semi_espejo_pos[0], ojo_pos[0]], [semi_espejo_pos[1], ojo_pos[1]], 'k:', alpha=0.3)
ax.plot([semi_espejo_pos[0]-0.5, semi_espejo_pos[0]+0.5],
        [semi_espejo_pos[1]-0.5, semi_espejo_pos[1]+0.5],
        'b-', linewidth=2, label='Semi-espejo')
ax.axvline(x=espejo_lejano_x, color='gray', linestyle='-', linewidth=3, label='Espejo lejano')
ax.plot(ojo_pos[0], ojo_pos[1], 'ko', markersize=8)
ax.text(ojo_pos[0]-0.2, ojo_pos[1]-0.2, 'Observador', fontsize=10)
circulo_rueda = plt.Circle(centro_rueda, radio_rueda, color='lightgray', fill=True, alpha=0.5)
ax.add_patch(circulo_rueda)
ax.plot(punto_paso[0], punto_paso[1], 'ro', markersize=6)

texto_c = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

dientes = []
pulsos = []
tiempo_anim = 0.0
omega = omega_inicial

def en_hueco(t, omega):
    phi = omega * t
    theta_rueda = (-phi) % (2*np.pi)
    sector = int(theta_rueda // (np.pi/N))
    return (sector % 2) == 1

def actualizar_texto_c(omega_val):
    f = omega_val / (2*np.pi)
    c_km_s = 4 * d_real_km * N * f
    texto_c.set_text(f'c = 4·d·N·f\nd = {d_real_km} km\nN = {N}\nω = {omega_val:.3f} rad/s\nf = {f:.3f} Hz\nc ≈ {c_km_s:.1f} km/s')
    return texto_c,

def init():
    global dientes, pulsos
    for d in dientes:
        d.remove()
    dientes.clear()
    for p in pulsos:
        p['linea'].remove()
    pulsos.clear()
    return []

def update(frame):
    global tiempo_anim, pulsos, dientes, omega
    dt = 0.02
    tiempo_anim += dt

    for d in dientes:
        d.remove()
    dientes.clear()
    phi = omega * tiempo_anim
    for i in range(2*N):
        if i % 2 == 0:
            theta_inicio = i * (np.pi/N) + phi
            theta_fin = (i+1) * (np.pi/N) + phi
            diente = patches.Wedge(centro_rueda, radio_rueda,
                                   theta_inicio*180/np.pi, theta_fin*180/np.pi,
                                   color='black', alpha=0.8)
            ax.add_patch(diente)
            dientes.append(diente)

    if int(tiempo_anim / 0.8) > int((tiempo_anim - dt) / 0.8):
        if en_hueco(tiempo_anim, omega):
            linea, = ax.plot([], [], 'y-', linewidth=2, alpha=0.8)
            pulsos.append({
                't_emision': tiempo_anim,
                't_inicio_tramo': tiempo_anim,
                'estado': 1,
                'linea': linea
            })

    for p in pulsos[:]:
        t_trans = tiempo_anim - p['t_inicio_tramo']
        estado = p['estado']

        if estado == 1:
            fraccion = t_trans / delta_t_viaje
            if fraccion >= 1.0:
                p['estado'] = -1
                p['t_inicio_tramo'] = tiempo_anim
            else:
                x_actual = punto_paso[0] + fraccion * (espejo_lejano_x - punto_paso[0])
                p['linea'].set_data([punto_paso[0], x_actual], [punto_paso[1], punto_paso[1]])
                p['linea'].set_color('yellow')

        elif estado == -1:
            fraccion = t_trans / delta_t_viaje
            if fraccion >= 1.0:
                if en_hueco(tiempo_anim, omega):
                    p['estado'] = -2
                    p['t_inicio_tramo'] = tiempo_anim
                else:
                    p['linea'].remove()
                    pulsos.remove(p)
                    continue
            else:
                x_actual = espejo_lejano_x - fraccion * (espejo_lejano_x - punto_paso[0])
                p['linea'].set_data([espejo_lejano_x, x_actual], [punto_paso[1], punto_paso[1]])
                p['linea'].set_color('orange')

        elif estado == -2:
            dist_primer_tramo = punto_paso[0] - semi_espejo_pos[0]
            dx2 = ojo_pos[0] - semi_espejo_pos[0]
            dy2 = ojo_pos[1] - semi_espejo_pos[1]
            dist_segundo_tramo = np.sqrt(dx2**2 + dy2**2)
            dist_total = dist_primer_tramo + dist_segundo_tramo
            fraccion = t_trans / delta_t_retorno
            if fraccion >= 1.0:
                p['linea'].remove()
                pulsos.remove(p)
                continue
            else:
                d_recorrida = fraccion * dist_total
                if d_recorrida <= dist_primer_tramo:
                    x_actual = punto_paso[0] - d_recorrida
                    y_actual = punto_paso[1]
                    p['linea'].set_data([punto_paso[0], x_actual], [punto_paso[1], y_actual])
                else:
                    d_restante = d_recorrida - dist_primer_tramo
                    frac2 = d_restante / dist_segundo_tramo
                    x_actual = semi_espejo_pos[0] + frac2 * dx2
                    y_actual = semi_espejo_pos[1] + frac2 * dy2
                    p['linea'].set_data([punto_paso[0], semi_espejo_pos[0], x_actual],
                                        [punto_paso[1], semi_espejo_pos[1], y_actual])
                p['linea'].set_color('green')

    actualizar_texto_c(omega)
    elementos = dientes + [texto_c] + [p['linea'] for p in pulsos]
    return elementos

ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Velocidad angular ω (rad/s)', 0.0, 1000.0, valinit=omega_inicial, valstep=0.1)

def update_omega(val):
    global omega
    omega = val
    actualizar_texto_c(omega)

slider.on_changed(update_omega)

ani = FuncAnimation(fig, update, init_func=init, interval=10, blit=False, cache_frame_data=False)
plt.show()
import numpy as np


media_recepcion = 4     # minutos
desv_recepcion = 1      # desviación estándar para recepción
min_pesado_mezcla = 3   # mínimo tiempo para pesado y mezcla
max_pesado_mezcla = 5   # máximo tiempo para pesado y mezcla
tasa_control_calidad = 0.25  # tasa para inspección de calidad
media_empaque = 6       # media para el tiempo de empaque
desv_empaque = 2        # desviación estándar para empaque


t = 0               # Tiempo actual
tmax = 480          # Tiempo máximo de simulación (minutos)
eventos = []        # Lista de eventos programados (cola de eventos)
cola_pedidos = []   # Cola de pedidos esperando con sus IDs de usuario


servidor_ocupado = False  # Estado del servidor (False = libre, True = ocupado)
usuarios_atendidos = 0    # Número de usuarios atendidos
usuarios_totales = 0      # Núemero total de usuarios que entraron al sistema
total_procesamiento = 0


def generar_tiempo_recepcion(distribucion="exponencial"):
    if distribucion == "normal":
        return max(0, np.random.normal(media_recepcion, desv_recepcion))
    elif distribucion == "exponencial":
        return np.random.exponential(1 / tasa_control_calidad)
    elif distribucion == "uniforme":
        return np.random.uniform(min_pesado_mezcla, max_pesado_mezcla)

def generar_tiempo_pesado_mezcla(distribucion="uniforme"):
    if distribucion == "normal":
        return max(0, np.random.normal((min_pesado_mezcla + max_pesado_mezcla) / 2, (max_pesado_mezcla - min_pesado_mezcla) / 2))
    elif distribucion == "exponencial":
        return np.random.exponential(1 / tasa_control_calidad)
    elif distribucion == "uniforme":
        return np.random.uniform(min_pesado_mezcla, max_pesado_mezcla)

def generar_tiempo_control_calidad(distribucion="normal"):
    if distribucion == "normal":
        return max(0, np.random.normal(1 / tasa_control_calidad, 0.5))
    elif distribucion == "exponencial":
        return np.random.exponential(1 / tasa_control_calidad)
    elif distribucion == "uniforme":
        return np.random.uniform(min_pesado_mezcla, max_pesado_mezcla)

def generar_tiempo_empaque(distribucion="exponencial"):
    if distribucion == "normal":
        return max(0, np.random.normal(media_empaque, desv_empaque))
    elif distribucion == "exponencial":
        return np.random.exponential(1 / tasa_control_calidad)
    elif distribucion == "uniforme":
        return np.random.uniform(min_pesado_mezcla, max_pesado_mezcla)


usuarios_totales += 1
eventos.append((generar_tiempo_recepcion(), "recepcion", "llegada", usuarios_totales))


while t < tmax and eventos:
    eventos.sort(key=lambda x: x[0])  # Ordenamos los eventos por tiempo
    t, etapa, estado, usuario_id = eventos.pop(0)

    if etapa == "recepcion" and estado == "llegada":
        print(f"[{t:.2f} min] Pedido recibdo del Usuario {usuario_id}.")

        if not servidor_ocupado:
            servidor_ocupado = True
            tiempo_pesado_mezcla = t + generar_tiempo_pesado_mezcla()
            eventos.append((tiempo_pesado_mezcla, "pesado_mezcla", "inicio", usuario_id))
        else:  #
            cola_pedidos.append(usuario_id)

        #
        usuarios_totales += 1
        eventos.append((t + generar_tiempo_recepcion(), "recepcion", "llegada", usuarios_totales))

    elif etapa == "pesado_mezcla" and estado == "inicio":
        print(f"[{t:.2f} min] Pesado y mezcla en proceso para el Usuario {usuario_id}.")
        tiempo_control_calidad = t + generar_tiempo_control_calidad()
        eventos.append((tiempo_control_calidad, "control_calidad", "inicio", usuario_id))

    elif etapa == "control_calidad" and estado == "inicio":
        print(f"[{t:.2f} min] Control de calidad en proceso para el Usuario {usuario_id}.")
        tiempo_empaque = t + generar_tiempo_empaque()
        eventos.append((tiempo_empaque, "empaque", "inicio", usuario_id))

    elif etapa == "empaque" and estado == "inicio":
        print(f"[{t:.2f} min] Empaque y etiquetado completado para el Usuario {usuario_id}.")
        usuarios_atendidos += 1
        total_procesamiento += t
        servidor_ocupado = False  # El servidor queda libre

        if cola_pedidos:  # Si hay pedidos en cola, atendemos el siguiente
            siguiente_usuario_id = cola_pedidos.pop(0)
            servidor_ocupado = True
            tiempo_pesado_mezcla = t + generar_tiempo_pesado_mezcla()
            eventos.append((tiempo_pesado_mezcla, "pesado_mezcla", "inicio", siguiente_usuario_id))


print("\nEstadísticas de Simulación:")
print(f"Total de pedidos que ingresaron al sistema: {usuarios_totales - 1}")
print(f"Total de pedidos atendidos: {usuarios_atendidos}")
print(f"Total de pedidos en espera: {len(cola_pedidos)}")
print(
    f"Tiempo promedio de procesamiento por pedido: {total_procesamiento / usuarios_atendidos if usuarios_atendidos > 0 else 0:.2f} min")
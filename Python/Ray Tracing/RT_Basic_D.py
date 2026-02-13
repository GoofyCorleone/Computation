# -*- coding: utf-8 -*-
"""
Trazado de rayos 3D para ovoides de Descartes (superficies cartesianas)
Autor: Adaptado para la consulta
Requisitos: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import root_scalar

# ----------------------------------------------------------------------
#   CLASE PRINCIPAL: Ovoide de Descartes
# ----------------------------------------------------------------------
class DescartesOvoid:
    """
    Representa un dioptrio cartesiano (ovoide de Descartes) que enfoca
    perfectamente un punto objeto O en un punto imagen I, con índices n1 y n2.
    El vértice V está sobre el eje OI.
    """
    def __init__(self, O, I, V, n1, n2):
        """
        Parámetros:
            O : tuple (x,y,z) - punto objeto (fuente perfecta)
            I : tuple (x,y,z) - punto imagen
            V : tuple (x,y,z) - vértice sobre el eje (debe estar entre O e I)
            n1: índice de refracción del medio incidente
            n2: índice de refracción del medio transmitido
        """
        self.O = np.array(O, dtype=float)
        self.I = np.array(I, dtype=float)
        self.V = np.array(V, dtype=float)
        self.n1 = n1
        self.n2 = n2

        # Vector unitario del eje óptico (de O a I)
        eje = self.I - self.O
        self.d = np.linalg.norm(eje)
        if self.d == 0:
            raise ValueError("O e I no pueden coincidir.")
        self.eje = eje / self.d

        # Constante de camino óptico: L = n1*|V-O| + n2*|V-I|
        self.L = n1 * np.linalg.norm(self.V - self.O) + n2 * np.linalg.norm(self.V - self.I)

        # Para facilitar la generación de la superficie, trabajaremos en coordenadas locales
        # con origen en O y eje x según OI. Esto simplifica el mallado.
        # En el sistema local: O_local = (0,0,0), I_local = (d,0,0), V_local = (v,0,0)
        # donde v = |V-O| (con signo, pues V está entre O e I)
        v_local = np.linalg.norm(self.V - self.O)
        # Aseguramos que V esté entre O e I (producto escalar positivo)
        if np.dot(self.V - self.O, self.I - self.O) < 0:
            v_local = -v_local
        self.v = v_local
        # Almacenamos también la matriz de cambio de base (por si se desea usar en el futuro)
        # No es estrictamente necesaria si fijamos O e I en el eje x para el trazado.
        # En este código simplificado asumimos O=(0,0,0), I=(d,0,0) y la fuente en cualquier lugar.
        # Pero permitimos que el usuario modifique O e I; para el mallado los proyectamos sobre el eje.

    # ----------------------------------------------------------------
    #   Función de nivel del ovoide: F(X) = n1*|X-O| + n2*|X-I| - L
    # ----------------------------------------------------------------
    def F(self, X):
        """Evalúa la función de nivel en el punto X (array numpy)."""
        dO = np.linalg.norm(X - self.O)
        dI = np.linalg.norm(X - self.I)
        return self.n1 * dO + self.n2 * dI - self.L

    # ----------------------------------------------------------------
    #   Gradiente de F (sin normalizar)
    # ----------------------------------------------------------------
    def grad_F(self, X):
        """Retorna el vector gradiente de F en X."""
        dO = np.linalg.norm(X - self.O)
        dI = np.linalg.norm(X - self.I)
        if dO == 0 or dI == 0:
            # En los focos el gradiente no está definido, pero nunca intersectaremos ahí.
            return np.zeros(3)
        grad = self.n1 * (X - self.O) / dO + self.n2 * (X - self.I) / dI
        return grad

    # ----------------------------------------------------------------
    #   Normal unitaria que apunta hacia el medio n2 (hacia la imagen)
    # ----------------------------------------------------------------
    def normal(self, X):
        """Normal unitaria en el punto X de la superficie, apuntando hacia n2."""
        grad = self.grad_F(X)
        # El gradiente apunta hacia el medio n1 (aumento de F). Lo invertimos.
        norm = -grad / np.linalg.norm(grad)
        return norm

    # ----------------------------------------------------------------
    #   Intersección rayo-superficie (búsqueda numérica)
    # ----------------------------------------------------------------
    def find_intersection(self, S, D):
        """
        Encuentra el punto de intersección del rayo (S + t*D, t>0) con el ovoide.
        Parámetros:
            S : array (3,) - punto de origen del rayo
            D : array (3,) - dirección del rayo (no necesariamente unitaria)
        Retorna:
            P : array (3,) - punto de intersección (None si no hay)
            t : float      - parámetro tal que P = S + t*D (None si no hay)
            N : array (3,) - normal unitaria en P (apuntando a n2)
            success : bool - True si se encontró intersección
        """
        D = D / np.linalg.norm(D)  # normalizamos
        # Función auxiliar para root_scalar
        def F_rayo(t):
            return self.F(S + t * D)

        # Búsqueda adaptativa de un intervalo donde F cambie de signo
        t = 0.0
        f0 = F_rayo(t)
        # Máxima distancia razonable (evitar bucles infinitos)
        t_max = 1e6
        step = 0.1  # paso inicial

        if f0 > 0:
            # Esperamos que el rayo entre al ovoide: F debe pasar de positiva a negativa
            while t < t_max:
                t += step
                ft = F_rayo(t)
                if ft <= 0:
                    t1 = t - step
                    t2 = t
                    break
                step *= 1.2
            else:
                return None, None, None, False
        elif f0 < 0:
            # El origen está dentro del ovoide; buscamos salida (F de negativa a positiva)
            while t < t_max:
                t += step
                ft = F_rayo(t)
                if ft >= 0:
                    t1 = t - step
                    t2 = t
                    break
                step *= 1.2
            else:
                return None, None, None, False
        else:
            # Origen justo sobre la superficie
            P = S.copy()
            t = 0.0
            N = self.normal(P)
            return P, t, N, True

        # Refinamos con brentq
        try:
            sol = root_scalar(F_rayo, bracket=[t1, t2], method='brentq')
            if sol.converged:
                t = sol.root
                P = S + t * D
                N = self.normal(P)
                return P, t, N, True
            else:
                return None, None, None, False
        except ValueError:
            return None, None, None, False

    # ----------------------------------------------------------------
    #   Ley de Snell vectorial
    # ----------------------------------------------------------------
    @staticmethod
    def snell_refraction(D_in, N, n1, n2):
        """
        Calcula la dirección del rayo refractado.
        Parámetros:
            D_in : array (3,) - dirección incidente (desde fuente hacia superficie), unitario.
            N    : array (3,) - normal unitaria en el punto, apuntando hacia el medio n2.
            n1   : índice del medio incidente
            n2   : índice del medio transmitido
        Retorna:
            D_ref : array (3,) - dirección refractada (unitaria), o None si hay reflexión total interna.
        """
        eta = n1 / n2
        cos_theta1 = -np.dot(D_in, N)  # >0 si el rayo incide desde n1 hacia la superficie
        if cos_theta1 < 0:
            # El rayo viene desde el otro lado; en nuestro contexto no debería ocurrir.
            # Si sucede, invertimos la normal y redefinimos n1,n2? Asumimos que no.
            # Para robustez, tomamos el valor absoluto y ajustamos.
            cos_theta1 = -cos_theta1
            N = -N   # Ahora la normal apunta hacia donde viene el rayo (medio n1)
            # Intercambiamos índices? Mejor devolver None.
            return None

        sin_theta1 = np.sqrt(max(0.0, 1.0 - cos_theta1**2))
        sin_theta2 = eta * sin_theta1
        if sin_theta2 > 1.0:
            return None  # Reflexión total interna

        cos_theta2 = np.sqrt(max(0.0, 1.0 - sin_theta2**2))
        D_ref = eta * D_in + (eta * cos_theta1 - cos_theta2) * N
        return D_ref / np.linalg.norm(D_ref)

    # ----------------------------------------------------------------
    #   Generación de la malla de la superficie (revolución alrededor del eje)
    # ----------------------------------------------------------------
    def generate_surface_points(self, n_phi=30, n_x=40):
        """
        Genera puntos (X,Y,Z) de la superficie del ovoide en el sistema de coordenadas global.
        Asume que O=(0,0,0), I=(d,0,0) para simplificar el mallado.
        Si el usuario cambia O e I, esta función debe modificarse o proyectarse.
        En este ejemplo, mantenemos esta suposición para el mallado.
        """
        # Parámetros locales
        d = self.d
        v = self.v
        n1 = self.n1
        n2 = self.n2
        L = self.L

        # Límite superior en x (punto de cierre sobre el eje)
        x_max = (L + n2 * d) / (n1 + n2)
        x_min = v  # desde el vértice

        x_vals = np.linspace(x_min, x_max, n_x)
        y_vals = []
        # Para cada x, resolver n1*sqrt(x^2+y^2) + n2*sqrt((x-d)^2+y^2) = L
        # con y>=0. Usamos root_scalar.
        def eq_y(y, x):
            return n1 * np.sqrt(x**2 + y**2) + n2 * np.sqrt((x - d)**2 + y**2) - L

        # Cota superior para y: (L/(n1+n2)) es seguro
        y_max_global = L / (n1 + n2)

        for x in x_vals:
            f0 = eq_y(0.0, x)
            if f0 >= 0:
                # Para x>=v, f0<0 (excepto en x=v que es 0). Si no es negativo, no hay raíz positiva.
                y_vals.append(0.0)
                continue
            try:
                sol = root_scalar(eq_y, args=(x,), bracket=[0.0, y_max_global], method='brentq')
                if sol.converged:
                    y_vals.append(sol.root)
                else:
                    y_vals.append(0.0)
            except ValueError:
                y_vals.append(0.0)

        # Convertir a arrays
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)

        # Generar malla por revolución alrededor del eje x
        phi = np.linspace(0, 2*np.pi, n_phi)
        X = np.outer(x_vals, np.ones(n_phi))
        Y = np.outer(y_vals, np.cos(phi))
        Z = np.outer(y_vals, np.sin(phi))

        return X, Y, Z


# ----------------------------------------------------------------------
#   FUNCIONES ADICIONALES: DISPERSIÓN (OPCIONAL)
# ----------------------------------------------------------------------
def sellmeier_BK7(lambda_um):
    """
    Índice de refracción del vidrio BK7 según la ecuación de Sellmeier.
    lambda_um: longitud de onda en micrómetros.
    """
    B1, B2, B3 = 1.03961212, 0.231792344, 1.01046945
    C1, C2, C3 = 0.00600069867, 0.0200179144, 103.560653
    l2 = lambda_um**2
    n2 = 1 + B1*l2/(l2 - C1) + B2*l2/(l2 - C2) + B3*l2/(l2 - C3)
    return np.sqrt(n2)


# ----------------------------------------------------------------------
#   CONFIGURACIÓN POR DEFECTO
# ----------------------------------------------------------------------
def configurar_por_defecto():
    """
    Establece parámetros por defecto:
    - O = (0,0,0)
    - I = (10,0,0)
    - V = (5,0,0)
    - n1 = 1.0 (vacío)
    - n2 = 1.5 (vidrio, índice fijo)
    - Fuente puntual en O.
    """
    O = (0, 0, 0)
    I = (10, 0, 0)
    V = (5, 0, 0)
    n1 = 1.0
    n2 = 1.5   # valor fijo, o se puede calcular con sellmeier si se da longitud de onda
    fuente = O   # fuente en el punto objeto
    return O, I, V, n1, n2, fuente


# ----------------------------------------------------------------------
#   TRAZADO DE RAYOS Y VISUALIZACIÓN 3D
# ----------------------------------------------------------------------
def trazar_rayos(ovoid, fuente, num_rayos=30, angulo_max=np.radians(60),
                 longitud_rayo=25, color_incidente='red', color_refractado='blue',
                 ax=None, white_lambda=None):
    """
    Traza rayos desde la fuente, calcula intersección y refracción, y los dibuja.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Direcciones de los rayos: distribuidos uniformemente en un cono alrededor del eje óptico.
    # Para simplicidad, generamos direcciones aleatorias dentro del cono.
    np.random.seed(42)  # reproducibilidad
    ejex = np.array([1, 0, 0])  # eje óptico en nuestra configuración

    for i in range(num_rayos):
        # Generar dirección aleatoria dentro de un cono de semiángulo angulo_max
        # Usamos el método de muestreo uniforme en ángulo sólido.
        theta = np.arccos(1 - np.random.rand() * (1 - np.cos(angulo_max)))
        phi = np.random.rand() * 2 * np.pi
        # Vector dirección en coordenadas esféricas con eje x como referencia
        dx = np.cos(theta)
        dy = np.sin(theta) * np.cos(phi)
        dz = np.sin(theta) * np.sin(phi)
        D = np.array([dx, dy, dz])
        D = D / np.linalg.norm(D)

        # Encontrar intersección
        P, t, N, success = ovoid.find_intersection(fuente, D)
        if not success:
            continue

        # Dibujar rayo incidente (desde fuente hasta P)
        ax.plot([fuente[0], P[0]], [fuente[1], P[1]], [fuente[2], P[2]],
                color=color_incidente, alpha=0.7, linewidth=1)

        # Calcular refracción
        D_in = D  # dirección incidente (de fuente a superficie)
        D_ref = ovoid.snell_refraction(D_in, N, ovoid.n1, ovoid.n2)
        if D_ref is None:
            continue  # reflexión total interna

        # Dibujar rayo refractado (desde P hacia adelante)
        t_ref = longitud_rayo
        Q = P + t_ref * D_ref
        ax.plot([P[0], Q[0]], [P[1], Q[1]], [P[2], Q[2]],
                color=color_refractado, alpha=0.7, linewidth=1)

    return ax


# ----------------------------------------------------------------------
#   PROGRAMA PRINCIPAL
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ----------------------------------------------------------------
    #   PARÁMETROS MODIFICABLES POR EL USUARIO
    # ----------------------------------------------------------------
    # 1. Parámetros del ovoide
    O = (0, 0, 0)
    I = (10, 0, 0)
    V = (5, 0, 0)      # vértice sobre el eje

    # 2. Índices de refracción
    # Opción A: índices fijos
    n1 = 1.0           # vacío
    n2 = 1.5           # vidrio (índice fijo)

    # Opción B: usar longitud de onda para calcular n2 (BK7) y n1=1.0
    # Descomentar las siguientes líneas para activar dispersión:
    # usar_dispersion = True
    # lambda_um = 0.55   # verde (micrómetros)
    # n2 = sellmeier_BK7(lambda_um)
    # # Para luz blanca, se pueden trazar varios rayos con distintos colores.
    # # Más abajo se muestra un ejemplo.

    # 3. Fuente puntual
    fuente = O   # por defecto en el objeto (foco perfecto)
    # Se puede cambiar, por ejemplo:
    # fuente = (2, 0, 0)   # desplazada sobre el eje
    # fuente = (0, 3, 0)   # fuera del eje

    # 4. Número de rayos a trazar
    num_rayos = 40

    # 5. Ángulo máximo del cono de rayos (en radianes)
    angulo_max = np.radians(60)

    # ----------------------------------------------------------------
    #   CREAR OVOIDE
    # ----------------------------------------------------------------
    ovoid = DescartesOvoid(O, I, V, n1, n2)

    # ----------------------------------------------------------------
    #   GENERAR MALLA DE LA SUPERFICIE
    # ----------------------------------------------------------------
    X, Y, Z = ovoid.generate_surface_points(n_phi=30, n_x=50)

    # ----------------------------------------------------------------
    #   VISUALIZACIÓN 3D
    # ----------------------------------------------------------------
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Dibujar superficie (wireframe para claridad)
    ax.plot_wireframe(X, Y, Z, color='cyan', alpha=0.3, rstride=1, cstride=1, linewidth=0.5)

    # Dibujar puntos O, I, V y la fuente
    ax.scatter(*O, color='green', s=50, label='Objeto (O)')
    ax.scatter(*I, color='orange', s=50, label='Imagen (I)')
    ax.scatter(*V, color='black', s=30, label='Vértice (V)')
    ax.scatter(*fuente, color='magenta', s=80, marker='*', label='Fuente puntual')

    # Trazar rayos
    trazar_rayos(ovoid, fuente, num_rayos=num_rayos, angulo_max=angulo_max,
                 color_incidente='red', color_refractado='blue', ax=ax)

    # Etiquetas y leyenda
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trazado de rayos en ovoide de Descartes')
    ax.legend(loc='upper right')

    # Ajustar límites para mejor visualización
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

    # ----------------------------------------------------------------
    #   EJEMPLO ADICIONAL: LUZ BLANCA (DISPERSIÓN)
    # ----------------------------------------------------------------
    # Descomentar el bloque siguiente para simular luz blanca trazando varios
    # rayos con distintas longitudes de onda y colores.
    """
    print("Simulando luz blanca (dispersión en BK7)...")
    fig2 = plt.figure(figsize=(12, 9))
    ax2 = fig2.add_subplot(111, projection='3d')
    # Malla del ovoide (mismos parámetros)
    X, Y, Z = ovoid.generate_surface_points(n_phi=30, n_x=50)
    ax2.plot_wireframe(X, Y, Z, color='gray', alpha=0.2, rstride=1, cstride=1, linewidth=0.5)
    ax2.scatter(*O, color='green', s=50)
    ax2.scatter(*I, color='orange', s=50)
    ax2.scatter(*V, color='black', s=30)
    ax2.scatter(*fuente, color='magenta', s=80, marker='*')

    longitudes_onda = [0.45, 0.55, 0.65]  # azul, verde, rojo (µm)
    colores = ['blue', 'green', 'red']
    n1_fijo = 1.0
    for lam, col in zip(longitudes_onda, colores):
        n2_disp = sellmeier_BK7(lam)
        ovoid_disp = DescartesOvoid(O, I, V, n1_fijo, n2_disp)
        trazar_rayos(ovoid_disp, fuente, num_rayos=12, angulo_max=angulo_max,
                     color_incidente=col, color_refractado=col, ax=ax2,
                     longitud_rayo=20)

    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.set_title('Luz blanca - Dispersión en BK7')
    plt.show()
    """
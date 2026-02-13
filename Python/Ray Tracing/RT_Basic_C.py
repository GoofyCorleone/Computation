import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# UTILIDADES FÍSICAS
# ============================================================

def normalize(v):
    return v / np.linalg.norm(v)

def snell_vector(n1, n2, normal, incident):
    """
    Refracción vectorial usando ley de Snell.
    normal apunta hacia el medio incidente.
    """
    normal = normalize(normal)
    incident = normalize(incident)

    cos_i = -np.dot(normal, incident)
    eta = n1 / n2

    sin_t2 = eta**2 * (1 - cos_i**2)

    # reflexión total interna
    if sin_t2 > 1:
        return None

    cos_t = np.sqrt(1 - sin_t2)

    refracted = eta * incident + (eta * cos_i - cos_t) * normal
    return normalize(refracted)

# ============================================================
# DISPERSIÓN (vidrio simple)
# ============================================================

def refractive_index_glass(wavelength_nm):
    """
    Modelo simple tipo Cauchy.
    """
    B = 1.5046
    C = 0.00420
    lam_um = wavelength_nm / 1000
    return B + C / lam_um**2

# ============================================================
# OVOIDE DE DESCARTES
# ============================================================

class DescartesOvoid:
    def __init__(self, F1, F2, n1, n2):
        self.F1 = np.array(F1)
        self.F2 = np.array(F2)
        self.n1 = n1
        self.n2 = n2

        # constante de la superficie
        self.k = n1*np.linalg.norm(self.F1) + n2*np.linalg.norm(self.F2)

    def surface_function(self, P):
        return self.n1*np.linalg.norm(P - self.F1) + \
               self.n2*np.linalg.norm(P - self.F2) - self.k

    def normal(self, P):
        grad = self.n1*(P-self.F1)/np.linalg.norm(P-self.F1) + \
               self.n2*(P-self.F2)/np.linalg.norm(P-self.F2)
        return normalize(grad)

# ============================================================
# INTERSECCIÓN RAYO-SUPERFICIE
# ============================================================

def intersect_ray(surface, origin, direction, tmax=200, steps=2000):
    t_values = np.linspace(0.001, tmax, steps)

    prev = surface.surface_function(origin)

    for t in t_values:
        P = origin + t*direction
        val = surface.surface_function(P)

        if prev * val < 0:
            return P

        prev = val

    return None

# ============================================================
# RAY-TRACING
# ============================================================

def trace_ray(surface, origin, direction, wavelength=None,
              white_light=False):

    rays = []

    if white_light:
        wavelengths = np.linspace(450, 650, 7)
    else:
        wavelengths = [wavelength]

    for lam in wavelengths:

        if lam is None:
            n2 = surface.n2
        else:
            n2 = refractive_index_glass(lam)

        hit = intersect_ray(surface, origin, direction)
        if hit is None:
            continue

        normal = surface.normal(hit)

        refracted = snell_vector(surface.n1, n2, normal, direction)
        if refracted is None:
            continue

        rays.append((hit, refracted, lam))

    return rays

# ============================================================
# GENERADOR DE RAYOS
# ============================================================

def generate_rays(source, num_rays=30, aperture=0.4):

    rays = []
    for theta in np.linspace(0, 2*np.pi, num_rays):
        for phi in np.linspace(-aperture, aperture, num_rays//4):

            dir_vec = np.array([
                np.cos(theta)*np.cos(phi),
                np.sin(theta)*np.cos(phi),
                np.sin(phi)
            ])

            rays.append(normalize(dir_vec))

    return rays

# ============================================================
# VISUALIZACIÓN
# ============================================================
def generate_ovoid_mesh(surface, x_range=(-2.5, 2.5), res_x=200, res_theta=120):
    """
    Genera la malla 3D del ovoide de Descartes por revolución.
    Eje óptico: x
    """

    F1 = surface.F1
    F2 = surface.F2
    n1 = surface.n1
    n2 = surface.n2
    k  = surface.k

    xs = np.linspace(x_range[0], x_range[1], res_x)
    thetas = np.linspace(0, 2*np.pi, res_theta)

    X, Y, Z = [], [], []

    for x in xs:
        # resolver radio r(x)
        # buscamos r tal que cumpla la ecuación
        r_vals = np.linspace(0, 3, 200)

        r_good = None
        prev_val = None

        for r in r_vals:
            P = np.array([x, r, 0])
            val = surface.surface_function(P)

            if prev_val is not None and prev_val*val < 0:
                r_good = r
                break

            prev_val = val

        if r_good is None:
            continue

        for theta in thetas:
            y = r_good*np.cos(theta)
            z = r_good*np.sin(theta)

            X.append(x)
            Y.append(y)
            Z.append(z)

    return np.array(X), np.array(Y), np.array(Z)


def plot_scene(surface, source, rays, white_light=False):

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')

    # =====================================================
    # SUPERFICIE DEL OVOIDE
    # =====================================================
    X, Y, Z = generate_ovoid_mesh(surface)

    ax.plot_trisurf(
        X, Y, Z,
        linewidth=0.2,
        alpha=0.25,
        color='gray'
    )

    # =====================================================
    # FUENTE
    # =====================================================
    ax.scatter(*source, color='red', s=80, label="Fuente")

    # =====================================================
    # RAYOS
    # =====================================================
    for d in rays:

        results = trace_ray(
            surface,
            source,
            d,
            wavelength=550,
            white_light=white_light
        )

        for hit, refr, lam in results:

            # rayo incidente
            ax.plot(
                [source[0], hit[0]],
                [source[1], hit[1]],
                [source[2], hit[2]],
                color='blue'
            )

            # rayo refractado
            end = hit + 4*refr

            if white_light:
                col = plt.cm.jet((lam-450)/200)
            else:
                col = 'green'

            ax.plot(
                [hit[0], end[0]],
                [hit[1], end[1]],
                [hit[2], end[2]],
                color=col
            )

    # =====================================================
    # FORMATO
    # =====================================================
    ax.set_xlabel("x (eje óptico)")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Ray-Tracing en Ovoide de Descartes")

    ax.legend()
    ax.set_box_aspect([1,1,1])

    plt.show()


# ============================================================
# PARÁMETROS POR DEFECTO
# ============================================================

if __name__ == "__main__":

    # Fuente y foco imagen
    F1 = np.array([-2.0,0,0])
    F2 = np.array([2.0,0,0])

    n1 = 1.0   # aire
    n2 = 1.5   # vidrio

    surface = DescartesOvoid(F1, F2, n1, n2)

    source = F1
    rays = generate_rays(source, num_rays=25)

    plot_scene(surface, source, rays, white_light=True)

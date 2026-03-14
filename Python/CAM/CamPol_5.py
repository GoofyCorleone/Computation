import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import os

# ==============================================================================
# CONFIGURACIÓN - Ajusta estas rutas a tus archivos
# ==============================================================================
FILE_PATHS = {
    'I0':    'CCDI(0,0).tif',      # Polarizador 0°,  retardador 0°
    'I45':   'CCDI(45,0).tif',     # Polarizador 45°, retardador 0°
    'I90':   'CCDI(90,0).tif',     # Polarizador 90°, retardador 0°
    'I4590': 'CCDI(45,90).tif',    # Polarizador 45°, retardador 90°
}

OUTPUT_DIR = 'resultados_polarimetria'
# ==============================================================================


def load_images(file_paths: dict) -> dict:
    """
    Carga imágenes TIFF y las retorna como arrays float64.
    Si son RGB, convierte a escala de grises promediando canales.
    """
    imgs = {}
    for name, path in file_paths.items():
        img = np.array(Image.open(path)).astype(np.float64)
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)
        imgs[name] = img
        print(f"  {name}: shape={img.shape}, dtype={img.dtype}, "
              f"min={img.min():.2f}, max={img.max():.2f}")
    return imgs


def compute_stokes(I0, I45, I90, I4590):
    """
    Calcula los parámetros de Stokes crudos:
        S0 = I0 + I90
        S1 = I0 - I90
        S2 = 2*I45 - I0 - I90
        S3 = 2*I4590 - I0 - I90
    """
    S0 = I0 + I90
    S1 = I0 - I90
    S2 = 2 * I45 - I0 - I90
    S3 = 2 * I4590 - I0 - I90
    return S0, S1, S2, S3


def normalize_stokes(S0, S1, S2, S3):
    """
    Normaliza los parámetros de Stokes dividiéndolos por S0
    (intensidad total), obteniendo el vector de Stokes normalizado:

        s0 = 1  (por definición)
        s1 = S1 / S0  ∈ [-1, 1]
        s2 = S2 / S0  ∈ [-1, 1]
        s3 = S3 / S0  ∈ [-1, 1]

    Los píxeles donde S0 = 0 se marcan como NaN para evitar
    divisiones por cero.
    """
    S0_safe = S0.copy()
    zero_mask = S0_safe == 0
    S0_safe[zero_mask] = np.finfo(float).eps  # evita división por cero

    s1 = S1 / S0_safe
    s2 = S2 / S0_safe
    s3 = S3 / S0_safe

    # Marcar píxeles sin señal como NaN
    s1[zero_mask] = np.nan
    s2[zero_mask] = np.nan
    s3[zero_mask] = np.nan

    # Clampear al rango físico [-1, 1]
    s1 = np.clip(s1, -1, 1)
    s2 = np.clip(s2, -1, 1)
    s3 = np.clip(s3, -1, 1)

    return s1, s2, s3


def compute_polarization_properties(s1, s2, s3):
    """
    Calcula propiedades de polarización usando los Stokes NORMALIZADOS:

    DoP  = sqrt(s1² + s2² + s3²)       ∈ [0, 1]
    AoLP = 0.5 * atan2(s2, s1)         ∈ [0°, 180°)
    DoCP = |s3|                         ∈ [0, 1]
    χ    = 0.5 * arcsin(s3 / DoP)      ∈ [-45°, 45°]
    """
    DOP = np.sqrt(s1**2 + s2**2 + s3**2)
    DOP = np.clip(DOP, 0, 1)

    AOLP = 0.5 * np.arctan2(s2, s1) * (180.0 / np.pi)
    AOLP = (AOLP + 180.0) % 180.0  # normalizar a [0°, 180°)

    DOCP = np.abs(s3)
    DOCP = np.clip(DOCP, 0, 1)

    # Ángulo de elipticidad
    DOP_safe = DOP.copy()
    DOP_safe[DOP_safe == 0] = np.finfo(float).eps
    chi = 0.5 * np.arcsin(np.clip(s3 / DOP_safe, -1, 1)) * (180.0 / np.pi)
    chi = np.clip(chi, -45.0, 45.0)

    # Enmascarar píxeles no polarizados (DoP < umbral)
    umbral = 0.1
    mask_unpolarized = DOP < umbral
    AOLP[mask_unpolarized] = np.nan
    chi[mask_unpolarized] = np.nan
    DOCP[mask_unpolarized] = 0.0

    return DOP, AOLP, DOCP, chi


def plot_stokes(S0, s1, s2, s3, output_dir):
    """Panel 2×2 con S0 (intensidad total) y s1, s2, s3 normalizados."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Parámetros de Stokes\n(s1, s2, s3 normalizados por S0)',
                 fontsize=14, fontweight='bold')

    datos = [
        (S0,  'S0 — Intensidad Total',       'gray',  None, None),
        (s1,  's1 = S1/S0 — Lineal H/V',     'RdBu',  -1,   1),
        (s2,  's2 = S2/S0 — Lineal Diagonal', 'RdBu', -1,   1),
        (s3,  's3 = S3/S0 — Circular',        'RdBu', -1,   1),
    ]

    for ax, (arr, titulo, cmap, vmin, vmax) in zip(axes.ravel(), datos):
        arr_plot = np.nan_to_num(arr, nan=0.0)
        kwargs = dict(cmap=cmap)
        if vmin is not None:
            kwargs.update(vmin=vmin, vmax=vmax)
        im = ax.imshow(arr_plot, **kwargs)
        plt.colorbar(im, ax=ax)
        ax.set_title(titulo, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(output_dir, 'stokes_normalizados.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {path}")


def plot_polarization_properties(DOP, AOLP, DOCP, chi, output_dir):
    """Panel 2×2 con DoP, AoLP, DoCP y ángulo de elipticidad."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Propiedades de Polarización', fontsize=14, fontweight='bold')

    datos = [
        (DOP,  'DoP — Grado de Polarización',     'viridis',  0,    1,    ''),
        (AOLP, 'AoLP — Ángulo Lineal de Pol.',     'hsv',      0,  180,  '°'),
        (DOCP, 'DoCP — Grado Polarización Circ.',  'plasma',   0,    1,    ''),
        (chi,  'χ — Ángulo de Elipticidad',        'coolwarm', -45,  45,  '°'),
    ]

    for ax, (arr, titulo, cmap, vmin, vmax, unidad) in zip(axes.ravel(), datos):
        arr_plot = np.nan_to_num(arr, nan=0.0)
        im = ax.imshow(arr_plot, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax)
        if unidad:
            cbar.set_label(unidad)
        ax.set_title(titulo, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(output_dir, 'propiedades_polarizacion.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {path}")


def plot_false_color(S0, DOP, AOLP, output_dir):
    """
    Imagen en falso color:
        Matiz (H)     = AoLP  → dirección de polarización lineal
        Saturación (S) = DoP   → qué tan polarizado está el píxel
        Valor (V)     = S0 normalizado → intensidad total
    """
    hsv = np.zeros((*S0.shape, 3))
    aolp_norm = np.nan_to_num(AOLP, nan=0.0) / 180.0
    hsv[..., 0] = aolp_norm
    hsv[..., 1] = DOP
    hsv[..., 2] = np.clip(S0 / S0.max(), 0, 1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv)

    # Píxeles sin polarización → gris neutro
    mask_unpolarized = DOP < 0.1
    rgb[mask_unpolarized] = 0.5

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(rgb)
    ax.set_title('Imagen en Falso Color\n'
                 'Matiz = AoLP | Saturación = DoP | Brillo = S0',
                 fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    path = os.path.join(output_dir, 'imagen_falso_color.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {path}")


def plot_histograms(DOP, AOLP, DOCP, chi, output_dir):
    """Histogramas de DoP, AoLP, DoCP y χ."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle('Distribuciones de Propiedades de Polarización', fontsize=13)

    datos = [
        (DOP,  'DoP',  'steelblue',  (0, 1)),
        (AOLP, 'AoLP (°)', 'darkorange', (0, 180)),
        (DOCP, 'DoCP', 'purple',     (0, 1)),
        (chi,  'χ (°)',   'green',    (-45, 45)),
    ]

    for ax, (arr, xlabel, color, xlim) in zip(axes, datos):
        data = arr[~np.isnan(arr)].ravel()
        ax.hist(data, bins=100, color=color, edgecolor='none', alpha=0.85)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Conteo de píxeles', fontsize=10)
        ax.set_xlim(xlim)
        ax.set_title(f'Distribución de {xlabel}')

    plt.tight_layout()
    path = os.path.join(output_dir, 'histogramas.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {path}")


def print_summary(S0, s1, s2, s3, DOP, AOLP, DOCP, chi):
    """Imprime estadísticas resumidas en consola."""
    print("\n=== RESUMEN ESTADÍSTICO ===")
    print(f"{'Parámetro':<30} {'Mín':>10} {'Máx':>10} {'Media':>10} {'Std':>10}")
    print("-" * 65)
    for nombre, arr in [('S0 (intensidad total)', S0),
                         ('s1 (norm.)', s1), ('s2 (norm.)', s2), ('s3 (norm.)', s3),
                         ('DoP', DOP), ('AoLP (°)', AOLP),
                         ('DoCP', DOCP), ('χ (°)', chi)]:
        data = arr[~np.isnan(arr)] if np.any(np.isnan(arr)) else arr.ravel()
        print(f"{nombre:<30} {data.min():>10.4f} {data.max():>10.4f} "
              f"{data.mean():>10.4f} {data.std():>10.4f}")


# ==============================================================================
# SCRIPT PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("1. Cargando imágenes...")
    imgs = load_images(FILE_PATHS)
    I0, I45, I90, I4590 = imgs['I0'], imgs['I45'], imgs['I90'], imgs['I4590']

    # Verificar que todas las imágenes tienen las mismas dimensiones
    shapes = [img.shape for img in imgs.values()]
    if len(set(shapes)) != 1:
        raise ValueError(f"Las imágenes tienen dimensiones distintas: {shapes}")

    print("\n2. Calculando parámetros de Stokes crudos...")
    S0, S1, S2, S3 = compute_stokes(I0, I45, I90, I4590)

    print("\n3. Normalizando parámetros de Stokes (s_i = S_i / S0)...")
    s1, s2, s3 = normalize_stokes(S0, S1, S2, S3)

    print("\n4. Calculando propiedades de polarización...")
    DOP, AOLP, DOCP, chi = compute_polarization_properties(s1, s2, s3)

    print_summary(S0, s1, s2, s3, DOP, AOLP, DOCP, chi)

    print(f"\n5. Generando figuras en '{OUTPUT_DIR}/'...")
    plot_stokes(S0, s1, s2, s3, OUTPUT_DIR)
    plot_polarization_properties(DOP, AOLP, DOCP, chi, OUTPUT_DIR)
    plot_false_color(S0, DOP, AOLP, OUTPUT_DIR)
    plot_histograms(DOP, AOLP, DOCP, chi, OUTPUT_DIR)

    print("\n¡Listo! Revisa la carpeta", OUTPUT_DIR)
    print("Archivos generados:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  - {f}")
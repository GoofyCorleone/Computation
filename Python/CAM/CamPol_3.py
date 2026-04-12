import numpy as np
import cv2
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# 1) Cargar imágenes BMP y normalizar a float64 en [0, 1]
#    Las imágenes son uint8 (8 bits) con 4 canales (BGRA).
#    Se extrae el canal de luminancia (conversión a escala de grises) y se
#    divide por 255 para obtener image_float con precisión de punto flotante.
# ──────────────────────────────────────────────────────────────────────────────
def cargar_imagen_float(ruta):
    img = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"No se encontró la imagen: {ruta}")
    # Convertir BGRA → escala de grises si tiene más de 1 canal
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if img.shape[2] == 4
                           else cv2.COLOR_BGR2GRAY)
    # Normalizar al rango [0, 1] según bit-depth
    max_val = np.iinfo(img.dtype).max
    image_float = img.astype(np.float64) / max_val
    return image_float

I0   = cargar_imagen_float('I_0_0.bmp')    # polarizador 0°,  retardador 0°
I45  = cargar_imagen_float('I_45_0.bmp')   # polarizador 45°, retardador 0°
I90  = cargar_imagen_float('I_90_0.bmp')   # polarizador 90°, retardador 0°
I4590 = cargar_imagen_float('I_45_90.bmp') # polarizador 45°, retardador 90°

print(f"Resolución: {I0.shape[1]}×{I0.shape[0]} px")
print(f"Rango I0: [{I0.min():.4f}, {I0.max():.4f}]")

# ──────────────────────────────────────────────────────────────────────────────
# 2) Parámetros de Stokes (en flotante, sin pérdida de precisión)
# ──────────────────────────────────────────────────────────────────────────────
S0 = I0 + I90
S1 = I0 - I90
S2 = 2*I45  - I0 - I90
S3 = 2*I4590 - I0 - I90

# ──────────────────────────────────────────────────────────────────────────────
# 3) Máscara por umbral en S0 (descarta píxeles sin señal)
# ──────────────────────────────────────────────────────────────────────────────
umbral = 0.05   # 5% de la intensidad máxima normalizada
mascara = S0 > umbral

# ──────────────────────────────────────────────────────────────────────────────
# 4) Stokes normalizados (s1, s2, s3 ∈ [-1, 1])
# ──────────────────────────────────────────────────────────────────────────────
S0_safe = np.where(mascara, S0, 1.0)   # evitar división por cero
s1 = S1 / S0_safe
s2 = S2 / S0_safe
s3 = S3 / S0_safe

# ──────────────────────────────────────────────────────────────────────────────
# 5) Propiedades de polarización (flotante completo)
# ──────────────────────────────────────────────────────────────────────────────
DoP  = np.sqrt(s1**2 + s2**2 + s3**2)           # Grado de polarización [0, 1]
AoLP = 0.5 * np.arctan2(s2, s1)                  # Ángulo de polarización lineal [rad]
AoLP_deg = np.degrees(AoLP) % 180.0              # → [0°, 180°)
DoCP = np.abs(s3)                                 # Grado de polarización circular [0, 1]

# Enmascarar
DoP[~mascara]     = 0.0
AoLP_deg[~mascara] = 0.0
DoCP[~mascara]    = 0.0

# ──────────────────────────────────────────────────────────────────────────────
# 6) Estadísticas en la región enmascarada
# ──────────────────────────────────────────────────────────────────────────────
print("\nEstado de polarización promedio (región con señal):")
print(f"  s1:   {s1[mascara].mean():.4f}")
print(f"  s2:   {s2[mascara].mean():.4f}")
print(f"  s3:   {s3[mascara].mean():.4f}")
print(f"  DoP:  {DoP[mascara].mean():.4f}")
print(f"  AoLP: {AoLP_deg[mascara].mean():.2f}°")
print(f"  DoCP: {DoCP[mascara].mean():.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 7) Imagen RGB de falso color (R=S1, G=S2, B=S3 reescalados a [0,1])
# ──────────────────────────────────────────────────────────────────────────────
def reescalar_canal(canal):
    cmin, cmax = canal.min(), canal.max()
    if cmax == cmin:
        return np.zeros_like(canal)
    return (canal - cmin) / (cmax - cmin)

S1_rgb = reescalar_canal(s1)
S2_rgb = reescalar_canal(s2)
S3_rgb = reescalar_canal(s3)
RGB_Stokes = np.stack([S1_rgb, S2_rgb, S3_rgb], axis=2)
RGB_Stokes[~mascara] = 0.0

# ──────────────────────────────────────────────────────────────────────────────
# 8) Guardar resultados como PNG (8-bit escalados)
# ──────────────────────────────────────────────────────────────────────────────
def guardar_png(nombre, datos_float, vmin=0.0, vmax=1.0):
    datos_clipped = np.clip((datos_float - vmin) / (vmax - vmin), 0, 1)
    cv2.imwrite(nombre, (datos_clipped * 255).astype(np.uint8))

guardar_png('DoLP_map.png',  DoP)
guardar_png('AoLP_map.png',  AoLP_deg, vmin=0.0, vmax=180.0)
guardar_png('DoCP_map.png',  DoCP)
guardar_png('Mask.png',      mascara.astype(np.float64))
cv2.imwrite('RGB_Stokes.png', (RGB_Stokes[:, :, ::-1] * 255).astype(np.uint8))

# ──────────────────────────────────────────────────────────────────────────────
# 9) Panel resumen con matplotlib
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Polarimetría — Parámetros de Stokes (BMP)", fontsize=13)

im0 = axes[0, 0].imshow(S0, cmap='gray');            axes[0, 0].set_title('S0 (intensidad total)');    plt.colorbar(im0, ax=axes[0, 0])
im1 = axes[0, 1].imshow(s1, cmap='RdBu', vmin=-1, vmax=1); axes[0, 1].set_title('s1 normalizado');    plt.colorbar(im1, ax=axes[0, 1])
im2 = axes[0, 2].imshow(s2, cmap='RdBu', vmin=-1, vmax=1); axes[0, 2].set_title('s2 normalizado');    plt.colorbar(im2, ax=axes[0, 2])
im3 = axes[1, 0].imshow(DoP,      cmap='hot', vmin=0, vmax=1);   axes[1, 0].set_title('DoP');          plt.colorbar(im3, ax=axes[1, 0])
im4 = axes[1, 1].imshow(AoLP_deg, cmap='hsv', vmin=0, vmax=180); axes[1, 1].set_title('AoLP (°)');     plt.colorbar(im4, ax=axes[1, 1])
im5 = axes[1, 2].imshow(RGB_Stokes);                              axes[1, 2].set_title('Falso color (s1/s2/s3)')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig('panel_polarimetria.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nResultados guardados: DoLP_map.png, AoLP_map.png, DoCP_map.png, Mask.png, RGB_Stokes.png, panel_polarimetria.png")

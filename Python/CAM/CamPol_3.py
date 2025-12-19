import numpy as np
import cv2

# 1) Cargar imágenes en 32 bits flotantes
I0 = cv2.imread('CCDI(0,0).tif', cv2.IMREAD_UNCHANGED).astype(np.float32)
I45 = cv2.imread('CCDI(45,0).tif', cv2.IMREAD_UNCHANGED).astype(np.float32)
I90 = cv2.imread('CCDI(90,0).tif', cv2.IMREAD_UNCHANGED).astype(np.float32)
I4590 = cv2.imread('CCDI(45,90).tif', cv2.IMREAD_UNCHANGED).astype(np.float32)

# 2) Calcular parámetros de Stokes
S0 = I0 + I90
S1 = I0 - I90
S2 = 2*I45 - I0 - I90
S3 = 2*I4590 - I0 - I90

# 3) Máscara por umbral en S0
threshold = 10.0
mask = (S0 > threshold).astype(np.uint8)

# 4) Evitar división por cero
S0_nonzero = S0.copy()
S0_nonzero[S0 == 0] = 1.0

# 5) Normalizar Stokes respecto a S0
S0_norm = S0_nonzero / S0_nonzero
S1_norm = S1 / S0_nonzero
S2_norm = S2 / S0_nonzero
S3_norm = S3 / S0_nonzero

# 6) Calcular DoLP y AoLP
DoLP = np.sqrt(S1**2 + S2**2 + S3**2) / S0_nonzero
AoLP = 0.5 * np.arctan2(S2, S1)

# 7) Reescalar para visualizar como 8-bits
DoLP_8bit = np.clip(255 * DoLP, 0, 255).astype(np.uint8)
AoLP_deg = (AoLP * 180.0 / np.pi) % 180.0
AoLP_8bit = np.clip(255 * (AoLP_deg / 180.0), 0, 255).astype(np.uint8)

# 8) Aplicar máscara
DoLP_8bit[mask == 0] = 0
AoLP_8bit[mask == 0] = 0

# 9) Estado de polarización promedio
S0_mean = np.mean(S0_norm[mask == 1])
S1_mean = np.mean(S1_norm[mask == 1])
S2_mean = np.mean(S2_norm[mask == 1])
S3_mean = np.mean(S3_norm[mask == 1])

print("Estado de polarización promedio:")
print(f"S0: {S0_mean:.2f}")
print(f"S1: {S1_mean:.2f}")
print(f"S2: {S2_mean:.2f}")
print(f"S3: {S3_mean:.2f}")

# 10) Construir imagen RGB basada en Stokes normalizados (R=S1, G=S2, B=S3)
# Reescalar S1_norm, S2_norm, S3_norm a rango [0, 255]
def normalize_channel(channel):
    channel_scaled = 255 * (channel - channel.min()) / (channel.max() - channel.min())
    return np.clip(channel_scaled, 0, 255).astype(np.uint8)

S1_rgb = normalize_channel(S1_norm)
S2_rgb = normalize_channel(S2_norm)
S3_rgb = normalize_channel(S3_norm)

# Asegurar que son 2D
S1_rgb = S1_rgb.squeeze()
S2_rgb = S2_rgb.squeeze()
S3_rgb = S3_rgb.squeeze()

RGB_Stokes = cv2.merge((S3_rgb, S2_rgb, S1_rgb))  # BGR en OpenCV

# Aplicar máscara - CORRECCIÓN AQUÍ
# Método 1: Usar indexación booleana directa
RGB_Stokes[mask == 0] = [0, 0, 0]

# 11) Guardar resultados
cv2.imwrite('DoLP_map.png', DoLP_8bit)
cv2.imwrite('AoLP_map.png', AoLP_8bit)
cv2.imwrite('Mask.png', mask * 255)
cv2.imwrite('RGB_Stokes.png', RGB_Stokes)
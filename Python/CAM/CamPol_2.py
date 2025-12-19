import numpy as np
import cv2  # OpenCV, por ejemplo
import glob

# 1) Cargar las 4 imágenes corregidas (bias y flat‐field hechos aparte)
#    Suponen que están en formato TIFF o PNG de 16‐bit, pero aquí
#    imaginamos ya corregidas y guardadas en 8‐bit.
I0   = cv2.imread('CCDI(0,0).tif', cv2.IMREAD_UNCHANGED).astype(np.float32)
I45  = cv2.imread('CCDI(45,0).tif', cv2.IMREAD_UNCHANGED).astype(np.float32)
I90  = cv2.imread('CCDI(90,0).tif', cv2.IMREAD_UNCHANGED).astype(np.float32)
I4590 = cv2.imread('CCDI(45,90).tif', cv2.IMREAD_UNCHANGED).astype(np.float32)

# 2) Calcular Stokes parciales
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

# 5) Calcular DoLP y AoLP en coma flotante
DoLP = np.sqrt(S1**2 + S2**2 + S3**2) / S0_nonzero
AoLP = 0.5 * np.arctan2(S2, S1)  # en radianes

# 6) Reescalar para visualizar como 8‐bits
DoLP_8bit = np.clip(255 * DoLP, 0, 255).astype(np.uint8)
AoLP_deg  = (AoLP * 180.0 / np.pi) % 180.0  # de rad a grados
AoLP_8bit = np.clip(255 * (AoLP_deg / 180.0), 0, 255).astype(np.uint8)

# 7) Aplicar máscara: poner a cero donde mask=0
DoLP_8bit[mask == 0] = 0
AoLP_8bit[mask == 0] = 0

# 8) Guardar resultados
cv2.imwrite('DoLP_map.png', DoLP_8bit)
cv2.imwrite('AoLP_map.png', AoLP_8bit)
cv2.imwrite('Mask.png', mask * 255)
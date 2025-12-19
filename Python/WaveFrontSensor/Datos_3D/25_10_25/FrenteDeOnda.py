import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Leer el archivo CSV con pandas
try:
    # Intentar diferentes codificaciones
    df_raw = pd.read_csv('Horizontal1/0.csv', encoding='latin-1', header=None)
except:
    try:
        df_raw = pd.read_csv('Horizontal1/0.csv', encoding='iso-8859-1', header=None)
    except:
        df_raw = pd.read_csv('Horizontal1/0.csv', encoding='utf-8',  header=None)

# Convertir a lista de líneas para procesar
lines = df_raw[0].astype(str).tolist()

# Encontrar el inicio de la sección WAVEFRONT
start_idx = 0
for i, line in enumerate(lines):
    if '*** WAVEFRONT ***' in line:
        # Buscar la línea que contiene las coordenadas X
        for j in range(i+1, len(lines)):
            if 'y / x [mm]' in lines[j]:
                start_idx = j
                break
        break

# Extraer coordenadas X
x_line = lines[start_idx]
x_parts = [x.strip() for x in x_line.split(',')[1:] if x.strip()]
x_coords = [float(x) for x in x_parts if x.replace('.', '').replace('-', '').isdigit()]

# Extraer coordenadas Y y datos Z
y_coords = []
wavefront_data = []

for i in range(start_idx + 1, len(lines)):
    line = lines[i].strip()
    if not line or '***' in line:
        break
    
    parts = [p.strip() for p in line.split(',')]
    if len(parts) < 2:
        continue
    
    try:
        # Primera columna es la coordenada Y
        y_val = float(parts[0])
        y_coords.append(y_val)
        
        # Procesar los valores Z
        row_data = []
        for val in parts[1:]:
            if not val or val == 'NaN':
                row_data.append(np.nan)
            else:
                try:
                    row_data.append(float(val))
                except:
                    row_data.append(np.nan)
        
        # Ajustar longitud de la fila
        while len(row_data) < len(x_coords):
            row_data.append(np.nan)
        while len(row_data) > len(x_coords):
            row_data.pop()
            
        wavefront_data.append(row_data)
    except ValueError:
        continue

# Crear DataFrame con los datos del wavefront
wavefront_df = pd.DataFrame(wavefront_data, columns=x_coords, index=y_coords)

print("Dimensiones del DataFrame:", wavefront_df.shape)
print("Primeras filas del DataFrame:")
print(wavefront_df.head())

# Crear mallas para el plot 3D
X, Y = np.meshgrid(wavefront_df.columns, wavefront_df.index)
Z = wavefront_df.values

# Configurar el estilo de los plots
plt.style.use('default')

# Plot 3D
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
surf = ax.plot_surface(X, Y, Z, 
                      cmap='viridis', 
                      alpha=0.9,
                      linewidth=0.5, 
                      antialiased=True,
                      rstride=1, 
                      cstride=1)

# Configuraciones del gráfico
ax.set_xlabel('X [mm]', fontsize=12, labelpad=15)
ax.set_ylabel('Y [mm]', fontsize=12, labelpad=15)
ax.set_zlabel('Wavefront [µm]', fontsize=12, labelpad=15)
ax.set_title('Shack-Hartmann Wavefront Surface Analysis\nModel: WFS150-5C', 
             fontsize=14, pad=20)

# Colorbar
cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
cbar.set_label('Wavefront Height [µm]', fontsize=11, rotation=270, labelpad=15)

# Vista y estilo
ax.view_init(elev=25, azim=45)
ax.grid(True, alpha=0.3)

# Información del análisis
info_text = f"""Análisis del Frente de Onda:
PV: 101.1 µm
RMS: 23.8 µm
Beam Diameter X: 3.027 mm
Beam Diameter Y: 3.211 mm"""

ax.text2D(0.02, 0.95, info_text, 
          transform=ax.transAxes, 
          fontsize=10, 
          verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.show()

# Visualizaciones 2D complementarias
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Contour plot
contour1 = axes[0,0].contourf(X, Y, Z, levels=50, cmap='viridis')
axes[0,0].set_xlabel('X [mm]', fontsize=11)
axes[0,0].set_ylabel('Y [mm]', fontsize=11)
axes[0,0].set_title('Contour Plot del Frente de Onda', fontsize=12)
axes[0,0].set_aspect('equal')
plt.colorbar(contour1, ax=axes[0,0], label='Wavefront [µm]')

# Heatmap
im = axes[0,1].imshow(Z, 
                     extent=[min(x_coords), max(x_coords), min(y_coords), max(y_coords)], 
                     cmap='viridis', 
                     origin='lower', 
                     aspect='auto')
axes[0,1].set_xlabel('X [mm]', fontsize=11)
axes[0,1].set_ylabel('Y [mm]', fontsize=11)
axes[0,1].set_title('Heatmap del Frente de Onda', fontsize=12)
plt.colorbar(im, ax=axes[0,1], label='Wavefront [µm]')

# Surface plot alternativo
axes[1,0].remove()
ax_3d = fig.add_subplot(2, 2, 3, projection='3d')
surf2 = ax_3d.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8)
ax_3d.set_xlabel('X [mm]', fontsize=11)
ax_3d.set_ylabel('Y [mm]', fontsize=11)
ax_3d.set_zlabel('Wavefront [µm]', fontsize=11)
ax_3d.set_title('Vista 3D Alternativa', fontsize=12)
ax_3d.view_init(elev=15, azim=60)

# Gráfico de perfil horizontal (corte central)
center_idx = len(y_coords) // 2
axes[1,1].plot(x_coords, Z[center_idx, :], 'b-', linewidth=2, label='Perfil horizontal')
axes[1,1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1,1].set_xlabel('X [mm]', fontsize=11)
axes[1,1].set_ylabel('Wavefront [µm]', fontsize=11)
axes[1,1].set_title(f'Perfil en Y = {y_coords[center_idx]:.2f} mm', fontsize=12)
axes[1,1].grid(True, alpha=0.3)
axes[1,1].legend()

plt.tight_layout()
plt.show()

# Análisis estadístico adicional
print("\n" + "="*50)
print("ANÁLISIS ESTADÍSTICO DEL FRENTE DE ONDA")
print("="*50)

# Crear Series de pandas para análisis
z_flat = pd.Series(Z.flatten())
z_clean = z_flat.dropna()

print(f"Rango de datos X: {min(x_coords):.3f} a {max(x_coords):.3f} mm")
print(f"Rango de datos Y: {min(y_coords):.3f} a {max(y_coords):.3f} mm")
print(f"Altura mínima del wavefront: {z_clean.min():.2f} µm")
print(f"Altura máxima del wavefront: {z_clean.max():.2f} µm")
print(f"Variación PV (Pico-Valle): {z_clean.max() - z_clean.min():.2f} µm")
print(f"Valor RMS: {z_clean.std():.2f} µm")
print(f"Valor medio: {z_clean.mean():.2f} µm")
print(f"Número de puntos válidos: {len(z_clean)} de {len(z_flat)} totales")
print(f"Porcentaje de área medida: {(len(z_clean)/len(z_flat)*100):.1f}%")

# Mostrar información del DataFrame
print("\nINFORMACIÓN DEL DATAFRAME:")
print(f"Shape: {wavefront_df.shape}")
print(f"Columnas (X): {len(wavefront_df.columns)} valores")
print(f"Filas (Y): {len(wavefront_df.index)} valores")
print("\nPrimeras 5 coordenadas X:", [f"{x:.3f}" for x in wavefront_df.columns[:5]])
print("Primeras 5 coordenadas Y:", [f"{y:.3f}" for y in wavefront_df.index[:5]])
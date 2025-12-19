import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy import interpolate

# Configuración de matplotlib para español y mejor calidad
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = [12, 8]

def analizar_aberracciones_shack_hartmann(archivo_csv):
    """
    Analiza datos de un sensor Shack-Hartmann y determina las aberraciones
    """
    
    # Leer el archivo CSV
    with open(archivo_csv, 'r') as f:
        lineas = f.readlines()
    
    # Extraer coeficientes de Zernike
    coeficientes = []
    en_seccion_zernike = False
    
    for linea in lineas:
        if '*** ZERNIKE FIT ***' in linea:
            en_seccion_zernike = True
            continue
        if '***' in linea and en_seccion_zernike:
            break
        if en_seccion_zernike and ',' in linea and 'Index' not in linea:
            partes = linea.split(',')
            if len(partes) >= 4:
                try:
                    coef = float(partes[3].strip())
                    coeficientes.append(coef)
                except:
                    continue
    
    # Nombres de las aberraciones de Zernike (OSA/ANSI standard)
    nombres_aberracciones = [
        "Pistón",                   # Z0 (no afecta calidad óptica)
        "Inclinación X (Tip)",      # Z1
        "Inclinación Y (Tilt)",     # Z2  
        "Desenfoque",               # Z3
        "Astigmatismo a 0°",        # Z4
        "Astigmatismo a 45°",       # Z5
        "Coma X",                   # Z6
        "Coma Y",                   # Z7
        "Aberración Esférica",      # Z8
        "Trefoil X",                # Z9
        "Trefoil Y",                # Z10
        "Astigmatismo Secundario a 0°",  # Z11
        "Astigmatismo Secundario a 45°", # Z12
        "Esférica Secundaria",      # Z13
        "Tetrafoil"                 # Z14
    ]
    
    # Extraer datos del wavefront
    datos_wavefront = []
    en_seccion_wavefront = False
    coordenadas_x = None
    
    for i, linea in enumerate(lineas):
        if '*** WAVEFRONT ***' in linea:
            en_seccion_wavefront = True
            continue
        if en_seccion_wavefront and 'y / x [mm]' in linea:
            # Extraer coordenadas X
            partes = linea.split(',')
            coordenadas_x = [float(x.strip()) for x in partes[1:] if x.strip()]
            continue
        if en_seccion_wavefront and ',' in linea and 'y / x [mm]' not in linea:
            partes = linea.split(',')
            if len(partes) == len(coordenadas_x) + 1:
                try:
                    y_val = float(partes[0].strip())
                    valores_z = [float(z.strip()) for z in partes[1:]]
                    datos_wavefront.append((y_val, valores_z))
                except:
                    continue
    
    # Convertir a arrays numpy
    y_coords = np.array([d[0] for d in datos_wavefront])
    x_coords = np.array(coordenadas_x)
    Z_medido = np.array([d[1] for d in datos_wavefront])
    
    # Crear malla para interpolación
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # =========================================================================
    # ANÁLISIS Y VISUALIZACIÓN
    # =========================================================================
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. MAPA DE FRENTE DE ONDA MEDIDO
    ax1 = fig.add_subplot(231)
    contour1 = ax1.contourf(X, Y, Z_medido, levels=50, cmap='seismic')
    ax1.set_xlabel('X [mm]')
    ax1.set_ylabel('Y [mm]')
    ax1.set_title('Mapa del Frente de Onda Medido')
    ax1.set_aspect('equal')
    plt.colorbar(contour1, ax=ax1, label='Desviación [µm]')
    
    # 2. GRÁFICO 3D DEL FRENTE DE ONDA
    ax2 = fig.add_subplot(232, projection='3d')
    surf = ax2.plot_surface(X, Y, Z_medido, cmap='seismic', 
                          linewidth=0, antialiased=True, alpha=0.8)
    ax2.set_xlabel('X [mm]')
    ax2.set_ylabel('Y [mm]')
    ax2.set_zlabel('Desviación [µm]')
    ax2.set_title('Frente de Onda 3D')
    
    # 3. COEFICIENTES DE ZERNIKE (BARRAS)
    ax3 = fig.add_subplot(233)
    indices = range(len(coeficientes))
    barras = ax3.bar(indices, coeficientes, color='skyblue', alpha=0.7)
    ax3.set_xlabel('Término de Zernike')
    ax3.set_ylabel('Coeficiente [µm]')
    ax3.set_title('Coeficientes de Zernike')
    ax3.grid(True, alpha=0.3)
    
    # Colorear barras significativas
    for i, (bar, coef) in enumerate(zip(barras, coeficientes)):
        if abs(coef) > 1.0:  # Resaltar coeficientes > 1µm
            bar.set_color('red')
            bar.set_alpha(0.8)
    
    # 4. ANÁLISIS DE ABERRACIONES PRINCIPALES
    ax4 = fig.add_subplot(234)
    aberraciones_significativas = []
    valores_significativos = []
    
    for i, (nombre, coef) in enumerate(zip(nombres_aberracciones, coeficientes)):
        if abs(coef) > 0.1:  # Solo considerar coeficientes > 0.1µm
            aberraciones_significativas.append(nombre)
            valores_significativos.append(coef)
    
    if aberraciones_significativas:
        colores = ['red' if abs(v) > 1.0 else 'blue' for v in valores_significativos]
        bars = ax4.barh(aberracciones_significativas, valores_significativos, color=colores, alpha=0.7)
        ax4.set_xlabel('Coeficiente [µm]')
        ax4.set_title('Aberraciones Significativas')
        ax4.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar, valor in zip(bars, valores_significativos):
            ax4.text(bar.get_width() + 0.01 * max(valores_significativos), 
                    bar.get_y() + bar.get_height()/2, 
                    f'{valor:.2f} µm', ha='left', va='center')
    else:
        ax4.text(0.5, 0.5, 'No hay aberraciones\nsignificativas (> 0.1 µm)', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Aberraciones Significativas')
    
    # 5. DISTRIBUCIÓN RADIAL (para ver simetría)
    ax5 = fig.add_subplot(235)
    r = np.sqrt(X**2 + Y**2)
    ax5.scatter(r.flatten(), Z_medido.flatten(), alpha=0.5, s=1)
    ax5.set_xlabel('Radio [mm]')
    ax5.set_ylabel('Desviación [µm]')
    ax5.set_title('Distribución Radial del Frente de Onda')
    ax5.grid(True, alpha=0.3)
    
    # 6. RESUMEN NUMÉRICO
    ax6 = fig.add_subplot(236)
    ax6.axis('off')
    
    # Información del análisis
    pv = np.max(Z_medido) - np.min(Z_medido)  # Peak-to-Valley
    rms = np.std(Z_medido)  # RMS
    
    info_text = f"""
    RESUMEN DEL ANÁLISIS
    
    Métricas del Frente de Onda:
    - PV (Peak-to-Valley): {pv:.2f} µm
    - RMS: {rms:.2f} µm
    - Diámetro de Pupila: {x_coords[-1]-x_coords[0]:.1f} mm
    
    Aberraciones Principales:
    """
    
    # Añadir aberraciones principales al resumen
    for i, coef in enumerate(coeficientes):
        if abs(coef) > 1.0:
            info_text += f"\n- {nombres_aberracciones[i]}: {coef:.2f} µm"
    
    if not any(abs(coef) > 1.0 for coef in coeficientes):
        info_text += "\n- No hay aberraciones dominantes (> 1 µm)"
    
    ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes, 
             fontfamily='monospace', verticalalignment='top', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # =========================================================================
    # IMPRIMIR ANÁLISIS DETALLADO EN CONSOLA
    # =========================================================================
    
    print("=" * 60)
    print("ANÁLISIS DETALLADO DE ABERRACIONES - SENSOR SHACK-HARTMANN")
    print("=" * 60)
    
    print(f"\nINFORMACIÓN GENERAL:")
    print(f"PV (Peak-to-Valley): {pv:.2f} µm")
    print(f"RMS: {rms:.2f} µm")
    
    print(f"\nCOEFICIENTES DE ZERNIKE:")
    print("-" * 50)
    for i, (nombre, coef) in enumerate(zip(nombres_aberracciones, coeficientes)):
        significado = ""
        if abs(coef) > 3.0:
            significado = " ← DOMINANTE"
        elif abs(coef) > 1.0:
            significado = " ← SIGNIFICATIVO"
        
        print(f"Z{i:2d}: {nombre:30} {coef:8.3f} µm{significado}")
    
    print(f"\nINTERPRETACIÓN:")
    print("-" * 50)
    
    # Análisis de aberraciones específicas
    if abs(coeficientes[3]) > 1.0 or abs(coeficientes[4]) > 1.0 or abs(coeficientes[5]) > 1.0:
        print("• La lente presenta ASTIGMATISMO significativo")
        print("  (diferente potencia en meridianos ortogonales)")
    
    if abs(coeficientes[6]) > 1.0 or abs(coeficientes[7]) > 1.0:
        print("• Se detecta COMA (asimetría en la formación de imagen)")
    
    if abs(coeficientes[8]) > 1.0:
        print("• ABERRACIÓN ESFÉRICA presente")
        print("  (enfoque diferente para rayos centrales vs marginales)")
    
    if abs(coeficientes[0]) > 10.0:
        print("• Pistón elevado - posible error de referencia")
    
    # Recomendación general basada en RMS
    if rms < 0.5:
        print("• CALIDAD ÓPTICA: Excelente")
    elif rms < 1.0:
        print("• CALIDAD ÓPTICA: Buena")
    elif rms < 2.0:
        print("• CALIDAD ÓPTICA: Aceptable")
    else:
        print("• CALIDAD ÓPTICA: Pobre - considerar recalibración")
    
    return {
        'coeficientes': coeficientes,
        'nombres_aberracciones': nombres_aberracciones,
        'wavefront': Z_medido,
        'X': X,
        'Y': Y,
        'pv': pv,
        'rms': rms
    }

# EJECUTAR EL ANÁLISIS
if __name__ == "__main__":
    # Reemplaza con la ruta a tu archivo CSV
    archivo = "lenteThorlabs.csv"
    
    try:
        resultados = analizar_aberracciones_shack_hartmann(archivo)
        print(f"\nAnálisis completado exitosamente!")
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo {archivo}")
        print("Asegúrate de que el archivo está en el directorio correcto.")
    except Exception as e:
        print(f"Error durante el análisis: {str(e)}")
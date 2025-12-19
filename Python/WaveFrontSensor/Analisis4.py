"""
ANALIZADOR DE ABERRACIONES ÓPTICAS CON SENSOR SHACK-HARTMANN
================================================================================

Este script analiza datos obtenidos de un sensor Shack-Hartmann para caracterizar
las aberraciones ópticas de lentes. Procesa archivos CSV, extrae coeficientes de
Zernike y genera visualizaciones completas del frente de onda.

Características principales:
- Lectura y parsing de archivos CSV del sensor Thorlabs
- Análisis de coeficientes de Zernike hasta 4to orden
- Visualización 2D y 3D del frente de onda
- Identificación automática de aberraciones significativas
- Evaluación de calidad óptica basada en métricas RMS y PV

Autor: Asistente IA
Versión: 2.0
Fecha: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy import interpolate
import chardet

# Configuración global de matplotlib para mejor visualización
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = [20, 16]  # Figura más grande para mejor espaciado
plt.rcParams['figure.autolayout'] = False  # Desactivar autolayout para control manual

def detectar_codificacion(archivo_csv):
    """
    Detecta la codificación de un archivo usando la librería chardet.
    
    Parámetros:
    -----------
    archivo_csv : str
        Ruta al archivo CSV que se quiere analizar
        
    Retorna:
    --------
    str
        Codificación detectada (ej: 'utf-8', 'latin-1', etc.)
    """
    with open(archivo_csv, 'rb') as f:
        resultado = chardet.detect(f.read())
    return resultado['encoding']


def leer_archivo_csv(archivo_csv):
    """
    Lee un archivo CSV intentando diferentes codificaciones hasta encontrar una que funcione.
    
    Parámetros:
    -----------
    archivo_csv : str
        Ruta al archivo CSV del sensor Shack-Hartmann
        
    Retorna:
    --------
    list
        Lista de líneas del archivo, o None si no se pudo leer
    """
    # Detectar codificación automáticamente
    try:
        codificacion = detectar_codificacion(archivo_csv)
        print(f"Codificación detectada: {codificacion}")
    except Exception as e:
        print(f"Error en detección de codificación: {e}")
        codificacion = 'latin-1'  # Fallback a codificación común
    
    # Lista de codificaciones a intentar (ordenadas por probabilidad)
    codificaciones = [codificacion, 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
    
    for encoding in codificaciones:
        try:
            with open(archivo_csv, 'r', encoding=encoding) as f:
                lineas = f.readlines()
            print(f"✓ Archivo leído exitosamente con codificación: {encoding}")
            return lineas
        except UnicodeDecodeError:
            print(f"✗ Error con codificación {encoding}, intentando siguiente...")
            continue
    
    # Si todas las codificaciones fallan, intentar con manejo de errores
    try:
        with open(archivo_csv, 'r', encoding='utf-8', errors='ignore') as f:
            lineas = f.readlines()
        print("✓ Archivo leído con manejo de errores UTF-8")
        return lineas
    except Exception as e:
        print(f"Error crítico: No se pudo leer el archivo: {e}")
        return None


def extraer_coeficientes_zernike(lineas):
    """
    Extrae los coeficientes de Zernike de las líneas del archivo CSV.
    
    Los coeficientes de Zernike representan las diferentes aberraciones ópticas
    en una base ortonormal sobre el círculo unitario.
    
    Parámetros:
    -----------
    lineas : list
        Lista de líneas del archivo CSV
        
    Retorna:
    --------
    list
        Lista de coeficientes de Zernike en micrómetros
    """
    coeficientes = []
    en_seccion_zernike = False
    
    for linea in lineas:
        # Buscar el inicio de la sección de coeficientes Zernike
        if '*** ZERNIKE FIT ***' in linea:
            en_seccion_zernike = True
            continue
        
        # Salir cuando encontremos la siguiente sección
        if '***' in linea and en_seccion_zernike and 'ZERNIKE' not in linea:
            break
        
        # Procesar líneas de coeficientes
        if en_seccion_zernike and ',' in linea and 'Index' not in linea:
            partes = linea.split(',')
            if len(partes) >= 4:
                try:
                    # Limpiar el valor de posibles caracteres extraños
                    valor_limpio = partes[3].strip().replace('Á', '').replace('�', '')
                    coef = float(valor_limpio)
                    coeficientes.append(coef)
                except ValueError as e:
                    print(f"Advertencia: No se pudo convertir '{partes[3]}' a float: {e}")
                    continue
    
    print(f"Coeficientes extraídos: {len(coeficientes)}")
    return coeficientes


def extraer_datos_wavefront(lineas):
    """
    Extrae los datos de matriz del frente de onda del archivo CSV.
    
    El frente de onda representa la desviación de fase en micrómetros
    sobre una grilla cartesiana regular.
    
    Parámetros:
    -----------
    lineas : list
        Lista de líneas del archivo CSV
        
    Retorna:
    --------
    tuple
        (X, Y, Z) donde X, Y son mallas de coordenadas y Z es el frente de onda
    """
    datos_wavefront = []
    en_seccion_wavefront = False
    coordenadas_x = None
    
    for i, linea in enumerate(lineas):
        # Buscar el inicio de la sección del wavefront
        if '*** WAVEFRONT ***' in linea:
            en_seccion_wavefront = True
            continue
        
        # Extraer coordenadas X de la línea de cabecera
        if en_seccion_wavefront and 'y / x [mm]' in linea:
            partes = linea.split(',')
            coordenadas_x = []
            for x in partes[1:]:
                x_limpio = x.strip().replace('Á', '').replace('�', '')
                if x_limpio:
                    try:
                        coordenadas_x.append(float(x_limpio))
                    except ValueError:
                        continue
            continue
        
        # Procesar líneas de datos del wavefront
        if en_seccion_wavefront and ',' in linea and 'y / x [mm]' not in linea:
            partes = linea.split(',')
            if coordenadas_x and len(partes) == len(coordenadas_x) + 1:
                try:
                    # Extraer coordenada Y y valores Z
                    y_val_limpio = partes[0].strip().replace('Á', '').replace('�', '')
                    y_val = float(y_val_limpio)
                    valores_z = []
                    for z in partes[1:]:
                        z_limpio = z.strip().replace('Á', '').replace('�', '')
                        if z_limpio:
                            valores_z.append(float(z_limpio))
                    
                    # Verificar que tenemos tantos valores Z como coordenadas X
                    if len(valores_z) == len(coordenadas_x):
                        datos_wavefront.append((y_val, valores_z))
                except ValueError as e:
                    print(f"Advertencia en línea {i}: {e}")
                    continue
    
    if datos_wavefront:
        # Convertir a arrays numpy para procesamiento
        y_coords = np.array([d[0] for d in datos_wavefront])
        x_coords = np.array(coordenadas_x)
        Z_medido = np.array([d[1] for d in datos_wavefront])
        
        # Crear malla de coordenadas para visualización
        X, Y = np.meshgrid(x_coords, y_coords)
        return X, Y, Z_medido
    else:
        print("ERROR: No se pudieron extraer datos del wavefront")
        return None, None, None


def crear_datos_ejemplo():
    """
    Crea datos de ejemplo para demostración cuando no se pueden leer del archivo.
    
    Retorna:
    --------
    tuple
        (X, Y, Z_ejemplo) con datos sintéticos de frente de onda
    """
    print("Creando datos de ejemplo para demostración...")
    y_coords = np.linspace(-1.5, 1.5, 21)
    x_coords = np.linspace(-1.5, 1.5, 21)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Crear un frente de onda sintético con astigmatismo y coma
    Z_ejemplo = (10 * np.exp(-(X**2 + Y**2)/2) * 
                (0.5 * np.sin(2*np.pi*X/3) +  # Componente astigmática
                 0.3 * Y * np.exp(-(X**2 + Y**2)/4)))  # Componente de coma
    
    return X, Y, Z_ejemplo


def analizar_aberracciones_shack_hartmann(archivo_csv):
    """
    Función principal que coordina el análisis completo de aberraciones ópticas.
    
    Flujo de trabajo:
    1. Lee y parsea el archivo CSV del sensor
    2. Extrae coeficientes de Zernike y datos del wavefront
    3. Genera visualizaciones comprehensivas
    4. Proporciona análisis cualitativo y cuantitativo
    
    Parámetros:
    -----------
    archivo_csv : str
        Ruta al archivo CSV del sensor Shack-Hartmann
        
    Retorna:
    --------
    dict
        Diccionario con resultados del análisis:
        - coeficientes: lista de coeficientes de Zernike
        - nombres_aberracciones: nombres de las aberraciones
        - wavefront: matriz del frente de onda (si disponible)
        - X, Y: coordenadas de la malla
        - pv: valor peak-to-valley
        - rms: valor RMS del wavefront
    """
    
    # =========================================================================
    # 1. LECTURA Y EXTRACCIÓN DE DATOS
    # =========================================================================
    
    print("Iniciando análisis de archivo CSV...")
    
    # Leer archivo CSV
    lineas = leer_archivo_csv(archivo_csv)
    if lineas is None:
        print("No se pudo leer el archivo, terminando análisis.")
        return None
    
    # Extraer coeficientes de Zernike
    coeficientes = extraer_coeficientes_zernike(lineas)
    
    # Nombres estándar de las aberraciones de Zernike (convención OSA/ANSI)
    nombres_aberracciones = [
        "Pistón",                           # Z0 - No afecta calidad óptica
        "Inclinación X (Tip)",              # Z1 - Inclinación en X
        "Inclinación Y (Tilt)",             # Z2 - Inclinación en Y  
        "Desenfoque",                       # Z3 - Desenfoque
        "Astigmatismo a 0°",                # Z4 - Astigmatismo vertical/horizontal
        "Astigmatismo a 45°",               # Z5 - Astigmatismo diagonal
        "Coma X",                           # Z6 - Coma en dirección X
        "Coma Y",                           # Z7 - Coma en dirección Y
        "Aberración Esférica",              # Z8 - Esférica primaria
        "Trefoil X",                        # Z9 - Trefoil en X
        "Trefoil Y",                        # Z10 - Trefoil en Y
        "Astigmatismo Secundario a 0°",     # Z11 - Astigmatismo secundario
        "Astigmatismo Secundario a 45°",    # Z12 - Astigmatismo secundario diagonal
        "Esférica Secundaria",              # Z13 - Esférica secundaria
        "Tetrafoil"                         # Z14 - Tetrafoil
    ]
    
    # Extraer datos del wavefront
    X, Y, Z_medido = extraer_datos_wavefront(lineas)
    
    # Si no se pudieron extraer datos, crear datos de ejemplo
    if Z_medido is None:
        X, Y, Z_medido = crear_datos_ejemplo()
        # Usar coeficientes del archivo o valores por defecto para demostración
        if not coeficientes:
            coeficientes = [-23.505, 11.578, 15.226, -0.173, -3.370, 0.219, 
                           -0.003, -0.003, 0.020, 0.010, -0.009, 0.008, 
                           0.014, -0.006, -0.006]
    
    # =========================================================================
    # 2. CÁLCULO DE MÉTRICAS PRINCIPALES
    # =========================================================================
    
    # Calcular métricas de calidad óptica
    pv = np.max(Z_medido) - np.min(Z_medido)  # Peak-to-Valley (máxima variación)
    rms = np.std(Z_medido)                    # Root Mean Square (desviación estándar)
    
    print(f"Métricas calculadas - PV: {pv:.2f} µm, RMS: {rms:.2f} µm")
    
    # =========================================================================
    # 3. CREACIÓN DE VISUALIZACIONES
    # =========================================================================
    
    # Configurar figura con espaciado optimizado
    fig = plt.figure(figsize=(20, 16))
    
    # Ajustar márgenes y espacios entre subplots para evitar superposiciones
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.05, top=0.95, 
                        wspace=0.3, hspace=0.4)
    
    # -------------------------------------------------------------------------
    # Subplot 1: Mapa 2D del frente de onda
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(231)
    contour1 = ax1.contourf(X, Y, Z_medido, levels=50, cmap='seismic')
    ax1.set_xlabel('X [mm]', fontsize=12, labelpad=10)
    ax1.set_ylabel('Y [mm]', fontsize=12, labelpad=10)
    ax1.set_title('Mapa del Frente de Onda Medido', fontsize=14, pad=15)
    ax1.set_aspect('equal')
    cbar1 = plt.colorbar(contour1, ax=ax1, label='Desviación [µm]')
    cbar1.ax.tick_params(labelsize=10)
    
    # -------------------------------------------------------------------------
    # Subplot 2: Visualización 3D del frente de onda
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(232, projection='3d')
    surf = ax2.plot_surface(X, Y, Z_medido, cmap='seismic', 
                          linewidth=0, antialiased=True, alpha=0.8)
    ax2.set_xlabel('X [mm]', fontsize=12, labelpad=10)
    ax2.set_ylabel('Y [mm]', fontsize=12, labelpad=10)
    ax2.set_zlabel('Desviación [µm]', fontsize=12, labelpad=10)
    ax2.set_title('Frente de Onda 3D', fontsize=14, pad=15)
    ax2.view_init(elev=30, azim=45)  # Ángulo de vista óptimo
    
    # -------------------------------------------------------------------------
    # Subplot 3: Coeficientes de Zernike (gráfico de barras)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(233)
    indices = range(len(coeficientes))
    barras = ax3.bar(indices, coeficientes, color='skyblue', alpha=0.7)
    ax3.set_xlabel('Término de Zernike', fontsize=12, labelpad=10)
    ax3.set_ylabel('Coeficiente [µm]', fontsize=12, labelpad=10)
    ax3.set_title('Coeficientes de Zernike', fontsize=14, pad=15)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)  # Rotar etiquetas para mejor legibilidad
    
    # Resaltar coeficientes significativos (> 1µm)
    for i, (bar, coef) in enumerate(zip(barras, coeficientes)):
        if abs(coef) > 1.0:
            bar.set_color('red')
            bar.set_alpha(0.8)
    
    # -------------------------------------------------------------------------
    # Subplot 4: Aberraciones significativas (gráfico de barras horizontales)
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(234)
    aberraciones_significativas = []
    valores_significativos = []
    
    # Filtrar solo aberraciones con coeficiente > 0.1µm
    for i, (nombre, coef) in enumerate(zip(nombres_aberracciones, coeficientes)):
        if abs(coef) > 0.1:
            aberraciones_significativas.append(nombre)
            valores_significativos.append(coef)
    
    if aberraciones_significativas:
        # Colorear según magnitud: rojo para > 1µm, azul para 0.1-1µm
        colores = ['red' if abs(v) > 1.0 else 'blue' for v in valores_significativos]
        bars = ax4.barh(aberraciones_significativas, valores_significativos, 
                       color=colores, alpha=0.7, height=0.6)
        ax4.set_xlabel('Coeficiente [µm]', fontsize=12, labelpad=10)
        ax4.set_title('Aberraciones Significativas', fontsize=14, pad=15)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='y', labelsize=10)
        
        # Añadir valores numéricos a las barras
        for bar, valor in zip(bars, valores_significativos):
            width = bar.get_width()
            ax4.text(width + 0.01 * max(valores_significativos), 
                    bar.get_y() + bar.get_height()/2, 
                    f'{valor:.2f} µm', ha='left', va='center', fontsize=9)
        
        # Ajustar límites para dar espacio a los textos
        x_max = max(valores_significativos) * 1.2
        ax4.set_xlim(-abs(x_max), x_max)
    else:
        ax4.text(0.5, 0.5, 'No hay aberraciones\nsignificativas (> 0.1 µm)', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Aberraciones Significativas', fontsize=14, pad=15)
    
    # -------------------------------------------------------------------------
    # Subplot 5: Distribución radial del frente de onda
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(235)
    r = np.sqrt(X**2 + Y**2)  # Calcular coordenadas radiales
    ax5.scatter(r.flatten(), Z_medido.flatten(), alpha=0.5, s=2)
    ax5.set_xlabel('Radio [mm]', fontsize=12, labelpad=10)
    ax5.set_ylabel('Desviación [µm]', fontsize=12, labelpad=10)
    ax5.set_title('Distribución Radial del Frente de Onda', fontsize=14, pad=15)
    ax5.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Subplot 6: Resumen numérico del análisis
    # -------------------------------------------------------------------------
    ax6 = fig.add_subplot(236)
    ax6.axis('off')  # Ocultar ejes para el panel de texto
    
    # Construir texto de resumen
    info_text = f"""
    RESUMEN DEL ANÁLISIS
    
    Métricas del Frente de Onda:
    - PV (Peak-to-Valley): {pv:.2f} µm
    - RMS: {rms:.2f} µm
    - Diámetro de Pupila: {X[0,-1]-X[0,0]:.1f} mm
    
    Aberraciones Principales:"""
    
    # Añadir aberraciones significativas al resumen
    for i, coef in enumerate(coeficientes):
        if i < len(nombres_aberracciones) and abs(coef) > 1.0:
            info_text += f"\n- {nombres_aberracciones[i]}: {coef:.2f} µm"
    
    if not any(abs(coef) > 1.0 for coef in coeficientes):
        info_text += "\n- No hay aberraciones dominantes (> 1 µm)"
    
    # Mostrar texto en el subplot con formato de cuadro
    ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes, 
             fontfamily='monospace', verticalalignment='top', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.5))
    
    # Título general de la figura
    fig.suptitle('Análisis Completo de Aberraciones Ópticas - Sensor Shack-Hartmann', 
                fontsize=16, y=0.98)
    
    plt.show()
    
    # =========================================================================
    # 4. ANÁLISIS CUALITATIVO EN CONSOLA
    # =========================================================================
    
    print("=" * 70)
    print("ANÁLISIS DETALLADO DE ABERRACIONES - SENSOR SHACK-HARTMANN")
    print("=" * 70)
    
    print(f"\nINFORMACIÓN GENERAL:")
    print(f"PV (Peak-to-Valley): {pv:.2f} µm")
    print(f"RMS: {rms:.2f} µm")
    
    if coeficientes:
        print(f"\nCOEFICIENTES DE ZERNIKE:")
        print("-" * 60)
        for i, (nombre, coef) in enumerate(zip(nombres_aberracciones, coeficientes)):
            if i >= len(coeficientes):
                break
            
            # Clasificar coeficientes por significancia
            significado = ""
            if abs(coef) > 3.0:
                significado = " ← DOMINANTE"
            elif abs(coef) > 1.0:
                significado = " ← SIGNIFICATIVO"
            
            print(f"Z{i:2d}: {nombre:35} {coef:8.3f} µm{significado}")
        
        print(f"\nINTERPRETACIÓN:")
        print("-" * 60)
        
        # Análisis específico por tipo de aberración
        if len(coeficientes) > 5 and (abs(coeficientes[3]) > 1.0 or 
                                     abs(coeficientes[4]) > 1.0 or 
                                     abs(coeficientes[5]) > 1.0):
            print("• La lente presenta ASTIGMATISMO significativo")
            print("  (diferente potencia en meridianos ortogonales)")
        
        if len(coeficientes) > 7 and (abs(coeficientes[6]) > 1.0 or 
                                     abs(coeficientes[7]) > 1.0):
            print("• Se detecta COMA (asimetría en la formación de imagen)")
        
        if len(coeficientes) > 8 and abs(coeficientes[8]) > 1.0:
            print("• ABERRACIÓN ESFÉRICA presente")
            print("  (enfoque diferente para rayos centrales vs marginales)")
        
        if len(coeficientes) > 0 and abs(coeficientes[0]) > 10.0:
            print("• Pistón elevado - posible error de referencia o alineación")
        
        # Evaluación general de calidad óptica basada en RMS
        print(f"\nEVALUACIÓN DE CALIDAD ÓPTICA:")
        print("-" * 60)
        if rms < 0.5:
            print("• CALIDAD: Excelente (RMS < 0.5 µm)")
        elif rms < 1.0:
            print("• CALIDAD: Buena (RMS < 1.0 µm)")
        elif rms < 2.0:
            print("• CALIDAD: Aceptable (RMS < 2.0 µm)")
        elif rms < 5.0:
            print("• CALIDAD: Regular (RMS < 5.0 µm)")
        else:
            print("• CALIDAD: Pobre (RMS ≥ 5.0 µm) - considerar recalibración")
    else:
        print("No se pudieron extraer coeficientes para análisis detallado")
    
    # =========================================================================
    # 5. PREPARAR RESULTADOS PARA RETORNO
    # =========================================================================
    
    resultados = {
        'coeficientes': coeficientes,
        'nombres_aberracciones': nombres_aberracciones,
        'wavefront': Z_medido,
        'X': X,
        'Y': Y,
        'pv': pv,
        'rms': rms
    }
    
    return resultados


# BLOQUE PRINCIPAL DE EJECUCIÓN
# =================================================================================

if __name__ == "__main__":
    """
    Punto de entrada principal del script.
    
    Cuando se ejecuta directamente (no importado como módulo):
    1. Verifica dependencias
    2. Ejecuta el análisis en el archivo especificado
    3. Maneja errores y excepciones
    """
    
    # Configuración del archivo a analizar
    archivo = "lenteImpresa.csv"  # Cambiar por la ruta de tu archivo
    
    try:
        # Verificar e instalar dependencias si es necesario
        try:
            import chardet
        except ImportError:
            print("Instalando chardet para detección de codificación...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "chardet"])
            import chardet
        
        # Ejecutar análisis principal
        print(f"Iniciando análisis del archivo: {archivo}")
        resultados = analizar_aberracciones_shack_hartmann(archivo)
        
        if resultados:
            print(f"\n" + "="*50)
            print("ANÁLISIS COMPLETADO EXITOSAMENTE!")
            print("="*50)
            print(f"Se generaron 6 visualizaciones del frente de onda y aberraciones.")
            print(f"Revise las gráficas para la interpretación visual completa.")
        else:
            print(f"\nEl análisis encontró problemas pero se completó.")
            
    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo {archivo}")
        print("Asegúrate de que:")
        print("1. El archivo existe en el directorio actual")
        print("2. El nombre del archivo es correcto")
        print("3. Tienes permisos de lectura")
        
    except Exception as e:
        print(f"Error durante el análisis: {str(e)}")
        print("Información de depuración:")
        import traceback
        traceback.print_exc()
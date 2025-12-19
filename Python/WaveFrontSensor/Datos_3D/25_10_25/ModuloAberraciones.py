"""
ANALIZADOR DE ABERRACIONES ÓPTICAS CON SENSOR SHACK-HARTMANN
================================================================================

Este módulo implementa un analizador completo de aberraciones ópticas usando
el paradigma de programación orientada a objetos. Procesa datos de sensores
Shack-Hartmann para caracterizar lentes mediante coeficientes de Zernike.

Características principales:
- Lectura robusta de archivos CSV con detección automática de codificación
- Análisis de coeficientes de Zernike hasta 4to orden
- Visualizaciones 2D y 3D del frente de onda
- Identificación automática de aberraciones significativas
- Evaluación de calidad óptica basada en métricas RMS y PV
- Exportación de resultados a PDF y TXT

Versión: 3.1 (Orientado a Objetos con Exportación)
Fecha: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy import interpolate
import chardet
import os
from datetime import datetime


class AnalizadorShackHartmann:
    """
    Clase principal para el análisis de aberraciones ópticas usando datos de sensor Shack-Hartmann.

    Esta clase encapsula toda la funcionalidad para procesar archivos CSV del sensor,
    extraer coeficientes de Zernike, generar visualizaciones y proporcionar análisis
    cualitativo de las aberraciones ópticas.

    Atributos:
        archivo_csv (str): Ruta al archivo CSV a analizar
        lineas (list): Líneas del archivo CSV leídas
        coeficientes (list): Coeficientes de Zernike extraídos
        wavefront_data (tuple): Datos del frente de onda (X, Y, Z)
        resultados (dict): Diccionario con todos los resultados del análisis
        guardar_archivos (bool): Flag para determinar si se guardan los archivos de salida
    """

    NOMBRES_ABERRACIONES = [
        "Pistón",  # Z0 - No afecta calidad óptica
        "Inclinación X (Tip)",  # Z1 - Inclinación en X
        "Inclinación Y (Tilt)",  # Z2 - Inclinación en Y
        "Desenfoque",  # Z3 - Desenfoque
        "Astigmatismo a 0°",  # Z4 - Astigmatismo vertical/horizontal
        "Astigmatismo a 45°",  # Z5 - Astigmatismo diagonal
        "Coma X",  # Z6 - Coma en dirección X
        "Coma Y",  # Z7 - Coma en dirección Y
        "Aberración Esférica",  # Z8 - Esférica primaria
        "Trefoil X",  # Z9 - Trefoil en X
        "Trefoil Y",  # Z10 - Trefoil en Y
        "Astigmatismo Secundario a 0°",  # Z11 - Astigmatismo secundario
        "Astigmatismo Secundario a 45°",  # Z12 - Astigmatismo secundario diagonal
        "Esférica Secundaria",  # Z13 - Esférica secundaria
        "Tetrafoil"  # Z14 - Tetrafoil
    ]

    def __init__(self, archivo_csv, guardar_archivos=False , nombre_carpeta='Nada', angulo = 0):
        """
        Inicializa el analizador con la ruta al archivo CSV.

        Parámetros:
            archivo_csv (str): Ruta al archivo CSV del sensor Shack-Hartmann
            guardar_archivos (bool): Si es True, guarda los resultados en archivos PDF y TXT
        """
        self.archivo_csv = archivo_csv
        self.lineas = None
        self.coeficientes = []
        self.wavefront_data = (None, None, None)
        self.resultados = {}
        self.guardar_archivos = guardar_archivos
        self.nombre_carpeta = nombre_carpeta  
        self.angulo = angulo
        self.nombre_base = os.path.splitext(os.path.basename(archivo_csv))[0]
        
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.figsize'] = [20, 16]
        plt.rcParams['figure.autolayout'] = False

        print(f"Analizador Shack-Hartmann inicializado para: {archivo_csv}")
        if guardar_archivos:
            print("Modo de guardado activado: Se generarán archivos PDF y TXT")

    def detectar_codificacion(self):
        """
        Detecta la codificación de un archivo usando la librería chardet.

        Esta función es crucial para manejar archivos CSV que pueden tener
        diferentes codificaciones dependiendo del sistema donde se generaron.

        Retorna:
            str: Codificación detectada (ej: 'utf-8', 'latin-1', etc.)

        Lanza:
            Exception: Si no se puede detectar la codificación
        """
        try:
            with open(self.archivo_csv, 'rb') as f:
                resultado = chardet.detect(f.read())
            codificacion = resultado['encoding']
            print(f"Codificación detectada: {codificacion}")
            return codificacion
        except Exception as e:
            print(f"Error en detección de codificación: {e}")
            return 'latin-1'  # Fallback a codificación común

    def leer_archivo_csv(self):
        """
        Lee un archivo CSV intentando diferentes codificaciones hasta encontrar una que funcione.

        El método intenta múltiples codificaciones comunes para archivos CSV
        y maneja errores de decodificación de manera robusta.

        Retorna:
            bool: True si la lectura fue exitosa, False en caso contrario
        """
        codificacion_detectada = self.detectar_codificacion()
        codificaciones = [codificacion_detectada, 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']

        for encoding in codificaciones:
            try:
                with open(self.archivo_csv, 'r', encoding=encoding) as f:
                    self.lineas = f.readlines()
                print(f"✓ Archivo leído exitosamente con codificación: {encoding}")
                return True
            except UnicodeDecodeError:
                print(f"✗ Error con codificación {encoding}, intentando siguiente...")
                continue
        try:
            with open(self.archivo_csv, 'r', encoding='utf-8', errors='ignore') as f:
                self.lineas = f.readlines()
            print("✓ Archivo leído con manejo de errores UTF-8")
            return True
        except Exception as e:
            print(f"Error crítico: No se pudo leer el archivo: {e}")
            self.lineas = None
            return False

    def extraer_coeficientes_zernike(self):
        """
        Extrae los coeficientes de Zernike de las líneas del archivo CSV.

        Los coeficientes de Zernike representan las diferentes aberraciones ópticas
        en una base ortonormal sobre el círculo unitario. Cada coeficiente corresponde
        a un modo específico de aberración.

        Retorna:
            list: Lista de coeficientes de Zernike en micrómetros
        """
        coeficientes = []
        en_seccion_zernike = False

        for linea in self.lineas:
            if '*** ZERNIKE FIT ***' in linea:
                en_seccion_zernike = True
                continue

            if '***' in linea and en_seccion_zernike and 'ZERNIKE' not in linea:
                break

            if en_seccion_zernike and ',' in linea and 'Index' not in linea:
                partes = linea.split(',')
                if len(partes) >= 4:
                    try:
                        valor_limpio = partes[3].strip().replace('Á', '').replace('�', '')
                        coef = float(valor_limpio)
                        coeficientes.append(coef)
                    except ValueError as e:
                        print(f"Advertencia: No se pudo convertir '{partes[3]}' a float: {e}")
                        continue

        self.coeficientes = coeficientes
        print(f"Coeficientes extraídos: {len(self.coeficientes)}")
        return self.coeficientes

    def extraer_datos_wavefront(self):
        """
        Extrae los datos de matriz del frente de onda del archivo CSV.

        El frente de onda representa la desviación de fase en micrómetros
        sobre una grilla cartesiana regular. Estos datos se utilizan para
        generar las visualizaciones 2D y 3D.

        Retorna:
            tuple: (X, Y, Z) donde X, Y son mallas de coordenadas y Z es el frente de onda
        """
        datos_wavefront = []
        en_seccion_wavefront = False
        coordenadas_x = None

        for i, linea in enumerate(self.lineas):
            if '*** WAVEFRONT ***' in linea:
                en_seccion_wavefront = True
                continue
            
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

            if en_seccion_wavefront and ',' in linea and 'y / x [mm]' not in linea:
                partes = linea.split(',')
                if coordenadas_x and len(partes) == len(coordenadas_x) + 1:
                    try:
                        
                        y_val_limpio = partes[0].strip().replace('Á', '').replace('�', '')
                        y_val = float(y_val_limpio)
                        valores_z = []
                        for z in partes[1:]:
                            z_limpio = z.strip().replace('Á', '').replace('�', '')
                            if z_limpio:
                                valores_z.append(float(z_limpio))

                        if len(valores_z) == len(coordenadas_x):
                            datos_wavefront.append((y_val, valores_z))
                    except ValueError as e:
                        print(f"Advertencia en línea {i}: {e}")
                        continue

        if datos_wavefront:
            
            y_coords = np.array([d[0] for d in datos_wavefront])
            x_coords = np.array(coordenadas_x)
            Z_medido = np.array([d[1] for d in datos_wavefront])
            
            X, Y = np.meshgrid(x_coords, y_coords)
            self.wavefront_data = (X, Y, Z_medido)
            return self.wavefront_data
        else:
            print("ERROR: No se pudieron extraer datos del wavefront")
            return None, None, None

    def crear_datos_ejemplo(self):
        """
        Crea datos de ejemplo para demostración cuando no se pueden leer del archivo.

        Genera un frente de onda sintético con características típicas de aberraciones
        ópticas como astigmatismo y coma. Útil para pruebas y demostraciones.

        Retorna:
            tuple: (X, Y, Z_ejemplo) con datos sintéticos de frente de onda
        """
        print("Creando datos de ejemplo para demostración...")
        y_coords = np.linspace(-1.5, 1.5, 21)
        x_coords = np.linspace(-1.5, 1.5, 21)
        X, Y = np.meshgrid(x_coords, y_coords)

        # Crear un frente de onda sintético con astigmatismo y coma
        Z_ejemplo = (10 * np.exp(-(X ** 2 + Y ** 2) / 2) *
                     (0.5 * np.sin(2 * np.pi * X / 3) +  # Componente astigmática
                      0.3 * Y * np.exp(-(X ** 2 + Y ** 2) / 4)))  # Componente de coma

        self.wavefront_data = (X, Y, Z_ejemplo)
        return self.wavefront_data

    def calcular_metricas(self):
        """
        Calcula las métricas principales de calidad óptica.

        Calcula:
        - PV (Peak-to-Valley): Diferencia entre máximo y mínimo del frente de onda
        - RMS (Root Mean Square): Desviación estándar del frente de onda

        Retorna:
            tuple: (pv, rms) con las métricas calculadas
        """
        if self.wavefront_data[2] is not None:
            Z = self.wavefront_data[2]
            pv = np.max(Z) - np.min(Z)  # Peak-to-Valley (máxima variación)
            rms = np.std(Z)  # Root Mean Square (desviación estándar)

            print(f"Métricas calculadas - PV: {pv:.2f} µm, RMS: {rms:.2f} µm")
            return pv, rms
        else:
            print("Error: No hay datos de wavefront para calcular métricas")
            return 0, 0

    def generar_visualizaciones(self):
        """
        Genera un panel completo de visualizaciones del análisis.

        Crea una figura con 6 subplots que muestran:
        1. Mapa 2D del frente de onda
        2. Visualización 3D del frente de onda
        3. Coeficientes de Zernike (gráfico de barras)
        4. Aberraciones significativas (barras horizontales)
        5. Distribución radial del frente de onda
        6. Resumen numérico del análisis

        Retorna:
            matplotlib.figure.Figure: La figura generada para posible guardado
        """
        X, Y, Z_medido = self.wavefront_data
        pv, rms = self.calcular_metricas()
        fig = plt.figure(figsize=(20, 16))

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
        indices = range(len(self.coeficientes))
        barras = ax3.bar(indices, self.coeficientes, color='skyblue', alpha=0.7)
        ax3.set_xlabel('Término de Zernike', fontsize=12, labelpad=10)
        ax3.set_ylabel('Coeficiente [µm]', fontsize=12, labelpad=10)
        ax3.set_title('Coeficientes de Zernike', fontsize=14, pad=15)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)  # Rotar etiquetas para mejor legibilidad

        # Resaltar coeficientes significativos (> 1µm)
        for i, (bar, coef) in enumerate(zip(barras, self.coeficientes)):
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
        for i, (nombre, coef) in enumerate(zip(self.NOMBRES_ABERRACIONES, self.coeficientes)):
            if i < len(self.coeficientes) and abs(coef) > 0.1:
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
                         bar.get_y() + bar.get_height() / 2,
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
        r = np.sqrt(X ** 2 + Y ** 2)  # Calcular coordenadas radiales
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
        - Diámetro de Pupila: {X[0, -1] - X[0, 0]:.1f} mm

        Aberraciones Principales:"""

        # Añadir aberraciones significativas al resumen
        for i, coef in enumerate(self.coeficientes):
            if i < len(self.NOMBRES_ABERRACIONES) and abs(coef) > 1.0:
                info_text += f"\n- {self.NOMBRES_ABERRACIONES[i]}: {coef:.2f} µm"

        if not any(abs(coef) > 1.0 for coef in self.coeficientes):
            info_text += "\n- No hay aberraciones dominantes (> 1 µm)"

        ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes,
                 fontfamily='monospace', verticalalignment='top', fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.5))

        fig.suptitle('Análisis Completo de Aberraciones Ópticas θ= {}° - Sensor Shack-Hartmann'.format(self.angulo),
                    fontsize=16, y=0.98)

        if self.guardar_archivos:
            self.guardar_plot_pdf(fig)

        plt.show()
        return fig


    def guardar_plot_pdf(self, fig):
        """
        Guarda el plot completo como archivo PDF en la carpeta self.nombre_carpeta.

        Parámetros:
            fig (matplotlib.figure.Figure): La figura a guardar
        """
        # Crear la ruta completa usando os.path.join
        nombre_pdf = f"{self.nombre_base}_analisis_aberracciones.pdf"
        ruta_completa = os.path.join(self.nombre_carpeta, nombre_pdf)
        
        try:
            # Asegurarse de que la carpeta existe
            os.makedirs(self.nombre_carpeta, exist_ok=True)
            
            fig.savefig(ruta_completa, format='pdf', dpi=300, bbox_inches='tight')
            print(f"✓ Gráficas guardadas como: {ruta_completa}")
        except Exception as e:
            print(f"✗ Error al guardar PDF: {e}")

    def guardar_datos_txt(self):
        """
        Guarda todos los datos calculados e información del análisis en un archivo TXT 
        en la carpeta nombre_carpeta.
        """
        nombre_txt = f"{self.nombre_base}_datos_analisis.txt"
        ruta_completa = os.path.join(self.nombre_carpeta, nombre_txt)
        
        pv, rms = self.calcular_metricas()
        
        try:
            os.makedirs(self.nombre_carpeta, exist_ok=True)
            
            with open(ruta_completa, 'w', encoding='utf-8') as f:
                
                f.write("""
                ██████╗  ██████╗ ████████╗███████╗    ██╗   ██╗██╗███████╗
                ██╔════╝ ██╔═══██╗╚══██╔══╝██╔════╝    ██║   ██║██║██╔════╝
                ██║  ███╗██║   ██║   ██║   ███████╗    ██║   ██║██║███████╗
                ██║   ██║██║   ██║   ██║   ╚════██║    ██║   ██║██║╚════██║
                ╚██████╔╝╚██████╔╝   ██║   ███████║    ╚██████╔╝██║███████║
                ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝     ╚═════╝ ╚═╝╚══════╝
                """)
                f.write("ANÁLISIS DE ABERRACIONES ÓPTICAS - SENSOR SHACK-HARTMANN\n")
                f.write("=" * 70 + "\n")
                f.write(f"Archivo analizado: {self.archivo_csv}\n")
                f.write(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n")
                
                # Métricas principales
                f.write("MÉTRICAS PRINCIPALES\n")
                f.write("-" * 40 + "\n")
                f.write(f"PV (Peak-to-Valley): {pv:.3f} µm\n")
                f.write(f"RMS: {rms:.3f} µm\n")
                if self.wavefront_data[0] is not None:
                    diametro = self.wavefront_data[0][0, -1] - self.wavefront_data[0][0, 0]
                    f.write(f"Diámetro de pupila: {diametro:.2f} mm\n")
                f.write("Ángulo de inlcinación: {}° \n".format(self.angulo))
                f.write("\n")
                
                # Coeficientes de Zernike
                f.write("COEFICIENTES DE ZERNIKE\n")
                f.write("-" * 40 + "\n")
                for i, (nombre, coef) in enumerate(zip(self.NOMBRES_ABERRACIONES, self.coeficientes)):
                    if i >= len(self.coeficientes):
                        break
                    significado = ""
                    if abs(coef) > 3.0:
                        significado = " [DOMINANTE]"
                    elif abs(coef) > 1.0:
                        significado = " [SIGNIFICATIVO]"
                    f.write(f"Z{i:2d}: {nombre:35} {coef:8.3f} µm{significado}\n")
                f.write("\n")
                
                # Aberraciones significativas
                f.write("ABERRACIONES SIGNIFICATIVAS (> 0.1 µm)\n")
                f.write("-" * 40 + "\n")
                aberraciones_filtradas = []
                for i, (nombre, coef) in enumerate(zip(self.NOMBRES_ABERRACIONES, self.coeficientes)):
                    if i < len(self.coeficientes) and abs(coef) > 0.1:
                        aberraciones_filtradas.append((nombre, coef))
                
                if aberraciones_filtradas:
                    for nombre, coef in aberraciones_filtradas:
                        f.write(f"- {nombre}: {coef:.3f} µm\n")
                else:
                    f.write("No se detectaron aberraciones significativas (> 0.1 µm)\n")
                f.write("\n")
                
                # Evaluación de calidad
                f.write("EVALUACIÓN DE CALIDAD ÓPTICA\n")
                f.write("-" * 40 + "\n")
                if rms < 0.5:
                    evaluacion = "Excelente (RMS < 0.5 µm)"
                elif rms < 1.0:
                    evaluacion = "Buena (RMS < 1.0 µm)"
                elif rms < 2.0:
                    evaluacion = "Aceptable (RMS < 2.0 µm)"
                elif rms < 5.0:
                    evaluacion = "Regular (RMS < 5.0 µm)"
                else:
                    evaluacion = "Pobre (RMS ≥ 5.0 µm) - considerar recalibración"
                f.write(f"Calidad: {evaluacion}\n")
                f.write("\n")
                
                # Interpretación detallada
                f.write("INTERPRETACIÓN DETALLADA\n")
                f.write("-" * 40 + "\n")
                if len(self.coeficientes) > 5 and (abs(self.coeficientes[3]) > 1.0 or 
                                                abs(self.coeficientes[4]) > 1.0 or 
                                                abs(self.coeficientes[5]) > 1.0):
                    f.write("• La lente presenta ASTIGMATISMO significativo\n")
                    f.write("  (diferente potencia en meridianos ortogonales)\n")
                
                if len(self.coeficientes) > 7 and (abs(self.coeficientes[6]) > 1.0 or 
                                                abs(self.coeficientes[7]) > 1.0):
                    f.write("• Se detecta COMA (asimetría en la formación de imagen)\n")
                
                if len(self.coeficientes) > 8 and abs(self.coeficientes[8]) > 1.0:
                    f.write("• ABERRACIÓN ESFÉRICA presente\n")
                    f.write("  (enfoque diferente para rayos centrales vs marginales)\n")
                
                if len(self.coeficientes) > 0 and abs(self.coeficientes[0]) > 10.0:
                    f.write("• Pistón elevado - posible error de referencia o alineación\n")
                
                if not any(abs(coef) > 1.0 for coef in self.coeficientes):
                    f.write("• No se detectaron aberraciones dominantes (> 1 µm)\n")
                
            print(f"✓ Datos guardados como: {ruta_completa}")
            
        except Exception as e:
            print(f"✗ Error al guardar archivo TXT: {e}")

    def imprimir_analisis_detallado(self):
        """
        Imprime un análisis detallado y cualitativo en la consola.

        Proporciona interpretación de los resultados, clasificación de la
        calidad óptica basada en métricas RMS, e identificación de los
        tipos de aberraciones más significativas.
        """
        pv, rms = self.calcular_metricas()

        print("=" * 70)
        print("ANÁLISIS DETALLADO DE ABERRACIONES - SENSOR SHACK-HARTMANN")
        print("=" * 70)

        print(f"\nINFORMACIÓN GENERAL:")
        print(f"PV (Peak-to-Valley): {pv:.2f} µm")
        print(f"RMS: {rms:.2f} µm")

        if self.coeficientes:
            print(f"\nCOEFICIENTES DE ZERNIKE:")
            print("-" * 60)
            for i, (nombre, coef) in enumerate(zip(self.NOMBRES_ABERRACIONES, self.coeficientes)):
                if i >= len(self.coeficientes):
                    break

                significado = ""
                if abs(coef) > 3.0:
                    significado = " ← DOMINANTE"
                elif abs(coef) > 1.0:
                    significado = " ← SIGNIFICATIVO"

                print(f"Z{i:2d}: {nombre:35} {coef:8.3f} µm{significado}")

            print(f"\nINTERPRETACIÓN:")
            print("-" * 60)

            if len(self.coeficientes) > 5 and (abs(self.coeficientes[3]) > 1.0 or
                                               abs(self.coeficientes[4]) > 1.0 or
                                               abs(self.coeficientes[5]) > 1.0):
                print("• La lente presenta ASTIGMATISMO significativo")
                print("  (diferente potencia en meridianos ortogonales)")

            if len(self.coeficientes) > 7 and (abs(self.coeficientes[6]) > 1.0 or
                                               abs(self.coeficientes[7]) > 1.0):
                print("• Se detecta COMA (asimetría en la formación de imagen)")

            if len(self.coeficientes) > 8 and abs(self.coeficientes[8]) > 1.0:
                print("• ABERRACIÓN ESFÉRICA presente")
                print("  (enfoque diferente para rayos centrales vs marginales)")

            if len(self.coeficientes) > 0 and abs(self.coeficientes[0]) > 10.0:
                print("• Pistón elevado - posible error de referencia o alineación")

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

    def analizar(self):
        """
        Ejecuta el análisis completo de aberraciones ópticas.

        Este es el método principal que coordina todo el flujo de trabajo:
        1. Lectura del archivo CSV
        2. Extracción de coeficientes de Zernike
        3. Extracción de datos del frente de onda
        4. Generación de visualizaciones
        5. Análisis cualitativo y cuantitativo
        6. Guardado de archivos (si está activado)

        Retorna:
            dict: Diccionario con todos los resultados del análisis
        """
        print("Iniciando análisis de archivo CSV...")

        if not self.leer_archivo_csv():
            print("No se pudo leer el archivo, creando datos de ejemplo...")
            self.crear_datos_ejemplo()
            
            if not self.coeficientes:
                self.coeficientes = [-23.505, 11.578, 15.226, -0.173, -3.370, 0.219,
                                     -0.003, -0.003, 0.020, 0.010, -0.009, 0.008,
                                     0.014, -0.006, -0.006]
            return self.resultados

        self.extraer_coeficientes_zernike()
        X, Y, Z_medido = self.extraer_datos_wavefront()
        if Z_medido is None:
            self.crear_datos_ejemplo()
        pv, rms = self.calcular_metricas()
        self.resultados = {
            'coeficientes': self.coeficientes,
            'nombres_aberracciones': self.NOMBRES_ABERRACIONES,
            'wavefront': self.wavefront_data[2],
            'X': self.wavefront_data[0],
            'Y': self.wavefront_data[1],
            'pv': pv,
            'rms': rms
        }
        
        # Generar visualizaciones (ya incluye guardado de PDF si está activado)
        self.generar_visualizaciones()
        
        # Guardar datos en TXT si está activado
        if self.guardar_archivos:
            self.guardar_datos_txt()
        
        self.imprimir_analisis_detallado()

        return self.resultados

    def main(self):
        """
        Punto de entrada principal para ejecutar el análisis.
        """
        try:
            print(f"Iniciando análisis del archivo: {self.archivo_csv}")
            resultados = self.analizar()

            if resultados:
                print(f"\n" + "=" * 50)
                print("ANÁLISIS COMPLETADO EXITOSAMENTE!")
                print("=" * 50)
                print(f"Se generaron 6 visualizaciones del frente de onda y aberraciones.")
                if self.guardar_archivos:
                    print(f"✓ Archivos guardados en carpeta: {self.nombre_carpeta}")
                    print(f"  - {self.nombre_base}_analisis_aberracciones.pdf")
                    print(f"  - {self.nombre_base}_datos_analisis.txt")
                else:
                    print(f"Revise las gráficas para la interpretación visual completa.")
            else:
                print(f"\nEl análisis encontró problemas pero se completó.")

        except FileNotFoundError:
            print(f"Error: No se pudo encontrar el archivo {self.archivo_csv}")
            print("Asegúrate de que:")
            print("1. El archivo existe en el directorio actual")
            print("2. El nombre del archivo es correcto")
            print("3. Tienes permisos de lectura")

        except Exception as e:
            print(f"Error durante el análisis: {str(e)}")
            print("Información de depuración:")
            import traceback
            traceback.print_exc()
                
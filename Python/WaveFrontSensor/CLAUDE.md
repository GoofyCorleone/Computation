# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Propósito del proyecto

Análisis de aberraciones ópticas mediante datos de un sensor Shack-Hartmann (Thorlabs). Los archivos CSV exportados por el sensor contienen secciones delimitadas por `*** ZERNIKE FIT ***` y `*** WAVEFRONT ***` con coeficientes de Zernike y la malla cartesiana del frente de onda respectivamente. El análisis extrae hasta 15 coeficientes (Z0–Z14, estándar OSA/ANSI) y calcula métricas PV y RMS en micrómetros.

## Ejecución

```bash
# Análisis de un solo CSV desde la raíz del proyecto
python ModuloAberraciones.py

# Análisis en lote (múltiples ángulos y direcciones)
cd Datos_3D/25_10_25
python Analisis.py
```

No hay entorno virtual ni `requirements.txt`. Instalar dependencias manualmente:
```bash
pip install numpy matplotlib scipy pandas chardet
```

## Arquitectura y evolución del código

El proyecto muestra una progresión de versiones que conviven:

| Archivo | Descripción |
|---|---|
| `Analisis.py` | v1 — función procedural simple, sin manejo de codificación |
| `Anallisis2.py` | v2 — añade detección de codificación con `chardet` |
| `Analisis3.py` | v2.5 — mejor manejo de errores y datos de ejemplo como fallback |
| `Analisis4.py` | v2 refactorizado en funciones separadas |
| `AberrationsModule.py` | v3.0 OO — clase `AnalizadorShackHartmann` sin exportación a archivo |
| `ModuloAberraciones.py` | v3.1 OO — clase `AnalizadorShackHartmann` con exportación PDF/TXT (versión actual) |

**`ModuloAberraciones.py` es el módulo canónico.** Los demás son versiones históricas.

### Clase `AnalizadorShackHartmann` (ModuloAberraciones.py)

Constructor: `AnalizadorShackHartmann(archivo_csv, guardar_archivos=False, nombre_carpeta=None, angulo=None)`

Método principal: `.main()` — ejecuta el pipeline completo:
1. Lectura del CSV con fallback multi-codificación (`chardet` + latin-1/iso-8859-1/cp1252)
2. Extracción de coeficientes de Zernike (busca `*** ZERNIKE FIT ***`)
3. Extracción de malla wavefront (busca `*** WAVEFRONT ***`, cabecera `y / x [mm]`)
4. Cálculo de métricas PV y RMS
5. Generación de 6 subplots (mapa 2D, superficie 3D, barras Zernike, aberraciones significativas, distribución radial, resumen)
6. Exportación opcional a PDF y TXT cuando `guardar_archivos=True`

### Datos en Datos_3D/25_10_25/

Mediciones angulares de lente en tres configuraciones: `Horizontal0_difpupila/`, `Horizontal1/`, `Horizontal2/`, `vertical/`. Cada subdirectorio contiene CSVs nombrados por ángulo en grados (ej. `-10.csv`, `+4.csv`, `0.csv`). `Analisis.py` procesa cada serie usando `ModuloAberraciones` con `angulo` como parámetro para etiquetar la salida.

### Archivos de datos en la raíz

- `lenteThorlabs.csv` — lente comercial de referencia (Thorlabs)
- `lenteImpresa.csv` — lente impresa en 3D
- `plano.csv` — referencia de plano óptico
- `lenteImpresa_analisis_aberracciones.pdf` y `lenteImpresa_datos_analisis.txt` — salidas previas de `ModuloAberraciones`

## Particularidades del formato CSV

Los CSVs del sensor Thorlabs pueden tener caracteres corruptos (`Á`, `\ufffd`) en valores numéricos. El parser los limpia con `.replace('Á', '').replace('�', '')` antes de convertir a `float`. Los valores NaN en la malla del wavefront se manejan explícitamente en `FrenteDeOnda.py`.

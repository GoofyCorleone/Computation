# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Polarimetric camera image analysis. Computes Stokes parameters and derived polarization properties (DoP, AoLP, DoCP, ellipticity) from four images captured at different polarizer/retarder angle combinations.

## Running Scripts

```bash
python CamPol_5.py   # Latest: full analysis pipeline, outputs to resultados_polarimetria/
python CamPol_4.py   # BMP input variant
python CamPolCalib.py  # Calibration analysis
```

No package manager files exist. Required libraries: `numpy`, `Pillow`, `opencv-python`, `matplotlib`.

## Architecture

Scripts are numbered versions of the same analysis pipeline, each an evolutionary improvement:

- **CamPol_2.py** — Experimental, pixel-by-pixel loops, noise subtraction from `ruido.png`
- **CamPol_3.py** — Vectorized Stokes calculation, TIFF inputs
- **CamPol_4.py** — BMP inputs (`I_0_0.bmp`, etc.), adds S0-based masking, RGB Stokes composite
- **CamPol_5.py** — Current main script: modular functions, full statistical output, 4 output PNGs
- **CamPolCalib.py** — Calibration-specific variant

## Input Data

Four images per measurement configuration (1024×1024 pixels):

| TIFF (CamPol_5) | BMP (CamPol_4) | Polarizer | Retarder |
|---|---|---|---|
| `CCDI(0,0).tif` | `I_0_0.bmp` | 0° | 0° |
| `CCDI(45,0).tif` | `I_45_0.bmp` | 45° | 0° |
| `CCDI(90,0).tif` | `I_90_0.bmp` | 90° | 0° |
| `CCDI(45,90).tif` | `I_45_90.bmp` | 45° | 90° |

## Stokes Parameter Convention

```
S0 = I0 + I90           (total intensity)
S1 = I0 - I90           (horizontal/vertical linear)
S2 = 2·I45 - I0 - I90  (diagonal linear)
S3 = 2·I45/90 - I0 - I90  (circular)

Normalized: s1=S1/S0, s2=S2/S0, s3=S3/S0

DoP  = sqrt(s1²+s2²+s3²)       ∈ [0,1]
AoLP = 0.5·arctan2(s2,s1)      ∈ [0°,180°)
DoCP = |s3|                     ∈ [0,1]
χ    = 0.5·arcsin(s3/DoP)      ∈ [-45°,45°]
```

## Output

`CamPol_5.py` writes to `resultados_polarimetria/`:
- `stokes_normalizados.png` — S0, s1, s2, s3 as 2×2 panel
- `propiedades_polarizacion.png` — DoP, AoLP, DoCP, ellipticity maps
- `imagen_falso_color.png` — HSV false-color composite
- `histogramas.png` — Distribution plots

Code and comments are in Spanish.

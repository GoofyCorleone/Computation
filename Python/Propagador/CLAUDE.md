# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Numerical physics simulation of Fresnel diffraction through rectangular apertures (single and double slits). The core problem is comparing the Fresnel paraxial approximation against the exact Huygens-Fresnel method, with tools for parameter optimization and performance benchmarking.

## Running the Code

```bash
# Run the parameter optimization workflow
python main.py

# Run a simple example/test
python Prueba1.py

# Benchmark performance on Apple Silicon (M-series)
python RunGPU.py

# Individual simulators can be run directly
python DobleDifraccion.py
python Propagador.py
```

No build step required. No test suite — `Prueba1.py` serves as a manual smoke test.

## Dependencies

No `requirements.txt` exists. Key packages used:
- `numpy`, `matplotlib`, `scipy` — core computation and visualization
- `numba` — JIT compilation (used in `DobleRendija3.py`, optional)
- `concurrent.futures`, `multiprocessing` — parallelism for Apple Silicon

## Architecture

### Physics Pipeline

```
Physical parameters (wavelength, aperture geometry)
  → Integration kernels (Fresnel or Huygens-Fresnel)
  → SimpsonAdaptativo (adaptive recursive Simpson integration)
  → Complex field amplitude
  → Intensity |E|²
  → matplotlib visualization
```

Two kernels are compared throughout the project:
- **Fresnel** (paraxial): `exp(1j * k * (x0-x)² / (2*d))` — fast approximation
- **Huygens-Fresnel** (exact): full spherical wave — ground truth

### Key Classes

| Class | File | Purpose |
|---|---|---|
| `SimpsonAdaptativo` | `Propagador.py` and variants | Recursive adaptive Simpson integration over complex integrands |
| `FresnelDiffractionSimulator` | `DobleDifraccion.py` | Primary double-slit simulator |
| `DifraccionRectangular2DCompleta` | `Propagador.py` | 2D rectangular aperture with both methods |
| `ParameterOptimizer` | `ParameterOptimizer.py` | Global + local parameter search |

### File Lineage

The codebase has parallel evolutionary series:
- `Propagador.py → Propagador1–5.py` — single/multi-aperture variants exploring different geometries
- `DobleRendija1–5.py` — double-slit series; version 3 adds Numba JIT
- `DobleDifraccion.py` — current main module; `DobleDifraccion_optimized.py` targets M-series Macs with ProcessPoolExecutor

When editing, prefer `DobleDifraccion.py` (primary) or the highest-numbered version of a series as the most current.

### Optimization Workflow (`main.py`)

1. **Global search**: `scipy.optimize.differential_evolution` over 9 parameters (`p, q, p2, q2, a, b, n, c, z0`)
2. **Local refinement**: Nelder-Mead polish
3. **Full simulation**: run with optimized params and save to JSON

### Performance Notes

`DobleDifraccion_optimized.py` was written specifically for Apple Silicon (M5 MacBook Pro). It uses `ProcessPoolExecutor` for row-level parallelism and includes GPU detection. `RunGPU.py` benchmarks sequential vs. parallel vs. FFT approaches and prints a recommendation.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Polarization Experiments (PE) — a set of scripts and Jupyter notebooks for studying the **generalized Malus law** with elliptical polarization, using both simulation and real single-photon counting hardware.

## Running Scripts

```bash
# Main photon-counting GUI (controls TOPTICA iBeam Smart laser + Thorlabs SPCM50A/M)
cd Python/PE
python malus_conteo_fotones.py

# Simple Malus law plot from raw Thorlabs .txt files
python SoloMalus.py

# Open notebooks with Jupyter
jupyter notebook DataAnalysis.ipynb
jupyter notebook MalusLaw_Simulation.ipynb
jupyter notebook Malus_Generalizada.ipynb
jupyter notebook Propuesta.ipynb
```

## Architecture

### Key files

| File | Purpose |
|---|---|
| `malus_conteo_fotones.py` | PyQt6 GUI that orchestrates a full 0–360° polarizer sweep. Simultaneously controls the laser (via `../TopasIbeamSmart/ibeam_gui.py`) and the SPCM counter (via `../TopasIbeamSmart/spcm_gui.py`). Exports timestamped results to `Malus_Tradicional_*/`. |
| `Distributions.py` | Standalone library: reads Stokes-parameter CSVs, renders publication-quality Poincaré sphere plots with Plotly, and fits cone-intersection geometry (law of elliptical birefringents). |
| `SoloMalus.py` | Minimal script: reads `Malus/*.txt` Thorlabs exports → normalizes counts → overlays on `cos²θ`. |
| `MalusLaw_Simulation.ipynb` | Simulates the generalized Malus law for partially-coherent beams (`ClassicalMalus`) and quantum single-photon regime (`QuantumMalus`), using the Poincaré / Bloch sphere formalism. |
| `Malus_Generalizada.ipynb` | Symbolic derivation of the generalized Malus law via SymPy; produces `malus_superficie_3D.png`. |
| `DataAnalysis.ipynb` | Fits real SPCM data from `DatosPE/` to the generalized Malus formula `C12(alpha, chi, alphap, chip)`. |
| `Propuesta.ipynb` | Jones-matrix derivation of a tunable elliptic polarizer; uses `Distributions` for Poincaré visualization. |
| `Malus.ipynb` | Theoretical derivation of the Malus law and its generalization (written in English; symbolic work). |
| `Classical.ipynb` | SymPy/quantum-mechanics derivations using tensor products; exploratory symbolic notebook. |
| `polarizador_eliptico.ipynb` | Jones-matrix construction of the elliptic polarizer (SymPy; companion to `Malus_Generalizada.ipynb`). |
| `Malus_Cuantica_Wigner.ipynb` | Quantum generalized Malus law: Stokes operators, the elliptic polarizer as a quantum channel, and photon-counting statistics via Stratonovich–Weyl Wigner functions on the Poincaré sphere. Companion to `Malus_Generalizada.ipynb` Parts I–II. |

### Hardware driver dependency

`malus_conteo_fotones.py` imports drivers from the sibling project:

```
../TopasIbeamSmart/ibeam_gui.py   → IBeamDriver, detectar_puerto
../TopasIbeamSmart/spcm_gui.py    → DriverSPCM, detectar_spcm
```

Without those modules on `sys.path`, the GUI will fail at import time.

### Data layout

| Directory | Contents |
|---|---|
| `Malus/` | Raw Thorlabs SPCM `.txt` files at 10° steps (used by `SoloMalus.py`) |
| `Malus_Tradicional*/` | Timestamped output folders auto-created by `malus_conteo_fotones.py` |
| `DatosConteoMalus/` | Output from photon-counting sweeps saved by `malus_conteo_fotones.py` |
| `DatosPE/` | SPCM `.txt` files at 5° steps + `incidente pl4.txt` for normalization (used by `DataAnalysis.ipynb`) |
| `DatosPE2/`, `DatosPE3/`, `DatosPE4/` | Additional experimental runs |
| `DATICOS/` | Stokes-parameter CSVs (`h*p.csv`, `v*p.csv`) consumed by `Distributions.py` for Poincaré-sphere fitting |
| `DATICOS/PElip/` | Elliptic-polarization runs (chi sweeps) for `Distributions.py` |

### Thorlabs SPCM data format

Files exported by the Thorlabs software have a **19-line header** before the tab-delimited data:

```python
pd.read_csv(ruta, header=19, delimiter='\t')  # columns: 'Bin Number', 'Counts per Bin'
```

### Generalized Malus law formula

The core intensity formula used throughout:

```
C12(α, χ, αₚ, χₚ) = cos²(χ)cos²(χₚ)cos²(α−αₚ) + sin²(χ)sin²(χₚ)
                    + (1/2)sin(2χ)sin(2χₚ)cos(2(α−αₚ))
```

where `(α, χ)` are the orientation and ellipticity angles of the incident beam and `(αₚ, χₚ)` of the polarizer.

## Dependencies

```
numpy, matplotlib, pandas, scipy   # all notebooks/scripts
plotly                             # Distributions.py, Propuesta.ipynb
sympy                              # Malus_Generalizada.ipynb, Propuesta.ipynb
PyQt6, matplotlib (QtAgg backend)  # malus_conteo_fotones.py
```

The shared venv lives at `../venv/` (Python at `/Users/jafertserrano/Desktop/U/Computation/Python/venv/bin/python`).

**Notebook kernel gotcha:** `Python/venv` does *not* have `jupyter`/`nbformat` installed. The registered `python3` kernel that actually executes these notebooks points at `Python/Depolarization Module Critian/.venv` (has numpy/scipy/sympy/matplotlib/nbclient). Use that interpreter when running notebooks non-interactively (e.g. via `nbclient`/`nbconvert`).

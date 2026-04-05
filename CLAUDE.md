# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

A multi-language, multi-project computational physics repository. Projects are grouped by programming language at the top level (`Python/`, `C/`, `C++/`, `Arduino/`, `Fortran/`, `FreeFem/`, `NodeJs/`, `Bash/`). Within `Python/`, each subdirectory is an independent project.

## Language & Style Convention

Code, comments, variable names, and docstrings are written in **Spanish** throughout this repo. Follow this convention in any new or modified code (e.g., `fuente` = source, `longitud_de_onda` = wavelength, `angulo_max` = max angle).

## Python Projects

Most Python projects have no build step — scripts run directly with `python <script>.py`. There is no shared virtual environment or `requirements.txt` at the repo root; dependencies vary by project.

Common dependencies across projects: `numpy`, `matplotlib`, `scipy`.

### Projects with their own CLAUDE.md

These subdirectories have detailed per-project guidance:

- **`Python/RayTracing/`** — Exact ray tracing through Cartesian oval surfaces (GOTS method). Main entry points: `ejemplos/ejemplo_lsoe.py`, `ejemplos/ejemplo_fig10.py`, `ejemplos/ejemplo_formas_lsoe.py`. See `RayTracing/CLAUDE.md`.
- **`Python/CAM/`** — Polarimetric camera image analysis (Stokes parameters). Main script: `CamPol_5.py`. See `CAM/CLAUDE.md`.
- **`Python/Propagador/`** — Fresnel diffraction simulations (single/double slits). Main entry: `main.py`. See `Propagador/CLAUDE.md`.

### Other notable Python projects

- **`Python/PE/`** — Polarization experiments and Malus law simulations. Contains Jupyter notebooks and standalone scripts.
- **`Python/Quantum/`** — Quantum mechanics computations.
- **`Python/Difracción/`** — Additional diffraction-related scripts.
- **`Python/Practicas Manim/`** — Manim animation practice scripts. Run with: `manim -pql <script>.py <ClassName>`
- **`Python/Random/`** — Miscellaneous experimental scripts.

## Running Manim Scripts

```bash
# Low quality preview (fast)
manim -pql Prueba1.py prueba

# High quality render
manim -pqh Prueba1.py prueba
```

## Running RayTracing Examples

```bash
cd Python/RayTracing
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python ejemplos/ejemplo_lsoe.py
python ejemplos/ejemplo_lsoe.py --no-stl  # disable STL export
```

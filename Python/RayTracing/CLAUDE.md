# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D ray-tracing simulations on **Descartes ovoids** (Cartesian oval surfaces) for optics research. The project models refraction through ovoid dioptric surfaces that produce perfect point-to-point imaging, with support for chromatic dispersion (white light / Sellmeier equation for BK7 glass).

Reference material: `TesisAlbertoDoc.pdf` (thesis document).

## Architecture

Two script variants existed (now deleted from `../Ray Tracing/`, being migrated here):

- **RT_Basic_C.py** — Simpler implementation using a Cauchy dispersion model and brute-force line-search for ray-surface intersection. `DescartesOvoid` defined by two foci F1, F2 and indices n1, n2.
- **RT_Basic_D.py** — More robust version using `scipy.optimize.root_scalar` (Brent's method) for intersection finding, Sellmeier dispersion (BK7), and a `DescartesOvoid` defined by Object O, Image I, and Vertex V points.

Core physics shared by both:
- **Surface function**: `F(P) = n1*|P-O| + n2*|P-I| - L` (optical path length condition)
- **Vectorial Snell's law** for 3D refraction
- **Surface mesh generation** by revolution around the optical axis
- **3D visualization** with matplotlib (`plot_trisurf` or `plot_wireframe`)

## Dependencies

- `numpy`, `matplotlib`, `scipy` (only RT_Basic_D uses scipy)

## Running

Scripts are standalone — run directly with Python:
```
python RT_Basic_D.py
```

All configurable parameters (O, I, V, n1, n2, num_rayos, angulo_max) are set in the `if __name__ == "__main__"` block.

## Language

Code comments and variable names are in **Spanish** (e.g., `fuente` = source, `trazar_rayos` = trace rays, `angulo_max` = max angle). Maintain this convention.

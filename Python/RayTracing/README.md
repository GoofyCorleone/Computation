# GOTS Ray Tracing

Exact ray tracing through **Cartesian oval surfaces** (Descartes ovoids) using the GOTS parametric formulation. Based on the doctoral thesis *"Aberraciones primarias a partir del estigmatismo riguroso"* by Alberto Silva Lora (Universidad Industrial de Santander, 2024).

## What it does

- Computes GOTS shape parameters (G, O, T, S) from physical system specifications (Eqs. 10-13)
- Finds exact ray-surface intersections by solving a quartic polynomial (Eq. 51)
- Applies vectorial Snell's law for 3D refraction (Eq. 68)
- Supports multi-surface sequential systems
- Designs LSOE (Lentes Singletes Ovoides Estigmaticas) with a shape factor parameter
- Generates 2D meridional and 3D visualizations
- Exports lens geometry to binary STL

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the examples

### LSOE from Table 4 (Figs. 11-12)

```bash
python ejemplos/ejemplo_lsoe.py
```

This traces rays through the thesis reference LSOE (object at z=0, image at z=200) and verifies rigorous stigmatism (on-axis rays converge to a single point with zero aberration). It generates 2D and 3D plots and exports an STL file by default.

To disable STL export:
```bash
python ejemplos/ejemplo_lsoe.py --no-stl
```

### Lens shapes vs shape factor (Fig. 5)

```bash
python ejemplos/ejemplo_formas_lsoe.py
```

Shows five LSOE shapes for different values of the shape factor sigma, from plano-convex to convex-plano.

### Ray tracing gallery (Fig. 10)

```bash
python ejemplos/ejemplo_fig10.py
```

A 3x3 grid of different LSOE configurations showing biconvex, meniscus, and other lens forms with their ray-tracing behavior. Exports an STL for each lens by default.

To disable STL export:
```bash
python ejemplos/ejemplo_fig10.py --no-stl
```

## Using the library

```python
import numpy as np
from gots import (
    SistemaOptico, SuperficieCartesiana, Rayo,
    calcular_gots, graficar_seccion_transversal, graficar_3d,
    exportar_sistema_stl
)

# Define a singlet lens from physical parameters
sistema, d1 = SistemaOptico.lsoe(
    zeta_0=80, zeta_1=90,      # vertex positions
    d_0=0, d_2=200,             # object and image positions
    n_0=1.0, n_1=1.6, n_2=1.0, # refractive indices
    sigma=0.0                   # shape factor (-1 to 1)
)

# Trace a fan of rays from a point source
fuente = np.array([0.0, 0.0, 0.0])
resultados = sistema.trazar_abanico(fuente, num_rayos=21, angulo_max=0.08)

# Visualize
colores = ['tab:blue'] * len(resultados)
graficar_seccion_transversal(sistema, resultados,
                              colores_rayos=colores, z_imagen=200.0)
graficar_3d(sistema, resultados,
            colores_rayos=colores, z_imagen=200.0)

# Export to STL (surfaces clipped at their mutual intersection)
exportar_sistema_stl(sistema, 'mi_lente.stl')
```

### Building a system manually

```python
from gots import calcular_gots, SuperficieCartesiana, SistemaOptico

# Compute GOTS parameters for each surface
p0 = calcular_gots(n_k=1.0, n_k1=1.6, zeta_k=80.0, d_k=0.0, d_k1=400.0)
p1 = calcular_gots(n_k=1.6, n_k1=1.0, zeta_k=90.0, d_k=400.0, d_k1=200.0)

# Build system
sistema = SistemaOptico()
sistema.agregar_superficie(SuperficieCartesiana(p0, n_k=1.0, n_k1=1.6))
sistema.agregar_superficie(SuperficieCartesiana(p1, n_k=1.6, n_k1=1.0))
```

## STL export

STL export is enabled by default in all examples. The surfaces are clipped at their physical intersection (lens aperture), not at the full oval extent. To disable it, pass `--no-stl` when running the examples.

The STL files are binary format and can be opened in any 3D viewer or slicer software (Blender, MeshLab, Cura, etc.).

## Package structure

```
gots/
    __init__.py              # Public API
    parametros_gots.py       # GOTS parameter computation (Eqs. 10-13)
    superficie_cartesiana.py # Surface geometry: z(rho), gradient, normal
    rayo.py                  # Ray-surface intersection via quartic (Eq. 51)
    snell.py                 # Vectorial Snell's law (Eq. 68)
    sistema_optico.py        # Multi-surface system, LSOE factory
    visualizacion.py         # 2D and 3D matplotlib plots
    exportar_stl.py          # Binary STL export
    utilidades.py            # Utilities (normalize, quartic solver)
ejemplos/
    ejemplo_lsoe.py          # Table 4 LSOE (Figs. 11-12)
    ejemplo_formas_lsoe.py   # Lens shapes vs sigma (Fig. 5)
    ejemplo_fig10.py         # Ray tracing gallery (Fig. 10)
```

## Reference

Silva Lora, A. L. (2024). *Estudio de las aberraciones primarias a partir de la teoria del estigmatismo riguroso*. Doctoral thesis, Universidad Industrial de Santander, Bucaramanga.

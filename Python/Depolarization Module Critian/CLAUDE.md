# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python module for simulating **light depolarization** in polarization optics. It models how partially coherent light loses polarization when passing through optical systems (waveplates, retarders), implementing both Jones and Mueller matrix formalisms.

## Running the Code

No build system — the module is imported directly:

```python
from Depo import State, Partial_State, Waveplate, Composite_Waveplate, Coherence
```

Dependencies: `numpy`, `sympy`, `pandas`, `matplotlib`, `scipy`, `mpmath`, `scienceplots`

Install with:
```bash
pip install numpy sympy pandas matplotlib scipy mpmath scienceplots
```

## Architecture

All code lives in `Depo.py`. The key class hierarchy:

### Polarization State Classes
- **`State`** — Fully polarized light as a Jones vector, parameterized by orientation angle `alpha` and ellipticity `chi`. Supports Poincaré sphere visualization and `operate()` to apply waveplate sequences.
- **`Partial_State`** — Partially polarized light defined by coherence matrix elements (J11, J12, J21, J22). Computes degree of polarization (DOP) and Stokes parameters.

### Optical Element Classes
- **`Waveplate`** — Jones matrix for a retarder with given OPD (optical path difference) and orientation angle. Has `operate(state, coherence)` to apply the plate accounting for coherence effects.
- **`Composite_Waveplate`** — Combines multiple `Waveplate` objects. The key complexity is `_subsetSums()`, which enumerates all interference OPD combinations between plates — this is where depolarization physics lives.
- **`Rotation`** — Jones/Mueller matrix for a polarization rotation.

### Source Coherence
- **`Coherence`** — Models temporal coherence with four profiles: `Gaussian`, `Lorentzian`, `GaussLorentz`, `BlackBody`. The `eval(OptPathDiff)` method returns the coherence degree at a given optical path difference, which modulates interference terms.

### Experimental Data
- **`Polarimeter_data`** — Reads CSV files from actual polarimeter measurements via `read_data()`.

### Key Helpers
- `Graphic()` — Renders a Poincaré sphere with polarization state points.
- `stokes_transformation()` — Returns the 4×4 matrix that converts Jones matrix to Mueller matrix via tensor product structure.

## Core Physics Pattern

The depolarization calculation follows this pattern:
1. Define a `Coherence` source (sets how quickly coherence decays with OPD)
2. Create `Waveplate` instances with specific OPDs and orientations
3. Combine into `Composite_Waveplate`
4. Call `operate(state, coherence)` — internally sums over all interference terms, weighting each by `coherence.eval(ΔOPD)`
5. Result is a `Partial_State` with reduced DOP

Both symbolic (`sympy`) and numeric (`numpy`/`mpmath`) computation are used — `sympy` for exact Mueller matrix derivations, numeric for evaluation.

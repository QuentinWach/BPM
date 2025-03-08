# BPM
Beam Propagation Method

**BPM** is a Python library for simulating beam propagation in integrated photonics using the Beam Propagation Method (BPM). The package provides functions to generate refractive index distributions for various structures (e.g., lenses, waveguides, and MMI splitters), a mode solver for slab waveguides, and BPM propagation routines with support for Perfectly Matched Layers (PML) for absorbing boundary conditions.

## Features

- Generate refractive index distributions:
  - Spherical lens
  - S-bend waveguide
  - MMI-based splitter
- Solve for guided slab waveguide modes (even/odd modes)
- BPM propagation using a Runge-Kutta integrator
- PML boundary absorption

## Installation

Clone the repository and install using pip:

```bash
git clone https://github.com/jwt625/bpm.git
cd bpm
pip install -e .
```

# References


Optical tomographic reconstruction based on multi-slice wave propagation method
- https://doi.org/10.1364/OE.25.022595

Light propagation through microlenses: a new simulation method
- https://doi.org/10.1364/AO.32.004984


Light propagation in graded-index optical fibers
- https://doi.org/10.1364/AO.17.003990
- this one is a classic


Numerical Simulation of Optical Wave Propagation with Examples in MATLAB
- https://spie.org/publications/book/866274

Photonic Devices for Telecommunications
- https://link.springer.com/book/10.1007/978-3-642-59889-0
- chapter 2

## Papers cited in chapter 2.2.1 of the book:

- Chung1990: An assessment of finite difference beam propagation method
	- https://ieeexplore.ieee.org/abstract/document/59679
- Vassallo1992: Improvement of finite difference methods for step-index optical waveguides
	- https://digital-library.theiet.org/doi/abs/10.1049/ip-j.1992.0024


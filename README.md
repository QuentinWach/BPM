# BPM (Beam Propagation Method)

**Update 202505**: I've "deployed" a [Gradio](https://www.gradio.app/) based GUI on Hugging Face, future development will likely continue there:
- [https://huggingface.co/spaces/jwt625/BPM](https://huggingface.co/spaces/jwt625/BPM)

**BPM** is a Python library for simulating beam propagation in integrated photonics using the Beam Propagation Method (BPM). The package provides functions to generate refractive index distributions for various structures (e.g., lenses, waveguides, and MMI splitters), a mode solver for slab waveguides, and BPM propagation routines with support for Perfectly Matched Layers (PML) for absorbing boundary conditions.

Currently it is 2D only, and use analytic solutions to launch slab modes. Propagation direction is upward and is called z. Transverse direction is x.

## Features

- Generate refractive index distributions:
  - Spherical lens
  - S-bend waveguide
  - MMI-based splitter
- Solve for guided slab waveguide modes (even/odd modes)
- BPM propagation using a Runge-Kutta integrator
- PML boundary absorption
- [] Import from GDSII

## Installation

Clone the repository and install using pip:

```bash
git clone https://github.com/jwt625/bpm.git
cd bpm
pip install -e .
```


## Examples

Slab mode solver and launcher:

![image](https://github.com/user-attachments/assets/5aad97f5-7521-431d-8578-9c7655831798)


Refractive index distribution of an MMI:

![image](https://github.com/user-attachments/assets/e15c0aac-7a6e-419b-9484-62c910e5ca1e)


Simulated example MMI:

![image](https://github.com/user-attachments/assets/d3ee4359-02d7-42cb-b568-d2f0eb55f7a0)


Simulated example S bend. The waveguide is multimode:

![image](https://github.com/user-attachments/assets/fdf6e0ba-5684-4312-8bb1-c2ab070e5de5)




## References


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

### Papers cited in chapter 2.2.1 of the book:

- Chung1990: An assessment of finite difference beam propagation method
	- https://ieeexplore.ieee.org/abstract/document/59679
- Vassallo1992: Improvement of finite difference methods for step-index optical waveguides
	- https://digital-library.theiet.org/doi/abs/10.1049/ip-j.1992.0024


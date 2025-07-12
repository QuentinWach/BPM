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

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager. Install it first if you haven't:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# or on macOS with Homebrew
brew install uv
```

Then clone and set up the project:

```bash
git clone https://github.com/jwt625/bpm.git
cd bpm

# Basic installation
uv pip install -e .

# With optional dependencies
uv pip install -e ".[test]"      # For running tests
uv pip install -e ".[examples]"  # For running examples
uv pip install -e ".[gui]"       # For GUI functionality
uv pip install -e ".[dev]"       # For development
uv pip install -e ".[all]"       # Install everything
```

### Using pip (Traditional)

```bash
git clone https://github.com/jwt625/bpm.git
cd bpm

# Basic installation
pip install -e .

# With optional dependencies
pip install -e ".[test]"      # For running tests
pip install -e ".[examples]"  # For running examples
pip install -e ".[gui]"       # For GUI functionality
pip install -e ".[dev]"       # For development
pip install -e ".[all]"       # Install everything
```

### Running the GUI

After installation, you can launch the interactive GUI:

```bash
# With uv
uv run bpm-gui

# With pip
bpm-gui
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

## Development

### Setting up the development environment

```bash
# Clone the repository
git clone https://github.com/jwt625/bpm.git
cd bpm

# Create virtual environment and install dependencies
uv sync --extra dev --extra gui

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=bpm --cov-report=html

# Format code
uv run black bpm/ tests/
uv run isort bpm/ tests/

# Type checking
uv run mypy bpm/

# Linting
uv run flake8 bpm/ tests/
```

### Project Structure

```
bpm/
├── bpm/                    # Main package
│   ├── __init__.py        # Package initialization
│   ├── core.py            # BPM propagation engine
│   ├── mode_solver.py     # Slab waveguide mode solver
│   ├── refractive_index.py # Structure generation
│   ├── pml.py             # Perfectly Matched Layer
│   └── app.py             # Gradio GUI application
├── tests/                 # Test suite
├── examples/              # Usage examples
├── pyproject.toml         # Project configuration
└── README.md              # This file
```




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

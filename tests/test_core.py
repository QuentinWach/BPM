#%%

import numpy as np
from bpm.core import run_bpm, compute_dE_dz
from bpm.pml import generate_sigma_x

def test_bpm_propagation():
    domain_size = 50
    wavelength = 0.532
    n0 = 1.0
    k0 = 2*np.pi/wavelength
    Nx = 256
    Nz = 100
    dx = domain_size / Nx
    dz = 2*dx
    x = np.linspace(-domain_size/2, domain_size/2, Nx)
    z = np.linspace(0, dz*Nz, Nz)
    
    # Create an initial Gaussian field
    E = np.exp(-x**2/4).astype(np.complex128)
    E = np.tile(E, (Nz, 1)).T  # shape (Nx, Nz)
    
    # Generate a dummy refractive index distribution (homogeneous)
    n_r2 = np.full((Nx, Nz), n0**2, dtype=np.float64)
    
    # Generate a PML damping profile in x
    sigma_x = generate_sigma_x(x, dx, wavelength, domain_size, sigma_max=0.5, pml_factor=5)
    
    E_out = run_bpm(E.copy(), n_r2, x, z, dx, dz, n0, sigma_x, wavelength)
    
    # Basic check: the field should remain finite
    assert np.all(np.isfinite(E_out))

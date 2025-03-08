import numpy as np
import matplotlib.pyplot as plt
from bpm.core import run_bpm, compute_dE_dz
from bpm.pml import generate_sigma_x

def test_core_propagation_plot():
    # Simulation parameters
    domain_size = 50  # um
    wavelength = 0.532
    n0 = 1.0
    Nx = 256
    Nz = 100
    dx = domain_size / Nx
    dz = 2 * dx
    x = np.linspace(-domain_size/2, domain_size/2, Nx)
    z = np.linspace(0, dz * Nz, Nz)
    
    # Create an initial Gaussian field
    E_init = np.exp(-x**2 / 2)
    # Make the field 2D (each column is the same as initial condition)
    E = np.tile(E_init, (Nz, 1)).T  # shape (Nx, Nz)
    
    # Use a homogeneous refractive index distribution (n0 everywhere)
    n_r2 = np.full((Nx, Nz), n0**2, dtype=np.float64)
    
    # Generate the PML damping profile in x
    sigma_x = generate_sigma_x(x, dx, wavelength, domain_size, sigma_max=0.5, pml_factor=5)
    
    # Run BPM propagation
    E_out = run_bpm(E.copy(), n_r2, x, z, dx, dz, n0, sigma_x, wavelength)
    
    # Plot the final intensity distribution
    plt.figure(figsize=(8, 6))
    plt.imshow(np.abs(E_out)**2, extent=[x[0], x[-1], z[0], z[-1]],
               origin='lower', aspect='auto', cmap='inferno')
    plt.xlabel("x (um)")
    plt.ylabel("z (um)")
    plt.title("Final Field Intensity from BPM Propagation")
    plt.colorbar(label="Intensity")
    plt.tight_layout()
    # Save the figure as a PNG file
    plt.savefig("core_test_field.png")
    plt.close()

if __name__ == "__main__":
    test_core_propagation_plot()

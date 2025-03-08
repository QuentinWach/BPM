import numpy as np
import matplotlib.pyplot as plt
from bpm.refractive_index import generate_MMI_n_r2

def test_refractive_index_distribution():
    # Simulation parameters
    domain_size = 50.0  # um, transverse extent
    Nx = 256
    Nz = 512
    x = np.linspace(-domain_size / 2, domain_size / 2, Nx)
    z = np.linspace(0, 200, Nz)
    
    # MMI structure parameters:
    z_MMI_start = 50.0    # MMI region begins at z = 50 um
    L_MMI = 130.0          # MMI region length = 40 um
    w_MMI = 8.0          # MMI region width = 40 um
    w_wg = 2.0            # input/output waveguide width = 4 um
    d = 4.0              # center-to-center separation of waveguides = 12 um
    
    # Refractive indices:
    n0 = 1.0      # background index
    n_WG = 1.1    # waveguide index
    n_MMI = 1.1   # MMI region index
    
    # Generate the refractive index distribution.
    n_r2 = generate_MMI_n_r2(x, z, z_MMI_start, L_MMI, w_MMI, w_wg, d, n_WG, n_MMI, n0)
    
    # Plot the refractive index distribution (plotting sqrt(n_r2) to get n_r).
    plt.figure(figsize=(8, 6))
    plt.imshow(np.sqrt(n_r2).T, extent=[x[0], x[-1], z[0], z[-1]],
               origin='lower', aspect='auto', cmap='viridis')
    plt.xlabel("x (um)")
    plt.ylabel("z (um)")
    plt.title("MMI Splitter Refractive Index Distribution (n_r)")
    plt.colorbar(label="n_r")
    plt.tight_layout()
    
    # Save the plot as a PNG file
    plt.savefig("refractive_index_test.png")
    plt.close()

if __name__ == "__main__":
    test_refractive_index_distribution()

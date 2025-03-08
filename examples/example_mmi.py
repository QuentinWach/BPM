import numpy as np
import matplotlib.pyplot as plt
from bpm.refractive_index import generate_MMI_n_r2
from bpm.mode_solver import slab_mode_source
from bpm.core import run_bpm
from bpm.pml import generate_sigma_x

# Simulation parameters
domain_size = 100.0   # um (transverse)
z_total = 200.0       # um (propagation length)
Nx = 512
Nz = 512
x = np.linspace(-domain_size/2, domain_size/2, Nx)
z = np.linspace(0, z_total, Nz)

# MMI structure parameters
z_MMI_start = 50.0    
L_MMI = 40.0          
w_MMI = 40.0          
w_wg = 4.0            
d = 12.0              

n0 = 1.0      
n_WG = 1.1    
n_MMI = 1.1   

n_r2 = generate_MMI_n_r2(x, z, z_MMI_start, L_MMI, w_MMI, w_wg, d, n_WG, n_MMI, n0)

# Launch a slab mode from the left input waveguide
# Shift the launched mode so that its center aligns with x = -d/2.
E0 = slab_mode_source(x, w=w_wg, n_WG=n_WG, n0=n0, wavelength=0.532, ind_m=0, x0=-d/2)

# Create initial field
E = np.zeros((Nx, Nz), dtype=np.complex128)
E[:, 0] = E0

# Generate PML profile in x
dx = domain_size / Nx
sigma_x = generate_sigma_x(x, dx, 0.532, domain_size, sigma_max=0.5, pml_factor=5)

# Run BPM propagation
E_out = run_bpm(E, n_r2, x, z, dx, z[1]-z[0], n0, sigma_x, 0.532)

# Plot final intensity
plt.figure(figsize=(8,6))
plt.imshow(np.abs(E_out)**2, extent=[x[0], x[-1], z[0], z[-1]], origin='lower', aspect='auto', cmap='inferno')
plt.xlabel("x (um)")
plt.ylabel("z (um)")
plt.title("MMI Splitter BPM Propagation")
plt.colorbar(label="Intensity")
plt.show()

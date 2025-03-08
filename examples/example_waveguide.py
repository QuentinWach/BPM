#%%
import numpy as np
import matplotlib.pyplot as plt
from bpm.refractive_index import generate_waveguide_n_r2
from bpm.mode_solver import slab_mode_source
from bpm.core import run_bpm
from bpm.pml import generate_sigma_x

# Simulation parameters
domain_size = 50.0   # um
z_total = 500.0      # um
Nx = 256
Nz = 2000
x = np.linspace(-domain_size/2, domain_size/2, Nx)
z = np.linspace(0, z_total, Nz)

# Waveguide parameters
l = 5.0   # lateral offset
L = 200.0  # length of S-bend
w = 1.0    # waveguide width
n0 = 1.0
n_WG = 1.1

n_r2 = generate_waveguide_n_r2(x, z, l, L, w, n_WG, n0)

# Launch a mode; here we use a mode source with no shift (x0 = 0)
E0 = slab_mode_source(x, w, n_WG, n0, wavelength=0.532, ind_m=0, x0=0)
E = np.zeros((Nx, Nz), dtype=np.complex128)
E[:, 0] = E0

dx = domain_size / Nx
sigma_x = generate_sigma_x(x, dx, 0.532, domain_size, sigma_max=0.5, pml_factor=5)

E_out = run_bpm(E, n_r2, x, z, dx, z[1]-z[0], n0, sigma_x, 0.532)

plt.figure(figsize=(8,6))
plt.imshow((np.abs(E_out)**2).T, extent=[x[0], x[-1], z[0], z[-1]],
           origin='lower', aspect='auto', cmap='inferno',
           vmin=0, vmax=0.6)
plt.xlabel("x (um)")
plt.ylabel("z (um)")
plt.title("Waveguide BPM Propagation")
plt.colorbar(label="Intensity")
plt.show()

# %%

#%%
import numpy as np
import plotly.graph_objects as go
from bpm.refractive_index import generate_MMI_n_r2
from bpm.mode_solver import slab_mode_source
from bpm.core import run_bpm
from bpm.pml import generate_sigma_x

# Simulation parameters
domain_size = 50.0   # um (transverse)
z_total = 250.0       # um (propagation length)
Nx = 256
Nz = 1024
x = np.linspace(-domain_size/2, domain_size/2, Nx)
z = np.linspace(0, z_total, Nz)

# MMI structure parameters
z_MMI_start = 50.0    
L_MMI = 130.0          # MMI region length = 40 um
w_MMI = 8.0          # MMI region width = 40 um
w_wg = 2.0            
d = 4.0              

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

# Plot final intensity using Plotly
fig1 = go.Figure(data=go.Heatmap(
    z=(np.abs(E_out)**2).T,
    x=x,
    y=z,
    colorscale='inferno',
    colorbar=dict(title='Intensity')
))

fig1.update_layout(
    title='MMI Splitter BPM Propagation',
    xaxis_title='x (um)',
    yaxis_title='z (um)',
    width=800,
    height=600
)

fig1.show()

# Plot refractive index profile using Plotly
fig2 = go.Figure(data=go.Heatmap(
    z=np.sqrt(n_r2).T,
    x=x,
    y=z,
    colorscale='inferno',
    colorbar=dict(title='Refractive Index')
))

fig2.update_layout(
    title='Refractive Index Profile',
    xaxis_title='x (um)',
    yaxis_title='z (um)',
    width=800,
    height=600
)

fig2.show()
# %%

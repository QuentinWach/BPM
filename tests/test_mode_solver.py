import numpy as np
from bpm.mode_solver import slab_mode_source

def test_slab_mode_normalization():
    x = np.linspace(-10, 10, 1000)
    w = 5.0
    n_WG = 1.1
    n0 = 1.0
    wavelength = 0.532
    E = slab_mode_source(x, w, n_WG, n0, wavelength, ind_m=0, x0=0)
    norm = np.trapz(np.abs(E)**2, x)
    # Check that the mode is normalized (within a tolerance)
    assert np.abs(norm - 1) < 1e-3

import numpy as np
from bpm.refractive_index import generate_MMI_n_r2

def test_generate_MMI_n_r2_shape():
    x = np.linspace(-50, 50, 256)
    z = np.linspace(0, 200, 256)
    n_r2 = generate_MMI_n_r2(x, z, z_MMI_start=50, L_MMI=40, w_MMI=40, w_wg=4, d=12, n_WG=1.1, n_MMI=1.1, n0=1.0)
    assert n_r2.shape == (256, 256)

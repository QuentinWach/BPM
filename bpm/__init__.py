"""
my_bpm: A beam propagation method (BPM) simulation library for integrated photonics.
"""

from .core import run_bpm, compute_dE_dz
from .refractive_index import generate_lens_n_r2, generate_waveguide_n_r2, generate_MMI_n_r2
from .mode_solver import slab_mode_source
from .pml import generate_sigma_x

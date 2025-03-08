import numpy as np

def generate_sigma_x(x, dx, wavelength, domain_size, sigma_max=0.5, pml_factor=5):
    """
    Generate the 1D PML damping profile sigma(x) for the x-dimension.
    
    Parameters:
      x : 1D numpy array of x coordinates.
      dx : grid spacing in x.
      wavelength : wavelength in microns.
      domain_size : total transverse domain size (in microns).
      sigma_max : maximum damping value (default 0.5).
      pml_factor : number of wavelengths to use for the PML thickness.
    
    Returns:
      sigma_x : 1D numpy array of damping values.
    """
    pml_thickness = int(pml_factor * wavelength / dx)
    pml_width = pml_thickness * dx
    x_edge = domain_size / 2 - pml_width
    sigma_x = np.where(np.abs(x) > x_edge,
                       sigma_max * ((np.abs(x) - x_edge) / pml_width) ** 2,
                       0)
    return sigma_x

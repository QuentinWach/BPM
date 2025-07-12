import numpy as np

def generate_sigma_x(x, dx, wavelength, domain_size, sigma_max=0.5, pml_factor=5):
    """
    Generate the 1D PML damping profile sigma(x) for the x-dimension.

    Parameters:
    -----------
    x : array_like
        1D numpy array of x coordinates
    dx : float
        Grid spacing in x (must be positive)
    wavelength : float
        Wavelength in microns (must be positive)
    domain_size : float
        Total transverse domain size in microns (must be positive)
    sigma_max : float, optional
        Maximum damping value (default 0.5, must be non-negative)
    pml_factor : float, optional
        Number of wavelengths to use for PML thickness (default 5, must be positive)

    Returns:
    --------
    sigma_x : ndarray
        1D numpy array of damping values
    """
    # Input validation
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be 1D array")
    if dx <= 0:
        raise ValueError("Grid spacing dx must be positive")
    if wavelength <= 0:
        raise ValueError("Wavelength must be positive")
    if domain_size <= 0:
        raise ValueError("Domain size must be positive")
    if sigma_max < 0:
        raise ValueError("sigma_max must be non-negative")
    if pml_factor <= 0:
        raise ValueError("pml_factor must be positive")

    pml_thickness = int(pml_factor * wavelength / dx)
    pml_width = pml_thickness * dx
    x_edge = domain_size / 2 - pml_width
    sigma_x = np.where(np.abs(x) > x_edge,
                       sigma_max * ((np.abs(x) - x_edge) / pml_width) ** 2,
                       0)
    return sigma_x

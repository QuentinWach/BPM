#!/usr/bin/env python3
"""
Tests for numerical stability conditions.
"""
import numpy as np
import pytest
import warnings
from bpm.core import validate_stability_conditions, run_bpm
from bpm.mode_solver import slab_mode_source
from bpm.pml import generate_sigma_x
from bpm.refractive_index import generate_waveguide_n_r2


def test_stability_conditions_pass():
    """Test that good parameters pass stability checks."""
    # Good parameters
    dx = 0.02  # μm - smaller to satisfy sampling
    dz = 0.001  # μm - smaller to satisfy CFL
    wavelength = 0.532  # μm
    n_max = 1.5
    
    # Should pass without warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        result = validate_stability_conditions(dx, dz, wavelength, n_max)
        assert result is True


def test_stability_conditions_sampling():
    """Test sampling condition violation."""
    # Bad sampling: dx too large
    dx = 1.0  # μm - way too large
    dz = 0.01  # μm
    wavelength = 0.532  # μm
    n_max = 1.5
    
    with pytest.warns(UserWarning, match="Sampling condition violated"):
        result = validate_stability_conditions(dx, dz, wavelength, n_max)
        assert result is False


def test_stability_conditions_cfl():
    """Test CFL-like condition violation."""
    # Bad CFL: dz too large
    dx = 0.05  # μm
    dz = 1.0   # μm - way too large
    wavelength = 0.532  # μm
    n_max = 1.5
    
    with pytest.warns(UserWarning, match="CFL-like condition violated"):
        result = validate_stability_conditions(dx, dz, wavelength, n_max)
        assert result is False


def test_stability_conditions_paraxial():
    """Test paraxial condition violation."""
    # Bad paraxial: dx too large for given wavelength and index
    dx = 0.5   # μm - too large
    dz = 0.01  # μm
    wavelength = 0.532  # μm
    n_max = 3.0  # High index
    
    with pytest.warns(UserWarning, match="Paraxial condition marginal"):
        result = validate_stability_conditions(dx, dz, wavelength, n_max)
        assert result is False


def test_stability_integration_with_bpm():
    """Test that stability checks are integrated into run_bpm."""
    # Create a simulation with bad parameters
    Nx, Nz = 50, 100
    domain_size = 10.0
    z_total = 20.0
    wavelength = 0.532
    n0 = 1.0
    
    x = np.linspace(-domain_size/2, domain_size/2, Nx)
    z = np.linspace(0, z_total, Nz)
    dx = x[1] - x[0]  # This will be ~0.2 μm
    dz = z[1] - z[0]  # This will be ~0.2 μm - too large!
    
    # Create structures
    n_r2 = generate_waveguide_n_r2(x, z, 0, z_total, 2.0, 1.5, n0)
    E0 = slab_mode_source(x, 2.0, 1.5, n0, wavelength)
    E = np.zeros((Nx, Nz), dtype=np.complex128)
    E[:, 0] = E0
    sigma_x = generate_sigma_x(x, dx, wavelength, domain_size)
    
    # Should issue stability warnings
    with pytest.warns(UserWarning):
        E_out = run_bpm(E, n_r2, x, z, dx, dz, n0, sigma_x, wavelength)
    
    # But should still complete
    assert E_out.shape == E.shape
    assert np.all(np.isfinite(E_out))


def test_stability_conditions_no_warn():
    """Test that warnings can be disabled."""
    # Bad parameters but warnings disabled
    dx = 1.0  # μm - too large
    dz = 1.0  # μm - too large
    wavelength = 0.532  # μm
    n_max = 1.5
    
    # Should not warn when warn=False
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        result = validate_stability_conditions(dx, dz, wavelength, n_max, warn=False)
        assert result is False  # Still returns False, just no warning


if __name__ == "__main__":
    print("Running stability condition tests...")
    test_stability_conditions_pass()
    test_stability_conditions_sampling()
    test_stability_conditions_cfl()
    test_stability_conditions_paraxial()
    test_stability_integration_with_bpm()
    test_stability_conditions_no_warn()
    print("✅ All stability tests passed!")

#!/usr/bin/env python3
"""
Performance benchmarks for BPM library.
"""
import time

import numpy as np

from bpm.core import compute_dE_dz, run_bpm
from bpm.mode_solver import slab_mode_source
from bpm.pml import generate_sigma_x
from bpm.refractive_index import generate_waveguide_n_r2


def benchmark_function(func, *args, **kwargs):
    """Benchmark a function and return execution time."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    return result, end_time - start_time


def test_laplacian_performance():
    """Benchmark the optimized Laplacian computation."""
    # Setup test data
    Nx = 1000
    x = np.linspace(-25, 25, Nx)
    dx = x[1] - x[0]
    E_slice = np.exp(-(x**2) / 4) * (1 + 0.1j)  # Complex Gaussian
    n_r2_slice = np.ones(Nx) * 1.5**2
    sigma_x = np.zeros(Nx)
    k0 = 2 * np.pi / 0.532
    n0 = 1.0

    # Benchmark the compute_dE_dz function
    _, exec_time = benchmark_function(
        compute_dE_dz, E_slice, n_r2_slice, dx, n0, sigma_x, k0
    )

    print(f"Laplacian computation time for {Nx} points: {exec_time:.4f} seconds")

    # Performance should be reasonable (< 1ms for 1000 points)
    assert exec_time < 0.001, f"Laplacian computation too slow: {exec_time:.4f}s"


def test_bpm_propagation_performance():
    """Benchmark BPM propagation for different grid sizes."""
    wavelength = 0.532
    n0 = 1.0

    grid_sizes = [(100, 50), (200, 100), (500, 200)]

    for Nx, Nz in grid_sizes:
        # Setup simulation
        domain_size = 20.0
        z_total = 10.0
        x = np.linspace(-domain_size / 2, domain_size / 2, Nx)
        z = np.linspace(0, z_total, Nz)
        dx = x[1] - x[0]
        dz = z[1] - z[0]

        # Create simple waveguide
        n_r2 = generate_waveguide_n_r2(x, z, 0, z_total, 2.0, 1.5, n0)

        # Initial field
        E0 = slab_mode_source(x, 2.0, 1.5, n0, wavelength)
        E = np.zeros((Nx, Nz), dtype=np.complex128)
        E[:, 0] = E0

        # PML
        sigma_x = generate_sigma_x(x, dx, wavelength, domain_size)

        # Benchmark propagation
        _, exec_time = benchmark_function(
            run_bpm, E.copy(), n_r2, x, z, dx, dz, n0, sigma_x, wavelength
        )

        points = Nx * Nz
        print(f"BPM propagation ({Nx}x{Nz}={points} points): {exec_time:.4f}s")

        # Performance scaling check (should be roughly linear with grid size)
        time_per_point = exec_time / points
        assert time_per_point < 1e-5, f"BPM too slow: {time_per_point:.2e}s/point"


def test_memory_usage():
    """Test memory efficiency of BPM propagation."""
    try:
        import os

        import psutil
    except ImportError:
        print("psutil not available, skipping memory test")
        return

    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Run a moderately large simulation
    Nx, Nz = 500, 1000
    domain_size = 20.0
    z_total = 50.0
    wavelength = 0.532
    n0 = 1.0

    x = np.linspace(-domain_size / 2, domain_size / 2, Nx)
    z = np.linspace(0, z_total, Nz)
    dx = x[1] - x[0]
    dz = z[1] - z[0]

    n_r2 = generate_waveguide_n_r2(x, z, 0, z_total, 2.0, 1.5, n0)
    E0 = slab_mode_source(x, 2.0, 1.5, n0, wavelength)
    E = np.zeros((Nx, Nz), dtype=np.complex128)
    E[:, 0] = E0
    sigma_x = generate_sigma_x(x, dx, wavelength, domain_size)

    # Run simulation
    E_out = run_bpm(E, n_r2, x, z, dx, dz, n0, sigma_x, wavelength)

    # Check final memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory

    print(
        f"Memory usage: {initial_memory:.1f} -> {final_memory:.1f} MB (+{memory_increase:.1f} MB)"
    )

    # Memory increase should be reasonable (< 100 MB for this test)
    assert memory_increase < 100, f"Excessive memory usage: {memory_increase:.1f} MB"

    # Verify result is valid
    assert np.all(np.isfinite(E_out)), "BPM result contains invalid values"


if __name__ == "__main__":
    print("Running BPM Performance Benchmarks...")
    test_laplacian_performance()
    test_bpm_propagation_performance()
    test_memory_usage()
    print("✅ All performance tests passed!")

# RFD-000: BPM Code Quality and Feature Improvements

**Status**: Draft  
**Author**: Augment Agent  
**Date**: 2025-07-12  
**Version**: 1.0  

## Abstract

This RFD outlines comprehensive improvements for the BPM (Beam Propagation Method) library to enhance code quality, fix existing issues, improve performance, and add missing features. The improvements are categorized by priority and include detailed implementation steps.

DO NOT WRITE REDUNDANT CODE OR SLOP CODE. BE CONCISE AND EFFICIENT.

## Background

The BPM library is a well-structured Python package for simulating electromagnetic wave propagation in integrated photonic devices. While the core functionality is solid, several areas need improvement:

1. **Code Quality Issues**: Warnings, deprecated functions, missing error handling
2. **Performance Bottlenecks**: Unoptimized loops, lack of vectorization
3. **Missing Features**: GDSII import, advanced structures, 3D support
4. **Documentation Gaps**: API docs, theoretical background, examples
5. **Testing Coverage**: Edge cases, performance tests, integration tests

## Detailed Improvement Plan

### Priority 1: Critical Fixes (Immediate)

#### 1.1 Fix Complex Casting Warning in core.py

**Issue**: Line 45 in `bpm/core.py` generates ComplexWarning when assigning complex values.

**Root Cause**: 
```python
E[:, zi] = E_prev + (k1 + 2*k2 + 2*k3 + k4) / 6  # Line 45
```
The field array `E` might be initialized as real-valued, causing casting issues.

**Solution Steps**:
1. **Investigate field initialization**
   - [ ] Check how `E` array is created in `run_bpm` function
   - [ ] Verify dtype is `complex128` throughout
   - [ ] Add explicit dtype specification

2. **Fix the assignment**
   ```python
   # Before (problematic)
   E[:, zi] = E_prev + (k1 + 2*k2 + 2*k3 + k4) / 6
   
   # After (fixed)
   E[:, zi] = E_prev + (k1 + 2*k2 + 2*k3 + k4) / 6
   # Ensure E is initialized as: E = np.zeros((Nx, Nz), dtype=np.complex128)
   ```

3. **Add input validation**
   - [ ] Validate input field is complex
   - [ ] Add dtype checks in function signature
   - [ ] Provide clear error messages

**TODO Items**:
- [x] Modify `run_bpm` to ensure complex field initialization
- [x] Add unit test for complex field handling
- [x] Update docstring with dtype requirements

#### 1.2 Replace Deprecated trapz Function

**Issue**: `np.trapz` is deprecated in favor of `np.trapezoid` (line 130 in `mode_solver.py`).

**Solution Steps**:
1. **Update function call**
   ```python
   # Before
   norm = np.sqrt(np.trapz(np.abs(E)**2, x))
   
   # After
   norm = np.sqrt(np.trapezoid(np.abs(E)**2, x))
   ```

2. **Add compatibility check**
   ```python
   # Add at top of file for backward compatibility
   try:
       from numpy import trapezoid
   except ImportError:
       from numpy import trapz as trapezoid
   ```

**TODO Items**:
- [x] Replace `np.trapz` with `np.trapezoid`
- [x] Add numpy version compatibility check
- [x] Test with different numpy versions
- [x] Update requirements to specify minimum numpy version

#### 1.3 Add Input Validation

**Issue**: Functions lack proper input validation, leading to potential runtime errors.

**Solution Steps**:
1. **Core function validation**
   ```python
   def run_bpm(E, n_r2, x, z, dx, dz, n0, sigma_x, wavelength):
       # Add validation
       if E.dtype != np.complex128:
           raise ValueError("Field E must be complex128")
       if E.shape != n_r2.shape:
           raise ValueError("E and n_r2 must have same shape")
       if len(x) != E.shape[0]:
           raise ValueError("x length must match E first dimension")
       # ... more validations
   ```

2. **Validation helper functions**
   ```python
   def validate_grid_spacing(dx, dz, wavelength, n_max):
       """Validate grid spacing for numerical stability."""
       # CFL condition check
       # Sampling requirements
       pass
   
   def validate_field_shape(E, n_r2, x, z):
       """Validate field and index arrays have consistent shapes."""
       pass
   ```

**TODO Items**:
- [x] Add validation to all public functions
- [x] Create validation helper module
- [x] Add stability condition checks (CFL, sampling)
- [x] Write validation tests

### Priority 2: Performance Improvements (High)

#### 2.1 Vectorize BPM Loop

**Issue**: The main BPM loop in `run_bpm` processes one z-slice at a time, missing vectorization opportunities.

**Current Implementation**:
```python
for zi in range(1, Nz):
    E_prev = E[:, zi-1]
    n_r2_slice = n_r2[:, zi-1]
    # RK4 steps...
    E[:, zi] = E_prev + (k1 + 2*k2 + 2*k3 + k4) / 6
```

**Optimized Implementation**:
```python
# Option 1: Batch processing
def run_bpm_vectorized(E, n_r2, x, z, dx, dz, n0, sigma_x, wavelength, batch_size=10):
    """Vectorized BPM with batch processing."""
    # Process multiple z-slices simultaneously
    pass

# Option 2: Numba JIT compilation
from numba import jit

@jit(nopython=True)
def compute_dE_dz_jit(E_slice, n_r2_slice, dx, n0, sigma_x, k0):
    """JIT-compiled derivative computation."""
    pass
```

**TODO Items**:
- [ ] Implement batch processing for multiple z-slices
- [ ] Add numba JIT compilation option
- [x] Benchmark performance improvements
- [x] Add performance tests
- [ ] Make optimization level configurable

#### 2.2 Optimize Finite Difference Operations

**Issue**: Laplacian computation uses `np.roll` which can be optimized.

**Current Implementation**:
```python
laplacian_E = (np.roll(E_slice, 1, axis=0) - 2 * E_slice + np.roll(E_slice, -1, axis=0)) / dx**2
```

**Optimized Implementation**:
```python
# Option 1: Direct indexing (faster for small arrays)
def compute_laplacian_direct(E_slice, dx):
    laplacian = np.zeros_like(E_slice)
    laplacian[1:-1] = (E_slice[2:] - 2*E_slice[1:-1] + E_slice[:-2]) / dx**2
    # Handle boundaries
    return laplacian

# Option 2: Scipy sparse matrices (for large arrays)
from scipy.sparse import diags
def create_laplacian_matrix(N, dx):
    """Create sparse Laplacian matrix."""
    diagonals = [1, -2, 1]
    offsets = [-1, 0, 1]
    return diags(diagonals, offsets, shape=(N, N)) / dx**2
```

**TODO Items**:
- [x] Implement direct indexing Laplacian
- [ ] Add sparse matrix option for large grids
- [x] Benchmark different approaches
- [x] Add boundary condition handling
- [ ] Make method selectable via parameter

### Priority 3: Feature Enhancements (Medium)

#### 3.1 Implement GDSII Import

**Issue**: GDSII import is marked as TODO but not implemented.

**Implementation Plan**:
```python
# New module: bpm/gdsii_import.py
import gdspy  # or gdstk for newer implementation

def import_gds_structure(gds_file, layer, x_grid, z_grid, n_core, n_clad):
    """
    Import structure from GDSII file and convert to refractive index map.
    
    Parameters:
    -----------
    gds_file : str
        Path to GDSII file
    layer : tuple
        (layer_number, datatype) to import
    x_grid, z_grid : array_like
        Coordinate grids for discretization
    n_core, n_clad : float
        Refractive indices for core and cladding
        
    Returns:
    --------
    n_r2 : ndarray
        Squared refractive index distribution
    """
    pass

def gds_to_index_map(polygons, x_grid, z_grid, n_core, n_clad):
    """Convert GDS polygons to refractive index map."""
    pass
```

**TODO Items**:
- [ ] Choose GDS library (gdspy vs gdstk)
- [ ] Implement polygon rasterization
- [ ] Add multi-layer support
- [ ] Handle curved structures (approximation)
- [ ] Add validation and error handling
- [ ] Write comprehensive tests
- [ ] Create example GDS files
- [ ] Document GDS workflow

#### 3.2 Add Advanced Photonic Structures

**Issue**: Limited structure library, missing common photonic components.

**New Structures to Implement**:

1. **Ring Resonators**
   ```python
   def generate_ring_resonator_n_r2(x, z, center, radius, width, gap, n_WG, n0):
       """Generate ring resonator structure."""
       pass
   ```

2. **Directional Couplers**
   ```python
   def generate_directional_coupler_n_r2(x, z, length, gap, width, n_WG, n0):
       """Generate directional coupler structure."""
       pass
   ```

3. **Bragg Gratings**
   ```python
   def generate_bragg_grating_n_r2(x, z, period, duty_cycle, n_high, n_low, n0):
       """Generate Bragg grating structure."""
       pass
   ```

4. **Tapered Waveguides**
   ```python
   def generate_taper_n_r2(x, z, w_start, w_end, length, n_WG, n0):
       """Generate tapered waveguide structure."""
       pass
   ```

**TODO Items**:
- [ ] Implement ring resonator generator
- [ ] Add directional coupler support
- [ ] Create Bragg grating function
- [ ] Add taper and bend structures
- [ ] Implement photonic crystal structures
- [ ] Add structure validation
- [ ] Create structure gallery examples
- [ ] Add structure optimization tools

#### 3.3 Enhanced Mode Solver

**Issue**: Current mode solver only handles step-index slab waveguides.

**Enhancements**:

1. **Graded-Index Support**
   ```python
   def solve_graded_index_modes(x, n_profile, wavelength, num_modes=5):
       """Solve modes for arbitrary refractive index profile."""
       # Use finite difference or shooting method
       pass
   ```

2. **2D Mode Solver**
   ```python
   def solve_2d_modes(x, y, n_r2, wavelength, num_modes=5):
       """Solve 2D waveguide modes using finite difference."""
       # Implement 2D eigenvalue solver
       pass
   ```

3. **Mode Overlap Calculations**
   ```python
   def calculate_mode_overlap(mode1, mode2, x):
       """Calculate overlap integral between two modes."""
       pass
   
   def calculate_coupling_coefficient(mode1, mode2, dn_perturbation, x):
       """Calculate coupling coefficient due to perturbation."""
       pass
   ```

**TODO Items**:
- [ ] Implement graded-index mode solver
- [ ] Add 2D mode solving capability
- [ ] Create mode analysis tools
- [ ] Add mode overlap calculations
- [ ] Implement effective index method
- [ ] Add mode visualization tools
- [ ] Create mode database system
- [ ] Add mode fitting algorithms

### Priority 4: Documentation and Testing (Medium)

#### 4.1 Comprehensive API Documentation

**Issue**: Missing detailed API documentation with examples.

**Documentation Structure**:
```
docs/
├── api/
│   ├── core.md
│   ├── mode_solver.md
│   ├── refractive_index.md
│   └── pml.md
├── theory/
│   ├── bpm_theory.md
│   ├── mode_theory.md
│   └── pml_theory.md
├── tutorials/
│   ├── getting_started.md
│   ├── advanced_structures.md
│   └── performance_optimization.md
└── examples/
    ├── basic_waveguide.md
    ├── mmi_splitter.md
    └── custom_structures.md
```

**TODO Items**:
- [ ] Set up documentation framework (Sphinx/MkDocs)
- [ ] Write comprehensive API documentation
- [ ] Add theoretical background sections
- [ ] Create step-by-step tutorials
- [ ] Add interactive examples
- [ ] Set up automatic documentation generation
- [ ] Add documentation tests
- [ ] Deploy documentation website

#### 4.2 Enhanced Testing Suite

**Issue**: Limited test coverage, missing edge cases and performance tests.

**Test Categories**:

1. **Unit Tests**
   ```python
   # tests/test_core_advanced.py
   def test_complex_field_handling():
       """Test complex field initialization and propagation."""
       pass
   
   def test_stability_conditions():
       """Test numerical stability under various conditions."""
       pass
   
   def test_boundary_conditions():
       """Test different boundary condition implementations."""
       pass
   ```

2. **Integration Tests**
   ```python
   # tests/test_integration.py
   def test_waveguide_propagation_accuracy():
       """Test against analytical solutions."""
       pass
   
   def test_mode_launching_efficiency():
       """Test mode launching and propagation."""
       pass
   ```

3. **Performance Tests**
   ```python
   # tests/test_performance.py
   def test_large_grid_performance():
       """Benchmark performance on large grids."""
       pass
   
   def test_memory_usage():
       """Monitor memory usage during simulation."""
       pass
   ```

**TODO Items**:
- [ ] Expand unit test coverage to >90%
- [ ] Add integration tests with analytical solutions
- [ ] Implement performance benchmarking
- [ ] Add property-based testing
- [ ] Create test data generators
- [ ] Add continuous integration tests
- [ ] Set up test coverage reporting
- [ ] Add regression tests

### Priority 5: Advanced Features (Low)

#### 5.1 3D BPM Implementation

**Issue**: Current implementation is 2D only.

**Implementation Approach**:
```python
# bpm/core_3d.py
def run_bpm_3d(E, n_r3, x, y, z, dx, dy, dz, n0, sigma_x, sigma_y, wavelength):
    """
    3D BPM propagation with alternating direction implicit (ADI) method.
    """
    # Implement ADI splitting
    # Handle 3D Laplacian
    # Manage memory efficiently
    pass
```

**TODO Items**:
- [ ] Design 3D data structures
- [ ] Implement ADI splitting method
- [ ] Add 3D PML boundaries
- [ ] Optimize memory usage
- [ ] Add 3D visualization
- [ ] Create 3D examples
- [ ] Benchmark 3D performance
- [ ] Add parallel processing support

#### 5.2 Nonlinear and Anisotropic Materials

**Issue**: Only linear, isotropic materials supported.

**Extensions**:
```python
def compute_nonlinear_response(E, chi3, intensity_threshold):
    """Compute nonlinear refractive index change."""
    pass

def apply_anisotropic_tensor(E, epsilon_tensor):
    """Apply anisotropic material tensor."""
    pass
```

**TODO Items**:
- [ ] Add Kerr nonlinearity support
- [ ] Implement anisotropic material tensors
- [ ] Add electro-optic effects
- [ ] Support dispersive materials
- [ ] Add gain/loss materials
- [ ] Create material database
- [ ] Add material fitting tools
- [ ] Implement coupled-mode theory

## Implementation Timeline

### Phase 1 (Weeks 1-2): Critical Fixes
- Fix complex casting warning
- Replace deprecated functions
- Add input validation
- Basic performance improvements

### Phase 2 (Weeks 3-4): Core Enhancements
- Vectorize BPM loop
- Optimize finite differences
- Enhanced error handling
- Improved testing

### Phase 3 (Weeks 5-8): Feature Development
- GDSII import functionality
- Advanced structures library
- Enhanced mode solver
- Comprehensive documentation

### Phase 4 (Weeks 9-12): Advanced Features
- 3D BPM implementation
- Nonlinear materials
- Performance optimization
- Production deployment

## Success Metrics

1. **Code Quality**
   - Zero warnings in test suite
   - >90% test coverage
   - All linting checks pass

2. **Performance**
   - 5x speedup for typical simulations
   - Memory usage optimization
   - Scalability to large grids

3. **Features**
   - GDSII import working
   - 10+ photonic structures available
   - 3D capability demonstrated

4. **Documentation**
   - Complete API documentation
   - 5+ comprehensive tutorials
   - Interactive examples

## Conclusion

This RFD provides a comprehensive roadmap for improving the BPM library. The improvements are prioritized to address critical issues first while building toward advanced features. Implementation should follow the phased approach to ensure stability and usability throughout the development process.

## Detailed Implementation Guides

### Code Quality Fixes - Step-by-Step

#### Fix 1: Complex Field Handling

**File**: `bpm/core.py`

**Current Issue**:
```python
# Line 45 - causes ComplexWarning
E[:, zi] = E_prev + (k1 + 2*k2 + 2*k3 + k4) / 6
```

**Step-by-step Fix**:

1. **Modify run_bpm function signature**:
   ```python
   def run_bpm(E, n_r2, x, z, dx, dz, n0, sigma_x, wavelength):
       """
       Run the BPM propagation using an RK4 integrator.

       Parameters:
         E: initial field (2D array, shape (len(x), len(z)), dtype=complex128)
         ...
       """
       # Add input validation
       if not np.iscomplexobj(E):
           E = E.astype(np.complex128)
           warnings.warn("Converting field to complex128", UserWarning)

       # Ensure proper dtype
       E = np.asarray(E, dtype=np.complex128)
   ```

2. **Update field initialization in examples**:
   ```python
   # In examples/example_waveguide.py and example_mmi.py
   # Change from:
   E = np.zeros((Nx, Nz), dtype=np.complex128)  # Make explicit
   E[:, 0] = E0
   ```

3. **Add validation helper**:
   ```python
   def _validate_field_array(E, name="E"):
       """Validate field array properties."""
       if not isinstance(E, np.ndarray):
           raise TypeError(f"{name} must be numpy array")
       if E.ndim != 2:
           raise ValueError(f"{name} must be 2D array")
       if not np.iscomplexobj(E):
           raise TypeError(f"{name} must be complex array")
       return True
   ```

**TODO Checklist**:
- [ ] Add input validation to `run_bpm`
- [ ] Update all example files
- [ ] Add validation helper functions
- [ ] Write unit tests for complex field handling
- [ ] Update docstrings with dtype requirements

#### Fix 2: Deprecated Function Replacement

**File**: `bpm/mode_solver.py`

**Current Issue**:
```python
# Line 130 - uses deprecated np.trapz
norm = np.sqrt(np.trapz(np.abs(E)**2, x))
```

**Step-by-step Fix**:

1. **Add compatibility import**:
   ```python
   # At top of mode_solver.py
   import numpy as np
   import warnings

   # Handle numpy version compatibility
   try:
       from numpy import trapezoid
   except ImportError:
       # Fallback for older numpy versions
       from numpy import trapz as trapezoid
       warnings.warn(
           "Using deprecated trapz. Please upgrade numpy >= 1.22.0",
           DeprecationWarning
       )
   ```

2. **Replace function call**:
   ```python
   # Line 130: Replace
   norm = np.sqrt(np.trapz(np.abs(E)**2, x))
   # With:
   norm = np.sqrt(trapezoid(np.abs(E)**2, x))
   ```

3. **Add version check in setup**:
   ```python
   # In pyproject.toml, update numpy requirement
   dependencies = [
       "numpy>=1.22.0",  # Ensures trapezoid is available
       "matplotlib>=3.5.0",
   ]
   ```

**TODO Checklist**:
- [ ] Add compatibility import
- [ ] Replace all instances of `trapz`
- [ ] Update numpy version requirement
- [ ] Test with different numpy versions
- [ ] Add deprecation handling tests

### Performance Optimization Details

#### Optimization 1: Numba JIT Compilation

**Implementation**:

1. **Create optimized core module**:
   ```python
   # bpm/core_optimized.py
   import numpy as np
   from numba import jit, prange

   @jit(nopython=True, parallel=True)
   def compute_dE_dz_jit(E_slice, n_r2_slice, dx, n0, sigma_x, k0):
       """JIT-compiled derivative computation."""
       Nx = E_slice.shape[0]
       dE_dz = np.zeros_like(E_slice)

       # Vectorized Laplacian computation
       for i in prange(1, Nx-1):
           laplacian = (E_slice[i+1] - 2*E_slice[i] + E_slice[i-1]) / (dx*dx)
           index_term = 1j * (k0 / (2 * n0)) * (n_r2_slice[i] - n0*n0) * E_slice[i]
           damping_term = -sigma_x[i] * E_slice[i]
           dE_dz[i] = (1j / (2 * k0 * n0)) * laplacian + index_term + damping_term

       # Handle boundaries
       # ... boundary condition code

       return dE_dz

   @jit(nopython=True)
   def run_bpm_jit(E, n_r2, dx, dz, n0, sigma_x, k0):
       """JIT-compiled BPM propagation."""
       Nx, Nz = E.shape

       for zi in range(1, Nz):
           E_prev = E[:, zi-1]
           n_r2_slice = n_r2[:, zi-1]

           # RK4 integration
           k1 = dz * compute_dE_dz_jit(E_prev, n_r2_slice, dx, n0, sigma_x, k0)
           k2 = dz * compute_dE_dz_jit(E_prev + k1/2, n_r2_slice, dx, n0, sigma_x, k0)
           k3 = dz * compute_dE_dz_jit(E_prev + k2/2, n_r2_slice, dx, n0, sigma_x, k0)
           k4 = dz * compute_dE_dz_jit(E_prev + k3, n_r2_slice, dx, n0, sigma_x, k0)

           E[:, zi] = E_prev + (k1 + 2*k2 + 2*k3 + k4) / 6

       return E
   ```

2. **Add performance selector**:
   ```python
   # In bpm/core.py
   def run_bpm(E, n_r2, x, z, dx, dz, n0, sigma_x, wavelength,
               method='auto', use_jit=None):
       """
       Run BPM propagation with performance options.

       Parameters:
       -----------
       method : str
           'auto', 'numpy', 'jit', 'sparse'
       use_jit : bool, optional
           Force JIT compilation (overrides method)
       """
       k0 = 2 * np.pi / wavelength

       # Auto-select method based on problem size
       if method == 'auto':
           total_points = E.size
           if total_points > 100000:
               method = 'jit'
           else:
               method = 'numpy'

       if method == 'jit' or use_jit:
           try:
               from .core_optimized import run_bmp_jit
               return run_bpm_jit(E, n_r2, dx, dz, n0, sigma_x, k0)
           except ImportError:
               warnings.warn("Numba not available, falling back to numpy")
               method = 'numpy'

       if method == 'numpy':
           return _run_bpm_numpy(E, n_r2, x, z, dx, dz, n0, sigma_x, k0)
   ```

**TODO Checklist**:
- [ ] Implement JIT-compiled functions
- [ ] Add performance benchmarking
- [ ] Create method selection logic
- [ ] Add optional numba dependency
- [ ] Write performance tests
- [ ] Document performance options

#### Optimization 2: Memory Management

**Implementation**:

1. **Add memory-efficient options**:
   ```python
   def run_bpm_memory_efficient(E_init, n_r2, x, z, dx, dz, n0, sigma_x,
                               wavelength, save_interval=10):
       """
       Memory-efficient BPM that doesn't store full field evolution.

       Parameters:
       -----------
       save_interval : int
           Save field every N steps (1 = save all, 10 = save every 10th)

       Returns:
       --------
       E_saved : ndarray
           Field at saved z-positions
       z_saved : ndarray
           Z-positions where field was saved
       """
       k0 = 2 * np.pi / wavelength
       Nx, Nz = len(x), len(z)

       # Only allocate memory for saved fields
       num_saved = (Nz - 1) // save_interval + 1
       E_saved = np.zeros((Nx, num_saved), dtype=np.complex128)
       z_saved = np.zeros(num_saved)

       # Current field (only need current and next)
       E_current = E_init[:, 0].copy()
       save_idx = 0
       E_saved[:, save_idx] = E_current
       z_saved[save_idx] = z[0]

       for zi in range(1, Nz):
           # BPM step (in-place to save memory)
           E_current = _bpm_step(E_current, n_r2[:, zi-1], dx, dz, n0, sigma_x, k0)

           # Save if needed
           if zi % save_interval == 0:
               save_idx += 1
               E_saved[:, save_idx] = E_current
               z_saved[save_idx] = z[zi]

       return E_saved, z_saved
   ```

**TODO Checklist**:
- [ ] Implement memory-efficient propagation
- [ ] Add memory usage monitoring
- [ ] Create chunked processing for large grids
- [ ] Add progress callbacks
- [ ] Write memory usage tests

### Feature Implementation Guides

#### GDSII Import Implementation

**Step 1: Choose and Setup GDS Library**

```python
# Add to pyproject.toml
[project.optional-dependencies]
gds = [
    "gdstk>=0.9.0",  # Modern, faster library
    "shapely>=2.0.0",  # For polygon operations
]
```

**Step 2: Create GDSII Module**

```python
# bpm/gdsii_import.py
import numpy as np
import warnings
from typing import Tuple, List, Optional, Union

try:
    import gdstk
    HAS_GDSTK = True
except ImportError:
    HAS_GDSTK = False
    warnings.warn("gdstk not available. Install with: pip install gdstk")

try:
    from shapely.geometry import Polygon, Point
    from shapely.ops import unary_union
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

class GDSImporter:
    """Import and process GDSII files for BPM simulation."""

    def __init__(self, gds_file: str):
        """Initialize with GDSII file."""
        if not HAS_GDSTK:
            raise ImportError("gdstk required for GDSII import")

        self.library = gdstk.read_gds(gds_file)
        self.cells = {cell.name: cell for cell in self.library.cells}

    def extract_layer(self, cell_name: str, layer: Tuple[int, int]) -> List:
        """Extract polygons from specific layer."""
        if cell_name not in self.cells:
            raise ValueError(f"Cell '{cell_name}' not found")

        cell = self.cells[cell_name]
        polygons = []

        for polygon in cell.polygons:
            if polygon.layer == layer[0] and polygon.datatype == layer[1]:
                polygons.append(polygon.points)

        return polygons

    def rasterize_layer(self, cell_name: str, layer: Tuple[int, int],
                       x_grid: np.ndarray, z_grid: np.ndarray,
                       n_core: float, n_clad: float) -> np.ndarray:
        """
        Rasterize layer to refractive index grid.

        Parameters:
        -----------
        cell_name : str
            Name of cell to process
        layer : tuple
            (layer_number, datatype)
        x_grid, z_grid : ndarray
            Coordinate grids
        n_core, n_clad : float
            Refractive indices for core and cladding

        Returns:
        --------
        n_r2 : ndarray
            Squared refractive index distribution
        """
        polygons = self.extract_layer(cell_name, layer)

        if not polygons:
            warnings.warn(f"No polygons found on layer {layer}")
            return np.full((len(x_grid), len(z_grid)), n_clad**2)

        # Create coordinate meshes
        X, Z = np.meshgrid(x_grid, z_grid, indexing='ij')
        n_r2 = np.full_like(X, n_clad**2)

        # Rasterize each polygon
        for poly_points in polygons:
            if HAS_SHAPELY:
                # Use shapely for accurate point-in-polygon
                poly = Polygon(poly_points)
                for i, x in enumerate(x_grid):
                    for j, z in enumerate(z_grid):
                        if poly.contains(Point(x, z)):
                            n_r2[i, j] = n_core**2
            else:
                # Fallback to simple bounding box
                warnings.warn("Shapely not available, using bounding box approximation")
                x_min, x_max = np.min(poly_points[:, 0]), np.max(poly_points[:, 0])
                z_min, z_max = np.min(poly_points[:, 1]), np.max(poly_points[:, 1])

                mask_x = (X >= x_min) & (X <= x_max)
                mask_z = (Z >= z_min) & (Z <= z_max)
                n_r2[mask_x & mask_z] = n_core**2

        return n_r2

# Convenience function
def import_gds_structure(gds_file: str, cell_name: str, layer: Tuple[int, int],
                        x_grid: np.ndarray, z_grid: np.ndarray,
                        n_core: float, n_clad: float) -> np.ndarray:
    """
    Import structure from GDSII file.

    Example:
    --------
    >>> x = np.linspace(-10, 10, 100)
    >>> z = np.linspace(0, 100, 500)
    >>> n_r2 = import_gds_structure('waveguide.gds', 'TOP', (1, 0),
    ...                            x, z, 1.5, 1.0)
    """
    importer = GDSImporter(gds_file)
    return importer.rasterize_layer(cell_name, layer, x_grid, z_grid, n_core, n_clad)
```

**Step 3: Add Example Usage**

```python
# examples/example_gds_import.py
import numpy as np
import matplotlib.pyplot as plt
from bpm.gdsii_import import import_gds_structure
from bpm.mode_solver import slab_mode_source
from bpm.core import run_bpm
from bpm.pml import generate_sigma_x

def example_gds_waveguide():
    """Example: Import waveguide from GDS and simulate."""

    # Define simulation grid
    domain_size = 20.0  # um
    z_total = 100.0     # um
    Nx, Nz = 200, 1000
    x = np.linspace(-domain_size/2, domain_size/2, Nx)
    z = np.linspace(0, z_total, Nz)

    # Import structure from GDS
    try:
        n_r2 = import_gds_structure(
            gds_file='structures/waveguide.gds',
            cell_name='WAVEGUIDE',
            layer=(1, 0),  # Layer 1, datatype 0
            x_grid=x,
            z_grid=z,
            n_core=1.5,
            n_clad=1.0
        )
        print("✅ GDS structure imported successfully")
    except FileNotFoundError:
        print("❌ GDS file not found, creating synthetic structure")
        # Fallback to synthetic structure
        from bpm.refractive_index import generate_waveguide_n_r2
        n_r2 = generate_waveguide_n_r2(x, z, 0, z_total, 2.0, 1.5, 1.0)

    # Rest of simulation...
    # (mode launching, BPM propagation, visualization)

if __name__ == "__main__":
    example_gds_waveguide()
```

**TODO Checklist**:
- [ ] Implement GDSImporter class
- [ ] Add polygon rasterization
- [ ] Handle curved structures (approximation)
- [ ] Add multi-layer support
- [ ] Create example GDS files
- [ ] Write comprehensive tests
- [ ] Add performance optimization
- [ ] Document GDS workflow

## Risk Assessment and Mitigation

### Technical Risks

1. **Performance Degradation**
   - **Risk**: New features slow down core functionality
   - **Mitigation**: Comprehensive benchmarking, optional features
   - **Monitoring**: Automated performance tests in CI

2. **Numerical Instability**
   - **Risk**: Optimizations introduce numerical errors
   - **Mitigation**: Validation against analytical solutions
   - **Monitoring**: Regression tests with known results

3. **Dependency Conflicts**
   - **Risk**: New dependencies conflict with existing ones
   - **Mitigation**: Optional dependencies, version pinning
   - **Monitoring**: Dependency scanning in CI

### Project Risks

1. **Scope Creep**
   - **Risk**: Feature requests expand beyond core improvements
   - **Mitigation**: Strict prioritization, phased implementation
   - **Monitoring**: Regular scope reviews

2. **Breaking Changes**
   - **Risk**: Improvements break existing user code
   - **Mitigation**: Semantic versioning, deprecation warnings
   - **Monitoring**: API compatibility tests

## References

1. Chung, Y., & Dagli, N. (1990). An assessment of finite difference beam propagation method. IEEE Journal of Quantum Electronics, 26(8), 1335-1339.
2. Vassallo, C. (1992). Improvement of finite difference methods for step-index optical waveguides. IEE Proceedings J, 139(2), 137-142.
3. Numerical Simulation of Optical Wave Propagation with Examples in MATLAB, SPIE Press.
4. Saleh, B. E., & Teich, M. C. (2019). Fundamentals of photonics. John Wiley & Sons.
5. Okamoto, K. (2021). Fundamentals of optical waveguides. Academic press.

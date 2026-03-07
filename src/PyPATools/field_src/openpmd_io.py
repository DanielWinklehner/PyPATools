"""
OpenPMD I/O Module for Particle Accelerator Field Data
=======================================================

Implements reading/writing electromagnetic field data using the OpenPMD standard.
Compatible with field_loaders.py data format.

Reference: https://github.com/openPMD/openPMD-standard
OpenPMD Standard Version: 2.0.0
"""

import numpy as np
import warnings
from typing import Dict, Union, Optional, Tuple

# Optional dependency handling
try:
    import openpmd_api as io

    HAS_OPENPMD = True
except ImportError:
    HAS_OPENPMD = False
    io = None


class OpenPMDError(Exception):
    """Custom exception for OpenPMD I/O errors."""
    pass


def _check_openpmd_available():
    """Raise informative error if openpmd-api is not installed."""
    if not HAS_OPENPMD:
        raise ImportError(
            "openpmd-api is required but not installed.\n"
            "Install with: pip install openPMD-api\n"
            "Or with conda: conda install -c conda-forge openpmd-api"
        )


def load_field_openpmd(
        filename: str,
        iteration: int = 0,
        field_name: str = 'B',
        component: Optional[str] = None
) -> Dict:
    """
    Load electromagnetic field data from OpenPMD format.

    Parameters
    ----------
    filename : str
        Path to OpenPMD file (e.g., 'fields.h5' or 'fields_%T.h5' for series)
    iteration : int, optional
        Iteration number to load from time series (default: 0)
    field_name : str, optional
        Field to load: 'E' (electric) or 'B' (magnetic) (default: 'B')
    component : str, optional
        Load specific component ('x', 'y', or 'z'). If None, loads all (default: None)

    Returns
    -------
    dict
        Field data dictionary with keys:
        - 'grid': dict with 'x', 'y', 'z' 1D arrays defining mesh points
        - 'values': dict with 'x', 'y', 'z' ndarrays of field components
        - 'dim': int, dimensionality (2 or 3)
        - 'field_type': str, 'E' or 'B'
        - 'metadata': dict with additional information (units, time, etc.)

    Example
    -------
    >>> field_data = load_field_openpmd('fields.h5', iteration=0, field_name='B')
    >>> Bx = field_data['values']['x']
    >>> grid_x = field_data['grid']['x']
    """
    _check_openpmd_available()

    if field_name not in ['E', 'B']:
        raise ValueError(f"field_name must be 'E' or 'B', got '{field_name}'")

    # Open OpenPMD series
    series = io.Series(filename, io.Access.read_only)

    if iteration not in series.iterations:
        available = list(series.iterations.keys())
        raise OpenPMDError(
            f"Iteration {iteration} not found in file.\n"
            f"Available iterations: {available}"
        )

    it = series.iterations[iteration]

    # Check if field exists
    if field_name not in it.meshes:
        available = list(it.meshes.keys())
        raise OpenPMDError(
            f"Field '{field_name}' not found in iteration {iteration}.\n"
            f"Available meshes: {available}"
        )

    mesh = it.meshes[field_name]

    # Determine components to load
    components_to_load = [component] if component else ['x', 'y', 'z']
    available_components = list(mesh.keys())

    # Load field data
    values = {}
    grid_spacing = None
    grid_offset = None
    shape = None
    units = {}

    for comp in components_to_load:
        if comp not in available_components:
            warnings.warn(f"Component '{comp}' not found in mesh. Available: {available_components}")
            continue

        mesh_comp = mesh[comp]

        # Load data
        chunk_data = mesh_comp.load_chunk()
        series.flush()
        values[comp] = chunk_data

        # Get metadata (same for all components)
        if grid_spacing is None:
            grid_spacing = mesh_comp.grid_spacing
            grid_offset = mesh_comp.grid_global_offset
            shape = mesh_comp.shape

        # Get units
        units[comp] = mesh_comp.unit_SI

    # Construct coordinate grids
    dim = len(shape)
    if dim not in [2, 3]:
        raise OpenPMDError(f"Only 2D and 3D meshes supported, got {dim}D")

    grid = {}
    axis_names = ['x', 'y', 'z'][:dim]

    for i, axis in enumerate(axis_names):
        # OpenPMD uses cell-centered data by default
        n_points = shape[i]
        dx = grid_spacing[i]
        offset = grid_offset[i]

        # Create grid: offset + (0.5 + i) * spacing for cell centers
        grid[axis] = offset + (np.arange(n_points) + 0.5) * dx

    # Get time information
    time = it.time
    time_unit_SI = it.time_unit_SI

    # Get additional metadata
    metadata = {
        'iteration': iteration,
        'time': time,
        'time_unit_SI': time_unit_SI,
        'grid_spacing': grid_spacing,
        'grid_offset': grid_offset,
        'shape': shape,
        'units': units,
        'geometry': mesh.geometry if hasattr(mesh, 'geometry') else 'cartesian',
        'axis_labels': mesh.axis_labels if hasattr(mesh, 'axis_labels') else axis_names,
    }

    series.close()

    return {
        'grid': grid,
        'values': values,
        'dim': dim,
        'field_type': field_name,
        'metadata': metadata
    }


def save_field_openpmd(
        filename: str,
        field_data: Dict,
        iteration: int = 0,
        field_name: str = 'B',
        time: float = 0.0,
        dt: Optional[float] = None
) -> None:
    """
    Save electromagnetic field data to OpenPMD format.

    Parameters
    ----------
    filename : str
        Output filename. Use '%T' for iteration wildcard (e.g., 'fields_%T.h5')
    field_data : dict
        Field data dictionary with keys:
        - 'grid': dict with 'x', 'y', 'z' 1D arrays
        - 'values': dict with 'x', 'y', 'z' ndarrays of field components
        - 'dim': int (optional, inferred from data)
        - 'metadata': dict (optional)
    iteration : int, optional
        Iteration number (default: 0)
    field_name : str, optional
        Field name: 'E' or 'B' (default: 'B')
    time : float, optional
        Physical time for this iteration (default: 0.0)
    dt : float, optional
        Time step (default: None, inferred from time)

    Example
    -------
    >>> field_data = {
    ...     'grid': {'x': x_array, 'y': y_array, 'z': z_array},
    ...     'values': {'x': Bx_array, 'y': By_array, 'z': Bz_array}
    ... }
    >>> save_field_openpmd('fields_%T.h5', field_data, iteration=0, field_name='B')
    """
    _check_openpmd_available()

    if field_name not in ['E', 'B']:
        raise ValueError(f"field_name must be 'E' or 'B', got '{field_name}'")

    # Validate input data
    if 'grid' not in field_data or 'values' not in field_data:
        raise ValueError("field_data must contain 'grid' and 'values' keys")

    grid = field_data['grid']
    values = field_data['values']
    metadata = field_data.get('metadata', {})

    # Determine dimensionality
    dim = field_data.get('dim', len(grid))
    if dim not in [2, 3]:
        raise ValueError(f"Only 2D and 3D fields supported, got dim={dim}")

    # Check data consistency
    components = list(values.keys())
    if not all(comp in ['x', 'y', 'z'] for comp in components):
        raise ValueError(f"Invalid components: {components}. Must be 'x', 'y', 'z'")

    # Get shape and validate
    ref_shape = values[components[0]].shape
    if not all(values[comp].shape == ref_shape for comp in components):
        raise ValueError("All field components must have the same shape")

    # Calculate grid spacing and offset
    grid_spacing = []
    grid_offset = []
    axis_names = ['x', 'y', 'z'][:dim]

    for axis in axis_names:
        if axis not in grid:
            raise ValueError(f"Grid axis '{axis}' missing from field_data['grid']")

        grid_1d = grid[axis]
        if len(grid_1d) < 2:
            raise ValueError(f"Grid axis '{axis}' must have at least 2 points")

        # Calculate spacing (assume uniform)
        dx = grid_1d[1] - grid_1d[0]
        grid_spacing.append(float(dx))

        # Offset to first cell center (assuming cell-centered data)
        grid_offset.append(float(grid_1d[0] - 0.5 * dx))

    # Create OpenPMD series
    series = io.Series(filename, io.Access.create)

    # Set OpenPMD version
    series.set_openPMD("2.0.0")
    series.set_openPMD_extension(0)

    # Set base path
    series.set_meshes_path("meshes")
    series.set_particles_path("particles")

    # Set software and author info
    series.set_software("openpmd_io.py", "1.0.0")
    series.set_author("OpenPMD Field I/O")

    # Create iteration
    it = series.iterations[iteration]
    it.set_time(time)
    it.set_time_unit_SI(1.0)  # seconds

    if dt is not None:
        it.set_dt(dt)

    # Create mesh
    mesh = it.meshes[field_name]

    # Set mesh properties
    mesh.set_geometry(io.Geometry.cartesian)
    mesh.set_data_order(io.Data_Order.C)  # Row-major (C-style)
    mesh.set_axis_labels(axis_names)
    mesh.set_grid_spacing(grid_spacing)
    mesh.set_grid_global_offset(grid_offset)

    # Determine unit SI (Tesla for B-field, V/m for E-field)
    if field_name == 'B':
        unit_SI = 1.0  # Tesla
        unit_dimension = {
            io.Unit_Dimension.M: 1,
            io.Unit_Dimension.I: -1,
            io.Unit_Dimension.T: -2
        }
    else:  # E-field
        unit_SI = 1.0  # V/m
        unit_dimension = {
            io.Unit_Dimension.L: 1,
            io.Unit_Dimension.M: 1,
            io.Unit_Dimension.I: -1,
            io.Unit_Dimension.T: -3
        }

    # Write components
    for comp in components:
        data = values[comp]

        # Create mesh component
        mesh_comp = mesh[comp]
        mesh_comp.set_unit_SI(unit_SI)
        mesh_comp.set_unit_dimension(unit_dimension)

        # Reset dataset
        dataset = io.Dataset(data.dtype, data.shape)
        mesh_comp.reset_dataset(dataset)

        # Store data
        mesh_comp.store_chunk(data)

    # Flush to disk
    series.flush()

    print(f"Saved {field_name}-field to {filename}")
    print(f"  Iteration: {iteration}, Time: {time}")
    print(f"  Shape: {ref_shape}, Dimension: {dim}D")
    print(f"  Components: {components}")
    print(f"  Grid spacing: {grid_spacing}")

    series.close()


def convert_h5part_to_openpmd(
        h5part_file: str,
        openpmd_file: str,
        iteration: int = 0,
        time: float = 0.0
) -> None:
    """
    Convert h5part format to OpenPMD format.

    Parameters
    ----------
    h5part_file : str
        Input h5part file path
    openpmd_file : str
        Output OpenPMD file path
    iteration : int, optional
        Iteration number (default: 0)
    time : float, optional
        Physical time (default: 0.0)

    Example
    -------
    >>> convert_h5part_to_openpmd('old_format.h5', 'new_format.h5')
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required for h5part conversion")

    _check_openpmd_available()

    # Load h5part file
    with h5py.File(h5part_file, 'r') as f:
        step = f['Step#0']
        block = step['Block']

        # Load H-field (magnetic field)
        h_field = block['Hfield']
        hx = h_field['0'][:]
        hy = h_field['1'][:]
        hz = h_field['2'][:]

        spacing = h_field.attrs['__Spacing__']
        origin = h_field.attrs['__Origin__']

    # Reconstruct grid (h5part stores in [z, y, x] order)
    shape_zyx = hx.shape
    nz, ny, nx = shape_zyx

    # Create coordinate arrays
    x = origin[0] + np.arange(nx) * spacing[0]
    y = origin[1] + np.arange(ny) * spacing[1]
    z = origin[2] + np.arange(nz) * spacing[2]

    # Transpose data from [z, y, x] to [x, y, z]
    bx = np.transpose(hx, (2, 1, 0))
    by = np.transpose(hy, (2, 1, 0))
    bz = np.transpose(hz, (2, 1, 0))

    # Create field_data dict
    field_data = {
        'grid': {'x': x, 'y': y, 'z': z},
        'values': {'x': bx, 'y': by, 'z': bz},
        'dim': 3,
        'metadata': {
            'source_format': 'h5part',
            'original_shape': shape_zyx
        }
    }

    # Save to OpenPMD
    save_field_openpmd(openpmd_file, field_data, iteration=iteration,
                       field_name='B', time=time)

    print(f"Converted {h5part_file} -> {openpmd_file}")


# =============================================================================
# Unit Tests
# =============================================================================

def test_openpmd_io():
    """Test OpenPMD I/O functions with synthetic data."""
    import tempfile
    import os

    if not HAS_OPENPMD:
        print("SKIP: openpmd-api not installed")
        return

    print("\n" + "=" * 70)
    print("Testing OpenPMD I/O")
    print("=" * 70)

    # Create synthetic 3D magnetic field
    nx, ny, nz = 20, 15, 10
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-0.5, 0.5, ny)
    z = np.linspace(0.0, 2.0, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Create a simple dipole-like field
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2) + 1e-10
    Bx = 3 * X * Z / r ** 5
    By = 3 * Y * Z / r ** 5
    Bz = (3 * Z ** 2 - r ** 2) / r ** 5

    field_data = {
        'grid': {'x': x, 'y': y, 'z': z},
        'values': {'x': Bx, 'y': By, 'z': Bz},
        'dim': 3,
        'metadata': {'test': 'synthetic dipole field'}
    }

    # Test 1: Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, 'test_fields_%T.h5')

        print("\n1. Testing save_field_openpmd...")
        save_field_openpmd(filename, field_data, iteration=0,
                           field_name='B', time=0.0)

        print("\n2. Testing load_field_openpmd...")
        loaded_data = load_field_openpmd(filename, iteration=0, field_name='B')

        print("\n3. Validating data...")
        # Check shapes
        assert loaded_data['dim'] == 3, "Dimension mismatch"
        assert loaded_data['field_type'] == 'B', "Field type mismatch"

        # Check grid
        for axis in ['x', 'y', 'z']:
            np.testing.assert_allclose(
                loaded_data['grid'][axis],
                field_data['grid'][axis],
                rtol=1e-10,
                err_msg=f"Grid {axis} mismatch"
            )

        # Check values
        for comp in ['x', 'y', 'z']:
            np.testing.assert_allclose(
                loaded_data['values'][comp],
                field_data['values'][comp],
                rtol=1e-10,
                err_msg=f"Component {comp} mismatch"
            )

        print("   PASS: All data validated successfully")

        # Test 2: Time series
        print("\n4. Testing time series...")
        for it in range(3):
            time = it * 0.1
            save_field_openpmd(filename, field_data, iteration=it,
                               field_name='B', time=time, dt=0.1)

        # Load different iterations
        data_t0 = load_field_openpmd(filename, iteration=0, field_name='B')
        data_t2 = load_field_openpmd(filename, iteration=2, field_name='B')

        assert data_t0['metadata']['time'] == 0.0
        assert data_t2['metadata']['time'] == 0.2
        print("   PASS: Time series working")

        # Test 3: 2D field
        print("\n5. Testing 2D field...")
        field_2d = {
            'grid': {'x': x, 'y': y},
            'values': {
                'x': Bx[:, :, nz // 2],
                'y': By[:, :, nz // 2],
                'z': Bz[:, :, nz // 2]
            },
            'dim': 2
        }

        filename_2d = os.path.join(tmpdir, 'test_2d_%T.h5')
        save_field_openpmd(filename_2d, field_2d, field_name='B')
        loaded_2d = load_field_openpmd(filename_2d, field_name='B')

        assert loaded_2d['dim'] == 2
        print("   PASS: 2D field working")

        # Test 4: E-field
        print("\n6. Testing E-field...")
        field_data['field_type'] = 'E'
        filename_e = os.path.join(tmpdir, 'test_efield_%T.h5')
        save_field_openpmd(filename_e, field_data, field_name='E')
        loaded_e = load_field_openpmd(filename_e, field_name='E')

        assert loaded_e['field_type'] == 'E'
        print("   PASS: E-field working")

    print("\n" + "=" * 70)
    print("All tests PASSED!")
    print("=" * 70)


def example_usage():
    """Demonstrate usage of OpenPMD I/O functions."""
    print("\n" + "=" * 70)
    print("OpenPMD I/O Example Usage")
    print("=" * 70)

    if not HAS_OPENPMD:
        print("\nERROR: openpmd-api not installed")
        print("Install with: pip install openPMD-api")
        return

    # Create example data
    print("\n1. Creating synthetic magnetic field data...")
    x = np.linspace(-0.1, 0.1, 50)  # meters
    y = np.linspace(-0.1, 0.1, 50)
    z = np.linspace(0.0, 0.5, 100)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Simple solenoid-like field
    Bx = -0.1 * X
    By = -0.1 * Y
    Bz = np.ones_like(Z) * 1.0  # 1 Tesla axial field

    field_data = {
        'grid': {'x': x, 'y': y, 'z': z},
        'values': {'x': Bx, 'y': By, 'z': Bz},
        'dim': 3,
        'metadata': {
            'description': 'Solenoid magnetic field',
            'device': 'Example RF cavity'
        }
    }

    # Save field
    print("\n2. Saving to OpenPMD format...")
    save_field_openpmd(
        'example_fields_%T.h5',
        field_data,
        iteration=0,
        field_name='B',
        time=0.0
    )

    # Load field
    print("\n3. Loading from OpenPMD format...")
    loaded = load_field_openpmd('example_fields_%T.h5', iteration=0, field_name='B')

    print("\n4. Field data summary:")
    print(f"   Dimension: {loaded['dim']}D")
    print(f"   Field type: {loaded['field_type']}")
    print(f"   Shape: {loaded['values']['x'].shape}")
    print(f"   Grid range X: [{loaded['grid']['x'][0]:.3f}, {loaded['grid']['x'][-1]:.3f}] m")
    print(f"   Grid range Y: [{loaded['grid']['y'][0]:.3f}, {loaded['grid']['y'][-1]:.3f}] m")
    print(f"   Grid range Z: [{loaded['grid']['z'][0]:.3f}, {loaded['grid']['z'][-1]:.3f}] m")
    print(f"   Bz (center): {loaded['values']['z'][25, 25, 50]:.3f} T")

    # Clean up
    import os
    try:
        os.remove('example_fields_0.h5')
        print("\n5. Cleaned up example file")
    except:
        pass

    print("\n" + "=" * 70)


if __name__ == '__main__':
    # Run tests
    test_openpmd_io()

    # Show example usage
    example_usage()

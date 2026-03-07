"""
field_loaders.py

Modular field loaders for particle accelerator simulation toolkit.
Each loader parses a specific field file format and returns a standardized dictionary.

Author: [Your name]
Date: 2024
"""

import numpy as np
import h5py
import os
import re
import gc
from scipy.interpolate import RegularGridInterpolator

# Unit conversion factors to SI (meters, Tesla, V/m)
UNIT_CONVERSIONS = {
    'length': {
        'mm': 0.001,
        'cm': 0.01,
        'm': 1.0,
    },
    'magnetic': {
        'gauss': 1e-4,
        'kgauss': 0.1,
        'tesla': 1.0,
        't': 1.0,
    },
    'electric': {
        'v/mm': 1000.0,
        'v/cm': 100.0,
        'v/m': 1.0,
    }
}


def _pol2cart(r, theta_deg):
    """Convert polar coordinates to Cartesian."""
    theta_rad = np.deg2rad(theta_deg)
    x = r * np.cos(theta_rad)
    y = r * np.sin(theta_rad)
    return x, y


def _cylinder_to_cartesian(r, theta_deg, z):
    """Convert cylindrical coordinates to Cartesian."""
    x, y = _pol2cart(r, theta_deg)
    return x, y, np.full_like(x, z)


def _cylinder_to_cartesian_field(r, theta_deg, z, br, btheta, bz):
    """Convert cylindrical field components to Cartesian."""
    theta_rad = np.deg2rad(theta_deg)
    x, y = _pol2cart(r, theta_deg)

    # Transform field components
    bx = br * np.cos(theta_rad) - btheta * np.sin(theta_rad)
    by = br * np.sin(theta_rad) + btheta * np.cos(theta_rad)

    return x, y, np.full_like(x, z), bx, by, bz


def load_opera_table(filename, extents=None, extents_dims=None):
    """
    Load OPERA .table format field map.

    File Format:
    ------------
    Line 1: Number of points in each dimension (nx ny nz)
    Lines 2-N: Column definitions with format:
        <col_num> <label> [<unit>]
        Example: "1 X [cm]"
                 "4 BX [Tesla]"
    Line N+1: "0" (end of header)
    Data lines: Space-separated values for each grid point

    Supported units:
    - Length: mm, cm, m
    - Magnetic field: Gauss, Tesla
    - Electric field: V/mm, V/cm, V/m

    Parameters:
    -----------
    filename : str
        Path to .table file
    extents : list of tuples, optional
        [(xmin, xmax), (ymin, ymax), (zmin, zmax)] in cm
        Used when spatial coordinates are not in file
    extents_dims : list of str, optional
        ['X', 'Y', 'Z'] - dimension labels for extents

    Returns:
    --------
    dict with keys:
        'grid': dict with 'x', 'y', 'z' arrays (1D, in meters)
        'values': dict with 'x', 'y', 'z' arrays (ND, in Tesla or V/m)
        'dim': int (1, 2, or 3)
        'field_type': str ('magnetic' or 'electric')
        'metadata': dict with 'units_original', 'n_points'

    Example:
    --------
    > field = load_opera_table('myfield.table')
    > bz_at_origin = field['values']['z'][nx//2, ny//2, nz//2]
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    if extents is not None and extents_dims is None:
        raise ValueError("extents_dims must be specified when using extents")

    try:
        with open(filename, 'r') as infile:
            # Read grid dimensions
            first_line = infile.readline().strip().split()
            n_dims = np.array([int(val) - 1 for val in first_line])[:3]

            data = {}
            spatial_dims = 0
            efield_dims = 0
            bfield_dims = 0

            # Parse header
            while True:
                line = infile.readline().strip()
                if line == "0":
                    break

                parts = line.split()
                col_no = int(parts[0]) - 1
                label = parts[1]

                # Extract unit from brackets
                unit_match = re.search(r'\[(.*?)]', line)  # TODO: removed redundant character escape here \]
                if unit_match:
                    unit_str = unit_match.group(1)
                else:
                    raise ValueError(f"No unit found for column {label}")

                data[label] = {"column": col_no, "unit": unit_str}

                # Categorize field type
                if unit_str.lower() in UNIT_CONVERSIONS['length']:
                    spatial_dims += 1
                elif unit_str.lower() in UNIT_CONVERSIONS['magnetic']:
                    bfield_dims += 1
                elif unit_str.lower() in UNIT_CONVERSIONS['electric']:
                    efield_dims += 1

            # Handle extents if provided
            if extents is not None:
                for i, label in enumerate(extents_dims):
                    data[label] = {
                        "column": col_no + i + 1,
                        "unit": "cm"
                    }
                    spatial_dims += 1

            # Determine grid sizes
            n = {}
            if "X" in data.keys():
                n["X"] = n_dims[0] + 1
                n["Y"] = n_dims[1] + 1 if "Y" in data.keys() else 1
                n["Z"] = n_dims[2] + 1 if "Z" in data.keys() else 1
            elif "Y" in data.keys():
                n["X"] = 1
                n["Y"] = n_dims[0] + 1
                n["Z"] = n_dims[1] + 1 if "Z" in data.keys() else 1
            elif "Z" in data.keys():
                n["X"] = 1
                n["Y"] = 1
                n["Z"] = n_dims[0] + 1

            # Determine dimensionality
            dim = len(np.where(n_dims > 0)[0])

            # Check for mixed fields
            if efield_dims > 0 and bfield_dims > 0:
                raise ValueError("Mixed E and B fields not supported")

            field_type = 'electric' if efield_dims > 0 else 'magnetic'

            # Read data
            array_len = n["X"] * n["Y"] * n["Z"]
            n_cols = spatial_dims + efield_dims + bfield_dims
            raw_data = np.zeros([array_len, n_cols])

            if extents is not None:
                # Generate spatial grids from extents
                xlims, ylims, zlims = extents[0], extents[1], extents[2]
                _x = np.linspace(xlims[0], xlims[1], n["X"])
                _y = np.linspace(ylims[0], ylims[1], n["Y"])
                _z = np.linspace(zlims[0], zlims[1], n["Z"])

                xv, yv, zv = np.meshgrid(_x, _y, _z, indexing='ij')
                raw_data[:, -3] = xv.ravel()
                raw_data[:, -2] = yv.ravel()
                raw_data[:, -1] = zv.ravel()

                # Read field values only
                for i, line in enumerate(infile):
                    raw_data[i, :-(spatial_dims)] = [float(item) for item in line.split()]
            else:
                # Read all columns
                for i, line in enumerate(infile):
                    raw_data[i, :] = [float(item) for item in line.split()]

    except Exception as e:
        raise RuntimeError(f"Error reading OPERA table file: {e}")

    # Process data
    raw_data = raw_data.T

    # Extract spatial grids and convert to meters
    grid_dict = {}
    for key in ['X', 'Y', 'Z']:
        if key in data and data[key]["unit"].lower() in UNIT_CONVERSIONS['length']:
            unique_vals = np.unique(raw_data[data[key]["column"]])
            scale = UNIT_CONVERSIONS['length'][data[key]["unit"].lower()]
            grid_dict[key.lower()] = unique_vals * scale

    # Extract field values and convert to SI
    values_dict = {}
    field_labels = ['BX', 'BY', 'BZ'] if field_type == 'magnetic' else ['EX', 'EY', 'EZ']
    unit_type = 'magnetic' if field_type == 'magnetic' else 'electric'

    for label in field_labels:
        if label in data:
            unit = data[label]["unit"]
            scale = UNIT_CONVERSIONS[unit_type].get(unit.lower(), 1.0)
            field_data = raw_data[data[label]["column"]].reshape([n["X"], n["Y"], n["Z"]])
            values_dict[label[-1].lower()] = field_data * scale
        else:
            # Create zero field for missing components
            values_dict[label[-1].lower()] = np.zeros([n["X"], n["Y"], n["Z"]])

    del raw_data
    gc.collect()

    # Build return dictionary
    result = {
        'grid': grid_dict,
        'values': values_dict,
        'dim': dim,
        'field_type': field_type,
        'metadata': {
            'n_points': n,
            'units_original': {k: v['unit'] for k, v in data.items()},
            'filename': os.path.basename(filename)
        }
    }

    return result


def load_comsol(filename):
    """
    Load COMSOL .comsol format field map.

    File Format:
    ------------
    9-line header followed by data:
    Line 1-2: Comments
    Line 3: % Model info
    Line 4: % Dimension: <n>
    Line 5: % Nodes: <n>
    Line 6-7: Additional info
    Line 8: % Length unit: <unit>
    Line 9: Column headers (%, x, y, z, mf.Bx (T), mf.By (T), mf.Bz (T))
    Data lines: Space-separated values

    Supports 1D, 2D, and 3D field maps:
    - 1D: Single spatial coordinate (typically z) with field components
    - 2D: Two spatial coordinates (e.g., x-y, x-z, r-z) with field components
    - 3D: Full 3D field map

    Notes:
    - Handles derived COMSOL variable names (removes 'mir', 'sec', 'side' prefixes)
    - Automatically detects magnetic field columns with (T) suffix
    - Automatically detects electric field columns with (V/cm) or (V/m) suffix

    Parameters:
    -----------
    filename : str
        Path to .comsol file

    Returns:
    --------
    dict with keys:
        'grid': dict with 'x', 'y', 'z' arrays (1D, in meters)
                Only includes dimensions present in file
        'values': dict with 'x', 'y', 'z' arrays (ND, in Tesla or V/m)
        'dim': int (1, 2, or 3)
        'field_type': str ('magnetic' or 'electric')
        'metadata': dict with 'units_original', 'array_length', 'spatial_dims'
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    label_map = {
        "x": "X", "y": "Y", "z": "Z",
        "r": "R", "phi": "PHI", "theta": "THETA",  # Cylindrical/spherical
        "mf.Bx": "BX", "mf.By": "BY", "mf.Bz": "BZ",
        "Bx": "BX", "By": "BY", "Bz": "BZ",
        "mf.Br": "BR", "mf.Bphi": "BPHI", "mf.Btheta": "BTHETA",
        "es.Ex": "EX", "es.Ey": "EY", "es.Ez": "EZ",
        "es.Er": "ER", "es.Ephi": "EPHI", "es.Etheta": "ETHETA"
    }

    try:
        # Parse header (first 9 lines)
        with open(filename, 'r') as infile:
            data = {}
            spatial_coords = []
            field_coords = []

            for i in range(9):
                line = infile.readline().strip()
                sline = line.split()

                if i == 3:
                    file_dim = int(sline[2])
                elif i == 4:
                    array_len = int(sline[2])
                elif i == 7:
                    l_unit = sline[3]
                elif i == 8:
                    # Detect field type and parse column headers
                    b_unit = None
                    e_unit = None

                    if "(T)" in line:
                        b_unit = "T"
                        field_type = 'magnetic'
                    elif "(V/cm)" in line:
                        e_unit = "V/cm"
                        field_type = 'electric'
                    elif "(V/m)" in line:
                        e_unit = "V/m"
                        field_type = 'electric'
                    else:
                        if "mf.B" in line:
                            b_unit = "T"
                            field_type = 'magnetic'
                        elif "es.E" in line:
                            e_unit = "V/m"
                            field_type = 'electric'
                        else:
                            raise ValueError("Cannot determine field type from header")

                    # Parse column headers
                    sline = sline[:4] + re.split(r'\([a-zA-Z/]+\)', ''.join(sline[4:]))

                    j = 0
                    for label in sline:
                        if label not in ["%", "(T)", "(V/cm)", "(V/m)", "", " "]:
                            clean_label = re.sub(r'side|mir|sec|cpl|\d+', '', label)

                            if clean_label in label_map:
                                nlabel = label_map[clean_label]
                                data[nlabel] = {"column": j}

                                if nlabel in ["X", "Y", "Z", "R", "PHI", "THETA"]:
                                    data[nlabel]["unit"] = l_unit
                                    spatial_coords.append(nlabel)
                                elif nlabel in ["BX", "BY", "BZ", "BR", "BPHI", "BTHETA"] and b_unit:
                                    data[nlabel]["unit"] = b_unit
                                    field_coords.append(nlabel)
                                elif nlabel in ["EX", "EY", "EZ", "ER", "EPHI", "ETHETA"] and e_unit:
                                    data[nlabel]["unit"] = e_unit
                                    field_coords.append(nlabel)

                                j += 1

        if not spatial_coords:
            raise ValueError("No spatial coordinates found in file")
        if not field_coords:
            raise ValueError("No field components found in file")

        dim = len(spatial_coords)

        # Read all data at once using numpy (MUCH faster)
        print(f"Loading {array_len} data points...")
        raw_data = np.loadtxt(filename, skiprows=9, max_rows=array_len)

        # If only one row, ensure it's 2D
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)

        print("Data loaded, processing...")

    except Exception as e:
        raise RuntimeError(f"Error reading COMSOL file: {e}")

    # Extract spatial columns for sorting
    spatial_indices = [data[coord]["column"] for coord in spatial_coords]

    # Single efficient sort using lexsort (sorts by last key first)
    # For 3D: sorts by z, then y, then x
    # For 2D: sorts by second coord, then first
    # For 1D: sorts by single coord
    if dim == 1:
        sort_idx = np.argsort(raw_data[:, spatial_indices[0]])
    else:
        # lexsort expects keys in reverse order (last dimension first)
        sort_keys = [raw_data[:, spatial_indices[i]] for i in range(dim - 1, -1, -1)]
        sort_idx = np.lexsort(sort_keys)

    raw_data = raw_data[sort_idx]

    # Determine grid dimensions
    n = {}
    grid_dict = {}
    l_scale = UNIT_CONVERSIONS['length'].get(l_unit.lower(), 1.0)

    for coord in spatial_coords:
        col_idx = data[coord]["column"]
        unique_vals = np.unique(raw_data[:, col_idx])
        n[coord] = len(unique_vals)
        grid_dict[coord.lower()] = unique_vals * l_scale

    # Verify total points
    expected_points = np.prod(list(n.values()))
    if expected_points != array_len:
        raise ValueError(f"Grid size mismatch: {expected_points} expected vs {array_len} in file")

    # Determine reshape order
    shape = [n[coord] for coord in spatial_coords]

    # Extract and reshape field values (vectorized - very fast)
    values_dict = {}
    unit_type = 'magnetic' if field_type == 'magnetic' else 'electric'

    field_map = {
        'BX': 'x', 'BY': 'y', 'BZ': 'z',
        'BR': 'r', 'BPHI': 'phi', 'BTHETA': 'theta',
        'EX': 'x', 'EY': 'y', 'EZ': 'z',
        'ER': 'r', 'EPHI': 'phi', 'ETHETA': 'theta'
    }

    for field_label in field_coords:
        unit = data[field_label]["unit"]
        scale = UNIT_CONVERSIONS[unit_type].get(unit.lower(), 1.0)
        col_idx = data[field_label]["column"]

        # Direct indexing and reshape (no intermediate arrays)
        field_data = raw_data[:, col_idx].reshape(shape) * scale

        output_key = field_map.get(field_label, field_label.lower())
        values_dict[output_key] = field_data

    # Fill in missing components with zeros
    for component in ['x', 'y', 'z']:
        if component not in values_dict:
            values_dict[component] = np.zeros(shape)

    # Fill in missing grid dimensions
    for coord in ['x', 'y', 'z']:
        if coord not in grid_dict:
            grid_dict[coord] = np.array([0.0])

    # Clear raw_data to free memory
    del raw_data
    gc.collect()

    print("Processing complete!")

    result = {
        'grid': grid_dict,
        'values': values_dict,
        'dim': dim,
        'field_type': field_type,
        'metadata': {
            'n_points': n,
            'spatial_dims': spatial_coords,
            'field_dims': field_coords,
            'units_original': {k: v['unit'] for k, v in data.items()},
            'filename': os.path.basename(filename)
        }
    }

    return result


def load_opal_midplane(filename, r_cutoff=None, scaling=1.0, cartesian_grid=True,
                       nx=None, ny=None):
    """
    Load OPAL CARBONCYCL midplane field (.dat format).

    File Format:
    ------------
    Line 1: Starting radius (mm)
    Line 2: Radius step (mm)
    Line 3: Starting angle (degrees)
    Line 4: Angle step (degrees)
    Line 5: Number of angle points
    Line 6: Number of radius points
    Lines 7+: Bz values (kGauss by default) in row-major order (theta varies fastest)

    The file contains a 2D midplane (z=0) field in cylindrical coordinates
    that is converted to Cartesian grid.

    Parameters:
    -----------
    filename : str
        Path to .dat file
    r_cutoff : float, optional
        Radial cutoff in meters. Bz set to 0 outside this radius
    scaling : float, optional
        Scaling factor for field values (default: 1.0)
        Default assumes kGauss input, converts to Tesla (factor 0.1)
    cartesian_grid : bool, optional
        If True, interpolate onto Cartesian grid (default: True)
        If False, return native cylindrical grid (more efficient)
    nx, ny : int, optional
        Number of points for Cartesian grid (default: auto-determined)

    Returns:
    --------
    dict with keys:
        'grid': dict with 'x', 'y', 'z' arrays (1D, in meters)
                OR 'r', 'theta', 'z' if cartesian_grid=False
        'values': dict with 'x', 'y', 'z' arrays (2D, in Tesla)
                  OR 'r', 'theta', 'z' if cartesian_grid=False (cylindrical components)
        'dim': int (2)
        'field_type': str ('magnetic')
        'metadata': dict with 'r_range', 'theta_range', 'n_points'
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    if os.path.splitext(filename)[1] != ".dat":
        raise ValueError("File extension must be '.dat'")

    try:
        with open(filename, 'r') as infile:
            sr = float(infile.readline().strip())  # mm
            dr = float(infile.readline().strip())  # mm
            sth = float(infile.readline().strip())  # degrees
            dth = float(infile.readline().strip())  # degrees
            nth = int(infile.readline().strip())
            nr = int(infile.readline().strip())

            raw_data = infile.readlines()

    except Exception as e:
        raise RuntimeError(f"Error reading OPAL midplane file: {e}")

    # Parse Bz values (convert kGauss to Tesla by default)
    bz_flat = scaling * 0.1 * np.array([np.fromstring(line, sep=" ") for line in raw_data]).flatten()

    expected_points = nth * nr
    if len(bz_flat) != expected_points:
        print(f"Warning: Found {len(bz_flat)} Bz values, expected {expected_points}")

    # Create cylindrical grid arrays
    r_unique = np.linspace(sr, sr + (nr - 1) * dr, nr)  # mm
    th_unique = np.linspace(sth, sth + (nth - 1) * dth, nth)  # degrees

    # Reshape Bz onto 2D grid (theta varies fastest, so reshape as (nr, nth))
    bz_cyl = bz_flat.reshape(nr, nth)

    # Apply radial cutoff if specified
    if r_cutoff is not None:
        r_mask = r_unique > r_cutoff * 1000.0
        bz_cyl[r_mask, :] = 0.0

    print(f"Load from OPAL: Max Bz = {np.max(bz_cyl)} T")

    # Option 1: Return in native cylindrical coordinates (most efficient)
    if not cartesian_grid:
        result = {
            'grid': {
                'r': r_unique * 0.001,  # mm -> m
                'theta': th_unique,  # degrees
                'z': np.array([0.0])
            },
            'values': {
                'r': np.zeros_like(bz_cyl),  # Br = 0 for midplane
                'theta': np.zeros_like(bz_cyl),  # Btheta = 0 for midplane
                'z': bz_cyl  # Bz
            },
            'dim': 2,
            'field_type': 'magnetic',
            'coordinate_system': 'cylindrical',
            'metadata': {
                'r_range': (sr, sr + (nr - 1) * dr),  # mm
                'theta_range': (sth, sth + (nth - 1) * dth),  # degrees
                'n_points': {'r': nr, 'theta': nth},
                'r_cutoff': r_cutoff,
                'scaling': scaling,
                'filename': os.path.basename(filename)
            }
        }
        return result

    # Option 2: Interpolate onto Cartesian grid using RegularGridInterpolator
    # Create interpolator on cylindrical grid (MUCH faster than griddata)
    bz_interp = RegularGridInterpolator(
        (r_unique, th_unique),
        bz_cyl,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )

    # Determine Cartesian grid extent
    r_max_m = r_unique[-1] * 0.001  # mm -> m
    if r_cutoff is not None:
        r_max_m = min(r_max_m, r_cutoff)

    # Auto-determine grid resolution if not specified
    if nx is None:
        # Match approximately the cylindrical grid resolution
        nx = int(2 * r_max_m / (dr * 0.001)) + 1
    if ny is None:
        ny = nx

    # Create Cartesian grid
    x_cart = np.linspace(-r_max_m, r_max_m, nx)
    y_cart = np.linspace(-r_max_m, r_max_m, ny)
    x_grid, y_grid = np.meshgrid(x_cart, y_cart, indexing='ij')

    # Convert Cartesian to cylindrical for interpolation
    r_cart = np.sqrt(x_grid ** 2 + y_grid ** 2) * 1000.0  # m -> mm
    theta_cart = np.rad2deg(np.arctan2(y_grid, x_grid))

    # Handle negative angles (make them positive)
    theta_cart = np.where(theta_cart < 0, theta_cart + 360.0, theta_cart)

    # Interpolate (vectorized - MUCH faster than griddata)
    points = np.column_stack([r_cart.ravel(), theta_cart.ravel()])
    bz_cart = bz_interp(points).reshape(x_grid.shape)

    # Zero out points outside r_cutoff
    if r_cutoff is not None:
        mask = np.sqrt(x_grid ** 2 + y_grid ** 2) > r_cutoff
        bz_cart[mask] = 0.0

    result = {
        'grid': {
            'x': x_cart,
            'y': y_cart,
            'z': np.array([0.0])
        },
        'values': {
            'x': np.zeros_like(bz_cart),
            'y': np.zeros_like(bz_cart),
            'z': bz_cart
        },
        'dim': 2,
        'field_type': 'magnetic',
        'coordinate_system': 'cartesian',
        'metadata': {
            'r_range': (sr, sr + (nr - 1) * dr),  # mm
            'theta_range': (sth, sth + (nth - 1) * dth),  # degrees
            'n_points': {'r': nr, 'theta': nth},
            'n_cartesian': {'x': nx, 'y': ny},
            'r_cutoff': r_cutoff,
            'scaling': scaling,
            'filename': os.path.basename(filename)
        }
    }

    return result


def load_aima_agora(path, mirror=False, interp_resolution=1000):
    """
    Load AIMA AGORA field map from multiple .map files.

    File Format:
    ------------
    Multiple files named: *Z<zpos>-<component>.map
    where <zpos> is z position in mm, <component> is 'br', 'bf', or 'bz'

    Each file contains:
    Line 1: Comment
    Line 2: n_theta n_r
    Line 3: start_r delta_r
    Line 4: start_theta delta_theta
    Lines 5+: Field values in row-major order (theta varies fastest)

    All files must have same r and theta grids. Units are cm and degrees.
    Field values are in unspecified units (assumed Gauss or Tesla based on magnitude).

    Parameters:
    -----------
    path : str
        Path to directory containing .map files or path to one .map file
    mirror : bool, optional
        If True, mirror field about z=0 (BX, BY flip sign; BZ doesn't)
    interp_resolution : int, optional
        Number of points for Cartesian interpolation grid (default: 1000)

    Returns:
    --------
    dict with keys:
        'grid': dict with 'x', 'y', 'z' arrays (1D, in meters)
        'values': dict with 'x', 'y', 'z' arrays (3D, in Tesla)
        'dim': int (3)
        'field_type': str ('magnetic')
        'metadata': dict with 'z_positions', 'n_files', 'mirrored'
    """

    # Determine directory
    if os.path.isfile(path):
        directory = os.path.dirname(path)
    elif os.path.isdir(path):
        directory = path
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    # Find all .map files
    map_files = []
    try:
        for entry in os.scandir(directory):
            if entry.is_file() and entry.name.endswith('.map'):
                map_files.append(entry.name)
    except Exception as e:
        raise RuntimeError(f"Error scanning directory: {e}")

    if not map_files:
        raise FileNotFoundError(f"No .map files found in {directory}")

    # Parse filenames to extract metadata
    metadata_dtype = np.dtype([
        ("fn", 'U1024'),  # filename
        ("fcomp", 'U2'),  # field component
        ("zpos", float)  # z position (mm)
    ])
    metadata = []

    for filename in map_files:
        try:
            z_pos_str, rest = filename.split("Z")[1].split("-")
            z_pos = float(z_pos_str)
            b_comp = rest.split(".")[0].lower()

            if b_comp not in ['br', 'bf', 'bz']:
                print(f"Warning: Skipping file with unknown component: {filename}")
                continue

            metadata.append((os.path.join(directory, filename), b_comp, z_pos))
        except Exception as e:
            print(f"Warning: Could not parse filename {filename}: {e}")
            continue

    if not metadata:
        raise ValueError("No valid .map files found")

    metadata = np.array(metadata, dtype=metadata_dtype)

    # Verify all components have same z positions
    z_br = set(metadata[metadata["fcomp"] == "br"]["zpos"])
    z_bf = set(metadata[metadata["fcomp"] == "bf"]["zpos"])
    z_bz = set(metadata[metadata["fcomp"] == "bz"]["zpos"])

    if not (z_br == z_bf == z_bz):
        raise ValueError("Not all components have the same z positions. Maybe a file is missing?")

    # Read first file to get grid parameters
    first_file = metadata["fn"][0]
    try:
        with open(first_file, 'r') as infile:
            infile.readline()  # Skip comment
            nth, nr = [int(val) for val in infile.readline().strip().split()]
            sr, dr = [float(val) for val in infile.readline().strip().split()]
            sth, dth = [float(val) for val in infile.readline().strip().split()]
    except Exception as e:
        raise RuntimeError(f"Error reading grid parameters from {first_file}: {e}")

    # Verify all files have same grid
    for fn in metadata["fn"][1:]:
        with open(fn, 'r') as infile:
            infile.readline()
            nth2, nr2 = [int(val) for val in infile.readline().strip().split()]
            sr2, dr2 = [float(val) for val in infile.readline().strip().split()]
            sth2, dth2 = [float(val) for val in infile.readline().strip().split()]

            if not (nth == nth2 and nr == nr2 and sr == sr2 and dr == dr2 and sth == sth2 and dth == dth2):
                raise ValueError(f"Grid mismatch in file {fn}")

    # Create cylindrical grid
    r_unique = np.linspace(sr, nr * dr, nr, endpoint=False)  # cm
    th_unique = np.linspace(sth, nth * dth, nth, endpoint=False)  # degrees
    z_unique = np.sort(np.unique(metadata["zpos"])) / 10.0  # mm -> cm

    # Create Cartesian grid for interpolation
    r_max = r_unique[-1]

    x_cart = np.linspace(-r_max, r_max, interp_resolution)  # cm
    y_cart = np.linspace(-r_max, r_max, interp_resolution)  # cm

    if mirror:
        z_cart = np.concatenate((-z_unique[:0:-1], z_unique))
        idx_offset = len(z_unique) - 1
    else:
        z_cart = z_unique
        idx_offset = 0

    # Initialize 3D field arrays
    grid_x, grid_y, grid_z = np.meshgrid(x_cart, y_cart, z_cart, indexing='ij')
    bx_3d = np.zeros(grid_x.shape)
    by_3d = np.zeros(grid_x.shape)
    bz_3d = np.zeros(grid_x.shape)

    # Precompute Cartesian -> cylindrical conversion for entire grid
    r_cart = np.sqrt(grid_x ** 2 + grid_y ** 2)
    theta_cart = np.rad2deg(np.arctan2(grid_y, grid_x))
    # Handle negative angles
    theta_cart = np.where(theta_cart < 0, theta_cart + 360.0, theta_cart)

    print(f"Processing {len(z_unique)} z-slices...")

    # Process each z slice
    for j, z_val in enumerate(z_unique):
        i = j + idx_offset
        k = idx_offset - j

        # Load all components at this z
        field_data = {}
        for comp in ['br', 'bf', 'bz']:
            matching = metadata[(metadata["fcomp"] == comp) & (metadata["zpos"] == 10.0 * z_val)]
            if len(matching) == 0:
                raise ValueError(f"Missing {comp} component at z={10.0 * z_val} mm")

            fn = matching["fn"][0]
            print(f"  [{j + 1}/{len(z_unique)}] Reading {comp} at z = {10.0 * z_val} mm")

            with open(fn, 'r') as infile:
                raw_lines = infile.readlines()[4:-1]

            field_flat = np.array([np.fromstring(line, sep=" ") for line in raw_lines]).flatten()

            # Reshape to 2D grid (theta varies fastest, so reshape as (nr, nth))
            field_data[comp] = field_flat.reshape(nr, nth)

        # Create interpolators on the native (r, theta) grid - MUCH FASTER than griddata
        print(f"  [{j + 1}/{len(z_unique)}] Creating interpolators...")

        br_interp = RegularGridInterpolator(
            (r_unique, th_unique),
            field_data["br"],
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        bf_interp = RegularGridInterpolator(
            (r_unique, th_unique),
            field_data["bf"],
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        bz_interp = RegularGridInterpolator(
            (r_unique, th_unique),
            field_data["bz"],
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        # Interpolate all points at once (vectorized - FAST!)
        print(f"  [{j + 1}/{len(z_unique)}] Interpolating...")

        # Extract slice and flatten
        r_slice = r_cart[:, :, i].ravel()
        theta_slice = theta_cart[:, :, i].ravel()
        points = np.column_stack([r_slice, theta_slice])

        # Interpolate cylindrical components
        br_cart = br_interp(points).reshape(interp_resolution, interp_resolution)
        bf_cart = bf_interp(points).reshape(interp_resolution, interp_resolution)
        bz_cart = bz_interp(points).reshape(interp_resolution, interp_resolution)

        # Convert cylindrical field components to Cartesian
        theta_rad = np.deg2rad(theta_cart[:, :, i])
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)

        bx_3d[:, :, i] = br_cart * cos_theta - bf_cart * sin_theta
        by_3d[:, :, i] = br_cart * sin_theta + bf_cart * cos_theta
        bz_3d[:, :, i] = bz_cart

        # Mirror if requested
        if mirror and j != 0:
            bx_3d[:, :, k] = -bx_3d[:, :, i]
            by_3d[:, :, k] = -by_3d[:, :, i]
            bz_3d[:, :, k] = bz_3d[:, :, i]

        print(f"  [{j + 1}/{len(z_unique)}] Done!")

    # Convert to SI units (cm -> m)
    result = {
        'grid': {
            'x': x_cart * 0.01,  # cm -> m
            'y': y_cart * 0.01,
            'z': z_cart * 0.01
        },
        'values': {
            'x': bx_3d,  # Assume already in Tesla
            'y': by_3d,
            'z': bz_3d
        },
        'dim': 3,
        'field_type': 'magnetic',
        'metadata': {
            'z_positions': z_unique,  # cm
            'n_files': len(map_files),
            'mirrored': mirror,
            'interp_resolution': interp_resolution,
            'r_range': (sr, nr * dr),  # cm
            'theta_range': (sth, nth * dth),  # degrees
            'filename': os.path.basename(directory)
        }
    }

    print(f"\nComplete! Loaded 3D field: {bx_3d.shape}")

    return result


def load_h5part(filename):
    """
    Load field map from H5Part format (HDF5).

    File Structure:
    ---------------
    /
    ├── Step#0/
    │   └── Block/
    │       ├── Efield/
    │       │   ├── 0 (Ex component)
    │       │   ├── 1 (Ey component)
    │       │   ├── 2 (Ez component)
    │       │   ├── __Spacing__ (attribute)
    │       │   └── __Origin__ (attribute)
    │       └── Hfield/
    │           ├── 0 (Hx component)
    │           ├── 1 (Hy component)
    │           ├── 2 (Hz component)
    │           ├── __Spacing__ (attribute)
    │           └── __Origin__ (attribute)

    The file contains both E and H fields on regular 3D grids.
    Data is stored with shape (nz, ny, nx) in the file.

    Parameters:
    -----------
    filename : str
        Path to .h5 or .h5part file

    Returns:
    --------
    dict with keys:
        'grid': dict with 'x', 'y', 'z' arrays (1D, in meters)
        'values': dict with 'x', 'y', 'z' arrays (3D, in Tesla)
                  Note: Returns H-field (magnetic) by default
        'dim': int (3)
        'field_type': str ('magnetic')
        'metadata': dict with 'spacing', 'origin', 'shape', 'has_efield'

    Notes:
    ------
    - If you need E-field instead, modify the 'field_group' variable
    - H-field is magnetic field in Tesla
    - Coordinates follow the storage convention: data[iz, iy, ix]
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    try:
        h5_file = h5py.File(filename, 'r')

        # Navigate to field data
        step0 = h5_file['Step#0']
        block = step0['Block']

        # Check which fields are available
        has_efield = 'Efield' in block
        has_hfield = 'Hfield' in block

        if not has_hfield:
            raise ValueError("No Hfield found in H5Part file")

        # Load H-field (magnetic field)
        field_group = block['Hfield']

        # Read field components
        hx = np.array(field_group['0'])  # Shape: (nz, ny, nx)
        hy = np.array(field_group['1'])
        hz = np.array(field_group['2'])

        # Read grid information from attributes
        spacing = field_group.attrs['__Spacing__']  # [dx, dy, dz] in meters
        origin = field_group.attrs['__Origin__']  # [x0, y0, z0] in meters

        # Determine grid dimensions
        nz, ny, nx = hx.shape

        # Create 1D grid arrays
        x_grid = origin[0] + np.arange(nx) * spacing[0]
        y_grid = origin[1] + np.arange(ny) * spacing[1]
        z_grid = origin[2] + np.arange(nz) * spacing[2]

        # Transpose data from (nz, ny, nx) to (nx, ny, nz) for consistency
        bx_3d = np.transpose(hx, (2, 1, 0))
        by_3d = np.transpose(hy, (2, 1, 0))
        bz_3d = np.transpose(hz, (2, 1, 0))

        h5_file.close()

    except Exception as e:
        if 'h5_file' in locals():
            h5_file.close()
        raise RuntimeError(f"Error reading H5Part file: {e}")

    result = {
        'grid': {
            'x': x_grid,
            'y': y_grid,
            'z': z_grid
        },
        'values': {
            'x': bx_3d,
            'y': by_3d,
            'z': bz_3d
        },
        'dim': 3,
        'field_type': 'magnetic',
        'metadata': {
            'spacing': spacing,
            'origin': origin,
            'shape': (nx, ny, nz),
            'has_efield': has_efield,
            'filename': os.path.basename(filename)
        }
    }

    return result


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_load_opera_table():
    """Test OPERA table loader with synthetic data."""
    import tempfile

    # Create test file
    test_data = """3 3 1
        1 X [cm]
        2 Y [cm]
        3 BZ [Tesla]
        0
        0.0 0.0 0.5
        1.0 0.0 0.6
        2.0 0.0 0.7
        0.0 1.0 0.8
        1.0 1.0 0.9
        2.0 1.0 1.0
        0.0 2.0 1.1
        1.0 2.0 1.2
        2.0 2.0 1.3
        """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.table', delete=False) as f:
        f.write(test_data)
        temp_file = f.name

    try:
        result = load_opera_table(temp_file)

        # Check structure
        assert 'grid' in result
        assert 'values' in result
        assert 'dim' in result
        assert 'field_type' in result
        assert 'metadata' in result

        # Check dimensions
        assert result['dim'] == 2
        assert result['field_type'] == 'magnetic'

        # Check grid (should be in meters)
        assert len(result['grid']['x']) == 3
        assert len(result['grid']['y']) == 3
        assert np.allclose(result['grid']['x'], [0.0, 0.01, 0.02])  # cm -> m

        # Check values
        assert result['values']['z'].shape == (3, 3, 1)
        assert result['values']['z'][0, 0, 0] == 0.5  # Already in Tesla

        print("test_load_opera_table: PASSED")

    finally:
        os.unlink(temp_file)


def test_load_comsol():
    """Test COMSOL loader with synthetic data."""
    import tempfile

    # Create test file
    test_data = """% Model: test
        % Version: COMSOL 5.0
        % Model info
        % Dimension: 3
        % Nodes: 8
        % Description: Test field
        % Extra line
        % Length unit: cm
        % x y z mf.Bx (T) mf.By (T) mf.Bz (T)
        0.0 0.0 0.0 0.1 0.2 0.3
        1.0 0.0 0.0 0.1 0.2 0.3
        0.0 1.0 0.0 0.1 0.2 0.3
        1.0 1.0 0.0 0.1 0.2 0.3
        0.0 0.0 1.0 0.1 0.2 0.3
        1.0 0.0 1.0 0.1 0.2 0.3
        0.0 1.0 1.0 0.1 0.2 0.3
        1.0 1.0 1.0 0.1 0.2 0.3
        """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.comsol', delete=False) as f:
        f.write(test_data)
        temp_file = f.name

    try:
        result = load_comsol(temp_file)

        # Check structure
        assert result['dim'] == 3
        assert result['field_type'] == 'magnetic'

        # Check grid (should be in meters)
        assert len(result['grid']['x']) == 2
        assert len(result['grid']['y']) == 2
        assert len(result['grid']['z']) == 2
        assert np.allclose(result['grid']['x'], [0.0, 0.01])  # cm -> m

        # Check values
        assert result['values']['x'].shape == (2, 2, 2)
        assert np.allclose(result['values']['x'], 0.1)

        print("test_load_comsol: PASSED")

    finally:
        os.unlink(temp_file)


def test_load_opal_midplane():
    """Test OPAL midplane loader with synthetic data."""
    import tempfile

    # Create test file
    test_data = """0.0
        10.0
        0.0
        45.0
        8
        3
        1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
        2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0
        3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0
        """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
        f.write(test_data)
        temp_file = f.name

    try:
        result = load_opal_midplane(temp_file)

        # Check structure
        assert result['dim'] == 2
        assert result['field_type'] == 'magnetic'

        # Check metadata
        assert result['metadata']['n_points']['r'] == 3
        assert result['metadata']['n_points']['theta'] == 8

        # Check that values are in Tesla (scaled from kGauss)
        assert np.max(result['values']['z']) < 1.0  # Should be 0.1-0.3 T

        print("test_load_opal_midplane: PASSED")

    finally:
        os.unlink(temp_file)


def test_load_h5part():
    """Test H5Part loader with synthetic data."""
    import tempfile

    # Create test HDF5 file
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        temp_file = f.name

    try:
        # Create synthetic H5Part file
        h5_file = h5py.File(temp_file, 'w')
        h5_file.attrs['Resonance Frequency(Hz)'] = 32800000.0

        step0 = h5_file.create_group('Step#0')
        block = step0.create_group('Block')

        # Create Hfield
        hfield = block.create_group('Hfield')

        # Create 3D field (nz=4, ny=3, nx=2)
        hx = np.ones((4, 3, 2)) * 0.1
        hy = np.ones((4, 3, 2)) * 0.2
        hz = np.ones((4, 3, 2)) * 0.3

        hfield.create_dataset('0', data=hx)
        hfield.create_dataset('1', data=hy)
        hfield.create_dataset('2', data=hz)

        hfield.attrs['__Spacing__'] = np.array([0.01, 0.01, 0.01])  # 1 cm spacing
        hfield.attrs['__Origin__'] = np.array([0.0, 0.0, 0.0])

        # Create empty Efield
        efield = block.create_group('Efield')
        efield.create_dataset('0', data=np.zeros((4, 3, 2)))
        efield.create_dataset('1', data=np.zeros((4, 3, 2)))
        efield.create_dataset('2', data=np.zeros((4, 3, 2)))
        efield.attrs['__Spacing__'] = np.array([0.01, 0.01, 0.01])
        efield.attrs['__Origin__'] = np.array([0.0, 0.0, 0.0])

        h5_file.close()

        # Test loading
        result = load_h5part(temp_file)

        # Check structure
        assert result['dim'] == 3
        assert result['field_type'] == 'magnetic'

        # Check grid
        assert len(result['grid']['x']) == 2
        assert len(result['grid']['y']) == 3
        assert len(result['grid']['z']) == 4

        # Check values (should be transposed to (nx, ny, nz))
        assert result['values']['x'].shape == (2, 3, 4)
        assert np.allclose(result['values']['x'], 0.1)
        assert np.allclose(result['values']['y'], 0.2)
        assert np.allclose(result['values']['z'], 0.3)

        # Check metadata
        assert result['metadata']['has_efield'] == True
        assert np.allclose(result['metadata']['spacing'], [0.01, 0.01, 0.01])

        print("test_load_h5part: PASSED")

    finally:
        os.unlink(temp_file)


def test_unit_conversions():
    """Test unit conversion functionality."""

    # Test length conversions
    assert UNIT_CONVERSIONS['length']['mm'] == 0.001
    assert UNIT_CONVERSIONS['length']['cm'] == 0.01
    assert UNIT_CONVERSIONS['length']['m'] == 1.0

    # Test magnetic field conversions
    assert UNIT_CONVERSIONS['magnetic']['gauss'] == 1e-4
    assert UNIT_CONVERSIONS['magnetic']['kgauss'] == 0.1
    assert UNIT_CONVERSIONS['magnetic']['tesla'] == 1.0

    # Test electric field conversions
    assert UNIT_CONVERSIONS['electric']['v/mm'] == 1000.0
    assert UNIT_CONVERSIONS['electric']['v/cm'] == 100.0
    assert UNIT_CONVERSIONS['electric']['v/m'] == 1.0

    print("test_unit_conversions: PASSED")


def test_coordinate_transforms():
    """Test coordinate transformation functions."""

    # Test polar to Cartesian
    r = np.array([1.0, 2.0])
    theta = np.array([0.0, 90.0])
    x, y = _pol2cart(r, theta)

    assert np.allclose(x, [1.0, 0.0], atol=1e-10)
    assert np.allclose(y, [0.0, 2.0], atol=1e-10)

    # Test cylindrical to Cartesian
    x, y, z = _cylinder_to_cartesian(r, theta, 5.0)
    assert np.allclose(z, 5.0)

    # Test field transformation
    br = np.array([1.0, 0.0])
    btheta = np.array([0.0, 1.0])
    bz_in = np.array([0.5, 0.5])

    x, y, z, bx, by, bz_out = _cylinder_to_cartesian_field(r, theta, 5.0, br, btheta, bz_in)

    # At theta=0: bx=br, by=btheta
    # At theta=90: bx=-btheta, by=br
    assert np.allclose(bx, [1.0, -1.0], atol=1e-10)
    assert np.allclose(by, [0.0, 0.0], atol=1e-10)
    assert np.allclose(bz_out, 0.5)

    print("test_coordinate_transforms: PASSED")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 60)
    print("Running field_loaders.py unit tests")
    print("=" * 60 + "\n")

    test_unit_conversions()
    test_coordinate_transforms()
    test_load_opera_table()
    test_load_comsol()
    test_load_opal_midplane()
    test_load_h5part()

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
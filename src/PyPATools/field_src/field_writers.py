"""
field_writers.py - Field File Writers for PyPATools

Writers mirroring the loaders in field_loaders.py, so every format PyPATools
can read can also be written (round-trip: write_x -> load_x -> identical
grid/values). All writers take the field on a regular grid as plain arrays:

    grid   : dict {'x': ndarray, 'y': ndarray, 'z': ndarray}  (1D, meters)
    values : dict {'x': ndarray, 'y': ndarray, 'z': ndarray}  (ND, Tesla or V/m)

Spatial dimensions are those grid axes with more than one point (matching
Field.from_arrays / the loaders). Value arrays must have shape
(len(axis_1), ..., len(axis_dim)) over the spatial axes, in 'ij' order.

Part of: PyPATools module
"""

import os

import numpy as np

# Spatial axes in canonical order (must match field_loaders / Field.from_arrays).
_AXES = ("x", "y", "z")

# Column-label prefix and unit string per field type.
_FIELD_STYLES = {
    "magnetic": ("B", "(T)"),
    "electric": ("E", "(V/m)"),
}


def _spatial_axes(grid):
    """Return the list of axis names with more than one grid point."""
    return [k for k in _AXES if k in grid and len(np.atleast_1d(grid[k])) > 1]


def write_comsol(filename, grid, values, *,
                 field_type="magnetic",
                 components=None,
                 model="PyPATools",
                 version="PyPATools field_writers",
                 description=None,
                 fmt="%.10e"):
    """
    Write a field to the COMSOL-style text format read by load_comsol().

    The header is exactly nine lines (load_comsol parses fixed header
    positions: line 3 -> dimension, line 4 -> node count, line 7 -> length
    unit, line 8 -> column labels). Data rows are written x-major (last
    spatial axis varies fastest); the loader re-sorts, so ordering is not
    load-bearing, but this matches the natural reshape order.

    Parameters
    ----------
    filename : str
        Output path. By convention '.comsol' (Field.from_file dispatches on it).
    grid : dict
        {'x','y','z'} 1D arrays in meters. Axes of length 1 are treated as a
        constant slice and are NOT written as a column (dimension drops).
    values : dict
        Field components on the spatial grid, in Tesla (magnetic) or V/m
        (electric). Each array must have shape (n_axis1, ..., n_axisD).
    field_type : str
        'magnetic' or 'electric' (sets column labels and unit string).
    components : str or sequence, optional
        Which components to write, e.g. 'z' (midplane Bz only) or 'xyz'.
        Default: all of x, y, z present in `values`.
    model, version, description : str
        Free-text header fields.
    fmt : str
        Number format for np.savetxt.

    Returns
    -------
    int : 0 on success.
    """
    if field_type not in _FIELD_STYLES:
        raise ValueError(f"field_type must be one of {list(_FIELD_STYLES)}, got {field_type!r}")
    prefix, unit_str = _FIELD_STYLES[field_type]

    axes = _spatial_axes(grid)
    if not axes:
        raise ValueError("write_comsol needs at least one grid axis with > 1 point.")
    dim = len(axes)
    axis_arrays = [np.asarray(grid[k], dtype=float) for k in axes]
    shape = tuple(len(a) for a in axis_arrays)
    n_nodes = int(np.prod(shape))

    if components is None:
        comps = [c for c in _AXES if c in values]
    else:
        comps = [c.lower() for c in components]
    if not comps:
        raise ValueError("No field components selected for writing.")

    comp_arrays = []
    for c in comps:
        if c not in values:
            raise KeyError(f"Component '{c}' not present in values dict.")
        arr = np.asarray(values[c], dtype=float)
        # Squeeze out non-spatial (singleton) axes so 2D slices of 3D arrays work.
        arr = arr.reshape(shape)
        comp_arrays.append(arr)

    # Coordinate columns: x-major ordering (last axis fastest, 'ij' + C-ravel).
    mesh = np.meshgrid(*axis_arrays, indexing="ij")
    columns = [m.ravel() for m in mesh] + [a.ravel() for a in comp_arrays]
    data = np.column_stack(columns)

    from datetime import date
    if description is None:
        description = f"{field_type.capitalize()} field components"

    coord_labels = "".join(f"{k:<20}" for k in axes)
    comp_labels = "".join(f"{prefix}{c} {unit_str}".ljust(20) for c in comps)
    header_lines = [
        f"% Model:              {model}",
        f"% Version:            {version}",
        f"% Date:               {date.today()}",
        f"% Dimension:          {dim}",
        f"% Nodes:              {n_nodes}",
        f"% Expressions:        {len(comps)}",
        f"% Description:        {description}",
        "% Length unit:        m",
        f"% {coord_labels}{comp_labels}".rstrip(),
    ]

    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(filename, "w") as outfile:
        outfile.write("\n".join(header_lines) + "\n")
        np.savetxt(outfile, data, fmt=fmt, delimiter="   ")

    return 0


def write_opal_midplane(filename, grid, values, *,
                        r_min=0.0, r_max=None, dr=1.0,
                        theta_min=0.0, theta_max=360.0, dtheta=0.5,
                        scaling=10.0,
                        values_per_line=5,
                        fmt="%.5e",
                        method="linear"):
    """
    Write an OPAL CARBONCYCL midplane map (.dat/.txt), the counterpart of
    load_opal_midplane().

    Takes a CARTESIAN midplane field in SI and resamples it onto the polar
    (r, theta) grid OPAL wants, in OPAL's units: **mm and kGauss**.

    File Format (matching load_opal_midplane):
    ------------------------------------------
    Line 1: starting radius   (mm)      '%.2e'
    Line 2: radius step       (mm)      '%.2e'
    Line 3: starting angle    (degrees) '%.2e'
    Line 4: angle step        (degrees) '%.2e'
    Line 5: number of angle points      '%i'
    Line 6: number of radius points     '%i'
    Lines 7+: Bz in kGauss, `values_per_line` per line, theta varying fastest
              (so the block reshapes as (n_r, n_theta)).

    A full turn is written WITHOUT the duplicate endpoint: if
    theta_max - theta_min == 360 the last angle is dropped, so the angles are
    theta_min .. theta_max - dtheta and n_theta == 360 / dtheta.

    Parameters
    ----------
    filename : str
        Output path.
    grid : dict
        {'x': ndarray, 'y': ndarray} 1D arrays in METERS. A singleton 'z' is
        ignored (this is a midplane map).
    values : dict
        Must contain 'z' -- Bz on the (x, y) grid in TESLA, shape
        (len(x), len(y)) in 'ij' order. 'x'/'y' components are ignored: the
        CARBONCYCL format stores Bz only.
    r_min, r_max, dr : float
        Radial grid in MILLIMETERS. r_max defaults to the largest radius fully
        contained in the Cartesian data, i.e. min(|x_min|, x_max, |y_min|, y_max).
    theta_min, theta_max, dtheta : float
        Azimuthal grid in DEGREES.
    scaling : float
        Applied to the Tesla values on the way out. Default 10.0 (T -> kGauss).
        Pass 1.0 to write Tesla instead.
    values_per_line : int
        Numbers per data line (OPAL does not care; 5 matches the usual files).
    fmt : str
        Number format for the data block.
    method : str
        Interpolation method handed to RegularGridInterpolator.

    Returns
    -------
    int : 0 on success.
    """
    from scipy.interpolate import RegularGridInterpolator

    if "z" not in values:
        raise KeyError("write_opal_midplane needs values['z'] (Bz) on the (x, y) grid.")
    for k in ("x", "y"):
        if k not in grid or len(np.atleast_1d(grid[k])) < 2:
            raise ValueError(f"write_opal_midplane needs a 2D Cartesian grid; "
                             f"axis '{k}' is missing or has < 2 points.")

    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid["y"], dtype=float)
    bz = np.asarray(values["z"], dtype=float).reshape(len(x), len(y))

    # Largest circle fully inside the Cartesian patch (meters -> mm).
    if r_max is None:
        r_max = 1000.0 * min(abs(x[0]), abs(x[-1]), abs(y[0]), abs(y[-1]))

    # A full turn would repeat theta_min at theta_max; drop the duplicate.
    if theta_max - theta_min == 360.0:
        theta_max -= dtheta

    n_theta = int(round((theta_max - theta_min) / dtheta)) + 1
    n_r = int(round((r_max - r_min) / dr)) + 1

    r_mm = r_min + dr * np.arange(n_r)
    th_deg = theta_min + dtheta * np.arange(n_theta)

    # Sample on the polar grid; r varies slowest so the flat block is
    # (n_r, n_theta) with theta fastest, which is what the loader expects.
    th_rad = np.deg2rad(th_deg)
    rr = (r_mm * 0.001)[:, None]                       # mm -> m, column
    xs = rr * np.cos(th_rad)[None, :]
    ys = rr * np.sin(th_rad)[None, :]

    interp = RegularGridInterpolator((x, y), bz, method=method,
                                     bounds_error=False, fill_value=0.0)
    bz_polar = interp(np.column_stack([xs.ravel(), ys.ravel()])) * scaling

    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(filename, "w") as outfile:
        # OPAL units here are mm and kGauss.
        outfile.write("%.2e\n" % r_min)
        outfile.write("%.2e\n" % dr)
        outfile.write("%.2e\n" % theta_min)
        outfile.write("%.2e\n" % dtheta)
        outfile.write("%i\n" % n_theta)
        outfile.write("%i\n" % n_r)

        # Rows of `values_per_line`, each number followed by two spaces, to
        # match the files the OPAL toolchain has always produced.
        n_full = (bz_polar.size // values_per_line) * values_per_line
        if n_full:
            np.savetxt(outfile, bz_polar[:n_full].reshape(-1, values_per_line),
                       fmt=fmt, delimiter="  ", newline="  \n")
        if n_full < bz_polar.size:
            outfile.write("".join((fmt + "  ") % v for v in bz_polar[n_full:]) + "\n")

    return 0

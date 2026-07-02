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

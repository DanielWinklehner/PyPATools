"""
field_visualization.py - Visualization Tools for Field Objects

Provides plotting utilities for Field objects including:
- Contour plots on arbitrary planes
- Vector field plots
- 3D isosurface plots
- Field line tracing

Usage:
    from field_visualization import plot_field_slice
    plot_field_slice(field, axis='z', intersect=0.0)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Optional, Tuple, Union
from ..field import Field


def plot_field_slice(field: Field,
                     axis: str = 'z',
                     intersect: float = 0.0,
                     limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                     resolution: int = 100,
                     levels: int = 20,
                     figsize: Tuple[float, float] = (14, 4),
                     cmap: str = 'RdBu_r',
                     show: bool = True,
                     save: Optional[str] = None) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot field components as contour plots on a plane.

    Creates three subplots showing Bx, By, Bz (or Ex, Ey, Ez) on a plane
    perpendicular to the specified axis at the given intersection point.

    Parameters
    ----------
    field : Field
        Field object to visualize
    axis : str
        Axis perpendicular to the plot plane ('x', 'y', or 'z')
    intersect : float
        Position along the perpendicular axis where plane intersects [m]
    limits : tuple of tuples, optional
        Plot limits as ((min1, max1), (min2, max2)) in meters.
        If None, uses field interpolator bounds.
        For axis='z': ((xmin, xmax), (ymin, ymax))
        For axis='y': ((xmin, xmax), (zmin, zmax))
        For axis='x': ((ymin, ymax), (zmin, zmax))
    resolution : int
        Number of points in each direction (default: 100)
    levels : int
        Number of contour levels (default: 20)
    figsize : tuple
        Figure size in inches (default: (14, 4))
    cmap : str
        Colormap name (default: 'RdBu_r' for diverging fields)
    show : bool
        Whether to display the plot (default: True)
    save : str, optional
        If provided, save figure to this filename

    Returns
    -------
    fig : matplotlib.Figure
        Figure object
    axes : np.ndarray
        Array of axis objects

    Examples
    --------
    >>> field = Field.from_file('magnetic_field.table')
    >>> fig, axes = plot_field_slice(field, axis='z', intersect=0.0)

    >>> # Custom limits: xy plane from -5 to 5 cm
    >>> plot_field_slice(field, axis='z', intersect=0.0,
    ...                  limits=((-0.05, 0.05), (-0.05, 0.05)))

    >>> # Save to file
    >>> plot_field_slice(field, axis='x', intersect=0.01, save='field_slice.png')
    """

    # Validate inputs
    axis = axis.lower()
    if axis not in ['x', 'y', 'z']:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")

    if field.dim == 0:
        raise ValueError("Cannot plot slice of 0D (constant) field")

    # Determine which coordinates to use
    axis_dict = {'x': 0, 'y': 1, 'z': 2}
    perp_axis = axis_dict[axis]

    # Get axis labels and coordinate arrays
    if axis == 'z':
        coord1_label, coord2_label = 'x', 'y'
        coord1_idx, coord2_idx = 0, 1
    elif axis == 'y':
        coord1_label, coord2_label = 'x', 'z'
        coord1_idx, coord2_idx = 0, 2
    else:  # axis == 'x'
        coord1_label, coord2_label = 'y', 'z'
        coord1_idx, coord2_idx = 1, 2

    # Get limits from interpolator or user input
    if limits is None:
        # Extract bounds from interpolator
        if field.dim == 1:
            # 1D field: limited options
            if axis != 'z':
                raise ValueError("1D field can only be sliced perpendicular to z-axis")
            limits = ((-0.1, 0.1), (-0.1, 0.1))  # Default for 1D
        else:
            # Get bounds from interpolator grid
            grid = field._field['z'].grid  # Access interpolator grid

            if field.dim == 2:
                # 2D field: only two dimensions available
                lim1 = (grid[0][0], grid[0][-1])
                lim2 = (grid[1][0], grid[1][-1])
                limits = (lim1, lim2)
            else:  # field.dim == 3
                # 3D field: extract appropriate dimensions
                all_lims = [(g[0], g[-1]) for g in grid]
                limits = (all_lims[coord1_idx], all_lims[coord2_idx])

    # Create coordinate arrays
    coord1 = np.linspace(limits[0][0], limits[0][1], resolution)
    coord2 = np.linspace(limits[1][0], limits[1][1], resolution)

    # Create meshgrid
    C1, C2 = np.meshgrid(coord1, coord2, indexing='ij')

    # Build 3D point array
    pts = np.zeros((resolution * resolution, 3))

    if axis == 'z':
        pts[:, 0] = C1.ravel()
        pts[:, 1] = C2.ravel()
        pts[:, 2] = intersect
    elif axis == 'y':
        pts[:, 0] = C1.ravel()
        pts[:, 1] = intersect
        pts[:, 2] = C2.ravel()
    else:  # axis == 'x'
        pts[:, 0] = intersect
        pts[:, 1] = C1.ravel()
        pts[:, 2] = C2.ravel()

    # Query field
    fx, fy, fz = field(pts)

    # Reshape to 2D
    fx = fx.reshape(resolution, resolution)
    fy = fy.reshape(resolution, resolution)
    fz = fz.reshape(resolution, resolution)

    # Determine field magnitude for colorbar scaling
    f_mag = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
    vmax = np.max(np.abs([fx, fy, fz]))

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Component names
    component_names = ['Bx', 'By', 'Bz']  # Assume magnetic, could detect from metadata
    component_data = [fx, fy, fz]

    # Plot each component
    for i, (ax, name, data) in enumerate(zip(axes, component_names, component_data)):
        # Contour plot
        contour = ax.contourf(C1, C2, data, levels=levels, cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.contour(C1, C2, data, levels=levels, colors='k', alpha=0.2, linewidths=0.5)

        # Colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label(f'{name} (T)', rotation=270, labelpad=20)

        # Labels
        ax.set_xlabel(f'{coord1_label} (m)')
        ax.set_ylabel(f'{coord2_label} (m)')
        ax.set_title(f'{name} at {axis}={intersect * 1000:.1f} mm')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')

    # Overall title
    fig.suptitle(f"Field Components on {axis}-plane at {axis}={intersect * 1000:.1f} mm",
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save if requested
    if save is not None:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save}")

    # Show if requested
    if show:
        plt.show()

    return fig, axes


def plot_field_magnitude(field: Field,
                         axis: str = 'z',
                         intersect: float = 0.0,
                         limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                         resolution: int = 100,
                         levels: int = 20,
                         figsize: Tuple[float, float] = (8, 7),
                         cmap: str = 'viridis',
                         show: bool = True,
                         save: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot field magnitude |B| on a plane.

    Similar to plot_field_slice but shows only the magnitude with vectors overlaid.

    Parameters
    ----------
    field : Field
        Field object to visualize
    axis : str
        Axis perpendicular to the plot plane
    intersect : float
        Position along perpendicular axis [m]
    limits : tuple of tuples, optional
        Plot limits
    resolution : int
        Number of points for contour (default: 100)
    levels : int
        Number of contour levels (default: 20)
    figsize : tuple
        Figure size (default: (8, 7))
    cmap : str
        Colormap (default: 'viridis')
    show : bool
        Display plot (default: True)
    save : str, optional
        Save filename

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """

    # Validate and setup (same as plot_field_slice)
    axis = axis.lower()
    if axis not in ['x', 'y', 'z']:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")

    # Determine coordinates
    if axis == 'z':
        coord1_label, coord2_label = 'x', 'y'
        coord1_idx, coord2_idx = 0, 1
        perp_idx = 2
    elif axis == 'y':
        coord1_label, coord2_label = 'x', 'z'
        coord1_idx, coord2_idx = 0, 2
        perp_idx = 1
    else:
        coord1_label, coord2_label = 'y', 'z'
        coord1_idx, coord2_idx = 1, 2
        perp_idx = 0

    # Get limits
    if limits is None:
        if field.dim >= 2:
            grid = field._field['z'].grid
            if field.dim == 2:
                limits = ((grid[0][0], grid[0][-1]), (grid[1][0], grid[1][-1]))
            else:
                all_lims = [(g[0], g[-1]) for g in grid]
                limits = (all_lims[coord1_idx], all_lims[coord2_idx])
        else:
            limits = ((-0.1, 0.1), (-0.1, 0.1))

    # Create grid
    coord1 = np.linspace(limits[0][0], limits[0][1], resolution)
    coord2 = np.linspace(limits[1][0], limits[1][1], resolution)
    C1, C2 = np.meshgrid(coord1, coord2, indexing='ij')

    # Build points
    pts = np.zeros((resolution * resolution, 3))
    pts[:, coord1_idx] = C1.ravel()
    pts[:, coord2_idx] = C2.ravel()
    pts[:, perp_idx] = intersect

    # Query field
    fx, fy, fz = field(pts)

    # Reshape
    fx = fx.reshape(resolution, resolution)
    fy = fy.reshape(resolution, resolution)
    fz = fz.reshape(resolution, resolution)

    # Calculate magnitude
    f_mag = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)

    # Extract in-plane components for vector plot
    if axis == 'z':
        f1, f2 = fx, fy
    elif axis == 'y':
        f1, f2 = fx, fz
    else:
        f1, f2 = fy, fz

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Contour plot of magnitude
    contour = ax.contourf(C1, C2, f_mag, levels=levels, cmap=cmap)

    # Add vector field (subsample for clarity)
    step = resolution // 15  # ~15 arrows per direction
    ax.quiver(C1[::step, ::step], C2[::step, ::step],
              f1[::step, ::step], f2[::step, ::step],
              color='white', alpha=0.6, width=0.003, scale_units='xy')

    # Colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('|B| (T)', rotation=270, labelpad=20)

    # Labels
    ax.set_xlabel(f'{coord1_label} (m)')
    ax.set_ylabel(f'{coord2_label} (m)')
    ax.set_title(f'Field Magnitude at {axis}={intersect * 1000:.1f} mm')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--', color='white')

    plt.tight_layout()

    if save is not None:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save}")

    if show:
        plt.show()

    return fig, ax


def plot_field_multiplane(field: Field,
                          positions: Optional[list] = None,
                          axis: str = 'z',
                          limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                          resolution: int = 80,
                          figsize: Tuple[float, float] = (15, 10),
                          cmap: str = 'RdBu_r',
                          show: bool = True,
                          save: Optional[str] = None) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot field magnitude at multiple positions along an axis.

    Creates a grid of subplots showing field evolution.

    Parameters
    ----------
    field : Field
        Field object to visualize
    positions : list of float, optional
        Positions along axis to plot. If None, automatically chooses 6 positions
    axis : str
        Axis along which to vary position
    limits : tuple of tuples, optional
        Plot limits for the plane
    resolution : int
        Grid resolution (default: 80)
    figsize : tuple
        Figure size (default: (15, 10))
    cmap : str
        Colormap (default: 'RdBu_r')
    show : bool
        Display plot (default: True)
    save : str, optional
        Save filename

    Returns
    -------
    fig, axes : matplotlib Figure and array of Axes

    Examples
    --------
    >>> # Plot solenoid field at z = -5, 0, 5 cm
    >>> plot_field_multiplane(field, positions=[-0.05, 0.0, 0.05], axis='z')
    """

    # Get positions
    if positions is None:
        # Auto-select positions
        if field.dim >= 3:
            axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
            grid = field._field['z'].grid
            z_min, z_max = grid[axis_idx][0], grid[axis_idx][-1]
            positions = np.linspace(z_min, z_max, 6)
        else:
            positions = np.linspace(-0.1, 0.1, 6)

    n_plots = len(positions)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).ravel()

    # Determine coordinates
    axis_lower = axis.lower()
    if axis_lower == 'z':
        coord1_label, coord2_label = 'x', 'y'
        coord1_idx, coord2_idx = 0, 1
        perp_idx = 2
    elif axis_lower == 'y':
        coord1_label, coord2_label = 'x', 'z'
        coord1_idx, coord2_idx = 0, 2
        perp_idx = 1
    else:
        coord1_label, coord2_label = 'y', 'z'
        coord1_idx, coord2_idx = 1, 2
        perp_idx = 0

    # Get limits
    if limits is None and field.dim >= 2:
        grid = field._field['z'].grid
        if field.dim == 2:
            limits = ((grid[0][0], grid[0][-1]), (grid[1][0], grid[1][-1]))
        else:
            all_lims = [(g[0], g[-1]) for g in grid]
            limits = (all_lims[coord1_idx], all_lims[coord2_idx])
    elif limits is None:
        limits = ((-0.1, 0.1), (-0.1, 0.1))

    # Create coordinate arrays
    coord1 = np.linspace(limits[0][0], limits[0][1], resolution)
    coord2 = np.linspace(limits[1][0], limits[1][1], resolution)
    C1, C2 = np.meshgrid(coord1, coord2, indexing='ij')

    # Find global vmax for consistent colorscale
    all_mags = []
    for pos in positions:
        pts = np.zeros((resolution * resolution, 3))
        pts[:, coord1_idx] = C1.ravel()
        pts[:, coord2_idx] = C2.ravel()
        pts[:, perp_idx] = pos

        fx, fy, fz = field(pts)
        f_mag = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
        all_mags.append(np.max(f_mag))

    vmax = max(all_mags)

    # Plot each position
    for idx, (ax, pos) in enumerate(zip(axes[:n_plots], positions)):
        pts = np.zeros((resolution * resolution, 3))
        pts[:, coord1_idx] = C1.ravel()
        pts[:, coord2_idx] = C2.ravel()
        pts[:, perp_idx] = pos

        fx, fy, fz = field(pts)
        f_mag = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2).reshape(resolution, resolution)

        contour = ax.contourf(C1, C2, f_mag, levels=15, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xlabel(f'{coord1_label} (m)')
        ax.set_ylabel(f'{coord2_label} (m)')
        ax.set_title(f'{axis}={pos * 1000:.1f} mm')
        ax.set_aspect('equal')

        if idx % n_cols == n_cols - 1 or idx == n_plots - 1:
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label('|B| (T)', rotation=270, labelpad=15)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'Field Magnitude at Multiple {axis.upper()} Positions',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save is not None:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save}")

    if show:
        plt.show()

    return fig, axes


if __name__ == "__main__":
    pass
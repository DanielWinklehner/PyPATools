"""
field.py - Electromagnetic Field Interpolation for Particle Tracking

Provides Field class for loading, interpolating, and manipulating EM fields
from various file formats used in accelerator simulations.

Author: Refactored for PyPATools cyclotron design suite
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .field_src.field_loaders import *
import h5py
import pickle
import os
import warnings
from typing import Optional, Union, Tuple, Dict, List, Callable
from abc import ABC, abstractmethod
from .field_src.interpolators import get_interpolator

try:
    import numba
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


    # Create dummy decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

try:
    import openpmd_api as io

    HAS_OPENPMD = True
except ImportError:
    HAS_OPENPMD = False

# ============================================================================
# Constants and Lookup Tables
# ============================================================================

UNIT_SCALES = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "Tesla": 1.0,
    "T": 1.0,
    "Gauss": 1e-4,
    "kGauss": 0.1,
    "V/m": 1.0,
    "V/cm": 100.0,
    "V/mm": 1000.0,
}


# ============================================================================
# Field Base Classes
# ============================================================================

class FieldBase(ABC):
    """Abstract base class for all field types"""

    @abstractmethod
    def __call__(self, pts: np.ndarray) -> np.ndarray:
        """
        Evaluate field at points.

        Parameters
        ----------
        pts : np.ndarray
            Points to evaluate, shape (M, 3)

        Returns
        -------
        field_values : np.ndarray(M, 3)
            Field components [Fx, Fy, Fz] at each point
        """
        pass


    def gradient(self, pts: np.ndarray) -> np.ndarray:
        """
        Compute field gradient (optional, not all fields support this)

        Returns
        -------
        grad : np.ndarray
            Gradient tensor, shape (N, 3, 3) - grad[i,j,k] = d(F_j)/d(x_k)
        """
        raise NotImplementedError("Gradient not implemented for this field type")


class Field(FieldBase):
    """
    Electromagnetic field interpolation class.

    Supports 0D (constant), 1D, 2D, and 3D fields with various interpolation
    methods and coordinate systems.

    Parameters
    ----------
    label : str
        Descriptive label for the field
    dim : int
        Spatial dimensions (0, 1, 2, or 3)
    field : dict, optional
        For 0D fields: {'x': float, 'y': float, 'z': float}
        For ND fields: {'x': interpolator, 'y': interpolator, 'z': interpolator}
    scaling : float
        Global scaling factor applied to field values
    units : str
        Spatial units ('m', 'cm', 'mm')
    debug : bool
        Enable debug output

    Examples
    --------
    # Constant field
    > B = Field(dim=0, field={'x': 0, 'y': 0, 'z': 1.0})
    > B([0, 0, 0])
    (0.0, 0.0, 1.0)

    # Load from file
    > B = Field.from_file('magnetic_field.table')
    > Bx, By, Bz = B(np.array([[0.1, 0.2, 0.3]]))
    """
    def __init__(self,
                 label: str = "Field",
                 dim: int = 0,
                 field: Optional[Dict] = None,
                 scaling: float = 1.0,
                 units: str = "m",
                 debug: bool = False,
                 method: str = "linear",
                 interpolator_backend: str = 'auto'):

        self._label = label
        self._dim = dim
        self._scaling = scaling
        self._debug = debug
        self._filename = None
        self._metadata = {}
        self._interpolator_backend = interpolator_backend
        self._method = method

        # Unit conversion
        if units not in UNIT_SCALES:
            raise ValueError(f"Unknown unit '{units}'. Must be one of {list(UNIT_SCALES.keys())}")
        self._unit_scale = UNIT_SCALES[units]

        # Initialize field storage
        self._field = {"x": None, "y": None, "z": None}
        if field is not None:
            self._field = field

        # Dispatch table for different dimensions
        self._call_dispatch = {
            0: self._get_field_0d,
            1: self._get_field_1d,
            2: self._get_field_2d,
            3: self._get_field_3d
        }

    def plot(self,
             axis: str = 'z',
             intersect: float = 0.0,
             limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
             **kwargs):
        """
        Quick plot of field on a plane (convenience method).

        Parameters
        ----------
        axis : str
            Axis perpendicular to plot plane ('x', 'y', or 'z')
        intersect : float
            Position along perpendicular axis [m]
        limits : tuple of tuples, optional
            Plot limits ((min1, max1), (min2, max2))
        **kwargs
            Additional arguments passed to plot_field_slice()

        Returns
        -------
        fig, axes : matplotlib Figure and Axes

        Examples
        --------
        > field = Field.from_file('magnetic.table')
        > field.plot(axis='z', intersect=0.0)

        > # Custom limits
        > field.plot(axis='x', intersect=0.01,
        ...            limits=((-0.05, 0.05), (-0.05, 0.05)))
        """
        from field_src.field_visualization import plot_field_slice

        return plot_field_slice(self, axis=axis, intersect=intersect,
                                limits=limits, **kwargs)

    # ========================================================================
    # Core API
    # ========================================================================

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        """
        Evaluate field at points.

        Parameters
        ----------
        pts : array_like
            Points to evaluate. Shape must be (M, 3) where M is number of points.
            Single points should be passed as [[x, y, z]] (shape (1, 3)).

        Returns
        -------
        field_values : np.ndarray(M, 3)
            Field components [Fx, Fy, Fz] at each point.
            Always returns 2D array, even for single point.

        Examples
        --------
        > field = Field(dim=0, field={'x': 1.0, 'y': 0.0, 'z': 0.0})
        > field([[0, 0, 0]])  # Single point
        array([[1., 0., 0.]])

        > field([[0, 0, 0], [1, 1, 1]])  # Multiple points
        array([[1., 0., 0.],
               [1., 0., 0.]])
        """
        pts = np.atleast_2d(pts)

        if pts.shape[1] != 3:
            raise ValueError(f"Points must have shape (M, 3), got {pts.shape}")

        return self._call_dispatch[self._dim](pts)

    def __str__(self) -> str:
        if self._dim == 0:
            vals = [self._field[k] for k in ['x', 'y', 'z']]
            return f"Field '{self._label}' (0D constant): {vals}"
        else:
            return f"Field '{self._label}' ({self._dim}D interpolated)"

    def __repr__(self) -> str:
        return f"Field(label='{self._label}', dim={self._dim}, scaling={self._scaling})"

    # ========================================================================
    # Field Algebra
    # ========================================================================

    def __add__(self, other):
        """Add two fields"""
        if isinstance(other, (int, float)):
            return ScaledField(self, offset=other)
        return CompositeField([self, other], [1.0, 1.0])

    def __mul__(self, scalar: float):
        """Scale field by constant"""
        return ScaledField(self, scale=scalar)

    def __rmul__(self, scalar: float):
        """Scale field by constant (reverse)"""
        return self.__mul__(scalar)

    def __sub__(self, other):
        """Subtract fields"""
        if isinstance(other, (int, float)):
            return ScaledField(self, offset=-other)
        return CompositeField([self, other], [1.0, -1.0])

    # ========================================================================
    # Class Methods (Loaders)
    # ========================================================================

    @classmethod
    def from_file(cls, filename: str, **kwargs) -> 'Field':
        """
        Load field from file, auto-detecting format from extension.

        Parameters
        ----------
        filename : str
            Path to field file
        **kwargs
            Additional arguments passed to specific loader

        Returns
        -------
        Field
            Loaded field object
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Field file not found: {filename}")

        _, ext = os.path.splitext(filename)

        field = cls(**kwargs)
        field._filename = filename

        if ext == ".pickle":
            field._load_pickle(filename)
            _data = None
        elif ext == ".h5" or ext == ".h5part":
            _data = field._load_h5part(filename, **kwargs)
        elif ext == ".table":
            _data = field._load_opera_table(filename, **kwargs)
        elif ext == ".comsol":
            _data = field._load_comsol(filename, **kwargs)
        elif ext == ".dat":
            _data = field._load_opal_midplane(filename, **kwargs)
        elif ext == ".map":
            _data = field._load_aima_agora(filename, **kwargs)
        else:
            raise ValueError(f"Unknown file extension: {ext}")

        if _data is not None:
            # Determine dimensionality
            grid_points = [_data['grid'][k] for k in ['x', 'y', 'z'] if k in _data['grid'] and len(_data['grid'][k]) > 1]
            field._dim = _data['dim']
            field._metadata = _data['metadata']

            # Create interpolators
            for component in ['x', 'y', 'z']:
                if component in _data['values']:
                    field._field[component] = get_interpolator(
                        points=tuple(grid_points),
                        values=_data['values'][component],
                        bounds_error=False,
                        fill_value=0.0,
                        method=field._method,
                        backend=field._interpolator_backend  # Use selected backend
                    )

        return field

    @classmethod
    def from_arrays(cls,
                    grid: Dict[str, np.ndarray],
                    values: Dict[str, np.ndarray],
                    **kwargs) -> 'Field':
        """
        Create field from numpy arrays.

        Parameters
        ----------
        grid : dict
            Grid definition: {'x': x_array, 'y': y_array, 'z': z_array}
            1D arrays defining grid points in each dimension
        values : dict
            Field values: {'x': fx_array, 'y': fy_array, 'z': fz_array}
            ND arrays with field components
        **kwargs
            Additional Field constructor arguments

        Returns
        -------
        Field
            Field object with interpolators
        """
        field = cls(**kwargs)

        # Determine dimensionality
        grid_points = [grid[k] for k in ['x', 'y', 'z'] if k in grid and len(grid[k]) > 1]
        field._dim = len(grid_points)

        # Create interpolators
        for component in ['x', 'y', 'z']:
            if component in values:
                field._field[component] = get_interpolator(
                    points=tuple(grid_points),
                    values=values[component],
                    bounds_error=False,
                    fill_value=0.0,
                    method=field._method,
                    backend=field._interpolator_backend  # Use selected backend
                )

        return field

    @classmethod
    def zero(cls, dim: int = 3) -> 'Field':
        """Create a zero field (for omitting E or B)"""
        if dim == 0:
            return cls(label="Zero Field", dim=0, field={'x': 0.0, 'y': 0.0, 'z': 0.0})
        else:
            # Create minimal interpolator
            grid = [np.array([-1e20, 1e20]) for _ in range(dim)]
            vals = np.zeros([2] * dim)
            field = cls(label="Zero Field", dim=dim)
            for component in ['x', 'y', 'z']:
                field._field[component] = get_interpolator(
                    points=tuple(grid),
                    values=vals,
                    bounds_error=False,
                    fill_value=0.0,
                    backend='scipy'  # Zero field is trivial, use scipy
                )
            return field

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str):
        self._label = value

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def scaling(self) -> float:
        return self._scaling

    @scaling.setter
    def scaling(self, value: float):
        self._scaling = value

    @property
    def metadata(self) -> dict:
        return self._metadata

    # ========================================================================
    # Internal Evaluation Methods
    # ========================================================================

    def _get_field_0d(self, pts: np.ndarray) -> np.ndarray:
        """
        Constant field (same everywhere).

        Returns
        -------
        field_values : np.ndarray(M, 3)
            Field components [Fx, Fy, Fz] for M points
        """
        M = len(pts)
        fx = self._scaling * self._field["x"] * np.ones(M)
        fy = self._scaling * self._field["y"] * np.ones(M)
        fz = self._scaling * self._field["z"] * np.ones(M)

        return np.column_stack([fx, fy, fz])

    def _get_field_1d(self, pts: np.ndarray) -> np.ndarray:
        """
        1D field (varies with z only).

        Returns
        -------
        field_values : np.ndarray(M, 3)
            Field components [Fx, Fy, Fz] for M points
        """
        z = pts[:, 2]

        fx = self._scaling * self._field["x"](z)
        fy = self._scaling * self._field["y"](z)
        fz = self._scaling * self._field["z"](z)

        return np.column_stack([fx, fy, fz])

    def _get_field_2d(self, pts: np.ndarray) -> np.ndarray:
        """
        2D field (varies with x, y).

        Returns
        -------
        field_values : np.ndarray(M, 3)
            Field components [Fx, Fy, Fz] for M points
        """
        xy = pts[:, :2]

        fx = self._scaling * self._field["x"](xy)
        fy = self._scaling * self._field["y"](xy)
        fz = self._scaling * self._field["z"](xy)

        return np.column_stack([fx, fy, fz])

    def _get_field_3d(self, pts: np.ndarray) -> np.ndarray:
        """
        3D field (varies with x, y, z).

        Returns
        -------
        field_values : np.ndarray(M, 3)
            Field components [Fx, Fy, Fz] for M points
        """
        fx = self._scaling * self._field["x"](pts)
        fy = self._scaling * self._field["y"](pts)
        fz = self._scaling * self._field["z"](pts)

        return np.column_stack([fx, fy, fz])

    # ========================================================================
    # File I/O (Stubs - will be populated by field_loaders.py)
    # ========================================================================

    def _load_pickle(self, filename: str):
        """Load field from pickle file"""
        with open(filename, "rb") as f:
            data = pickle.load(f)

        # Restore attributes
        for key in ['_label', '_dim', '_scaling', '_field', '_unit_scale', '_metadata']:
            if key in data:
                setattr(self, key, data[key])

    def save_pickle(self, filename: str):
        """Save field to pickle file"""
        data = {
            '_label': self._label,
            '_dim': self._dim,
            '_scaling': self._scaling,
            '_field': self._field,
            '_unit_scale': self._unit_scale,
            '_metadata': self._metadata
        }

        with open(filename, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved field to {filename}")

    @staticmethod
    def _load_h5part(filename: str, **kwargs):
        """Load from H5Part format"""
        return load_h5part(filename)

    @staticmethod
    def _load_opera_table(filename: str, **kwargs):
        """Load from OPERA table format"""
        return load_opera_table(filename)

    @staticmethod
    def _load_comsol(filename: str, **kwargs):
        """Load from COMSOL export format"""
        return load_comsol(filename)

    @staticmethod
    def _load_opal_midplane(filename: str, **kwargs):
        """Load from OPAL midplane format"""
        return load_opal_midplane(filename)

    @staticmethod
    def _load_aima_agora(filename: str, **kwargs):
        """Load from AIMA Agora format"""
        return load_aima_agora(filename)


# ============================================================================
# Numba Kernel for Composite Fields
# ============================================================================

@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _composite_field_add_kernel(field_out, field_in, weight):
    """
    Numba kernel for weighted field addition (parallelized across points).

    Accumulates weighted field contributions: field_out += weight * field_in

    Parameters
    ----------
    field_out : np.ndarray(M, 3)
        Output array (modified in-place)
    field_in : np.ndarray(M, 3)
        Input field values
    weight : float
        Weight factor
    """
    M = field_out.shape[0]

    for i in prange(M):
        # Unrolled loop for 3 components (faster than inner loop)
        field_out[i, 0] += weight * field_in[i, 0]
        field_out[i, 1] += weight * field_in[i, 1]
        field_out[i, 2] += weight * field_in[i, 2]


# ============================================================================
# Composite Fields
# ============================================================================

class CompositeField(FieldBase):
    """
    Linear combination of multiple fields.

    Allows field algebra: F_total = a*F1 + b*F2 + c*F3

    Uses Numba JIT parallelization for efficient batch queries when M > 100.

    Parameters
    ----------
    fields : list of Field
        Component fields
    weights : list of float
        Weights for each field

    Examples
    --------
    > B_main = Field.from_file('main_magnet.table')
    > B_corr = Field.from_file('correction_coils.table')
    > B_total = CompositeField([B_main, B_corr], [1.0, 0.8])
    > B_values = B_total([[0, 0, 0]])  # Returns (1, 3) array
    """

    def __init__(self, fields: List[FieldBase], weights: List[float]):
        if len(fields) != len(weights):
            raise ValueError("Number of fields must match number of weights")

        self.fields = fields
        self.weights = np.array(weights, dtype=np.float64)
        self.n_fields = len(fields)

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        """
        Evaluate composite field at points.

        Uses Numba parallelization for large queries (M > 100).

        Returns
        -------
        field_values : np.ndarray(M, 3)
            Sum of weighted field components
        """
        pts = np.atleast_2d(pts)
        M = len(pts)

        # Initialize output array
        field_total = np.zeros((M, 3), dtype=np.float64)

        # Choose algorithm based on query size
        if HAS_NUMBA and M > 100:
            # Use parallel Numba kernel for large queries
            for field, weight in zip(self.fields, self.weights):
                field_values = field(pts)  # Returns (M, 3)
                _composite_field_add_kernel(field_total, field_values, weight)
        else:
            # Use simple NumPy for small queries (less overhead)
            for field, weight in zip(self.fields, self.weights):
                field_values = field(pts)  # Returns (M, 3)
                field_total += weight * field_values

        return field_total

    def __str__(self):
        return f"CompositeField with {len(self.fields)} components"


class ScaledField(FieldBase):
    """
    Field with scaling and offset: F_new = scale * F_old + offset

    Parameters
    ----------
    field : FieldBase
        Base field
    scale : float
        Multiplicative factor
    offset : float
        Additive offset (applied to all components)
    """

    def __init__(self, field: FieldBase, scale: float = 1.0, offset: float = 0.0):
        self.field = field
        self.scale = scale
        self.offset = offset

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        """
        Evaluate scaled field at points.

        Returns
        -------
        field_values : np.ndarray(M, 3)
            Scaled and offset field values
        """
        field_values = self.field(pts)  # always returns (M, 3)
        return self.scale * field_values + self.offset


# ============================================================================
# Field Map Container
# ============================================================================

class FieldMap:
    """
    Container for electric and magnetic fields.

    Used when both E and B fields are defined in the same simulation region.
    Commonly loaded from single HDF5 file containing both fields.

    Parameters
    ----------
    efield : FieldBase, optional
        Electric field
    bfield : FieldBase, optional
        Magnetic field
    metadata : dict, optional
        Additional information (frequency, geometry, etc.)

    Examples
    --------
    # Load from H5Part file with both E and B
    > fieldmap = FieldMap.from_h5part('rf_cavity.h5')
    > Ex, Ey, Ez = fieldmap.efield([0, 0, 0])
    > Bx, By, Bz = fieldmap.bfield([0, 0, 0])
    """

    def __init__(self,
                 efield: Optional[FieldBase] = None,
                 bfield: Optional[FieldBase] = None,
                 metadata: Optional[dict] = None):
        self.efield = efield if efield is not None else Field.zero()
        self.bfield = bfield if bfield is not None else Field.zero()
        self.metadata = metadata if metadata is not None else {}

    @classmethod
    def from_h5part(cls, filename: str, iteration: int = 0) -> 'FieldMap':
        """
        Load E and B fields from H5Part file.

        Parameters
        ----------
        filename : str
            Path to H5Part file
        iteration : int
            Time step to load (default: 0)

        Returns
        -------
        FieldMap
            Container with both fields
        """
        # Stub - will be implemented with openpmd_io.py
        raise NotImplementedError("H5Part FieldMap loader will be implemented by agent")

    @classmethod
    def from_fields(cls, efield: FieldBase, bfield: FieldBase, **metadata) -> 'FieldMap':
        """Create FieldMap from existing Field objects"""
        return cls(efield=efield, bfield=bfield, metadata=metadata)

    def __str__(self):
        return f"FieldMap(E: {self.efield}, B: {self.bfield})"


# ============================================================================
# Utility Functions
# ============================================================================

def cartesian_to_cylindrical(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian to cylindrical coordinates.

    Parameters
    ----------
    x, y, z : np.ndarray
        Cartesian coordinates

    Returns
    -------
    r, theta, z : np.ndarray
        Cylindrical coordinates (theta in radians)
    """
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return r, theta, z


def cylindrical_to_cartesian(r: np.ndarray, theta: np.ndarray, z: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert cylindrical to Cartesian coordinates.

    Parameters
    ----------
    r, theta, z : np.ndarray
        Cylindrical coordinates (theta in radians)

    Returns
    -------
    x, y, z : np.ndarray
        Cartesian coordinates
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z


def transform_cylindrical_field_to_cartesian(r: np.ndarray, theta: np.ndarray, z: np.ndarray,
                                             fr: np.ndarray, ftheta: np.ndarray, fz: np.ndarray) -> Tuple:
    """
    Transform field components from cylindrical to Cartesian.

    Parameters
    ----------
    r, theta, z : np.ndarray
        Position in cylindrical coords (theta in radians)
    fr, ftheta, fz : np.ndarray
        Field components in cylindrical basis

    Returns
    -------
    x, y, z, fx, fy, fz : np.ndarray
        Position and field in Cartesian coords
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    fx = fr * np.cos(theta) - ftheta * np.sin(theta)
    fy = fr * np.sin(theta) + ftheta * np.cos(theta)

    return x, y, z, fx, fy, fz


if __name__ == "__main__":
    # Basic tests
    print("Testing Field class...")

    # Test 0D field
    B_const = Field(label="Constant B", dim=0, field={'x': 0, 'y': 0, 'z': 1.0})
    print(B_const)
    print("B at origin:", B_const(np.array([0, 0, 0])))

    # Test field algebra
    B_double = 2.0 * B_const
    print("2*B at origin:", B_double(np.array([0, 0, 0])))

    # Test composite
    B1 = Field(dim=0, field={'x': 1, 'y': 0, 'z': 0})
    B2 = Field(dim=0, field={'x': 0, 'y': 1, 'z': 0})
    B_sum = B1 + B2
    print("B1 + B2 at origin:", B_sum(np.array([0, 0, 0])))

    # Test loading
    B3 = Field.from_file(filename=r"backup/uCyclo_v2_Midplane_Res1mm_400x400mm_CARBONCYCL.dat")
    print("B3 at origin:", B3(np.array([0, 0, 0])))
    # B3.plot()

    B4 = Field.from_file(filename=r"backup/uCyclo_v2_Midplane_Res0.5mm_400x400mm.comsol")
    print("B4 at origin:", B4(np.array([0, 0, 0])))
    # B4.plot()

    print("\nAll basic tests passed!")

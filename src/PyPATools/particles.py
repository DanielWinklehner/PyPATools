"""
particles.py - Particle Distribution Class

Holds particle distribution data with efficient (N, 3) array storage and provides
functions to calculate collective beam properties (emittance, Twiss parameters, etc.).

Author: Daniel Winklehner, PyPATools Development Team
"""

import os
from .global_variables import *
import numpy as np
from .species import IonSpecies
from typing import Optional, Tuple, List, Literal
import warnings

from .particles_src import particle_io
from .particles_src import particle_visualization as pv
from .particles_src.distribution_generators import generate_distribution as gen_dist

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn("Numba not available. Performance will be reduced.")

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    def prange(*args, **kwargs):
        return range(*args, **kwargs)

# ============================================================================
# Numba-Accelerated Kernels
# ============================================================================

@njit(fastmath=True, cache=True)
def _calculate_energy_single(px, py, pz, mass_mev, z_energy, relativistic):
    """Calculate kinetic energy for single particle."""
    if z_energy:
        pr = pz
    else:
        pr = np.sqrt(px ** 2 + py ** 2 + pz ** 2)

    if relativistic:
        return np.sqrt((pr * mass_mev) ** 2 + mass_mev ** 2) - mass_mev
    else:
        return mass_mev * pr ** 2 / 2.0


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _calculate_energies_batch(px, py, pz, mass_mev, z_energy, relativistic):
    """Calculate kinetic energies for batch of particles (parallelized)."""
    N = len(px)
    energies = np.empty(N, dtype=np.float64)

    for i in prange(N):
        energies[i] = _calculate_energy_single(px[i], py[i], pz[i],
                                               mass_mev, z_energy, relativistic)

    return energies


@njit(fastmath=True, cache=True)
def _velocity_from_momentum(p, relativistic, clight):
    """Convert momentum (β·γ) to velocity."""
    if relativistic:
        return clight * p / np.sqrt(p ** 2 + 1.0)
    else:
        return clight * p


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _velocities_from_momenta_batch(momenta, relativistic, clight):
    """Convert momenta to velocities (parallelized)."""
    N = momenta.shape[0]
    velocities = np.empty((N, 3), dtype=np.float64)

    for i in prange(N):
        for j in range(3):
            velocities[i, j] = _velocity_from_momentum(momenta[i, j],
                                                       relativistic, clight)

    return velocities


@njit(fastmath=True, cache=True)
def _momentum_from_velocity(v, relativistic, clight):
    """Convert velocity to momentum (β·γ)."""
    if relativistic:
        v_sq = v ** 2
        if v_sq >= clight ** 2:
            v_sq = 0.99999 * clight ** 2
        return v / np.sqrt(clight ** 2 - v_sq)
    else:
        return v / clight


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _momenta_from_velocities_batch(velocities, relativistic, clight):
    """Convert velocities to momenta (parallelized)."""
    N = velocities.shape[0]
    momenta = np.empty((N, 3), dtype=np.float64)

    for i in prange(N):
        for j in range(3):
            momenta[i, j] = _momentum_from_velocity(velocities[i, j],
                                                    relativistic, clight)

    return momenta

# ============================================================================
# ParticleDistribution Class
# ============================================================================

class ParticleDistribution(object):
    """
    A class that holds particle distribution data with efficient (N, 3) array storage.

    Provides functions to calculate collective beam properties like emittance and
    Twiss parameters.

    Parameters
    ----------
    species : IonSpecies, optional
        Ion species. Defaults to protons.
    x, y, z : np.ndarray, optional
        Position coordinates [m]. Must all be same length if provided.
    px, py, pz : np.ndarray, optional
        Momentum components as β·γ. Must all be same length if provided.
    x_vec : np.ndarray(N, 3), optional
        Position vectors [m]. Alternative to x, y, z.
    p_vec : np.ndarray(N, 3), optional
        Momentum vectors as β·γ. Alternative to px, py, pz.
    q : float
        Bunch charge [C]
    f : float
        Bunch frequency [Hz]
    recalculate : bool
        Calculate collective properties on initialization

    Notes
    -----
    Either provide (x, y, z, px, py, pz) OR (x_vec, p_vec), not both.
    Internal storage uses efficient (N, 3) arrays for better performance.

    Examples
    --------
    > # Old-style initialization (backward compatible)
    > dist = ParticleDistribution(
    ...     species=IonSpecies('H2_1+'),
    ...     x=x_array, y=y_array, z=z_array,
    ...     px=px_array, py=py_array, pz=pz_array
    ... )

    > # New-style initialization (recommended)
    > positions = np.column_stack([x, y, z])
    > momenta = np.column_stack([px, py, pz])
    > dist = ParticleDistribution(
    ...     species=IonSpecies('H2_1+'),
    ...     x_vec=positions, p_vec=momenta
    ... )
    """

    def __init__(self,
                 species: IonSpecies = IonSpecies('proton'),
                 x: Optional[np.ndarray] = None,
                 y: Optional[np.ndarray] = None,
                 z: Optional[np.ndarray] = None,
                 px: Optional[np.ndarray] = None,
                 py: Optional[np.ndarray] = None,
                 pz: Optional[np.ndarray] = None,
                 x_vec: Optional[np.ndarray] = None,
                 p_vec: Optional[np.ndarray] = None,
                 q: float = 0.0,
                 f: float = 0.0,
                 recalculate: bool = True):

        self._species = species

        # Check for conflicting input methods
        separate_given = (x is not None or y is not None or z is not None or
                          px is not None or py is not None or pz is not None)
        vector_given = (x_vec is not None or p_vec is not None)

        if separate_given and vector_given:
            raise ValueError(
                "Cannot specify both separate coordinates (x,y,z,px,py,pz) "
                "AND vector coordinates (x_vec, p_vec). Choose one method."
            )

        # Initialize from vector format
        if vector_given:
            if x_vec is None or p_vec is None:
                raise ValueError("Both x_vec and p_vec must be provided together.")

            if x_vec.shape != p_vec.shape:
                raise ValueError(
                    f"x_vec and p_vec must have same shape. "
                    f"Got x_vec: {x_vec.shape}, p_vec: {p_vec.shape}"
                )

            if x_vec.ndim != 2 or x_vec.shape[1] != 3:
                raise ValueError(
                    f"x_vec and p_vec must have shape (N, 3). Got: {x_vec.shape}"
                )

            self._x_vec = x_vec.copy()
            self._p_vec = p_vec.copy()

        # Initialize from separate coordinates (backward compatibility)
        else:
            # Handle defaults
            if x is None:
                x = np.zeros(1)
            if y is None:
                y = np.zeros(1)
            if z is None:
                z = np.zeros(1)
            if px is None:
                px = np.zeros(1)
            if py is None:
                py = np.zeros(1)
            if pz is None:
                pz = np.zeros(1)

            # Check all arrays have same length
            lengths = [len(arr) for arr in [x, y, z, px, py, pz]]
            if len(set(lengths)) != 1:
                raise ValueError(
                    f"All coordinate arrays must have same length. "
                    f"Got lengths: x={len(x)}, y={len(y)}, z={len(z)}, "
                    f"px={len(px)}, py={len(py)}, pz={len(pz)}"
                    f"Note that this can also happen if you forget one of the 6D coordinates."
                )

            # Store as (N, 3) arrays
            self._x_vec = np.column_stack([x, y, z])
            self._p_vec = np.column_stack([px, py, pz])

        # Bunch parameters
        self.q = q  # C
        self.f = f  # Hz

        self.numpart = len(self._x_vec)

        # Alive Mask
        self.alive_mask = np.ones(self.numpart, dtype=bool)

        # --- Collective data (computed on demand) --- #
        self._centroid = None  # [xm, ym, zm] in m
        self._mean_momentum = None  # [pxm, pym, pzm] as β·γ
        self._ekin_mean = None  # Mean energy (MeV)
        self._ekin_stdev = None  # RMS energy spread (MeV)

        # Standard deviations
        self._x_std_vec = None
        self._xp_std_vec = None

        self._xxp_std = None
        self._yyp_std = None
        self._xyp_std = None
        self._yxp_std = None

        if recalculate:
            self.recalculate_all()

    # ========================================================================
    # Properties for Backward Compatibility (x, y, z, px, py, pz)
    # ========================================================================

    @property
    def x(self) -> np.ndarray:
        """X positions [m]."""
        return self._x_vec[:, 0]

    @x.setter
    def x(self, value: np.ndarray):
        self._x_vec[:, 0] = value
        self._invalidate_cache()

    @property
    def y(self) -> np.ndarray:
        """Y positions [m]."""
        return self._x_vec[:, 1]

    @y.setter
    def y(self, value: np.ndarray):
        self._x_vec[:, 1] = value
        self._invalidate_cache()

    @property
    def z(self) -> np.ndarray:
        """Z positions [m]."""
        return self._x_vec[:, 2]

    @z.setter
    def z(self, value: np.ndarray):
        self._x_vec[:, 2] = value
        self._invalidate_cache()

    @property
    def px(self) -> np.ndarray:
        """X momentum component [β·γ]."""
        return self._p_vec[:, 0]

    @px.setter
    def px(self, value: np.ndarray):
        self._p_vec[:, 0] = value
        self._invalidate_cache()

    @property
    def py(self) -> np.ndarray:
        """Y momentum component [β·γ]."""
        return self._p_vec[:, 1]

    @py.setter
    def py(self, value: np.ndarray):
        self._p_vec[:, 1] = value
        self._invalidate_cache()

    @property
    def pz(self) -> np.ndarray:
        """Z momentum component [β·γ]."""
        return self._p_vec[:, 2]

    @pz.setter
    def pz(self, value: np.ndarray):
        self._p_vec[:, 2] = value
        self._invalidate_cache()

    # ========================================================================
    # New Efficient Properties (N, 3 arrays)
    # ========================================================================

    @property
    def x_vec(self) -> np.ndarray:
        """Position vectors as (N, 3) array [x, y, z] in meters."""
        return self._x_vec

    @x_vec.setter
    def x_vec(self, value: np.ndarray):
        if value.shape != self._x_vec.shape:
            raise ValueError(
                f"Positions must have shape {self._x_vec.shape}, got {value.shape}"
            )
        self._x_vec = value.copy()
        self._invalidate_cache()

    @property
    def p_vec(self) -> np.ndarray:
        """Momentum vectors as (N, 3) array [px, py, pz] as β·γ."""
        return self._p_vec

    @p_vec.setter
    def p_vec(self, value: np.ndarray):
        if value.shape != self._p_vec.shape:
            raise ValueError(
                f"Momenta must have shape {self._p_vec.shape}, got {value.shape}"
            )
        self._p_vec = value.copy()
        self._invalidate_cache()

    @property
    def x_vec_p_vec(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._x_vec, self._p_vec

    @x_vec_p_vec.setter
    def x_vec_p_vec(self, value: Tuple[np.ndarray, np.ndarray]):
        if value[0].shape != value[1].shape:
            raise ValueError(
                f"Momenta must have the same shape got {value[0].shape} and {value[1].shape}"
            )
        self._x_vec, self._p_vec = value
        self._invalidate_cache()

    @property
    def dist6d(self) -> np.ndarray:
        """6-D phase space as (N, 6) array [x, y, z, px, py, pz] in m and beta*gamma"""
        return np.column_stack((self._x_vec, self._p_vec))

    # ========================================================================
    # Velocity Properties
    # ========================================================================

    @property
    def v_vec(self) -> np.ndarray:
        """Velocity vectors as (N, 3) array [vx, vy, vz] in m/s."""
        if HAS_NUMBA:
            return _velocities_from_momenta_batch(self._p_vec, RELATIVISTIC, CLIGHT)
        else:
            if RELATIVISTIC:
                return CLIGHT * self._p_vec / np.sqrt(self._p_vec ** 2.0 + 1.0)
            else:
                return CLIGHT * self._p_vec

    @property
    def vx(self) -> np.ndarray:
        """X velocity component [m/s]."""
        return self.v_vec[:, 0]

    @property
    def vy(self) -> np.ndarray:
        """Y velocity component [m/s]."""
        return self.v_vec[:, 1]

    @property
    def vz(self) -> np.ndarray:
        """Z velocity component [m/s]."""
        return self.v_vec[:, 2]

    def set_p_from_v(self, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray):
        """
        Set velocity data and calculate momenta.

        Parameters
        ----------
        vx, vy, vz : np.ndarray
            Velocity components [m/s]
        """
        v_vec = np.column_stack([vx, vy, vz])

        if HAS_NUMBA:
            self._p_vec = _momenta_from_velocities_batch(v_vec, RELATIVISTIC, CLIGHT)
        else:
            if RELATIVISTIC:
                self._p_vec = v_vec / np.sqrt(CLIGHT ** 2 - v_vec ** 2)
            else:
                self._p_vec = v_vec / CLIGHT

        self._invalidate_cache()

    def set_p_from_v_vec(self, v_vec: np.ndarray):
        """
        Set velocity data and calculate momenta.

        Parameters
        ----------
        v_vec : np.ndarray
            Velocity components [m/s]
        """
        if HAS_NUMBA:
            self._p_vec = _momenta_from_velocities_batch(v_vec, RELATIVISTIC, CLIGHT)
        else:
            if RELATIVISTIC:
                self._p_vec = v_vec / np.sqrt(CLIGHT ** 2 - v_vec ** 2)
            else:
                self._p_vec = v_vec / CLIGHT

        self._invalidate_cache()

    # ========================================================================
    # Angle Properties (x', y')
    # ========================================================================
    @property
    def xp_vec(self) -> np.ndarray:
        vz_safe = np.where(np.abs(self.v_vec[:, 2]) < 1e-10,
                          EPSILON, self.v_vec[:, 2])
        vz_safe_view = np.broadcast_to(vz_safe[:, np.newaxis], (len(vz_safe), 3))
        return self.v_vec / vz_safe_view

    @property
    def xp(self) -> np.ndarray:
        """X angle x' = vx/vz [rad]."""
        return self.xp_vec[:, 0]

    @property
    def yp(self) -> np.ndarray:
        """Y angle y' = vy/vz [rad]."""
        return self.xp_vec[:, 1]

    # ========================================================================
    # Momentum (SI units)
    # ========================================================================

    @property
    def px_si(self) -> np.ndarray:
        """X momentum in SI units [kg·m/s]."""
        return self.px * self._species.mass_kg * CLIGHT

    @property
    def py_si(self) -> np.ndarray:
        """Y momentum in SI units [kg·m/s]."""
        return self.py * self._species.mass_kg * CLIGHT

    @property
    def pz_si(self) -> np.ndarray:
        """Z momentum in SI units [kg·m/s]."""
        return self.pz * self._species.mass_kg * CLIGHT

    @property
    def mean_p_si(self) -> float:
        """Mean momentum magnitude in SI units [kg·m/s]."""
        return np.mean(np.linalg.norm(self._p_vec, axis=1)) * self._species.mass_kg * CLIGHT

    # ========================================================================
    # Energy Properties
    # ========================================================================

    @property
    def ekin_mev(self) -> np.ndarray:
        """Kinetic energy for each particle [MeV]."""
        if HAS_NUMBA:
            return _calculate_energies_batch(self.px, self.py, self.pz,
                                             self._species.mass_mev,
                                             Z_ENERGY, RELATIVISTIC)
        else:
            if Z_ENERGY:
                pr = self.pz
            else:
                pr = np.linalg.norm(self._p_vec, axis=1)

            m_mev = self._species.mass_mev

            if RELATIVISTIC:
                return np.sqrt((pr * m_mev) ** 2 + m_mev ** 2) - m_mev
            else:
                return m_mev * pr ** 2 / 2.0

    @property
    def mean_energy_mev(self) -> float:
        """Mean kinetic energy [MeV]."""
        if self._ekin_mean is None:
            self.calculate_mean_energy_mev()
        return self._ekin_mean

    @property
    def mean_energy_mev_per_amu(self) -> float:
        """Mean kinetic energy per nucleon [MeV/u]."""
        return self.mean_energy_mev / self.species.a

    @property
    def rms_energy_spread_mev(self) -> float:
        """RMS energy spread [MeV]."""
        if self._ekin_stdev is None:
            self.calculate_mean_energy_mev()
        return self._ekin_stdev

    def get_mean_energy_mev(self) -> float:
        """Get mean energy [MeV]."""
        if self._ekin_mean is None:
            self.calculate_mean_energy_mev()
        return self._ekin_mean

    # ========================================================================
    # Other Collective Properties
    # ========================================================================

    @property
    def current(self) -> float:
        """Beam current [A]."""
        return self.q * self.f

    @property
    def bunch_charge(self) -> float:
        """Bunch charge [C]."""
        return self.q

    @bunch_charge.setter
    def bunch_charge(self, q: float):
        self.q = q

    @property
    def bunch_freq(self) -> float:
        """Bunch frequency [Hz]."""
        return self.f

    @bunch_freq.setter
    def bunch_freq(self, f: float):
        self.f = f

    @property
    def species(self) -> IonSpecies:
        """Ion species."""
        return self._species

    @species.setter
    def species(self, species: IonSpecies):
        if isinstance(species, IonSpecies):
            self._species = species
            self._invalidate_cache()
        else:
            raise TypeError(f"Expected IonSpecies, got {type(species)}")

    @property
    def centroid(self) -> np.ndarray:
        """Beam centroid [xm, ym, zm] in meters."""
        if self._centroid is None:
            self.calculate_centroid()
        return self._centroid

    @property
    def mean_momentum_betagamma(self) -> float:
        """Mean momentum magnitude [β·γ]."""
        if Z_ENERGY:
            return np.mean(self.pz)
        else:
            return np.mean(np.linalg.norm(self._p_vec, axis=1))

    @property
    def v_mean_m_per_s(self) -> float:
        """Mean velocity magnitude [m/s]."""
        if Z_ENERGY:
            v = self.vz
        else:
            v = np.linalg.norm(self.v_vec, axis=1)
        return np.mean(v)

    @property
    def v_mean_cm_per_s(self) -> float:
        """Mean velocity magnitude [cm/s]."""
        return self.v_mean_m_per_s * 1.0e2

    @property
    def mean_b_rho(self) -> float:
        """Mean magnetic rigidity B·ρ [T·m]."""
        if Z_ENERGY:
            pr = self.pz
        else:
            pr = np.linalg.norm(self._p_vec, axis=1)

        return np.mean(pr) * self.species.mass_mev * 1.0e6 / (self.species.q * CLIGHT)

    @property
    def alive(self):
        """Mask for alive/terminated particles"""
        return self.alive_mask

    @alive.setter
    def alive(self, alive):
        """Mask for alive/terminated particles"""
        self.alive_mask = alive

    # ========================================================================
    # Standard Deviation Properties
    # ========================================================================

    @property
    def x_std(self) -> float:
        """RMS beam size in x [m]."""
        if self._x_std_vec is None:
            self.calculate_stdevs()
        return self._x_std_vec[0]

    @property
    def y_std(self) -> float:
        """RMS beam size in y [m]."""
        if self._x_std_vec is None:
            self.calculate_stdevs()
        return self._x_std_vec[1]

    @property
    def z_std(self) -> float:
        """RMS bunch length [m]."""
        if self._x_std_vec is None:
            self.calculate_stdevs()
        return self._x_std_vec[2]

    # ========================================================================
    # Calculation Methods
    # ========================================================================

    def _invalidate_cache(self):
        """Invalidate cached collective properties."""
        self._centroid = None
        self._mean_momentum = None
        self._ekin_mean = None
        self._ekin_stdev = None

        self._xp_std_vec = None
        self._x_std_vec = None

        self._xxp_std = None
        self._yyp_std = None
        self._xyp_std = None
        self._yxp_std = None

    def recalculate_all(self):
        """Recalculate all collective properties."""
        self.numpart = len(self._x_vec)
        self.calculate_centroid()
        self.calculate_mean_pr()
        self.calculate_mean_energy_mev()
        self.calculate_stdevs()

    def calculate_centroid(self):
        """Calculate mean position (centroid)."""
        self._centroid = np.mean(self._x_vec, axis=0)

    def calculate_mean_pr(self):
        """Calculate mean momentum."""
        self._mean_momentum = np.mean(self._p_vec, axis=0)

    def calculate_mean_energy_mev(self):
        """Calculate mean energy and RMS spread."""
        energies = self.ekin_mev
        self._ekin_mean = np.mean(energies)
        self._ekin_stdev = np.std(energies)

    def calculate_stdevs(self):
        """Calculate standard deviations and correlations."""

        self._x_std_vec = np.std(self._x_vec, axis=0)
        self._xp_std_vec = np.std(self.xp_vec, axis=0)

        centroid = self.centroid
        xp_mean_vec = np.mean(self.xp_vec, axis=0)

        self._xxp_std = np.mean((self.x - centroid[0]) * (self.xp - xp_mean_vec[0]))
        self._yyp_std = np.mean((self.y - centroid[1]) * (self.yp - xp_mean_vec[1]))
        self._xyp_std = np.mean((self.x - centroid[0]) * (self.yp - xp_mean_vec[1]))
        self._yxp_std = np.mean((self.y - centroid[1]) * (self.xp - xp_mean_vec[0]))

    # ========================================================================
    # Beam Analysis Methods
    # ========================================================================

    def get_beam_edges(self, mode: str = "1rms") -> np.ndarray:
        """
        Get beam extent in each dimension.

        Parameters
        ----------
        mode : str
            '1rms', '2rms', or 'full'

        Returns
        -------
        edges : np.ndarray(6,)
            [xmin, xmax, ymin, ymax, zmin, zmax]
        """
        if mode == "1rms":

            min_x_vec = self.centroid - self._x_std_vec
            max_x_vec = self.centroid + self._x_std_vec

            return np.column_stack((min_x_vec, max_x_vec)).flatten()

        elif mode == "2rms":
            min_x_vec = self.centroid - 2.0 * self._x_std_vec
            max_x_vec = self.centroid + 2.0 * self._x_std_vec

            return np.column_stack((min_x_vec, max_x_vec)).flatten()

        elif mode == "full":
            min_x_vec = np.min(self._x_vec, axis=0)
            max_x_vec = np.max(self._x_vec, axis=0)

            return np.column_stack((min_x_vec, max_x_vec)).flatten()

        else:
            raise ValueError(f"Unknown mode: {mode}. Use '1rms', '2rms', or 'full'")

    def get_emittances(self, normalized: bool = True) -> np.ndarray:
        """
        Calculate RMS emittances.

        Parameters
        ----------
        normalized : bool
            Return normalized emittance (ε_n = β·γ·ε) if True

        Returns
        -------
        emittances : np.ndarray(4,)
            [ε_x, ε_y, ε_xy, ε_yx] in m·rad (or mm·mrad if normalized)
        """
        if self._x_std_vec is None:
            self.calculate_stdevs()

        e_xxp_1rms = np.sqrt((self._x_std_vec[0] * self._xp_std_vec[0]) ** 2 - self._xxp_std ** 2)
        e_yyp_1rms = np.sqrt((self._x_std_vec[1] * self._xp_std_vec[1]) ** 2 - self._yyp_std ** 2)
        e_xyp_1rms = np.sqrt((self._x_std_vec[0] * self._xp_std_vec[1]) ** 2 - self._xyp_std ** 2)
        e_yxp_1rms = np.sqrt((self._x_std_vec[1] * self._xp_std_vec[0]) ** 2 - self._yxp_std ** 2)

        emittances = np.array([e_xxp_1rms, e_yyp_1rms, e_xyp_1rms, e_yxp_1rms])

        if normalized:
            return emittances * self.mean_momentum_betagamma
        else:
            return emittances

    def get_twiss_parameters(self) -> np.ndarray:
        """
        Calculate Twiss parameters from RMS beam properties.

        Returns
        -------
        twiss : np.ndarray(6,)
            [α_x, β_x, γ_x, α_y, β_y, γ_y]
            where γ = (1 + α²)/β (Twiss gamma, not Lorentz factor)

        Notes
        -----
        Prints percentage of beam within 4-RMS emittance ellipse.
        """
        e_xxp_1rms, e_yyp_1rms, _, _ = self.get_emittances(normalized=False)

        # Twiss parameters
        beta_x = (self._x_std_vec[0] ** 2) / e_xxp_1rms
        gamma_x = (self._xp_std_vec[0] ** 2) / e_xxp_1rms

        if self._xxp_std < 0:
            alpha_x = np.sqrt(beta_x * gamma_x - 1.0)
        else:
            alpha_x = -np.sqrt(beta_x * gamma_x - 1.0)

        beta_y = (self._x_std_vec[1] ** 2) / e_yyp_1rms
        gamma_y = (self._xp_std_vec[1] ** 2) / e_yyp_1rms

        if self._yyp_std < 0:
            alpha_y = np.sqrt(beta_y * gamma_y - 1.0)
        else:
            alpha_y = -np.sqrt(beta_y * gamma_y - 1.0)

        # Calculate percentage inside 4-RMS ellipse
        xp = self.xp
        yp = self.yp

        ellipse_x = (gamma_x * self.x ** 2 +
                     2.0 * alpha_x * self.x * xp +
                     beta_x * xp ** 2)
        ellipse_y = (gamma_y * self.y ** 2 +
                     2.0 * alpha_y * self.y * yp +
                     beta_y * yp ** 2)

        inside_4rms_x = np.sum(ellipse_x < 4.0 * e_xxp_1rms)
        inside_4rms_y = np.sum(ellipse_y < 4.0 * e_yyp_1rms)

        perc_x = 100.0 * inside_4rms_x / self.numpart
        perc_y = 100.0 * inside_4rms_y / self.numpart

        print(f"4-RMS emittances include {perc_x:.1f}% and {perc_y:.1f}% "
              f"of the beam in x and y direction")

        return np.array([alpha_x, beta_x, gamma_x, alpha_y, beta_y, gamma_y])

        # ========================================================================
        # Beam Manipulation Methods
        # ========================================================================

    def set_z_momentum_from_b_rho(self, b_rho: float):
        """
        Set all pz to correspond to given magnetic rigidity.

        Parameters
        ----------
        b_rho : float
            Magnetic rigidity B·ρ [T·m]

        Returns
        -------
        mean_energy : float
            Resulting mean energy [MeV]
        """
        pz_value = (b_rho * np.abs(self.species.q) * CLIGHT /
                    (1.0e6 * self.species.mass_mev))

        self._p_vec[:, 2] = pz_value
        self.recalculate_all()

        return self.mean_energy_mev

    def set_mean_energy_z_mev(self, energy: float):
        """
        Set mean energy by adjusting pz (currently only affects z direction).

        Parameters
        ----------
        energy : float
            Target mean energy [MeV]

        Notes
        -----
        This adjusts all pz values to achieve the target mean energy.
        Preserves energy spread structure.
        """
        # Calculate target pz for each particle
        ekin_mev = self.ekin_mev - self.mean_energy_mev + energy

        mysign = np.sign(ekin_mev)
        gamma = np.abs(ekin_mev) / self.species.mass_mev + 1.0
        beta = np.sqrt(1.0 - gamma ** (-2.0))

        self._p_vec[:, 2] = mysign * beta * gamma

        self.recalculate_all()

        # ========================================================================
        # Time-Dependent Formulation (NEW)
        # ========================================================================

    def get_as_timed_injection(self, reference_frequency_hz: float,
                               reference_radius_m: Optional[float] = None,
                               reference_velocity_m_per_s: Optional[float] = None) -> np.ndarray:
        """
        Convert z-coordinate to injection time/phase for cyclotron tracking.

        Transforms spatial bunch (z-distribution) into temporal distribution
        for time-dependent injection at cyclotron center.

        Parameters
        ----------
        reference_frequency_hz : float
            RF frequency or revolution frequency [Hz]
        reference_radius_m : float, optional
            Reference radius for velocity calculation [m]
            If None, uses mean velocity from momenta
        reference_velocity_m_per_s : float, optional
            Reference velocity [m/s]
            If None, calculates from mean momentum

        Returns
        -------
        injection_data : np.ndarray(N, 7)
            Columns: [x, y, phi_deg, px, py, pz, t_inject]
            where phi_deg is RF phase in degrees
            and t_inject is injection time in seconds

        Examples
        --------
        > dist = ParticleDistribution(...)
        > injection_data = dist.get_as_timed_injection(
        ...     reference_frequency_hz=42e6,  # 42 MHz
        ...     reference_radius_m=0.4
        ... )
        """
        # Determine reference velocity
        if reference_velocity_m_per_s is None:
            if reference_radius_m is not None:
                # For cyclotron: v = ω*r = 2πfr
                reference_velocity_m_per_s = (2.0 * np.pi * reference_frequency_hz *
                                              reference_radius_m)
            else:
                # Use mean velocity from distribution
                reference_velocity_m_per_s = self.v_mean_m_per_s

        # Convert z to time
        t_inject = self.z / reference_velocity_m_per_s

        # Convert time to RF phase
        phi_deg = (reference_frequency_hz * t_inject) * 360.0

        # Wrap phase to [-180, 180] degrees
        phi_deg = np.mod(phi_deg + 180.0, 360.0) - 180.0

        # Stack output: [x, y, phi, px, py, pz, t]
        injection_data = np.column_stack([
            self.x, self.y, phi_deg,
            self.px, self.py, self.pz,
            t_inject
        ])

        return injection_data

    def set_mean_momentum(self, px_mean: Optional[float] = None,
                          py_mean: Optional[float] = None,
                          pz_mean: Optional[float] = None):
        """
        Set mean momentum (subtracts current mean first).

        Shifts distribution so mean momentum equals specified values.
        Only specified axes are updated.

        Parameters
        ----------
        px_mean, py_mean, pz_mean : float, optional
            Target mean momentum [β·γ]. None = don't change this axis.

        Examples
        --------
        > dist.set_mean_momentum(pz_mean=0.1)  # Sets longitudinal momentum
        > dist.set_mean_momentum(px_mean=0.0, py_mean=0.0, pz_mean=0.15)
        """
        current_mean = np.mean(self._p_vec, axis=0)

        if px_mean is not None:
            self._p_vec[:, 0] += (px_mean - current_mean[0])
        if py_mean is not None:
            self._p_vec[:, 1] += (py_mean - current_mean[1])
        if pz_mean is not None:
            self._p_vec[:, 2] += (pz_mean - current_mean[2])

        self._invalidate_cache()

    def add_mean_momentum(self, dpx: float = 0.0, dpy: float = 0.0, dpz: float = 0.0):
        """
        Add momentum boost to entire distribution.

        Shifts all particles by specified momentum delta. Does not change spread.

        Parameters
        ----------
        dpx, dpy, dpz : float
            Momentum boost to add [β·γ]

        Examples
        --------
        > dist.add_mean_momentum(dpz=0.05)  # Boost by Δpz = 0.05
        > dist.add_mean_momentum(dpx=0.01, dpz=0.1)
        """
        self._p_vec[:, 0] += dpx
        self._p_vec[:, 1] += dpy
        self._p_vec[:, 2] += dpz

        self._invalidate_cache()

    def add_energy(self, energy_mev: float,
                   direction: Optional[np.ndarray] = None,
                   relativistic: bool = True):
        """
        Add kinetic energy boost to distribution.

        Each particle gains the specified kinetic energy. The momentum change
        depends on the particle's current momentum via the energy-momentum relation.

        Parameters
        ----------
        energy_mev : float
            Kinetic energy to add to each particle [MeV]
        direction : np.ndarray(3,), optional
            Direction of momentum change (will be normalized).
            Default: [0, 0, 1] (z-direction)
        relativistic : bool
            Use relativistic energy-momentum relation (default: True)

        Notes
        -----
        For each particle with current kinetic energy T_i and momentum p_i:
        - New kinetic energy: T_i' = T_i + ΔT
        - New momentum: p_i' = sqrt((T_i' + mc²)² - (mc²)²) / c
        - Momentum change: Δp_i = p_i' - p_i
        - Boost in direction: (Δp_i / |Δp_i|) · Δp_i · direction_normalized

        Examples
        --------
        >>> # Each particle gains 3 MeV in z-direction
        >>> dist.add_energy(3.0)

        >>> # Each particle gains 2 MeV at 45° in x-z plane
        >>> dist.add_energy(2.0, direction=[1, 0, 1])
        """
        if direction is None:
            direction = np.array([0.0, 0.0, 1.0])
        else:
            direction = np.array(direction, dtype=float)

        # Normalize direction
        dir_mag = np.linalg.norm(direction)
        if dir_mag < 1e-10:
            raise ValueError("Direction vector must have non-zero magnitude")
        direction_normalized = direction / dir_mag

        mass_mev = self._species.mass_mev

        # Calculate current kinetic energy for each particle
        current_ekin = self.ekin_mev

        # New kinetic energy for each particle
        new_ekin = current_ekin + energy_mev

        if relativistic:
            # Current momentum: E² = (pc)² + (mc²)²
            # p = sqrt(E² - (mc²)²) / c, where E = T + mc²
            E_old = current_ekin + mass_mev
            p_old_betagamma = np.sqrt(E_old ** 2 - mass_mev ** 2)

            # New momentum
            E_new = new_ekin + mass_mev
            p_new_betagamma = np.sqrt(E_new ** 2 - mass_mev ** 2)

            # Momentum change for each particle
            dp_betagamma = p_new_betagamma - p_old_betagamma
        else:
            # Non-relativistic: p = sqrt(2mT)
            p_old_betagamma = np.sqrt(2.0 * current_ekin / mass_mev)
            p_new_betagamma = np.sqrt(2.0 * new_ekin / mass_mev)
            dp_betagamma = p_new_betagamma - p_old_betagamma

        # Apply momentum change in specified direction
        # For each particle, scale by its momentum change magnitude
        dp_vec = dp_betagamma[:, np.newaxis] * direction_normalized

        # Add to particle momenta
        self._p_vec += dp_vec

        self._invalidate_cache()

    def set_centroid(self, x_mean: Optional[float] = None,
                     y_mean: Optional[float] = None,
                     z_mean: Optional[float] = None):
        """
        Set beam centroid position (subtracts current mean first).

        Shifts distribution so mean position equals specified values.

        Parameters
        ----------
        x_mean, y_mean, z_mean : float, optional
            Target mean position [m]. None = don't change this axis.

        Examples
        --------
        > dist.set_centroid(x_mean=0.4, y_mean=0.0, z_mean=0.0)  # Place at injection radius
        """
        current_centroid = self.centroid

        if x_mean is not None:
            self._x_vec[:, 0] += (x_mean - current_centroid[0])
        if y_mean is not None:
            self._x_vec[:, 1] += (y_mean - current_centroid[1])
        if z_mean is not None:
            self._x_vec[:, 2] += (z_mean - current_centroid[2])

        self._invalidate_cache()

        # ========================================================================
        # Distribution Generation Interface (NEW)
        # ========================================================================

    @classmethod
    def generate_distribution(
            cls,
            species: IonSpecies,
            type: List[str],
            s_direction: Literal['x', 'y', 'z'] = 'z',
            n_particles: int = 1000,

            # Method 1: Correlation matrix + RMS spreads (OPAL style)
            correlation_matrix: Optional[np.ndarray] = None,
            sigma_x: Optional[float] = None,
            sigma_px: Optional[float] = None,
            sigma_y: Optional[float] = None,
            sigma_py: Optional[float] = None,
            sigma_z: Optional[float] = None,
            sigma_pz: Optional[float] = None,

            # Method 2: Full covariance matrix
            sigma_matrix: Optional[np.ndarray] = None,

            # Method 3: Twiss parameters
            alpha_x: Optional[float] = None,
            beta_x: Optional[float] = None,
            emittance_x: Optional[float] = None,
            alpha_y: Optional[float] = None,
            beta_y: Optional[float] = None,
            emittance_y: Optional[float] = None,
            alpha_z: Optional[float] = None,
            beta_z: Optional[float] = None,
            emittance_z: Optional[float] = None,
            reference_momentum: Optional[float] = None,

            # Cutoffs (Gaussian only)
            cutoff_x: Optional[float] = None,
            cutoff_y: Optional[float] = None,
            cutoff_z: Optional[float] = None,
            cutoff_px: Optional[float] = None,
            cutoff_py: Optional[float] = None,
            cutoff_pz: Optional[float] = None,

            # Flattop specific
            flattop_length: Optional[float] = None,
            dist_type_end: Optional[str] = None,
            alpha_end: Optional[float] = None,
            beta_end: Optional[float] = None,
            emittance_end: Optional[float] = None,

            # Bunch parameters
            bunch_charge: float = 0.0,
            bunch_freq: float = 0.0,

            **kwargs
    ) -> 'ParticleDistribution':
        """
        Generate particle distribution from collective parameters.

        Main user-facing classmethod. Supports multiple input methods and distribution types.
        Generated with mean position = 0, mean momentum = 0 (add reference values later).

        Parameters
        ----------
        species : IonSpecies
            Particle species
        type : List[str]
            Distribution type for each axis [x, y, z].
            Options: 'gaussian', 'kv', 'waterbag', 'flattop'
            Length determines dimensionality: 1, 2, or 3
        s_direction : str
            Longitudinal direction: 'x', 'y', or 'z' (default: 'z')
        n_particles : int
            Number of macroparticles

        (See distribution_generators.py for other parameters)

        Returns
        -------
        ParticleDistribution
            Generated distribution with mean = 0

        Examples
        --------
        > from PyPATools.particles import IonSpecies, ParticleDistribution
        > ion = IonSpecies('H2_1+', energy_mev=3.0)
        > dist = ParticleDistribution.generate_distribution(
        ...     species=ion,
        ...     type=['gaussian', 'gaussian', 'gaussian'],
        ...     alpha_x=0.5, beta_x=2.0, emittance_x=1e-6,
        ...     alpha_y=-0.3, beta_y=1.5, emittance_y=1e-6,
        ...     alpha_z=0.0, beta_z=0.5, emittance_z=1e-3,
        ...     reference_momentum=0.1,
        ...     n_particles=10000,
        ...     bunch_charge=1e-12,
        ...     bunch_freq=42e6
        ... )
        > dist.set_mean_momentum(pz_mean=0.1)  # Set reference momentum
        """

        # Call generator
        positions, momenta = gen_dist(
            type=type,
            s_direction=s_direction,
            n_particles=n_particles,
            correlation_matrix=correlation_matrix,
            sigma_x=sigma_x, sigma_px=sigma_px,
            sigma_y=sigma_y, sigma_py=sigma_py,
            sigma_z=sigma_z, sigma_pz=sigma_pz,
            sigma_matrix=sigma_matrix,
            alpha_x=alpha_x, beta_x=beta_x, emittance_x=emittance_x,
            alpha_y=alpha_y, beta_y=beta_y, emittance_y=emittance_y,
            alpha_z=alpha_z, beta_z=beta_z, emittance_z=emittance_z,
            reference_momentum=reference_momentum,
            cutoff_x=cutoff_x, cutoff_y=cutoff_y, cutoff_z=cutoff_z,
            cutoff_px=cutoff_px, cutoff_py=cutoff_py, cutoff_pz=cutoff_pz,
            flattop_length=flattop_length,
            dist_type_end=dist_type_end,
            alpha_end=alpha_end, beta_end=beta_end, emittance_end=emittance_end,
            **kwargs
        )

        # Create ParticleDistribution
        return cls(
            species=species,
            x_vec=positions,
            p_vec=momenta,
            q=bunch_charge,
            f=bunch_freq,
            recalculate=True
        )

    @classmethod
    def from_file(cls, filename: str, format: str = 'auto', **kwargs) -> 'ParticleDistribution':
        """
        Load distribution from file.

        Parameters
        ----------
        filename : str
            Path to file
        format : str
            'auto' (detect from extension), 'opal', 'openpmd', 'tracewin',
            'npz', 'aima'
        **kwargs
            Format-specific arguments

        Returns
        -------
        ParticleDistribution
            Loaded distribution
        """

        # Auto-detect format
        if format == 'auto':
            ext = os.path.splitext(filename)[1].lower()
            format_map = {
                '.h5': 'opal',
                '.bp': 'openpmd',
                '.dst': 'tracewin',
                '.ini': 'tracewin',
                '.npz': 'npz',
                '.lst': 'aima'
            }
            format = format_map.get(ext, 'opal')

        # Load data
        if format == 'opal':
            positions, momenta, metadata = particle_io.load_opal_h5(filename, **kwargs)
        elif format == 'openpmd':
            positions, momenta, metadata = particle_io.load_openpmd(filename, **kwargs)
        elif format == 'tracewin':
            positions, momenta, metadata = particle_io.load_tracewin(filename, **kwargs)
        elif format == 'npz':
            positions, momenta, metadata = particle_io.load_npz(filename)
        elif format == 'aima':
            positions, momenta, metadata = particle_io.load_aima(filename, **kwargs)
        else:
            raise ValueError(f"Unknown format: {format}")

        # Extract species if in metadata
        species = metadata.get('species', IonSpecies('proton'))

        return cls(
            species=species,
            x_vec=positions, p_vec=momenta,
            q=metadata.get('bunch_charge', 0.0),
            f=metadata.get('bunch_freq', 0.0),
            recalculate=True
        )

    def save_to_file(self, filename: str, format: str = 'auto', **metadata):
        """
        Save distribution to file.

        Parameters
        ----------
        filename : str
            Output filename
        format : str
            'auto', 'opal', 'openpmd', 'tracewin', 'npz', 'aima'
        **metadata
            Additional metadata to save
        """
        # Auto-detect format
        if format == 'auto':
            ext = os.path.splitext(filename)[1].lower()
            format_map = {
                '.h5': 'opal',
                '.bp': 'openpmd',
                '.dst': 'tracewin',
                '.npz': 'npz',
                '.lst': 'aima'
            }
            format = format_map.get(ext, 'npz')

        species_data = {
            'mass_mev': self.species.mass_mev,
            'charge_state': self.species.q,
            'a': self.species.a,
            'name': self.species.name
        }

        metadata.update({
            'bunch_charge': self.q,
            'bunch_freq': self.f,
            'species': self.species
        })

        # Save
        if format == 'opal':
            particle_io.save_opal_h5(filename, self._x_vec, self._p_vec, species_data, **metadata)
        elif format == 'openpmd':
            particle_io.save_openpmd(filename, self._x_vec, self._p_vec, species_data, **metadata)
        elif format == 'tracewin':
            particle_io.save_tracewin(filename, self._x_vec, self._p_vec, species_data,
                                      reference_energy_mev=self.mean_energy_mev,
                                      frequency_mhz=self.f / 1e6)
        elif format == 'npz':
            particle_io.save_npz(filename, self._x_vec, self._p_vec, **metadata)
        elif format == 'aima':
            particle_io.save_aima(filename, self._x_vec, self._p_vec, species_data,
                                  reference_energy_mev=self.mean_energy_mev,
                                  frequency_hz=self.f)
        else:
            raise ValueError(f"Unknown format: {format}")

    def plot(self, plot_type: str = 'phase_space', plane: str = 'x', **kwargs):
        """
        Quick visualization of distribution.

        Parameters
        ----------
        plot_type : str
            'phase_space', 'emittance', 'histogram', 'all'
        plane : str
            'x', 'y', or 'both'
        **kwargs
            Plot-specific arguments

        Returns
        -------
        fig, axes : matplotlib Figure and Axes
        """

        if plot_type == 'phase_space':
            return pv.plot_phase_space(self._x_vec, self._p_vec, plane=plane,
                                       species_mass_mev=self.species.mass_mev, **kwargs)

        elif plot_type == 'emittance':
            twiss = self.get_twiss_parameters()
            emittances = self.get_emittances(normalized=False)

            if plane == 'x':
                alpha, beta = twiss[0], twiss[1]
                emittance = emittances[0]
            else:
                alpha, beta = twiss[3], twiss[4]
                emittance = emittances[1]

            return pv.plot_emittance(self._x_vec, self._p_vec, alpha, beta, emittance,
                                     plane=plane, **kwargs)

        elif plot_type == 'histogram':
            return pv.plot_particle_histogram(self._x_vec, self._p_vec, **kwargs)

        elif plot_type == 'all':
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            pv.plot_phase_space(self._x_vec, self._p_vec, plane='x',
                                species_mass_mev=self.species.mass_mev,
                                ax=axes[0, 0], **kwargs)
            pv.plot_phase_space(self._x_vec, self._p_vec, plane='y',
                                species_mass_mev=self.species.mass_mev,
                                ax=axes[0, 1], **kwargs)
            pv.plot_particle_histogram(self._x_vec, self._p_vec, variable='x',
                                       ax=axes[1, 0], **kwargs)
            pv.plot_particle_histogram(self._x_vec, self._p_vec, variable='y',
                                       ax=axes[1, 1], **kwargs)

            plt.tight_layout()
            return fig, axes

        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")


if __name__ == '__main__':
    # Basic tests
    print("Testing refactored ParticleDistribution...")

    # Test 1: Old-style initialization
    print("\n1. Old-style initialization (backward compatible):")
    x = np.random.randn(1000) * 0.001
    y = np.random.randn(1000) * 0.001
    z = np.random.randn(1000) * 0.01
    px = np.random.randn(1000) * 0.001
    py = np.random.randn(1000) * 0.001
    pz = np.ones(1000) * 0.1

    dist1 = ParticleDistribution(
        species=IonSpecies('proton'),
        x=x, y=y, z=z, px=px, py=py, pz=pz
    )
    print(f"   ✓ Created {dist1.numpart} particles")
    print(f"   ✓ Mean energy: {dist1.mean_energy_mev:.3f} MeV")

    # Test 2: New-style initialization
    print("\n2. New-style initialization (efficient):")
    positions = np.column_stack([x, y, z])
    momenta = np.column_stack([px, py, pz])

    dist2 = ParticleDistribution(
        species=IonSpecies('proton'),
        x_vec=positions, p_vec=momenta
    )
    print(f"   ✓ Created {dist2.numpart} particles")
    print(f"   ✓ Positions shape: {dist2.x_vec.shape}")
    print(f"   ✓ Momenta shape: {dist2.p_vec.shape}")

    # Test 3: Backward compatibility
    print("\n3. Backward compatibility:")
    print(f"   ✓ dist2.x.shape = {dist2.x.shape}")
    print(f"   ✓ dist2.px[0] = {dist2.px[0]:.6f}")

    # Test 4: Emittance calculation
    print("\n4. Emittance calculation:")
    emittances = dist2.get_emittances(normalized=True)
    print(f"   ✓ ε_x = {emittances[0] * 1e6:.3f} mm·mrad")
    print(f"   ✓ ε_y = {emittances[1] * 1e6:.3f} mm·mrad")

    # Test 5: Twiss parameters
    print("\n5. Twiss parameters:")
    twiss = dist2.get_twiss_parameters()
    print(f"   ✓ α_x = {twiss[0]:.3f}, β_x = {twiss[1]:.3f} m")

    print("\n✓ All tests passed!")
    print(f"Numba available: {HAS_NUMBA}")

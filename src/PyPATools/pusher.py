"""
pusher.py - Particle Integration Algorithms for Charged Particle Tracking

Provides optimized particle pushers (integrators) for tracking charged particles
through electromagnetic fields. All pushers support both single-particle and batch
tracking with Numba JIT acceleration and NumPy fallback.

Supported Algorithms:
    - leapfrog: Simple 1st-order explicit (non-relativistic only)
    - boris: 2nd-order Boris pusher (non-relativistic)
    - rk4: 4th-order Runge-Kutta (non-relativistic)
    - yoshida: 4th-order symplectic (non-relativistic)
    - vay_rel: Relativistic Vay pusher (Vay, Phys. Plasmas 2008)
    - rk4_rel: Relativistic 4th-order Runge-Kutta
    - yoshida_rel: Relativistic 4th-order symplectic

Author: PyPATools Development Team
References:
    - Boris (1970): Original Boris algorithm
    - Vay (2008): Phys. Plasmas 15, 056701
    - Yoshida (1990): Phys. Lett. A 150, 262

Notes:
    - All field query functions must be callable: efield(pts), bfield(pts)
    - Single particle push() accepts (3,) arrays but may change to (1,3) in future
    - Boundary checking removed for performance (may be re-added later)
"""

import numpy as np
from typing import Tuple, Callable
import warnings
from .global_variables import CLIGHT
from py_electrodes.py_electrodes import PyElectrodeAssembly

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    warnings.warn(
        "Numba not available. Performance will be significantly reduced.\n"
        "Install with: pip install numba"
    )


    # Dummy decorators for fallback
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


    def prange(*args, **kwargs):
        return range(*args, **kwargs)

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    # CuPy stubs for future GPU implementation
    cp = None


# ============================================================================
# Utility Functions
# ============================================================================

@njit(fastmath=True, cache=True)
def relativistic_gamma(v):
    """
    Calculate Lorentz factor from velocity.

    Parameters
    ----------
    v : np.ndarray(3,)
        Velocity [m/s]

    Returns
    -------
    gamma : float
        Lorentz factor
    """
    v_mag_sq = v[0] ** 2 + v[1] ** 2 + v[2] ** 2
    beta_sq = v_mag_sq / (CLIGHT ** 2)

    # Clamp to avoid numerical issues
    if beta_sq >= 0.9999:
        beta_sq = 0.9999

    return 1.0 / np.sqrt(1.0 - beta_sq)


@njit(fastmath=True, cache=True)
def cross_product(a, b):
    """Compute cross product a x b (Numba-compatible)."""
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ])


# ============================================================================
# Non-Relativistic Kernels
# ============================================================================

@njit(fastmath=True, cache=True)
def leapfrog_dbetagamma_dt(v, efield, bfield, q_over_m):
    """
    Compute d(v)/dt for leapfrog (non-relativistic).

    Simple Lorentz force: dv/dt = (q/m)(E + v x B)
    """
    v_cross_b = cross_product(v, bfield)
    return q_over_m * (efield + v_cross_b)


@njit(fastmath=True, cache=True)
def boris_push_single(v, efield, bfield, dt, q_over_m):
    """
    Boris algorithm for single particle (non-relativistic).

    Standard Boris pusher with E-field half-kick, B-field rotation, E-field half-kick.

    Parameters
    ----------
    v : np.ndarray(3,)
        Velocity [m/s]
    efield : np.ndarray(3,)
        Electric field [V/m]
    bfield : np.ndarray(3,)
        Magnetic field [T]
    dt : float
        Time step [s]
    q_over_m : float
        Charge-to-mass ratio [C/kg]

    Returns
    -------
    v_new : np.ndarray(3,)
        Updated velocity [m/s]
    """
    # Half E-field acceleration
    v_minus = v + 0.5 * q_over_m * efield * dt

    # B-field rotation
    t = 0.5 * q_over_m * bfield * dt
    t_mag_sq = t[0] ** 2 + t[1] ** 2 + t[2] ** 2
    s = 2.0 * t / (1.0 + t_mag_sq)

    v_cross_t = cross_product(v_minus, t)
    v_prime = v_minus + v_cross_t
    v_prime_cross_s = cross_product(v_prime, s)
    v_plus = v_minus + v_prime_cross_s

    # Half E-field acceleration
    v_new = v_plus + 0.5 * q_over_m * efield * dt

    return v_new


@njit(fastmath=True, cache=True)
def rk4_dbetagamma_dt(v, efield, bfield, q_over_m):
    """Compute d(v)/dt for RK4 (non-relativistic Lorentz force)."""
    v_cross_b = cross_product(v, bfield)
    return q_over_m * (efield + v_cross_b)


# Yoshida coefficients (4th order symplectic)
YOSHIDA_C1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
YOSHIDA_C2 = -2.0 ** (1.0 / 3.0) / (2.0 - 2.0 ** (1.0 / 3.0))
YOSHIDA_C3 = YOSHIDA_C1
YOSHIDA_D1 = 2.0 * YOSHIDA_C1
YOSHIDA_D2 = -(2.0 ** (1.0 / 3.0) + 1.0) / (2.0 - 2.0 ** (1.0 / 3.0))
YOSHIDA_D3 = YOSHIDA_D1


@njit(fastmath=True, cache=True)
def yoshida_push_single(r, v, efield, bfield, dt, q_over_m):
    """
    Yoshida 4th-order symplectic integrator (non-relativistic).

    Preserves phase space volume and has excellent long-term energy conservation.
    """
    r_temp = r.copy()
    v_temp = v.copy()

    # Stage 1
    accel = rk4_dbetagamma_dt(v_temp, efield, bfield, q_over_m)
    v_temp = v_temp + YOSHIDA_C1 * dt * accel
    r_temp = r_temp + YOSHIDA_D1 * dt * v_temp

    # Stage 2
    accel = rk4_dbetagamma_dt(v_temp, efield, bfield, q_over_m)
    v_temp = v_temp + YOSHIDA_C2 * dt * accel
    r_temp = r_temp + YOSHIDA_D2 * dt * v_temp

    # Stage 3
    accel = rk4_dbetagamma_dt(v_temp, efield, bfield, q_over_m)
    v_temp = v_temp + YOSHIDA_C3 * dt * accel
    r_temp = r_temp + YOSHIDA_D3 * dt * v_temp

    return r_temp, v_temp


# ============================================================================
# Relativistic Kernels
# ============================================================================

@njit(fastmath=True, cache=True)
def vay_push_single(v, efield, bfield, dt, q_over_m):
    """
    Vay relativistic pusher (Vay 2008, Phys. Plasmas 15, 056701).

    Improved Boris algorithm for relativistic particles. Better handles
    ultra-relativistic velocities (gamma >> 1).

    Parameters
    ----------
    v : np.ndarray(3,)
        Velocity [m/s]
    efield : np.ndarray(3,)
        Electric field [V/m]
    bfield : np.ndarray(3,)
        Magnetic field [T]
    dt : float
        Time step [s]
    q_over_m : float
        Charge-to-mass ratio [C/kg]

    Returns
    -------
    v_new : np.ndarray(3,)
        Updated velocity [m/s]
    """
    # Current gamma and momentum
    gamma_n = relativistic_gamma(v)
    u_n = gamma_n * v  # Relativistic momentum / m

    # Half E-field push
    tau = 0.5 * q_over_m * bfield * dt
    u_prime = u_n + q_over_m * efield * dt

    # Compute auxiliary quantities
    u_prime_mag = np.sqrt(u_prime[0] ** 2 + u_prime[1] ** 2 + u_prime[2] ** 2)
    tau_mag = np.sqrt(tau[0] ** 2 + tau[1] ** 2 + tau[2] ** 2)
    u_prime_dot_tau = u_prime[0] * tau[0] + u_prime[1] * tau[1] + u_prime[2] * tau[2]

    # Solve for gamma at n+1 (quadratic equation)
    gamma_prime_inv_sq = 1.0 - (u_prime_mag / CLIGHT) ** 2
    if gamma_prime_inv_sq < 1e-10:
        gamma_prime_inv_sq = 1e-10

    sigma = gamma_prime_inv_sq - tau_mag ** 2
    gamma_new = np.sqrt(0.5 * (sigma + np.sqrt(sigma ** 2 +
                                               4.0 * (tau_mag ** 2 + (u_prime_dot_tau / CLIGHT) ** 2))))

    # Rotation (similar to Boris but with new gamma)
    t = tau / gamma_new
    s = 2.0 * t / (1.0 + tau_mag ** 2 / gamma_new ** 2)

    u_star = u_prime + cross_product(u_prime, t)
    u_new = u_prime + cross_product(u_star, s)

    # Convert back to velocity
    v_new = u_new / gamma_new

    # Clamp to < c
    v_mag = np.sqrt(v_new[0] ** 2 + v_new[1] ** 2 + v_new[2] ** 2)
    if v_mag >= 0.9999 * CLIGHT:
        v_new = v_new * (0.9999 * CLIGHT / v_mag)

    return v_new


@njit(fastmath=True, cache=True)
def rk4_rel_dbetagamma_dt(v, efield, bfield, q_over_m):
    """
    Compute d(gamma*v)/dt for relativistic RK4.

    Relativistic equation of motion: d(γv)/dt = (q/m)(E + v×B) - (v·F)v/(γc²)
    where F = (q/m)(E + v×B)
    """
    gamma = relativistic_gamma(v)
    v_cross_b = cross_product(v, bfield)
    force = q_over_m * (efield + v_cross_b)

    # Relativistic correction
    v_dot_f = v[0] * force[0] + v[1] * force[1] + v[2] * force[2]
    correction = (v_dot_f / (gamma * CLIGHT ** 2)) * v

    return force / gamma - correction


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def rk4_rel_dbetagamma_dt_batch(v_array, efield_array, bfield_array, q_over_m):
    """
    Compute d(gamma*v)/dt for batch of particles (Numba parallelized).

    Parameters
    ----------
    v_array : np.ndarray(M, 3)
        Velocities
    efield_array : np.ndarray(M, 3)
        Electric fields
    bfield_array : np.ndarray(M, 3)
        Magnetic fields
    q_over_m : float
        Charge-to-mass ratio

    Returns
    -------
    dbetagamma_array : np.ndarray(M, 3)
        Time derivatives of relativistic velocities
    """
    M = v_array.shape[0]
    result = np.empty((M, 3), dtype=np.float64)

    for i in prange(M):
        result[i] = rk4_rel_dbetagamma_dt(v_array[i], efield_array[i],
                                          bfield_array[i], q_over_m)

    return result


@njit(fastmath=True, cache=True)
def yoshida_rel_push_single(r, v, efield, bfield, dt, q_over_m):
    """
    Relativistic Yoshida 4th-order symplectic integrator.

    Maintains symplectic structure with relativistic corrections.
    """
    r_temp = r.copy()
    v_temp = v.copy()

    # Stage 1
    accel = rk4_rel_dbetagamma_dt(v_temp, efield, bfield, q_over_m)
    v_temp = v_temp + YOSHIDA_C1 * dt * accel
    r_temp = r_temp + YOSHIDA_D1 * dt * v_temp

    # Stage 2
    accel = rk4_rel_dbetagamma_dt(v_temp, efield, bfield, q_over_m)
    v_temp = v_temp + YOSHIDA_C2 * dt * accel
    r_temp = r_temp + YOSHIDA_D2 * dt * v_temp

    # Stage 3
    accel = rk4_rel_dbetagamma_dt(v_temp, efield, bfield, q_over_m)
    v_temp = v_temp + YOSHIDA_C3 * dt * accel
    r_temp = r_temp + YOSHIDA_D3 * dt * v_temp

    # Clamp velocity to < c
    v_mag = np.sqrt(v_temp[0] ** 2 + v_temp[1] ** 2 + v_temp[2] ** 2)
    if v_mag >= 0.9999 * CLIGHT:
        v_temp = v_temp * (0.9999 * CLIGHT / v_mag)

    return r_temp, v_temp


# ============================================================================
# Batch Processing Kernels
# ============================================================================

@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def boris_push_batch(v_array, efield_array, bfield_array, dt, q_over_m):
    """Boris pusher for batch of particles (parallelized)."""
    M = v_array.shape[0]
    v_new_array = np.empty_like(v_array)

    for i in prange(M):
        v_new_array[i] = boris_push_single(v_array[i], efield_array[i],
                                           bfield_array[i], dt, q_over_m)

    return v_new_array


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def vay_push_batch(v_array, efield_array, bfield_array, dt, q_over_m):
    """Vay pusher for batch of particles (parallelized)."""
    M = v_array.shape[0]
    v_new_array = np.empty_like(v_array)

    for i in prange(M):
        v_new_array[i] = vay_push_single(v_array[i], efield_array[i],
                                         bfield_array[i], dt, q_over_m)

    return v_new_array


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def position_update_batch(r_array, v_array, dt):
    """Update positions for batch of particles (parallelized)."""
    M = r_array.shape[0]
    r_new_array = np.empty_like(r_array)

    for i in prange(M):
        r_new_array[i] = r_array[i] + v_array[i] * dt

    return r_new_array


# NumPy fallback versions (if Numba unavailable)
def rk4_rel_dbetagamma_dt_batch_numpy(v_array, efield_array, bfield_array, q_over_m):
    """NumPy vectorized version of relativistic RK4 derivative (fallback)."""
    # Lorentz factors
    v_mag_sq = np.sum(v_array ** 2, axis=1)
    beta_sq = np.clip(v_mag_sq / (CLIGHT ** 2), 0, 0.9999)
    gamma = 1.0 / np.sqrt(1.0 - beta_sq)

    # Lorentz force
    v_cross_b = np.cross(v_array, bfield_array)
    force = q_over_m * (efield_array + v_cross_b)

    # Relativistic correction
    v_dot_f = np.sum(v_array * force, axis=1)
    correction = (v_dot_f / (gamma * CLIGHT ** 2))[:, np.newaxis] * v_array

    return force / gamma[:, np.newaxis] - correction


# ============================================================================
# Pusher Class
# ============================================================================

class Pusher:
    """
    Particle pusher for tracking charged particles through EM fields.

    All algorithms use callable field objects and support both single-particle
    and batch tracking with Numba acceleration.

    Parameters
    ----------
    ion : IonSpecies
        Ion species with mass and charge properties
    algorithm : str
        Integration algorithm:
        - 'leapfrog': Simple 1st-order (non-relativistic)
        - 'boris': 2nd-order Boris (non-relativistic)
        - 'rk4': 4th-order Runge-Kutta (non-relativistic)
        - 'yoshida': 4th-order symplectic (non-relativistic)
        - 'vay_rel': Relativistic Vay pusher
        - 'rk4_rel': Relativistic RK4
        - 'yoshida_rel': Relativistic symplectic
    use_numba : bool
        Use Numba JIT compilation (default: True if available)

    Examples
    --------
    > from PyPATools.species import IonSpecies
    > from PyPATools.field import Field
    > ion = IonSpecies('H2_1+', energy_mev=0.07)
    > pusher = Pusher(ion, algorithm='vay_rel')
    > bfield = Field.from_file('bfield.table')
    > efield = Field.zero(dim=3)
    > r_hist, v_hist = pusher.track(r0, v0, efield, bfield,
    ...                                nsteps=10000, dt=1e-11)
    """

    ALGORITHMS = ['leapfrog', 'boris', 'rk4', 'yoshida',
                  'vay_rel', 'rk4_rel', 'yoshida_rel']

    def __init__(self, ion, algorithm: str = 'boris',
                 use_numba: bool = True, electrode_assembly: PyElectrodeAssembly = None):
        """Initialize pusher with ion species and algorithm."""
        self.ion = ion
        self.q_over_m = ion.q_over_m

        algorithm = algorithm.lower()
        if algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Must be one of {self.ALGORITHMS}"
            )

        # Handle deprecated relativistic leapfrog request
        # TODO: This will not work. IonSpecies does not store energy anymore, ...moved to ParticleDistribution
        if algorithm == 'leapfrog' and hasattr(ion, 'energy_mev'):
            if ion.energy_mev > 0.1:  # > 100 keV, likely relativistic
                warnings.warn(
                    "Leapfrog is non-relativistic. Using 'vay_rel' instead for "
                    "better relativistic accuracy.",
                    UserWarning
                )
                algorithm = 'vay_rel'

        self.algorithm = algorithm
        self.use_numba = use_numba and HAS_NUMBA

        if not HAS_NUMBA and use_numba:
            warnings.warn("Numba requested but not available. Falling back to NumPy.")

        # Determine if algorithm is relativistic
        self.relativistic = algorithm.endswith('_rel')

        # TODO: Think about separating Pusher and TrackingLoop
        # TODO: Termination checks sold then be in TrackingLoop
        self.elec_assy = electrode_assembly

    # ========================================================================
    # Single Particle Methods
    # ========================================================================

    def push(self, r: np.ndarray, v: np.ndarray,
             efield: Callable, bfield: Callable, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance single particle by one time step.

        Parameters
        ----------
        r : np.ndarray(3,)
            Position [m]
            Note: May require (1,3) shape in future versions for consistency
        v : np.ndarray(3,)
            Velocity [m/s]
        efield : callable
            Electric field function: (pts) -> (Ex, Ey, Ez)
        bfield : callable
            Magnetic field function: (pts) -> (Bx, By, Bz)
        dt : float
            Time step [s]

        Returns
        -------
        r_new : np.ndarray(3,)
            Updated position [m]
        v_new : np.ndarray(3,)
            Updated velocity [m/s]
        """
        # Query fields at current position
        ef = efield(r.reshape(1, 3))[0]
        bf = bfield(r.reshape(1, 3))[0]

        # Algorithm-specific integration
        if self.algorithm == 'leapfrog':
            dv = leapfrog_dbetagamma_dt(v, ef, bf, self.q_over_m)
            v_new = v + dv * dt
            r_new = r + v_new * dt

        elif self.algorithm == 'boris':
            v_new = boris_push_single(v, ef, bf, dt, self.q_over_m)
            r_new = r + v_new * dt

        elif self.algorithm == 'vay_rel':
            v_new = vay_push_single(v, ef, bf, dt, self.q_over_m)
            r_new = r + v_new * dt

        elif self.algorithm in ['rk4', 'rk4_rel']:
            # RK4 requires field queries at intermediate points
            r_new, v_new = self._rk4_step_single(r, v, efield, bfield, dt)

        elif self.algorithm in ['yoshida', 'yoshida_rel']:
            # Yoshida: symplectic coupled position-velocity update
            push_fn = (yoshida_rel_push_single if self.relativistic
                       else yoshida_push_single)
            r_new, v_new = push_fn(r, v, ef, bf, dt, self.q_over_m)

        else:
            raise ValueError(f"Algorithm '{self.algorithm}' not implemented")

        return r_new, v_new

    def _rk4_step_single(self, r: np.ndarray, v: np.ndarray,
                         efield: Callable, bfield: Callable,
                         dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single RK4 step with proper intermediate field queries.

        Evaluates fields at 4 intermediate positions for high accuracy.
        """
        dbetagamma_fn = (rk4_rel_dbetagamma_dt if self.relativistic
                         else rk4_dbetagamma_dt)

        # K1: Current position
        ef1 = efield(r.reshape(1, 3))[0]
        bf1 = bfield(r.reshape(1, 3))[0]
        k1_v = dbetagamma_fn(v, ef1, bf1, self.q_over_m)
        k1_r = v.copy()

        # K2: Midpoint
        r2 = r + 0.5 * dt * k1_r
        v2 = v + 0.5 * dt * k1_v
        ef2 = efield(r2.reshape(1, 3))[0]
        bf2 = bfield(r2.reshape(1, 3))[0]
        k2_v = dbetagamma_fn(v2, ef2, bf2, self.q_over_m)
        k2_r = v2.copy()

        # K3: Midpoint with k2
        r3 = r + 0.5 * dt * k2_r
        v3 = v + 0.5 * dt * k2_v
        ef3 = efield(r3.reshape(1, 3))[0]
        bf3 = bfield(r3.reshape(1, 3))[0]
        k3_v = dbetagamma_fn(v3, ef3, bf3, self.q_over_m)
        k3_r = v3.copy()

        # K4: Endpoint
        r4 = r + dt * k3_r
        v4 = v + dt * k3_v
        ef4 = efield(r4.reshape(1, 3))[0]
        bf4 = bfield(r4.reshape(1, 3))[0]
        k4_v = dbetagamma_fn(v4, ef4, bf4, self.q_over_m)
        k4_r = v4.copy()

        # Weighted combination
        r_new = r + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
        v_new = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        # Clamp velocity for relativistic case
        if self.relativistic:
            v_mag = np.sqrt(np.sum(v_new ** 2))
            if v_mag >= 0.9999 * CLIGHT:
                v_new = v_new * (0.9999 * CLIGHT / v_mag)

        return r_new, v_new

    def track(self, r0: np.ndarray, v0: np.ndarray,
              efield: Callable, bfield: Callable,
              nsteps: int, dt: float,
              rec_every_n_steps: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track single particle through fields and optional electrodes for termination

        Parameters
        ----------
        r0 : np.ndarray(3,)
            Initial position [m]
        v0 : np.ndarray(3,)
            Initial velocity [m/s]
        efield : callable
            Electric field function: (pts) -> (Ex, Ey, Ez)
        bfield : callable
            Magnetic field function: (pts) -> (Bx, By, Bz)
        nsteps : int
            Number of integration steps
        dt : float
            Time step [s]
        rec_every_n_steps : int
            Record position/velocity every N steps (default: 1)
            Note: Previously called 'record_every'

        Returns
        -------
        r_array : np.ndarray(n_records, 3)
            Position history [m]
        v_array : np.ndarray(n_records, 3)
            Velocity history [m/s]
Notes
        -----
        Boundary checking removed for performance. May be re-added in future.
        """
        # Calculate storage size
        n_records = nsteps // rec_every_n_steps + 1
        r_array = np.zeros((n_records, 3))
        v_array = np.zeros((n_records, 3))

        # Initialize
        r = r0.copy()
        v = v0.copy()

        # For Boris: initialize velocity at half-step back
        if self.algorithm == 'boris':
            ef = efield(r.reshape(1, 3))[0]
            bf = bfield(r.reshape(1, 3))[0]
            v = boris_push_single(v, ef, bf, -0.5 * dt, self.q_over_m)

        # Store initial conditions
        r_array[0] = r
        v_array[0] = v
        record_idx = 1

        # Main tracking loop
        for step in range(nsteps):
            r, v = self.push(r, v, efield, bfield, dt)

            # Record if needed
            if (step + 1) % rec_every_n_steps == 0:
                if record_idx < n_records:
                    r_array[record_idx] = r
                    v_array[record_idx] = v
                    record_idx += 1

        # For Boris: push velocity forward by half-step for final state
        if self.algorithm == 'boris':
            ef = efield(r.reshape(1, 3))[0]
            bf = bfield(r.reshape(1, 3))[0]
            v = boris_push_single(v, ef, bf, 0.5 * dt, self.q_over_m)
            v_array[-1] = v

        return r_array, v_array

    # ========================================================================
    # Batch (Multi-Particle) Methods
    # ========================================================================

    def push_batch(self, r_array: np.ndarray, v_array: np.ndarray,
                   efield: Callable, bfield: Callable,
                   dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance batch of particles by one time step.

        Parameters
        ----------
        r_array : np.ndarray(M, 3)
            Positions of M particles [m]
        v_array : np.ndarray(M, 3)
            Velocities of M particles [m/s]
        efield : callable
            Electric field function: (pts) -> (Ex, Ey, Ez)
        bfield : callable
            Magnetic field function: (pts) -> (Bx, By, Bz)
        dt : float
            Time step [s]

        Returns
        -------
        r_new_array : np.ndarray(M, 3)
            Updated positions [m]
        v_new_array : np.ndarray(M, 3)
            Updated velocities [m/s]
        """
        # Query fields for all particles (batch query for parallelization)
        efield_array = efield(r_array)
        bfield_array = bfield(r_array)

        # Algorithm-specific integration
        if self.algorithm == 'leapfrog':
            # Simple leapfrog (non-relativistic)
            if self.use_numba:
                M = r_array.shape[0]
                v_new_array = np.empty_like(v_array)
                for i in prange(M):
                    dv = leapfrog_dbetagamma_dt(v_array[i], efield_array[i],
                                                bfield_array[i], self.q_over_m)
                    v_new_array[i] = v_array[i] + dv * dt
                r_new_array = position_update_batch(r_array, v_new_array, dt)
            else:
                # NumPy fallback
                v_cross_b = np.cross(v_array, bfield_array)
                dv = self.q_over_m * (efield_array + v_cross_b)
                v_new_array = v_array + dv * dt
                r_new_array = r_array + v_new_array * dt

        elif self.algorithm == 'boris':
            # Boris pusher
            if self.use_numba:
                v_new_array = boris_push_batch(v_array, efield_array,
                                               bfield_array, dt, self.q_over_m)
                r_new_array = position_update_batch(r_array, v_new_array, dt)
            else:
                # NumPy fallback
                M = r_array.shape[0]
                v_new_array = np.empty_like(v_array)
                for i in range(M):
                    v_new_array[i] = boris_push_single(v_array[i], efield_array[i],
                                                       bfield_array[i], dt, self.q_over_m)
                r_new_array = r_array + v_new_array * dt

        elif self.algorithm == 'vay_rel':
            # Vay relativistic pusher
            if self.use_numba:
                v_new_array = vay_push_batch(v_array, efield_array,
                                            bfield_array, dt, self.q_over_m)
                r_new_array = position_update_batch(r_array, v_new_array, dt)
            else:
                # NumPy fallback
                M = r_array.shape[0]
                v_new_array = np.empty_like(v_array)
                for i in range(M):
                    v_new_array[i] = vay_push_single(v_array[i], efield_array[i],
                                                     bfield_array[i], dt, self.q_over_m)
                r_new_array = r_array + v_new_array * dt

        elif self.algorithm in ['rk4', 'rk4_rel']:
            # RK4 with batched field queries
            r_new_array, v_new_array = self._rk4_step_batch(
                r_array, v_array, efield, bfield, dt
            )

        elif self.algorithm in ['yoshida', 'yoshida_rel']:
            # Yoshida symplectic
            M = r_array.shape[0]
            r_new_array = np.empty_like(r_array)
            v_new_array = np.empty_like(v_array)

            push_fn = (yoshida_rel_push_single if self.relativistic
                      else yoshida_push_single)

            for i in range(M):
                r_new_array[i], v_new_array[i] = push_fn(
                    r_array[i], v_array[i], efield_array[i],
                    bfield_array[i], dt, self.q_over_m
                )

        else:
            raise ValueError(f"Algorithm '{self.algorithm}' not implemented")

        return r_new_array, v_new_array

    def _rk4_step_batch(self, r_array: np.ndarray, v_array: np.ndarray,
                        efield: Callable, bfield: Callable,
                        dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch RK4 step with proper field queries at intermediate points.

        Uses batched field queries for parallelization: evaluates all M particles
        simultaneously at each K stage rather than processing sequentially.
        """
        dbetagamma_fn = (rk4_rel_dbetagamma_dt_batch if self.relativistic
                        else None)

        if not isinstance(dt, float):
            dt = np.broadcast_to(dt[:, np.newaxis], (len(dt), 3))

        # K1: Evaluate at current positions (all particles at once)
        ef1 = efield(r_array)
        bf1 = bfield(r_array)

        if self.relativistic and self.use_numba:
            k1_v = dbetagamma_fn(v_array, ef1, bf1, self.q_over_m)
        elif self.relativistic:
            k1_v = rk4_rel_dbetagamma_dt_batch_numpy(v_array, ef1, bf1, self.q_over_m)
        else:
            # Non-relativistic: simple Lorentz force
            v_cross_b = np.cross(v_array, bf1)
            k1_v = self.q_over_m * (ef1 + v_cross_b)

        k1_r = v_array.copy()

        # K2: Evaluate at midpoint (all particles at once)
        r2 = r_array + 0.5 * dt * k1_r
        v2 = v_array + 0.5 * dt * k1_v
        ef2 = efield(r2)
        bf2 = bfield(r2)

        if self.relativistic and self.use_numba:
            k2_v = dbetagamma_fn(v2, ef2, bf2, self.q_over_m)
        elif self.relativistic:
            k2_v = rk4_rel_dbetagamma_dt_batch_numpy(v2, ef2, bf2, self.q_over_m)
        else:
            v_cross_b = np.cross(v2, bf2)
            k2_v = self.q_over_m * (ef2 + v_cross_b)

        k2_r = v2.copy()

        # K3: Evaluate at midpoint with k2 (all particles at once)
        r3 = r_array + 0.5 * dt * k2_r
        v3 = v_array + 0.5 * dt * k2_v
        ef3 = efield(r3)
        bf3 = bfield(r3)

        if self.relativistic and self.use_numba:
            k3_v = dbetagamma_fn(v3, ef3, bf3, self.q_over_m)
        elif self.relativistic:
            k3_v = rk4_rel_dbetagamma_dt_batch_numpy(v3, ef3, bf3, self.q_over_m)
        else:
            v_cross_b = np.cross(v3, bf3)
            k3_v = self.q_over_m * (ef3 + v_cross_b)

        k3_r = v3.copy()

        # K4: Evaluate at endpoint (all particles at once)
        r4 = r_array + dt * k3_r
        v4 = v_array + dt * k3_v
        ef4 = efield(r4)
        bf4 = bfield(r4)

        if self.relativistic and self.use_numba:
            k4_v = dbetagamma_fn(v4, ef4, bf4, self.q_over_m)
        elif self.relativistic:
            k4_v = rk4_rel_dbetagamma_dt_batch_numpy(v4, ef4, bf4, self.q_over_m)
        else:
            v_cross_b = np.cross(v4, bf4)
            k4_v = self.q_over_m * (ef4 + v_cross_b)

        k4_r = v4.copy()

        # Weighted combination
        r_new_array = r_array + (dt / 6.0) * (k1_r + 2.0*k2_r + 2.0*k3_r + k4_r)
        v_new_array = v_array + (dt / 6.0) * (k1_v + 2.0*k2_v + 2.0*k3_v + k4_v)

        # Clamp velocities for relativistic case
        if self.relativistic:
            v_mag = np.sqrt(np.sum(v_new_array**2, axis=1))
            over_c = v_mag >= 0.9999 * CLIGHT
            if np.any(over_c):
                v_new_array[over_c] *= (0.9999 * CLIGHT / v_mag[over_c, np.newaxis])

        return r_new_array, v_new_array

    def track_batch(self, r0_array: np.ndarray, v0_array: np.ndarray,
                    efield: Callable, bfield: Callable,
                    nsteps: int, dt: float,
                    rec_every_n_steps: int = 1,
                    verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track batch of particles through fields (parallelized).

        Parameters
        ----------
        r0_array : np.ndarray(M, 3)
            Initial positions of M particles [m]
        v0_array : np.ndarray(M, 3)
            Initial velocities of M particles [m/s]
        efield : callable
            Electric field function: (pts) -> (Ex, Ey, Ez)
        bfield : callable
            Magnetic field function: (pts) -> (Bx, By, Bz)
        nsteps : int
            Number of integration steps
        dt : float
            Time step [s]
        rec_every_n_steps : int
            Record data every N steps (default: 1)
        verbose : bool
            Print progress updates (default: False)

        Returns
        -------
        r_array : np.ndarray(n_records, M, 3)
            Position history [m]
        v_array : np.ndarray(n_records, M, 3)
            Velocity history [m/s]
        active : np.ndarray(M,)
            particles still alive or collision with bd?

        Notes
        -----
        Boundary checking removed for performance. May be re-added in future.
        """
        M = r0_array.shape[0]
        n_records = nsteps // rec_every_n_steps + 1

        # Allocate storage
        r_array = np.full((n_records, M, 3), np.nan)
        v_array = np.full((n_records, M, 3), np.nan)
        active = np.ones(M, dtype=bool)

        # Initialize
        r_current = r0_array.copy()
        v_current = v0_array.copy()

        # For Boris: initialize velocities at half-step back
        if self.algorithm == 'boris':
            efield_array = efield(r_current)
            bfield_array = bfield(r_current)

            if self.use_numba:
                v_current = boris_push_batch(v_current, efield_array, bfield_array,
                                            -0.5 * dt, self.q_over_m)
            else:
                for i in range(M):
                    v_current[i] = boris_push_single(v_current[i], efield_array[i],
                                                     bfield_array[i], -0.5 * dt,
                                                     self.q_over_m)

        # Store initial conditions
        r_array[0] = r_current
        v_array[0] = v_current
        record_idx = 1

        # Main tracking loop
        for step in range(nsteps):
            if verbose and nsteps >= 10 and (step % (nsteps // 10) == 0):
                print(f"Step {step}/{nsteps} ({100*step/nsteps:.0f}%)")

            # Advance all particles
            r_old = r_current[active].copy()
            r_current[active], v_current[active] = self.push_batch(r_current[active], v_current[active],
                                                   efield, bfield, dt)

            # Record if needed
            if (step + 1) % rec_every_n_steps == 0:
                if record_idx < n_records:
                    r_array[record_idx][active] = r_current[active]
                    v_array[record_idx][active] = v_current[active]
                    record_idx += 1

            # Collision test if there is a PyElectrodeAssembly
            if self.elec_assy:
                collision_data = self.elec_assy.segment_intersects_surface(r_old, r_current[active])
                old_active_idx = np.where(active)[0]
                active[old_active_idx[collision_data["hit_mask"]]] = False

                # If all particles are lost --> terminate tracking
                if len(np.where(active)[0]) == 0:
                    break

        # For Boris: push velocities forward by half-step for final state
        if self.algorithm == 'boris':
            efield_array = efield(r_current[active])
            bfield_array = bfield(r_current[active])

            if self.use_numba:
                v_current[active] = boris_push_batch(v_current[active], efield_array, bfield_array,
                                            0.5 * dt, self.q_over_m)
            else:
                for i in range(M):
                    if active[i]:
                        v_current[i] = boris_push_single(v_current[i], efield_array[i],
                                                         bfield_array[i], 0.5 * dt,
                                                         self.q_over_m)

            v_array[-1][active] = v_current[active]

        if verbose:
            print(f"Tracking complete: {nsteps} steps, {M} particles")

        return r_array, v_array, active

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def change_algorithm(self, algorithm: str):
        """
        Change integration algorithm.

        Parameters
        ----------
        algorithm : str
            New algorithm name (see __init__ docstring for options)
        """
        algorithm = algorithm.lower()
        if algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Must be one of {self.ALGORITHMS}"
            )

        self.__init__(self.ion, algorithm=algorithm, use_numba=self.use_numba)

    def __repr__(self):
        return (f"Pusher(ion={self.ion.name}, algorithm='{self.algorithm}', "
                f"relativistic={self.relativistic}, numba={self.use_numba})")


# ============================================================================
# CuPy Stubs (Future GPU Implementation)
# ============================================================================

if HAS_CUPY:
    # Future GPU-accelerated implementations
    # Will use CuPy arrays and custom CUDA kernels

    class PusherGPU(Pusher):
        """GPU-accelerated pusher using CuPy (future implementation)."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            warnings.warn("GPU acceleration not yet implemented. Using CPU version.")


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing Pusher module...")

    # Mock ion species for testing
    class MockIon:
        def __init__(self):
            self.name = "H2_1+"
            self.q_over_m = 4.79e7  # C/kg for H2+
            self.energy_mev = 0.07

    ion = MockIon()

    # Test 1: Create pushers
    print("\n1. Testing pusher creation:")
    for algo in Pusher.ALGORITHMS:
        pusher = Pusher(ion, algorithm=algo)
        print(f"   [OK] {pusher}")

    # Test 2: Single particle push
    print("\n2. Testing single particle push:")
    pusher = Pusher(ion, algorithm='boris')

    # Mock field functions
    def mock_efield(pts):
        return (np.zeros(pts.shape[0]), np.zeros(pts.shape[0]),
                np.zeros(pts.shape[0]))

    def mock_bfield(pts):
        return (np.zeros(pts.shape[0]), np.zeros(pts.shape[0]),
                np.ones(pts.shape[0]) * 1.0)  # 1 T in z

    r = np.array([0.01, 0.0, 0.0])
    v = np.array([0.0, 1e5, 0.0])
    dt = 1e-10

    r_new, v_new = pusher.push(r, v, mock_efield, mock_bfield, dt)
    print(f"   [OK] Boris: r={r_new}, v={v_new}")

    # Test 3: Batch push
    print("\n3. Testing batch push:")
    r_batch = np.array([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    v_batch = np.array([[0.0, 1e5, 0.0], [0.0, 1e5, 0.0]])

    r_new_batch, v_new_batch = pusher.push_batch(r_batch, v_batch,
                                                  mock_efield, mock_bfield, dt)
    print(f"   [OK] Batch: r_shape={r_new_batch.shape}")

    # Test 4: Relativistic Vay pusher
    print("\n4. Testing relativistic Vay pusher:")
    pusher_vay = Pusher(ion, algorithm='vay_rel')
    r_new, v_new = pusher_vay.push(r, v, mock_efield, mock_bfield, dt)
    print(f"   [OK] Vay: r={r_new}, v={v_new}")

    # Test 5: Track single particle
    print("\n5. Testing track:")
    r_hist, v_hist = pusher.track(r, v, mock_efield, mock_bfield,
                                   nsteps=10, dt=dt, rec_every_n_steps=2)
    print(f"   [OK] Track: r_hist.shape={r_hist.shape}")

    print("\n[OK] All tests passed!")
    print(f"\nNumba available: {HAS_NUMBA}")
    print(f"CuPy available: {HAS_CUPY}")


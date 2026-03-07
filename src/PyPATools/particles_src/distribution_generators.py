"""
distribution_generators.py - Generate particle distributions from collective parameters

Provides generators for particle distributions from Twiss parameters, sigma matrices,
or direct RMS specifications. Follows OPAL convention of momentum-space generation
with mean position and momentum = 0.

Supported distribution types:
- Gaussian: Normal distribution with optional hyperelliptical cutoffs
- KV (Kapchinskij-Vladimirskij): Uniform in 4D phase space, sharp boundary
- Waterbag: Uniform inside emittance boundary
- Flattop: Uniform core with optional Gaussian/KV/Waterbag caps

Author: PyPATools Development Team
"""

import numpy as np
from typing import Tuple, Optional, Literal, List
import warnings

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

# ============================================================================
# Constants
# ============================================================================

CLIGHT = 299792458.0  # m/s


# ============================================================================
# Coordinate Transformation Helpers
# ============================================================================

def twiss_to_sigma_momentum_space(alpha: float, beta: float, emittance: float,
                                  p_ref: float) -> Tuple[float, float, float]:
    """
    Convert Twiss parameters to momentum-space sigma matrix elements.

    Twiss parameters describe (x, x') phase space. This converts to (x, px)
    using the relationship px = p_ref * x' (small angle approximation).

    Parameters
    ----------
    alpha : float
        Twiss alpha (correlation parameter)
    beta : float
        Twiss beta [m]
    emittance : float
        RMS geometric emittance [m·rad]
    p_ref : float
        Reference momentum [β·γ] for converting angles to momenta

    Returns
    -------
    sigma_xx : float
        Position variance [m²]
    sigma_xpx : float
        Position-momentum covariance [m·β·γ]
    sigma_pxpx : float
        Momentum variance [β·γ²]

    Notes
    -----
    Beam matrix in angle space: Σ_angle = [[β·ε, -α·ε], [-α·ε, γ·ε]]
    where γ = (1 + α²)/β (Twiss gamma, not Lorentz factor)

    Transformation: px = p_ref * x'
    """
    gamma_twiss = (1.0 + alpha ** 2) / beta

    # Position variance unchanged
    sigma_xx = beta * emittance

    # Momentum variance: σ_px² = p_ref² * σ_x'²
    sigma_pxpx = (gamma_twiss * emittance) * p_ref ** 2

    # Covariance: σ_xpx = -α·ε·p_ref
    sigma_xpx = -alpha * emittance * p_ref

    return sigma_xx, sigma_xpx, sigma_pxpx


def sigma_matrix_to_correlation_and_rms(sigma_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose covariance matrix into correlation matrix and RMS spreads.

    Σ = D · R · D where D = diag(σ_i), R = correlation matrix

    Parameters
    ----------
    sigma_matrix : np.ndarray(n, n)
        Covariance matrix

    Returns
    -------
    correlation_matrix : np.ndarray(n, n)
        Correlation matrix (diagonal = 1, off-diagonal = ρ_ij ∈ [-1, 1])
    rms_spreads : np.ndarray(n,)
        RMS spreads [σ_0, σ_1, ..., σ_n-1]
    """
    rms_spreads = np.sqrt(np.diag(sigma_matrix))

    # Avoid division by zero
    rms_spreads = np.where(rms_spreads > 0, rms_spreads, 1.0)

    # R = D^(-1) · Σ · D^(-1)
    D_inv = np.diag(1.0 / rms_spreads)
    correlation_matrix = D_inv @ sigma_matrix @ D_inv

    # Ensure diagonal is exactly 1.0
    np.fill_diagonal(correlation_matrix, 1.0)

    return correlation_matrix, rms_spreads


def _energy_to_betagamma(energy_mev: float, mass_mev: float,
                         relativistic: bool = True) -> float:
    """
    Convert kinetic energy to relativistic momentum β·γ.

    Parameters
    ----------
    energy_mev : float
        Kinetic energy [MeV]
    mass_mev : float
        Rest mass [MeV/c²]
    relativistic : bool
        Use relativistic formula (default: True)

    Returns
    -------
    betagamma : float
        Relativistic momentum β·γ
    """
    if relativistic:
        gamma = energy_mev / mass_mev + 1.0
        beta = np.sqrt(1.0 - gamma ** (-2.0))
        return beta * gamma
    else:
        # Non-relativistic: p = sqrt(2mE) / (mc)
        return np.sqrt(2.0 * energy_mev / mass_mev)


def _rotate_to_s_direction(positions: np.ndarray, momenta: np.ndarray,
                           s_direction: Literal['x', 'y', 'z']) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate coordinates so s_direction becomes z-axis.

    Generation is done with z as longitudinal direction. This rotates
    if user wants x or y as longitudinal.

    Parameters
    ----------
    positions : np.ndarray(n, 3)
        Position vectors [x, y, z]
    momenta : np.ndarray(n, 3)
        Momentum vectors [px, py, pz]
    s_direction : str
        Which axis should be longitudinal

    Returns
    -------
    positions_rotated, momenta_rotated : np.ndarray
        Rotated coordinates
    """
    if s_direction == 'z':
        return positions, momenta  # Already correct

    # Coordinate permutations
    if s_direction == 'x':
        # x → z, y → x, z → y
        positions = positions[:, [1, 2, 0]]
        momenta = momenta[:, [1, 2, 0]]
    elif s_direction == 'y':
        # y → z, z → x, x → y
        positions = positions[:, [2, 0, 1]]
        momenta = momenta[:, [2, 0, 1]]
    else:
        raise ValueError(f"s_direction must be 'x', 'y', or 'z', got '{s_direction}'")

    return positions, momenta


# ============================================================================
# Sampling Functions (Core Distribution Generators)
# ============================================================================

def _sample_gaussian(n_particles: int,
                     cholesky_matrix: np.ndarray,
                     cutoff_r: np.ndarray,
                     cutoff_p: np.ndarray,
                     dim: int) -> np.ndarray:
    """
    Sample from multivariate Gaussian with optional hyperelliptical cutoffs.

    Uses Cholesky decomposition for correlated sampling and rejection
    sampling for cutoffs.

    Parameters
    ----------
    n_particles : int
        Number of particles to generate
    cholesky_matrix : np.ndarray(2*dim, 2*dim)
        Lower triangular Cholesky decomposition of correlation matrix
    cutoff_r : np.ndarray(3,)
        Position cutoffs [cutoff_x, cutoff_y, cutoff_z] in σ units
        0 or None means no cutoff
    cutoff_p : np.ndarray(3,)
        Momentum cutoffs [cutoff_px, cutoff_py, cutoff_pz] in σ units
    dim : int
        Dimensionality: 1, 2, or 3

    Returns
    -------
    samples : np.ndarray(n_particles, 2*dim)
        Sampled coordinates in normalized space (before scaling by sigma)
        Format: [x, px, y, py, z, pz] (only first 2*dim elements used)

    Notes
    -----
    Cutoffs form hyperellipsoids in position and momentum space:
    - Dim 1: |z| ≤ cutoff_z AND |pz| ≤ cutoff_pz
    - Dim 2: (x/cutoff_x)² + (y/cutoff_y)² ≤ 1 AND (px/...)² + (py/...)² ≤ 1
    - Dim 3: (x/...)² + (y/...)² + (z/...)² ≤ 1 AND same for momenta
    """
    # Determine which dimensions are active
    if dim == 1:
        active_r_idx = [2]  # z only
        active_p_idx = [2]  # pz only
    elif dim == 2:
        active_r_idx = [0, 1]  # x, y
        active_p_idx = [0, 1]  # px, py
    else:  # dim == 3
        active_r_idx = [0, 1, 2]  # x, y, z
        active_p_idx = [0, 1, 2]  # px, py, pz

    # Extract active cutoffs
    cutoff_r_active = cutoff_r[active_r_idx]
    cutoff_p_active = cutoff_p[active_p_idx]

    # Check if any cutoffs requested
    has_r_cutoff = np.any(cutoff_r_active > 0)
    has_p_cutoff = np.any(cutoff_p_active > 0)

    if not has_r_cutoff and not has_p_cutoff:
        # No cutoffs - fast path (no rejection sampling)
        z = np.random.randn(n_particles, 2 * dim)
        return z @ cholesky_matrix.T

    # Rejection sampling with cutoffs
    samples = []
    max_attempts = n_particles * 1000  # Safety limit
    attempts = 0

    # Estimate acceptance rate for warning
    if has_r_cutoff:
        avg_cutoff_r = np.mean(cutoff_r_active[cutoff_r_active > 0])
        if avg_cutoff_r < 2.0:
            warnings.warn(
                f"Small position cutoff ({avg_cutoff_r:.1f}σ) will cause slow sampling. "
                f"Consider using cutoff ≥ 3σ for better performance."
            )

    while len(samples) < n_particles and attempts < max_attempts:
        # Generate candidate
        z = np.random.randn(2 * dim)
        candidate = cholesky_matrix @ z

        # Extract position and momentum (interleaved: [x,px,y,py,z,pz])
        pos = candidate[::2][:dim]  # [x, y, z][:dim]
        mom = candidate[1::2][:dim]  # [px, py, pz][:dim]

        accept = True

        # Position hyperellipse check
        if has_r_cutoff:
            # Only non-zero cutoffs contribute
            r_terms = []
            for i, idx in enumerate(active_r_idx[:dim]):
                if cutoff_r_active[i] > 0:
                    r_terms.append((pos[i] / cutoff_r_active[i]) ** 2)

            if len(r_terms) > 0:
                r_norm_sq = np.sum(r_terms)
                if r_norm_sq > 1.0:
                    accept = False

        # Momentum hyperellipse check
        if accept and has_p_cutoff:
            p_terms = []
            for i, idx in enumerate(active_p_idx[:dim]):
                if cutoff_p_active[i] > 0:
                    p_terms.append((mom[i] / cutoff_p_active[i]) ** 2)

            if len(p_terms) > 0:
                p_norm_sq = np.sum(p_terms)
                if p_norm_sq > 1.0:
                    accept = False

        if accept:
            samples.append(candidate)

        attempts += 1

    if len(samples) < n_particles:
        raise RuntimeError(
            f"Rejection sampling failed: generated only {len(samples)}/{n_particles} "
            f"particles after {max_attempts} attempts. Cutoffs may be too restrictive."
        )

    return np.array(samples)


def _sample_kv(n_particles: int, dim: int) -> np.ndarray:
    """
    Sample from Kapchinskij-Vladimirskij distribution.

    Uniform density inside 4-RMS emittance ellipse with sharp boundary.

    Parameters
    ----------
    n_particles : int
        Number of particles
    dim : int
        Dimensionality (1, 2, or 3)

    Returns
    -------
    samples : np.ndarray(n_particles, 2*dim)
        Sampled coordinates in normalized space

    Notes
    -----
    KV distribution has constant density inside phase space ellipse:
    (x/σ_x)² + (x'/σ_x')² = constant

    Boundary is at 4-RMS (4σ equivalent).
    """
    # Sample uniformly in (2*dim)-dimensional hypersphere, then project
    # to get uniform distribution in phase space

    # Radius: uniform in [0, 4] (4-RMS boundary)
    r = 4.0 * np.random.uniform(0, 1, n_particles) ** (1.0 / (2 * dim))

    # Angular coordinates: uniform on (2*dim-1)-sphere
    # Generate normal then normalize (standard method)
    angles = np.random.randn(n_particles, 2 * dim)
    angles = angles / np.linalg.norm(angles, axis=1, keepdims=True)

    # Scale by radius
    samples = r[:, np.newaxis] * angles

    return samples


def _sample_waterbag(n_particles: int, dim: int) -> np.ndarray:
    """
    Sample from waterbag distribution.

    Uniform density inside emittance boundary (4D/6D sphere in normalized space).

    Parameters
    ----------
    n_particles : int
        Number of particles
    dim : int
        Dimensionality (1, 2, or 3)

    Returns
    -------
    samples : np.ndarray(n_particles, 2*dim)
        Sampled coordinates in normalized space
    """
    # Similar to KV but in full phase space
    # Uniform in (2*dim)-D ball

    r = 4.0 * np.random.uniform(0, 1, n_particles) ** (1.0 / (2 * dim))
    angles = np.random.randn(n_particles, 2 * dim)
    angles = angles / np.linalg.norm(angles, axis=1, keepdims=True)

    samples = r[:, np.newaxis] * angles

    return samples


# ============================================================================
# Core Generator Functions
# ============================================================================

def generate_dist_from_sigma_matrix(
        n_particles: int,
        sigma_matrix: np.ndarray,
        dist_type: Literal['gaussian', 'kv', 'waterbag'],
        cutoff_r: Optional[np.ndarray] = None,
        cutoff_p: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate distribution from covariance matrix in momentum space.

    Parameters
    ----------
    n_particles : int
        Number of particles to generate
    sigma_matrix : np.ndarray
        Covariance matrix in (x, px, y, py, z, pz) coordinates
        Shape (2, 2) for 1D, (4, 4) for 2D, (6, 6) for 3D
    dist_type : str
        'gaussian', 'kv', or 'waterbag'
    cutoff_r : np.ndarray(3,), optional
        Position cutoffs [cutoff_x, cutoff_y, cutoff_z]
    cutoff_p : np.ndarray(3,), optional
        Momentum cutoffs [cutoff_px, cutoff_py, cutoff_pz]

    Returns
    -------
    samples : np.ndarray(n_particles, 2*dim)
        Particle coordinates in (x, px, y, py, z, pz) format
    """
    dim = sigma_matrix.shape[0] // 2

    if sigma_matrix.shape[0] != 2 * dim or sigma_matrix.shape[1] != 2 * dim:
        raise ValueError(
            f"Sigma matrix must be square with even dimensions. "
            f"Got shape {sigma_matrix.shape}"
        )

    # Decompose into correlation matrix and RMS spreads
    correlation_matrix, rms_spreads = sigma_matrix_to_correlation_and_rms(sigma_matrix)

    # Cholesky decomposition of correlation matrix
    try:
        cholesky_corr = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Sigma matrix is not positive definite. Check for physical validity."
        )

    # Handle cutoffs
    if cutoff_r is None:
        cutoff_r = np.zeros(3)
    if cutoff_p is None:
        cutoff_p = np.zeros(3)

    # Sample based on distribution type
    if dist_type == 'gaussian':
        samples_normalized = _sample_gaussian(
            n_particles, cholesky_corr, cutoff_r, cutoff_p, dim
        )
    elif dist_type == 'kv':
        if np.any(cutoff_r > 0) or np.any(cutoff_p > 0):
            warnings.warn(
                "Cutoffs specified for KV distribution but ignored "
                "(KV has natural boundary at 4σ)"
            )
        samples_normalized = _sample_kv(n_particles, dim)
    elif dist_type == 'waterbag':
        if np.any(cutoff_r > 0) or np.any(cutoff_p > 0):
            warnings.warn(
                "Cutoffs specified for waterbag distribution but ignored "
                "(waterbag has natural boundary)"
            )
        samples_normalized = _sample_waterbag(n_particles, dim)
    else:
        raise ValueError(f"Unknown dist_type: {dist_type}")

    # Scale by RMS spreads
    samples = samples_normalized * rms_spreads[:2 * dim]

    return samples


def generate_dist_from_twiss_parameters(
        n_particles: int,
        alpha_x: float, beta_x: float, emittance_x: float,
        alpha_y: Optional[float] = None,
        beta_y: Optional[float] = None,
        emittance_y: Optional[float] = None,
        alpha_z: Optional[float] = None,
        beta_z: Optional[float] = None,
        emittance_z: Optional[float] = None,
        reference_momentum: float = 1.0,
        dist_type: Literal['gaussian', 'kv', 'waterbag'] = 'gaussian',
        cutoff_r: Optional[np.ndarray] = None,
        cutoff_p: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate distribution from Twiss parameters.

    Converts Twiss parameters to momentum-space covariance matrix,
    then calls generate_dist_from_sigma_matrix().

    Parameters
    ----------
    n_particles : int
        Number of particles
    alpha_x, beta_x, emittance_x : float
        Horizontal Twiss parameters and emittance [m·rad]
    alpha_y, beta_y, emittance_y : float, optional
        Vertical Twiss parameters (for 2D/3D)
    alpha_z, beta_z, emittance_z : float, optional
        Longitudinal Twiss parameters (for 3D)
    reference_momentum : float
        Reference momentum [β·γ] for converting emittance to momentum spread
    dist_type : str
        'gaussian', 'kv', or 'waterbag'
    cutoff_r, cutoff_p : np.ndarray, optional
        Position and momentum cutoffs

    Returns
    -------
    samples : np.ndarray(n_particles, 2*dim)
        Particle coordinates
    """
    # Determine dimensionality
    if alpha_z is not None and beta_z is not None and emittance_z is not None:
        dim = 3
    elif alpha_y is not None and beta_y is not None and emittance_y is not None:
        dim = 2
    else:
        dim = 1

    # Build covariance matrix
    sigma_matrix = np.zeros((2 * dim, 2 * dim))

    # X plane (always present)
    s_xx, s_xpx, s_pxpx = twiss_to_sigma_momentum_space(
        alpha_x, beta_x, emittance_x, reference_momentum
    )
    sigma_matrix[0, 0] = s_xx
    sigma_matrix[0, 1] = s_xpx
    sigma_matrix[1, 0] = s_xpx
    sigma_matrix[1, 1] = s_pxpx

    # Y plane (if 2D or 3D)
    if dim >= 2:
        s_yy, s_ypy, s_pypy = twiss_to_sigma_momentum_space(
            alpha_y, beta_y, emittance_y, reference_momentum
        )
        sigma_matrix[2, 2] = s_yy
        sigma_matrix[2, 3] = s_ypy
        sigma_matrix[3, 2] = s_ypy
        sigma_matrix[3, 3] = s_pypy

    # Z plane (if 3D)
    if dim == 3:
        s_zz, s_zpz, s_pzpz = twiss_to_sigma_momentum_space(
            alpha_z, beta_z, emittance_z, reference_momentum
        )
        sigma_matrix[4, 4] = s_zz
        sigma_matrix[4, 5] = s_zpz
        sigma_matrix[5, 4] = s_zpz
        sigma_matrix[5, 5] = s_pzpz

    return generate_dist_from_sigma_matrix(
        n_particles, sigma_matrix, dist_type, cutoff_r, cutoff_p
    )


def generate_dist_longitudinal_flattop(
        n_particles: int,
        flattop_length: float,
        sigma_pz: float,
        dist_type_end: Literal['gaussian', 'kv', 'waterbag', 'flat'] = 'gaussian',
        alpha_end: float = 0.0,
        beta_end: float = 1.0,
        emittance_end: float = 0.0,
        reference_momentum: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate longitudinal flat-top distribution.

    Creates uniform core with optional Gaussian/KV/Waterbag caps at ends.

    Parameters
    ----------
    n_particles : int
        Number of particles
    flattop_length : float
        Length of uniform core [m]
    sigma_pz : float
        RMS momentum spread [β·γ]
    dist_type_end : str
        End cap distribution: 'gaussian', 'kv', 'waterbag', or 'flat'
        'flat' means no caps (perfect cylinder)
    alpha_end, beta_end, emittance_end : float
        Twiss parameters for end caps (only if dist_type_end != 'flat')
    reference_momentum : float
        Reference momentum [β·γ]

    Returns
    -------
    z : np.ndarray(n_particles,)
        Longitudinal positions [m]
    pz : np.ndarray(n_particles,)
        Longitudinal momenta [β·γ]
    """
    # Input validation
    if flattop_length <= 0:
        raise ValueError(f"flattop_length must be positive, got {flattop_length}")
    if sigma_pz <= 0:
        raise ValueError(f"sigma_pz must be positive, got {sigma_pz}")

    if dist_type_end == 'flat':
        # Perfect cylinder: uniform z, Gaussian pz
        z = np.random.uniform(-flattop_length / 2, flattop_length / 2, n_particles)
        pz = np.random.randn(n_particles) * sigma_pz
        return z, pz

    # With end caps: 3-part distribution
    # Convert Twiss to sigma for ends
    s_zz, s_zpz, s_pzpz = twiss_to_sigma_momentum_space(
        alpha_end, beta_end, emittance_end, reference_momentum
    )
    sigma_z_end = np.sqrt(s_zz)

    # Partition particles: core vs caps
    cap_extend = 3.0 * sigma_z_end  # Extend each cap 3σ beyond flattop edge
    frac_core = flattop_length / (flattop_length + 2.0 * cap_extend)
    n_core = int(n_particles * frac_core)
    n_caps_total = n_particles - n_core
    n_left_cap = n_caps_total // 2
    n_right_cap = n_caps_total - n_left_cap

    # Generate core (uniform in position, Gaussian in momentum)
    z_core = np.random.uniform(-flattop_length / 2.0, flattop_length / 2.0, n_core)
    pz_core = np.random.randn(n_core) * sigma_pz

    # Generate left cap (centered at -flattop_length/2 - sigma_z_end)
    sigma_matrix_end = np.array([
        [s_zz, s_zpz],
        [s_zpz, s_pzpz]
    ])

    samples_left = generate_dist_from_sigma_matrix(
        n_left_cap, sigma_matrix_end, dist_type_end
    )
    z_left = samples_left[:, 0] - flattop_length / 2.0  # Shift down
    pz_left = samples_left[:, 1]

    # Keep only particles actually in left cap region (z < -flattop_length/2)
    mask_left = z_left < -flattop_length / 2.0
    z_left = z_left[mask_left]
    pz_left = pz_left[mask_left]

    # Generate right cap (centered at +flattop_length/2 + sigma_z_end)
    samples_right = generate_dist_from_sigma_matrix(
        n_right_cap, sigma_matrix_end, dist_type_end
    )
    z_right = samples_right[:, 0] + flattop_length / 2.0  # Shift up
    pz_right = samples_right[:, 1]

    # Keep only particles actually in right cap region (z > +flattop_length/2)
    mask_right = z_right > flattop_length / 2.0
    z_right = z_right[mask_right]
    pz_right = pz_right[mask_right]

    # Combine all parts
    z = np.concatenate([z_left, z_core, z_right])
    pz = np.concatenate([pz_left, pz_core, pz_right])

    # Pad or trim to exact particle count
    n_generated = len(z)

    if n_generated < n_particles:
        # Lost particles to masking; pad with core particles
        n_missing = n_particles - n_generated
        z_pad = np.random.uniform(-flattop_length / 2.0, flattop_length / 2.0, n_missing)
        pz_pad = np.random.randn(n_missing) * sigma_pz
        z = np.concatenate([z, z_pad])
        pz = np.concatenate([pz, pz_pad])

    elif n_generated > n_particles:
        # Trim to exact size (keep core + first caps)
        z = z[:n_particles]
        pz = pz[:n_particles]

    return z, pz



# ============================================================================
# Main Distribution Generator (User-Facing API)
# ============================================================================

def generate_distribution(
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

        #
        # Flattop specific
        flattop_length: Optional[float] = None,
        dist_type_end: Optional[str] = None,
        alpha_end: Optional[float] = None,
        beta_end: Optional[float] = None,
        emittance_end: Optional[float] = None,

        **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate particle distribution from collective parameters.

    Main user-facing function. Supports multiple input methods and distribution types.
    Generated with mean position = 0, mean momentum = 0 (add reference values later).

    Parameters
    ----------
    type : List[str]
        Distribution type for each axis [x, y, z].
        Options: 'gaussian', 'kv', 'waterbag', 'flattop'
        Length determines dimensionality: 1, 2, or 3

    s_direction : str
        Longitudinal direction: 'x', 'y', or 'z'

    Returns
    -------
    positions : np.ndarray(n_particles, 3)
        Positions [x, y, z] in meters. Unused dimensions are zero.
    momenta : np.ndarray(n_particles, 3)
        Momenta [px, py, pz] as β·γ. Unused dimensions are zero.

    Examples
    --------
    # 6D Gaussian from Twiss
    > pos, mom = generate_distribution(
    ...     type=['gaussian', 'gaussian', 'gaussian'],
    ...     alpha_x=0.5, beta_x=2.0, emittance_x=1e-6,
    ...     alpha_y=-0.3, beta_y=1.5, emittance_y=1e-6,
    ...     alpha_z=0.0, beta_z=0.5, emittance_z=1e-3,
    ...     reference_momentum=0.1,
    ...     n_particles=10000
    ... )

    # 4D transverse + flattop longitudinal
    > pos, mom = generate_distribution(
    ...     type=['gaussian', 'gaussian', 'flattop'],
    ...     alpha_x=0.5, beta_x=2.0, emittance_x=1e-6,
    ...     alpha_y=-0.3, beta_y=1.5, emittance_y=1e-6,
    ...     flattop_length=0.05,
    ...     dist_type_end='gaussian',
    ...     sigma_pz=0.001,
    ...     reference_momentum=0.1,
    ...     n_particles=10000
    ... )

    # OPAL-style: correlation + spreads
    > pos, mom = generate_distribution(
    ...     type=['gaussian', 'gaussian', 'gaussian'],
    ...     correlation_matrix=corr_6x6,
    ...     sigma_x=0.001, sigma_px=0.01,
    ...     sigma_y=0.001, sigma_py=0.01,
    ...     sigma_z=0.01, sigma_pz=0.001,
    ...     n_particles=10000
    ... )
    """
    dim = len(type)

    if dim < 1 or dim > 3:
        raise ValueError(f"type list must have length 1, 2, or 3. Got {dim}")

    # Validate transverse types are consistent
    if dim >= 2:
        transverse_types = [type[0], type[1]] if dim == 2 else [type[0], type[1]]
        if len(set(transverse_types)) > 1:
            raise ValueError(
                f"Mixed transverse distribution types not supported. "
                f"Got {transverse_types}"
            )

    # Prepare cutoff arrays
    cutoff_r = np.array([
        cutoff_x if cutoff_x is not None else 0.0,
        cutoff_y if cutoff_y is not None else 0.0,
        cutoff_z if cutoff_z is not None else 0.0
    ])
    cutoff_p = np.array([
        cutoff_px if cutoff_px is not None else 0.0,
        cutoff_py if cutoff_py is not None else 0.0,
        cutoff_pz if cutoff_pz is not None else 0.0
    ])

    # Determine longitudinal index
    long_idx = {'x': 0, 'y': 1, 'z': 2}[s_direction]

    # ==== CASE 1: All same type (uniform distribution) ====
    if len(set(type)) == 1:
        dist_type = type[0]

        # Determine input method and build sigma matrix
        if sigma_matrix is not None:
            # Method 2: Direct sigma matrix
            pass  # Use as-is

        elif correlation_matrix is not None:
            # Method 1: Correlation + spreads
            spreads = []
            spread_names = ['sigma_x', 'sigma_px', 'sigma_y', 'sigma_py', 'sigma_z', 'sigma_pz']
            spread_vars = [sigma_x, sigma_px, sigma_y, sigma_py, sigma_z, sigma_pz]

            for i in range(2 * dim):
                if spread_vars[i] is None:
                    raise ValueError(
                        f"Missing {spread_names[i]} for correlation matrix method"
                    )
                spreads.append(spread_vars[i])

            spreads = np.array(spreads)

            # Check correlation matrix size
            if correlation_matrix.shape != (2 * dim, 2 * dim):
                raise ValueError(
                    f"Correlation matrix must be ({2 * dim}, {2 * dim}) for dim={dim}. "
                    f"Got {correlation_matrix.shape}"
                )

            # Reconstruct sigma matrix: Σ = D·R·D
            D = np.diag(spreads)
            sigma_matrix = D @ correlation_matrix @ D

        elif alpha_x is not None:
            # Method 3: Twiss parameters
            if reference_momentum is None:
                raise ValueError(
                    "reference_momentum required when using Twiss parameters"
                )

            # Use generate_dist_from_twiss_parameters
            samples = generate_dist_from_twiss_parameters(
                n_particles, alpha_x, beta_x, emittance_x,
                alpha_y, beta_y, emittance_y,
                alpha_z, beta_z, emittance_z,
                reference_momentum, dist_type,
                cutoff_r, cutoff_p
            )

            # Convert samples to positions and momenta
            pos_full = np.zeros((n_particles, 3))
            mom_full = np.zeros((n_particles, 3))

            # Fill in generated dimensions (in z-longitudinal frame)
            if dim >= 1:
                pos_full[:, 0] = samples[:, 0]  # x
                mom_full[:, 0] = samples[:, 1]  # px
            if dim >= 2:
                pos_full[:, 1] = samples[:, 2]  # y
                mom_full[:, 1] = samples[:, 3]  # py
            if dim == 3:
                pos_full[:, 2] = samples[:, 4]  # z
                mom_full[:, 2] = samples[:, 5]  # pz

            # Rotate to desired s_direction
            pos_full, mom_full = _rotate_to_s_direction(pos_full, mom_full, s_direction)

            return pos_full, mom_full

        else:
            raise ValueError(
                "Must provide one of: sigma_matrix, "
                "(correlation_matrix + sigma_x/px/...), or Twiss parameters"
            )

        # Generate from sigma matrix
        samples = generate_dist_from_sigma_matrix(
            n_particles, sigma_matrix, dist_type, cutoff_r, cutoff_p
        )

        # Convert to 3D arrays
        pos_full = np.zeros((n_particles, 3))
        mom_full = np.zeros((n_particles, 3))

        if dim >= 1:
            pos_full[:, 0] = samples[:, 0]
            mom_full[:, 0] = samples[:, 1]
        if dim >= 2:
            pos_full[:, 1] = samples[:, 2]
            mom_full[:, 1] = samples[:, 3]
        if dim == 3:
            pos_full[:, 2] = samples[:, 4]
            mom_full[:, 2] = samples[:, 5]

        # Rotate to desired s_direction
        pos_full, mom_full = _rotate_to_s_direction(pos_full, mom_full, s_direction)

        return pos_full, mom_full

    # ==== CASE 2: Different longitudinal (e.g., gaussian + gaussian + flattop) ====
    elif 'flattop' in type:
        # Generate transverse (2D) + longitudinal (1D) separately

        if dim != 3:
            raise ValueError(
                "Flattop distribution requires 3D (transverse + longitudinal)"
            )

        # Check that transverse types match
        transverse_types = [type[i] for i in range(3) if i != long_idx]
        if len(set(transverse_types)) > 1:
            raise ValueError(
                f"Transverse types must match when using flattop. Got {transverse_types}"
            )

        trans_type = transverse_types[0]

        # Generate transverse 2D
        if alpha_x is not None and alpha_y is not None:
            if reference_momentum is None:
                raise ValueError("reference_momentum required for Twiss parameters")

            samples_trans = generate_dist_from_twiss_parameters(
                n_particles,
                alpha_x, beta_x, emittance_x,
                alpha_y, beta_y, emittance_y,
                None, None, None,  # No z
                reference_momentum, trans_type,
                cutoff_r, cutoff_p
            )
        else:
            raise NotImplementedError(
                "Flattop currently only supports Twiss parameter input for transverse"
            )

        # Generate longitudinal flattop
        if flattop_length is None:
            raise ValueError("flattop_length required for flattop distribution")

        if sigma_pz is None:
            raise ValueError("sigma_pz required for flattop distribution")

        if dist_type_end is None:
            dist_type_end = 'gaussian'

        z_long, pz_long = generate_dist_longitudinal_flattop(
            n_particles, flattop_length, sigma_pz,
            dist_type_end,
            alpha_end if alpha_end is not None else 0.0,
            beta_end if beta_end is not None else 1.0,
            emittance_end if emittance_end is not None else 0.0,
            reference_momentum if reference_momentum is not None else 1.0
        )

        # Combine transverse and longitudinal (assuming z is longitudinal in generation)
        pos_full = np.zeros((n_particles, 3))
        mom_full = np.zeros((n_particles, 3))

        pos_full[:, 0] = samples_trans[:, 0]  # x
        mom_full[:, 0] = samples_trans[:, 1]  # px
        pos_full[:, 1] = samples_trans[:, 2]  # y
        mom_full[:, 1] = samples_trans[:, 3]  # py
        pos_full[:, 2] = z_long  # z
        mom_full[:, 2] = pz_long  # pz

        # Rotate to desired s_direction
        pos_full, mom_full = _rotate_to_s_direction(pos_full, mom_full, s_direction)

        return pos_full, mom_full

    else:
        raise ValueError(
            f"Unsupported combination of distribution types: {type}"
        )


if __name__ == "__main__":
    # Basic tests
    print("Testing distribution_generators.py...")

    # Test 1: 2D Gaussian from Twiss
    print("\n1. 2D Gaussian from Twiss parameters:")
    pos, mom = generate_distribution(
        type=['gaussian', 'gaussian'],
        alpha_x=0.5, beta_x=2.0, emittance_x=1e-6,
        alpha_y=-0.3, beta_y=1.5, emittance_y=1e-6,
        reference_momentum=0.1,
        n_particles=1000
    )
    print(f"   ✓ Generated {len(pos)} particles")
    print(f"   ✓ Position RMS: x={np.std(pos[:, 0]) * 1e3:.2f} mm, y={np.std(pos[:, 1]) * 1e3:.2f} mm")
    print(f"   ✓ Momentum RMS: px={np.std(mom[:, 0]):.4f}, py={np.std(mom[:, 1]):.4f}")

    # Test 2: 6D Gaussian with cutoffs
    print("\n2. 6D Gaussian with 3σ cutoffs:")
    pos, mom = generate_distribution(
        type=['gaussian', 'gaussian', 'gaussian'],
        alpha_x=0.5, beta_x=2.0, emittance_x=1e-6,
        alpha_y=-0.3, beta_y=1.5, emittance_y=1e-6,
        alpha_z=0.0, beta_z=0.5, emittance_z=1e-3,
        reference_momentum=0.1,
        cutoff_x=3.0, cutoff_y=3.0, cutoff_z=3.0,
        cutoff_px=3.0, cutoff_py=3.0, cutoff_pz=3.0,
        n_particles=1000
    )
    print(f"   ✓ Generated {len(pos)} particles with cutoffs")

    # Test 3: KV distribution
    print("\n3. 2D KV distribution:")
    pos, mom = generate_distribution(
        type=['kv', 'kv'],
        alpha_x=0.5, beta_x=2.0, emittance_x=1e-6,
        alpha_y=-0.3, beta_y=1.5, emittance_y=1e-6,
        reference_momentum=0.1,
        n_particles=1000
    )
    print(f"   ✓ Generated KV distribution")

    print("\n✓ All basic tests passed!")
"""
poisson_fft.py - Open-boundary FFT Poisson solver for bunch space charge.

Solves the free-space Poisson equation for the charge of a particle bunch on a
moving Cartesian window and returns the electric field at the particles:

    phi(r) = 1 / (4 pi eps0) * sum_j q_j <1/|r - r'|>_cell(j)      E = -grad phi

The sum is a discrete convolution of the cloud-in-cell (CIC) charge with the
Green's function of 1/r, evaluated with FFTs on a grid doubled in every
direction (Hockney & Eastwood): the zero padding makes the circular
convolution equal to the open, aperiodic one, so no boundary condition is
imposed at the edge of the window - the potential is that of the bunch in
free space. The Green's function is INTEGRATED over the source cell
(Qiang, Lidia, Ryne, Limborg-Deprey, PRSTAB 9, 044204 (2006)): with the
closed-form triple antiderivative F of 1/r,

    G(x, y, z) = 1/(hx hy hz) * sum over the 8 cell corners of +-F,

which is the exact potential of a uniformly charged cell at the observation
node. Unlike the point-charge Green's function it stays accurate when the
cells are anisotropic (hx != hy != hz) and has no singularity to patch at
r = 0. Its FFT is cached and only recomputed when the grid shape or spacing
changes.

Window
------
The grid follows the live particles: their bounding box plus ``pad_cells``
cells on every side, quantised to FFT-friendly lengths (scipy.fft's
next_fast_len) with hysteresis (an axis keeps its node count while the bunch
still fits and has not shrunk to half of it; growth comes in 20 % steps), so
that a slowly growing or breathing bunch does not force a Green's function
recompute at every solve. Nodes sit at ``origin + i * h``; particles
are at least ``pad_cells`` (>= 2) cells inside, so CIC deposition and the
central-difference gradient never touch the boundary layer. If the bunch
outgrows ``max_cells`` the cell size is coarsened (warning).

Approximations (stated deliberately)
------------------------------------
* FREE SPACE. No image charges in the electrodes (dees, housing, poles) and no
  neighbouring RF buckets: the bunch is alone in the universe. For a bunch a
  few mm across that is several mm from metal this is the leading-order
  space-charge field; the electrode images are a correction the Shortley-
  Weller solver in ``poisson_amg`` can supply when needed.
* NON-RELATIVISTIC (electrostatic, rest frame == lab frame). The solve is done
  in the lab frame on the lab-frame positions and only E is returned; the
  magnetic self-field (which cancels the electric one to order beta^2) is
  neglected. Adequate for beta < 0.1 (H2+ up to ~10 MeV: beta^2 < 1e-2).

  TODO(relativistic): boost to the bunch rest frame along the mean velocity
  v_mean = <v>: gamma = 1/sqrt(1 - v_mean^2/c^2); positions r' = r_perp +
  gamma * r_par (stretch along v_hat about the bunch centre); solve the SAME
  electrostatic problem for E' with the SAME charges; transform back
  E_par = E'_par, E_perp = gamma * E'_perp, B = (v_mean x E) / c^2. Callers
  already receive (E, B) from ``solve_eb`` and apply q (E + v x B), so the
  boost slots into ``solve_eb`` behind the ``relativistic`` flag without any
  caller change; ``solve`` keeps returning E only. Not implemented yet:
  ``relativistic=True`` raises NotImplementedError.
* CIC deposition / trilinear gather (same kernels as ``poisson_amg``); E by
  second-order central differences of phi on the nodes.

GPU
---
``use_gpu=True`` runs deposition, the FFTs, the gradient and the gather on
the GPU with CuPy (input and output stay NumPy): fused ElementwiseKernels for
deposit (atomic adds) and gather, cuFFT plans cached per window shape next to
the Green spectrum. CPU: numba kernels for deposit / gather / gradient,
scipy.fft with all workers for the transforms.

Usage
-----
    solver = FFTPoissonSolver(h=1.5e-3, pad_cells=2, use_gpu=True)
    E = solver.solve(positions_m, charges_c)          # (N, 3) V/m at the particles
    E, B = solver.solve_eb(positions_m, charges_c)    # B == 0 (non-relativistic)
    phi, (x, y, z) = solver.phi, solver.grid_axes     # potential [V] on the window
    solver.gather(points)                             # E of the LAST solve at points
    field = solver.to_field()                         # PyPATools Field for plots

Author: PyPATools Team (2026-09-11)
"""
import logging
import time
import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import numba as nb
from scipy import fft as sp_fft

from .global_variables import EPS0

try:
    import cupy as cp
    import cupyx.scipy.fft as cu_fft

    CUPY_AVAILABLE = True
except ImportError:                                                        # pragma: no cover
    cp = None
    cu_fft = None
    CUPY_AVAILABLE = False


# Fused GPU kernels: one launch each for the CIC deposit (atomic adds) and the
# trilinear gather, instead of ~100 small cupy operations per solve. Same index
# and clamping conventions as the numba kernels above. Built lazily on first use.
_GPU_KERNELS = {}


def _gpu_kernels():
    if not _GPU_KERNELS:
        _GPU_KERNELS['deposit'] = cp.ElementwiseKernel(
            'float64 px, float64 py, float64 pz, float64 q, int64 nx, int64 ny, int64 nz',
            'raw float64 rho',                     # accumulated in place (raw inputs are const)
            '''
            long long ix = (long long)floor(px), iy = (long long)floor(py), iz = (long long)floor(pz);
            double fx = px - ix, fy = py - iy, fz = pz - iz;
            ix = min(max(ix, 0LL), nx - 2); iy = min(max(iy, 0LL), ny - 2); iz = min(max(iz, 0LL), nz - 2);
            for (int dx = 0; dx < 2; ++dx) { double wx = dx ? fx : 1.0 - fx;
              for (int dy = 0; dy < 2; ++dy) { double wy = dy ? fy : 1.0 - fy;
                for (int dz = 0; dz < 2; ++dz) { double wz = dz ? fz : 1.0 - fz;
                  long long idx = (ix + dx) * (ny * nz) + (iy + dy) * nz + (iz + dz);
                  atomicAdd(&rho[idx], q * wx * wy * wz);
            } } }
            ''', 'pypatools_cic_deposit')
        _GPU_KERNELS['gather'] = cp.ElementwiseKernel(
            'float64 px, float64 py, float64 pz, raw float64 Ex, raw float64 Ey, raw float64 Ez, int64 nx, int64 ny, int64 nz',
            'float64 ex, float64 ey, float64 ez',
            '''
            long long ix = (long long)floor(px), iy = (long long)floor(py), iz = (long long)floor(pz);
            double fx = px - ix, fy = py - iy, fz = pz - iz;
            ix = min(max(ix, 0LL), nx - 2); iy = min(max(iy, 0LL), ny - 2); iz = min(max(iz, 0LL), nz - 2);
            ex = 0.0; ey = 0.0; ez = 0.0;
            for (int dx = 0; dx < 2; ++dx) { double wx = dx ? fx : 1.0 - fx;
              for (int dy = 0; dy < 2; ++dy) { double wy = dy ? fy : 1.0 - fy;
                for (int dz = 0; dz < 2; ++dz) { double wz = dz ? fz : 1.0 - fz;
                  long long idx = (ix + dx) * (ny * nz) + (iy + dy) * nz + (iz + dz);
                  double w = wx * wy * wz;
                  ex += w * Ex[idx]; ey += w * Ey[idx]; ez += w * Ez[idx];
            } } }
            ''', 'pypatools_trilinear_gather')
    return _GPU_KERNELS

# The CIC deposition and the E = -grad(phi) kernels are verbatim copies of the
# PyAMG solver's (poisson_amg.PyAMGPoissonSolver._cic_deposit_numba and
# poisson_amg.compute_field_from_potential_numba): same conventions, same
# charge bookkeeping; tests/test_poisson_fft.py checks the two stay identical.
# They are NOT imported from there because poisson_amg imports py_electrodes at
# module level, which initialises MPI on import - with mpi4py's auto-init
# disabled (MPI4PY_RC_INITIALIZE=false, needed on some machines) that aborts
# the whole process rather than raising ImportError, and this solver must be
# importable in a tracker that never needs electrodes.
@nb.jit(nopython=True, parallel=True, cache=True)
def compute_field_from_potential_numba(phi_3d, hx, hy, hz):
    """E = -grad(phi) by central differences on the interior nodes; the
    boundary layer stays zero (copy of poisson_amg's kernel)."""
    nx, ny, nz = phi_3d.shape
    Ex = np.zeros((nx, ny, nz), dtype=np.float64)
    Ey = np.zeros((nx, ny, nz), dtype=np.float64)
    Ez = np.zeros((nx, ny, nz), dtype=np.float64)
    for i in nb.prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                Ex[i, j, k] = -(phi_3d[i + 1, j, k] - phi_3d[i - 1, j, k]) / (2 * hx)
                Ey[i, j, k] = -(phi_3d[i, j + 1, k] - phi_3d[i, j - 1, k]) / (2 * hy)
                Ez[i, j, k] = -(phi_3d[i, j, k + 1] - phi_3d[i, j, k - 1]) / (2 * hz)
    return Ex, Ey, Ez


@nb.jit(nopython=True, cache=True)
def _cic_deposit_numba(px, py, pz, charges, rho, nx, ny, nz):
    """Serial CIC deposition onto the flat (nx*ny*nz,) array ``rho``; node
    (i, j, k) is at i*(ny*nz) + j*nz + k. Positions in node units. Serial on
    purpose: the scatter-add is a data race under prange (copy of
    poisson_amg's kernel)."""
    for pid in range(len(charges)):
        ix = int(np.floor(px[pid]))
        iy = int(np.floor(py[pid]))
        iz = int(np.floor(pz[pid]))
        fx = px[pid] - ix
        fy = py[pid] - iy
        fz = pz[pid] - iz
        ix = min(max(ix, 0), nx - 2)
        iy = min(max(iy, 0), ny - 2)
        iz = min(max(iz, 0), nz - 2)
        for dix in [0, 1]:
            for diy in [0, 1]:
                for diz in [0, 1]:
                    jx = (ix + dix) % nx
                    jy = (iy + diy) % ny
                    jz = (iz + diz) % nz
                    idx = jx * (ny * nz) + jy * nz + jz
                    w = (1.0 - fx if dix == 0 else fx) * \
                        (1.0 - fy if diy == 0 else fy) * \
                        (1.0 - fz if diz == 0 else fz)
                    rho[idx] += charges[pid] * w


@nb.jit(nopython=True, parallel=True, cache=True)
def _trilinear_gather_numba(px, py, pz, Ex, Ey, Ez, out):
    """Trilinear interpolation of the node fields to the particles.

    ``px, py, pz`` are positions in node units (node i at i.0). Read-only on
    the grids, one output row per particle: safe under prange. The clamp
    mirrors the deposition kernel so gather and deposit see the same cell.
    """
    nx, ny, nz = Ex.shape
    for p in nb.prange(len(px)):
        ix = int(np.floor(px[p]))
        iy = int(np.floor(py[p]))
        iz = int(np.floor(pz[p]))
        fx = px[p] - ix
        fy = py[p] - iy
        fz = pz[p] - iz
        ix = min(max(ix, 0), nx - 2)
        iy = min(max(iy, 0), ny - 2)
        iz = min(max(iz, 0), nz - 2)
        ex = 0.0
        ey = 0.0
        ez = 0.0
        for dix in range(2):
            wx = fx if dix == 1 else 1.0 - fx
            for diy in range(2):
                wy = fy if diy == 1 else 1.0 - fy
                for diz in range(2):
                    wz = fz if diz == 1 else 1.0 - fz
                    w = wx * wy * wz
                    ex += w * Ex[ix + dix, iy + diy, iz + diz]
                    ey += w * Ey[ix + dix, iy + diy, iz + diz]
                    ez += w * Ez[ix + dix, iy + diy, iz + diz]
        out[p, 0] = ex
        out[p, 1] = ey
        out[p, 2] = ez


# ============================================================================
# Integrated Green's function of 1/r
# ============================================================================
def _antiderivative_1_over_r(x, y, z):
    """F with d^3 F / dx dy dz = 1 / sqrt(x^2 + y^2 + z^2).

    F = -z^2/2 atan(xy/(zr)) - y^2/2 atan(xz/(yr)) - x^2/2 atan(yz/(xr))
        + yz ln(x + r) + xz ln(y + r) + xy ln(z + r)

    (Qiang et al. 2006, eq. 10). Evaluated only at cell CORNERS, which sit at
    half-integer multiples of the spacing, so no coordinate is ever zero and
    x + r > 0 always: no singularity to guard. Works for numpy and cupy.
    """
    xp = cp.get_array_module(x) if CUPY_AVAILABLE else np
    r = xp.sqrt(x * x + y * y + z * z)
    return (-0.5 * z * z * xp.arctan(x * y / (z * r))
            - 0.5 * y * y * xp.arctan(x * z / (y * r))
            - 0.5 * x * x * xp.arctan(y * z / (x * r))
            + y * z * xp.log(x + r) + x * z * xp.log(y + r) + x * y * xp.log(z + r))


def integrated_green_1_over_r(x, y, z, hx, hy, hz):
    """Cell-averaged inverse distance <1/|r - r'|> over the cell of size
    (hx, hy, hz) centred on the origin, for observation points (x, y, z) that
    are node offsets (integer multiples of the spacing, 0 included).

    Multiply by 1/(4 pi eps0) for the potential of a unit charge spread
    uniformly over the cell. At the origin this is the self-potential
    constant (2.38008 / h for a cube); far away it tends to 1/r with an
    O((h/r)^4) error (the quadrupole moment of a cube vanishes).
    """
    ax, ay, az = 0.5 * hx, 0.5 * hy, 0.5 * hz
    F = _antiderivative_1_over_r
    g = (F(x + ax, y + ay, z + az) - F(x - ax, y + ay, z + az)
         - F(x + ax, y - ay, z + az) + F(x - ax, y - ay, z + az)
         - F(x + ax, y + ay, z - az) + F(x - ax, y + ay, z - az)
         + F(x + ax, y - ay, z - az) - F(x - ax, y - ay, z - az))
    return g / (hx * hy * hz)


def hockney_green_function(shape, h, xp=np):
    """Integrated Green's function on the DOUBLED grid (2nx, 2ny, 2nz).

    Index i maps to the separation ``h * min(i, 2n - i)`` (mirror-symmetric),
    so the circular convolution with the zero-padded charge reproduces the
    open-space sum for every pair of nodes of the physical grid. Even in each
    axis, hence a real spectrum. Returns <1/r> (no 1/(4 pi eps0)).
    """
    nx, ny, nz = (int(v) for v in shape)
    hx, hy, hz = (float(v) for v in h)
    # G depends on |x|, |y|, |z| only: evaluate the closed form on one octant
    # (nx+1, ny+1, nz+1) and mirror it onto the doubled grid by indexing -
    # 8x fewer transcendental evaluations (the expensive part on a GPU in
    # float64, and what a window-shape change costs during tracking).
    X, Y, Z = xp.meshgrid(hx * xp.arange(nx + 1, dtype=xp.float64),
                          hy * xp.arange(ny + 1, dtype=xp.float64),
                          hz * xp.arange(nz + 1, dtype=xp.float64), indexing='ij', sparse=True)
    G_oct = integrated_green_1_over_r(X, Y, Z, hx, hy, hz)
    ix = xp.arange(2 * nx)
    iy = xp.arange(2 * ny)
    iz = xp.arange(2 * nz)
    ix = xp.minimum(ix, 2 * nx - ix)
    iy = xp.minimum(iy, 2 * ny - iy)
    iz = xp.minimum(iz, 2 * nz - iz)
    return xp.ascontiguousarray(G_oct[ix[:, None, None], iy[None, :, None], iz[None, None, :]])


# ============================================================================
# Solver
# ============================================================================
class FFTPoissonSolver:
    """Open-boundary FFT Poisson solver on a moving window (see module doc).

    Parameters
    ----------
    h : float or (hx, hy, hz)
        Cell size [m]. Default 1.5 mm. Anisotropic cells are fine (integrated
        Green's function).
    pad_cells : int
        Empty cells between the particle bounding box and the window edge on
        every side (>= 2: CIC and the gradient stencil need one clear layer,
        the second keeps the gather away from the zero boundary layer of E).
        Larger values make ``gather`` / ``phi`` available further out (plots).
    use_gpu : bool
        CuPy for deposit, FFTs, gradient and gather. Falls back to the CPU
        with a warning when CuPy is not importable.
    max_cells : int
        Cap on nx*ny*nz of the physical window. Beyond it the cell size is
        coarsened by 25 % steps until the window fits (warning, once per
        coarsening). The doubled FFT grid holds 8x as many points.
    min_cells : int
        Minimum nodes per axis (keeps a very thin bunch from degenerating to a
        2-node axis).
    relativistic : bool
        Reserved for the rest-frame boost (see the module TODO). ``True``
        raises NotImplementedError today.
    cache_size : int
        Number of Green's-function spectra kept (one per distinct window
        shape / spacing; each is (2nx)(2ny)(nz+1) complex128, i.e. up to a few
        hundred MB for a window at ``max_cells``; on the GPU the cuFFT plans
        of that shape are kept with it).
    window_quantile : float
        0 (default): the window spans the exact bounding box of the particles.
        q > 0: it spans the [q, 1 - q] quantile range per axis and the
        stragglers outside it neither deposit nor receive a field (E = 0) -
        a handful of far-away particles must not blow the window (and the
        cell size, via ``max_cells``) up for the whole bunch. 1e-3 excludes
        at most ~0.6 % of the particles.
    verbose : bool
        Log one line per solve (logging.INFO).

    Attributes (after a solve)
    --------------------------
    origin, shape, h    window geometry: node (i, j, k) is at origin + (i hx, j hy, k hz)
    phi                 potential [V] on the window, numpy (nx, ny, nz)
    efield_grid         (Ex, Ey, Ez) [V/m] on the nodes, numpy
    grid_axes           (x, y, z) 1D node coordinates [m]
    n_solves, solve_times, last_timing, green_recomputes
    """

    def __init__(self,
                 h: Union[float, Sequence[float]] = 1.5e-3,
                 pad_cells: int = 2,
                 use_gpu: bool = False,
                 max_cells: int = 4_000_000,
                 min_cells: int = 8,
                 relativistic: bool = False,
                 cache_size: int = 4,
                 window_quantile: float = 0.0,
                 verbose: bool = False):
        h = np.atleast_1d(np.asarray(h, dtype=float))
        if not 0.0 <= window_quantile < 0.5:
            raise ValueError("window_quantile must be in [0, 0.5)")
        self.window_quantile = float(window_quantile)
        if h.size == 1:
            h = np.repeat(h, 3)
        if h.shape != (3,) or np.any(h <= 0.0):
            raise ValueError("h must be a positive cell size or a 3-tuple of them")
        self.h0 = h.copy()              # requested spacing (coarsening starts from here)
        self.h = h.copy()               # spacing actually used by the last solve
        if pad_cells < 2:
            raise ValueError("pad_cells must be >= 2")
        self.pad_cells = int(pad_cells)
        self.max_cells = int(max_cells)
        self.min_cells = max(4, int(min_cells))
        if relativistic:
            raise NotImplementedError(
                "relativistic (rest-frame boost) solve is not implemented yet; see the module TODO")
        self.relativistic = bool(relativistic)
        self.cache_size = max(1, int(cache_size))
        self.verbose = bool(verbose)

        self.use_gpu = bool(use_gpu) and CUPY_AVAILABLE
        if use_gpu and not CUPY_AVAILABLE:
            warnings.warn("FFTPoissonSolver: GPU requested but CuPy is not available, using the CPU",
                          stacklevel=2)
        self.xp = cp if self.use_gpu else np

        self.growth = 1.2               # headroom when an axis of the window has to grow
        self.shrink_factor = 0.5        # shrink an axis only below this fraction of the window
        self._last_fit = (None, None)   # (h, shape) of the previous window, for the hysteresis

        self.origin = None
        self.shape = None
        self._phi = None                # (nx, ny, nz) on the device
        self._E = None                  # (Ex, Ey, Ez) on the device
        self._green_cache = {}          # (shape, h) -> spectrum (device array)
        self._green_order = []
        self.green_recomputes = 0
        self.n_solves = 0
        self.solve_times = []
        self.last_timing = {}
        self._coarsen_warned = False

    # ------------------------------------------------------------------ window
    def _fft_len(self, n: int) -> int:
        """Smallest 5-smooth length >= n (the doubled grid 2n is then 5-smooth too)."""
        return int(sp_fft.next_fast_len(max(int(n), self.min_cells), real=True))

    def _fit_window(self, lo: np.ndarray, hi: np.ndarray):
        """Choose spacing, shape and origin for a bunch spanning [lo, hi].

        Hysteresis keeps the Green's function cache useful while the bunch
        grows or breathes: an axis keeps its previous node count while the
        bunch still fits and has not shrunk below ``shrink_factor`` of it;
        when it has to grow it gets ``growth`` headroom, so a steadily
        expanding bunch changes the window shape in ~20 % steps, not by a
        cell or two every solve (each shape change costs a Green's recompute).
        """
        h = self.h0.copy()
        extent = np.maximum(hi - lo, 0.0)
        prev_h, prev_shape = self._last_fit
        while True:
            n_needed = np.ceil(extent / h).astype(int) + 2 * self.pad_cells + 1
            shape = []
            for a in range(3):
                n_prev = prev_shape[a] if (prev_shape is not None and np.allclose(prev_h, h)) else None
                if n_prev is None:
                    shape.append(self._fft_len(int(n_needed[a])))                  # first fit: exact
                elif n_prev >= n_needed[a] and n_needed[a] >= self.shrink_factor * n_prev:
                    shape.append(int(n_prev))                                       # still fits: keep
                elif n_prev < n_needed[a]:
                    shape.append(self._fft_len(int(np.ceil(self.growth * n_needed[a]))))   # grow with headroom
                else:
                    shape.append(self._fft_len(int(n_needed[a])))                  # shrunk a lot: refit
            shape = tuple(shape)
            if int(np.prod(shape)) <= self.max_cells:
                break
            # too big with the kept / headroom axes: an exact refit at this spacing
            # comes before coarsening (a kept axis from a formerly larger bunch must
            # not push the whole window into a coarser cell size)
            exact = tuple(self._fft_len(int(n)) for n in n_needed)
            if int(np.prod(exact)) <= self.max_cells:
                shape = exact
                break
            h = h * 1.25
            if not self._coarsen_warned:
                warnings.warn(
                    f"FFTPoissonSolver: window of {tuple(int(v) for v in n_needed)} cells exceeds "
                    f"max_cells={self.max_cells}; coarsening the cell size (h -> {h * 1e3} mm)",
                    stacklevel=3)
                self._coarsen_warned = True
        self._last_fit = (h.copy(), shape)
        # centre the window on the bunch: the padding is then >= pad_cells on both sides
        centre = 0.5 * (lo + hi)
        origin = centre - 0.5 * (np.asarray(shape) - 1) * h
        return h, shape, origin

    def _green_spectrum(self, shape, h):
        """rfftn of the doubled-grid Green's function, cached by (shape, h).

        On the GPU the entry also carries the two cuFFT plans (R2C forward,
        C2R inverse) for this doubled shape: CuPy's own plan cache holds 16
        plans and a moving window with a few live shapes thrashes it, at
        ~10 ms per plan creation - as much as the FFTs themselves.
        Returns (G_hat, plan_r2c, plan_c2r); the plans are None on the CPU.
        """
        key = (tuple(int(v) for v in shape), tuple(float(v) for v in h))
        entry = self._green_cache.get(key)
        if entry is None:
            t0 = time.perf_counter()
            xp = self.xp
            G = hockney_green_function(shape, h, xp=xp)
            if self.use_gpu:
                plan_r2c = cu_fft.get_fft_plan(G, axes=(0, 1, 2), value_type='R2C')
                G_hat = cu_fft.rfftn(G, plan=plan_r2c)
                plan_c2r = cu_fft.get_fft_plan(G_hat, shape=G.shape, axes=(0, 1, 2), value_type='C2R')
                entry = (G_hat, plan_r2c, plan_c2r)
            else:
                entry = (sp_fft.rfftn(G, workers=-1), None, None)
            del G
            self._green_cache[key] = entry
            self._green_order.append(key)
            while len(self._green_order) > self.cache_size:
                # evicted device arrays go back to CuPy's pool, which reuses them
                # (and frees them itself if an allocation would otherwise fail)
                self._green_cache.pop(self._green_order.pop(0), None)
            self.green_recomputes += 1
            self.last_timing['green_s'] = time.perf_counter() - t0
        else:
            self.last_timing['green_s'] = 0.0
        return entry

    # ------------------------------------------------------------------ pieces
    def _deposit(self, pos, q, shape, origin, h):
        """CIC charge [C] per node, flat (nx*ny*nz,) on the device."""
        nx, ny, nz = shape
        if self.use_gpu:
            g = (cp.asarray(pos, dtype=cp.float64) - cp.asarray(origin)) / cp.asarray(h)
            g = cp.ascontiguousarray(g)
            rho = cp.zeros(nx * ny * nz, dtype=cp.float64)
            _gpu_kernels()['deposit'](g[:, 0], g[:, 1], g[:, 2], cp.asarray(q, dtype=cp.float64), nx, ny, nz, rho)
            return rho
        pos = np.ascontiguousarray(pos, dtype=np.float64)
        px = (pos[:, 0] - origin[0]) / h[0]
        py = (pos[:, 1] - origin[1]) / h[1]
        pz = (pos[:, 2] - origin[2]) / h[2]
        rho = np.zeros(nx * ny * nz, dtype=np.float64)
        _cic_deposit_numba(px, py, pz, np.ascontiguousarray(q, dtype=np.float64), rho, nx, ny, nz)
        return rho

    def _potential(self, rho_flat, shape, h):
        """phi [V] on the physical window from the node charges (Hockney convolution)."""
        nx, ny, nz = shape
        xp = self.xp
        G_hat, plan_r2c, plan_c2r = self._green_spectrum(shape, h)
        rho_pad = xp.zeros((2 * nx, 2 * ny, 2 * nz), dtype=xp.float64)
        rho_pad[:nx, :ny, :nz] = rho_flat.reshape(nx, ny, nz)
        if self.use_gpu:
            spec = cu_fft.rfftn(rho_pad, plan=plan_r2c)
            spec *= G_hat
            phi = cu_fft.irfftn(spec, s=rho_pad.shape, plan=plan_c2r)
        else:
            phi = sp_fft.irfftn(sp_fft.rfftn(rho_pad, workers=-1) * G_hat, s=rho_pad.shape, workers=-1)
        phi = phi[:nx, :ny, :nz]
        phi *= 1.0 / (4.0 * np.pi * EPS0)
        return xp.ascontiguousarray(phi)

    def _gradient(self, phi, h):
        """E = -grad(phi) by central differences; zero on the boundary layer."""
        if self.use_gpu:
            Ex = cp.zeros_like(phi)
            Ey = cp.zeros_like(phi)
            Ez = cp.zeros_like(phi)
            Ex[1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / (2.0 * h[0])
            Ey[:, 1:-1, :] = -(phi[:, 2:, :] - phi[:, :-2, :]) / (2.0 * h[1])
            Ez[:, :, 1:-1] = -(phi[:, :, 2:] - phi[:, :, :-2]) / (2.0 * h[2])
            # match the CPU kernel exactly: the boundary layer of EVERY axis is zero
            for arr in (Ex, Ey, Ez):
                arr[0, :, :] = 0.0
                arr[-1, :, :] = 0.0
                arr[:, 0, :] = 0.0
                arr[:, -1, :] = 0.0
                arr[:, :, 0] = 0.0
                arr[:, :, -1] = 0.0
            return Ex, Ey, Ez
        return compute_field_from_potential_numba(phi, float(h[0]), float(h[1]), float(h[2]))

    def _gather(self, pos):
        """Trilinear gather of the stored node field at ``pos`` (device -> numpy)."""
        Ex, Ey, Ez = self._E
        nx, ny, nz = self.shape
        if self.use_gpu:
            g = (cp.asarray(pos, dtype=cp.float64) - cp.asarray(self.origin)) / cp.asarray(self.h)
            g = cp.ascontiguousarray(g)
            ex, ey, ez = _gpu_kernels()['gather'](g[:, 0], g[:, 1], g[:, 2], Ex, Ey, Ez, nx, ny, nz)
            return cp.asnumpy(cp.stack([ex, ey, ez], axis=1))
        pos = np.ascontiguousarray(pos, dtype=np.float64)
        px = (pos[:, 0] - self.origin[0]) / self.h[0]
        py = (pos[:, 1] - self.origin[1]) / self.h[1]
        pz = (pos[:, 2] - self.origin[2]) / self.h[2]
        out = np.empty((len(pos), 3), dtype=np.float64)
        _trilinear_gather_numba(px, py, pz, Ex, Ey, Ez, out)
        return out

    # -------------------------------------------------------------------- API
    def solve(self, positions: np.ndarray, charges: np.ndarray) -> np.ndarray:
        """Space-charge field E [V/m] at the particles (N, 3), free space,
        non-relativistic. Also leaves phi / efield_grid / grid_axes on the
        solver for inspection and ``gather`` for other points.

        Parameters
        ----------
        positions : (N, 3) [m]
        charges : (N,) [C]  charge per macro-particle (sign included)
        """
        t_start = time.perf_counter()
        pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        q = np.asarray(charges, dtype=np.float64).reshape(-1)
        if len(q) != len(pos):
            raise ValueError(f"positions ({len(pos)}) and charges ({len(q)}) differ in length")
        if len(pos) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        if not np.all(np.isfinite(pos)):
            raise ValueError("non-finite particle positions")

        # window from the bunch extent; with window_quantile > 0 the extent is
        # the [q, 1-q] quantile range per axis and the stragglers outside it
        # neither deposit nor receive a field (E = 0): a few far-away particles
        # must not blow the window up for everybody else
        inside = None
        if self.window_quantile > 0.0 and len(pos) > 2:
            lo = np.quantile(pos, self.window_quantile, axis=0)
            hi = np.quantile(pos, 1.0 - self.window_quantile, axis=0)
            inside = np.all((pos >= lo) & (pos <= hi), axis=1)
            if inside.all():
                inside = None
        else:
            lo, hi = pos.min(axis=0), pos.max(axis=0)
        h, shape, origin = self._fit_window(lo, hi)
        self.h, self.shape, self.origin = h, shape, origin
        pos_in = pos if inside is None else pos[inside]
        q_in = q if inside is None else q[inside]
        t_win = time.perf_counter()

        rho = self._deposit(pos_in, q_in, shape, origin, h)
        t_dep = time.perf_counter()

        self._phi = self._potential(rho, shape, h)
        t_phi = time.perf_counter()

        self._E = self._gradient(self._phi, h)
        t_grad = time.perf_counter()

        if inside is None:
            E = self._gather(pos)
        else:
            E = np.zeros((len(pos), 3), dtype=np.float64)
            E[inside] = self._gather(pos_in)
        t_end = time.perf_counter()

        self.n_solves += 1
        self.solve_times.append(t_end - t_start)
        self.last_timing.update({
            'window_s': t_win - t_start, 'deposit_s': t_dep - t_win, 'potential_s': t_phi - t_dep,
            'gradient_s': t_grad - t_phi, 'gather_s': t_end - t_grad, 'total_s': t_end - t_start,
            'shape': tuple(int(v) for v in shape), 'h_m': tuple(float(v) for v in h), 'n_particles': int(len(pos)),
            'n_excluded': 0 if inside is None else int((~inside).sum()),
        })
        if self.verbose:
            logging.info(
                "FFTPoissonSolver solve %d: %d particles, window %s cells (h = %s mm), %.1f ms "
                "(deposit %.1f, fft %.1f, grad %.1f, gather %.1f; green %.1f)",
                self.n_solves, len(pos), shape, tuple(np.round(1e3 * h, 3)), 1e3 * (t_end - t_start),
                1e3 * (t_dep - t_win), 1e3 * (t_phi - t_dep), 1e3 * (t_grad - t_phi), 1e3 * (t_end - t_grad),
                1e3 * self.last_timing.get('green_s', 0.0))
        return E

    def solve_eb(self, positions: np.ndarray, charges: np.ndarray,
                 velocities: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(E, B) at the particles. The interface for the tracker hooks: they
        apply q (E + v x B) whatever the solver's frame treatment.

        Non-relativistic version: E from ``solve``, B identically zero;
        ``velocities`` is accepted (the relativistic boost will need the mean
        velocity) and ignored.
        """
        E = self.solve(positions, charges)
        return E, np.zeros_like(E)

    def gather(self, positions: np.ndarray) -> np.ndarray:
        """E [V/m] of the LAST solve interpolated to ``positions`` (N, 3).
        Points outside the window get zero (the field there is unknown,
        not small: enlarge ``pad_cells`` if you need it)."""
        if self._E is None:
            raise RuntimeError("no solve yet")
        pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if len(pos) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        out = self._gather(pos)
        hi = self.origin + (np.asarray(self.shape) - 1) * self.h
        outside = np.any((pos < self.origin) | (pos > hi), axis=1)
        if np.any(outside):
            out[outside] = 0.0
        return out

    # -------------------------------------------------------------- accessors
    @property
    def phi(self) -> Optional[np.ndarray]:
        """Potential [V] on the window of the last solve (numpy, (nx, ny, nz))."""
        if self._phi is None:
            return None
        return cp.asnumpy(self._phi) if self.use_gpu else self._phi

    @property
    def efield_grid(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """(Ex, Ey, Ez) [V/m] on the nodes of the last solve (numpy)."""
        if self._E is None:
            return None
        if self.use_gpu:
            return tuple(cp.asnumpy(a) for a in self._E)
        return self._E

    @property
    def grid_axes(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """1D node coordinates (x, y, z) [m] of the last window."""
        if self.shape is None:
            return None
        return tuple(self.origin[a] + np.arange(self.shape[a]) * self.h[a] for a in range(3))

    @property
    def window(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(lo, hi) corners [m] of the last window."""
        if self.shape is None:
            return None
        return self.origin.copy(), self.origin + (np.asarray(self.shape) - 1) * self.h

    def to_field(self, label: str = "Space-charge E-field (FFT)", interpolator_backend: str = 'scipy',
                 method: str = 'linear'):
        """The node field of the last solve as a PyPATools ``Field`` (zero
        outside the window), for plots or a CompositeField slot."""
        from .field import Field
        if self._E is None:
            raise RuntimeError("no solve yet")
        x, y, z = self.grid_axes
        Ex, Ey, Ez = self.efield_grid
        return Field.from_arrays(grid={'x': x, 'y': y, 'z': z}, values={'x': Ex, 'y': Ey, 'z': Ez},
                                 label=label, dim=3, scaling=1.0, units='m', method=method,
                                 interpolator_backend=interpolator_backend)

    def summary(self) -> dict:
        """Solve count and timing statistics."""
        st = np.asarray(self.solve_times, dtype=float)
        return {
            'n_solves': int(self.n_solves),
            'total_solve_s': float(st.sum()) if st.size else 0.0,
            'mean_solve_ms': float(1e3 * st.mean()) if st.size else 0.0,
            'max_solve_ms': float(1e3 * st.max()) if st.size else 0.0,
            'green_recomputes': int(self.green_recomputes),
            'last_shape': tuple(int(v) for v in self.shape) if self.shape is not None else None,
            'last_h_m': tuple(float(v) for v in self.h),
            'pad_cells': self.pad_cells,
            'window_quantile': self.window_quantile,
            'gpu': bool(self.use_gpu),
            'relativistic': bool(self.relativistic),
        }

    def __repr__(self):
        return (f"FFTPoissonSolver(h={tuple(np.round(1e3 * self.h0, 4))} mm, pad_cells={self.pad_cells}, "
                f"gpu={self.use_gpu}, solves={self.n_solves}, shape={self.shape})")

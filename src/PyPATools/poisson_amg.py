"""
poisson_amg.py - PyAMG Poisson Solver for Cyclotron Space-Charge

Solves the 3D Poisson equation ∇²φ = -ρ using PyAMG with Shortley-Weller
boundary treatment for arbitrary conductor geometry.

GPU Acceleration Strategy:
    - GPU (CuPy): Particle binning (CIC deposition), field interpolation
    - CPU (PyAMG + scipy.sparse.linalg): GMRES with AMG preconditioner

    Rationale: GMRES and preconditioner are tightly coupled iterative
    operations. Moving data between CPU and GPU per iteration would exceed
    the cost of the computation itself. Instead, we transfer RHS and solution
    vectors once (small overhead), keep the solve loop on CPU.

    Future Optimization:
        GPU GMRES without preconditioner could be tested using
        cupyx.scipy.sparse.linalg.gmres. This would require:
        - GPU-accelerated preconditioner (e.g., GPU-based AMG via cuSPARSE)
        - Or relaxed convergence tolerances (accepting more iterations)
        Currently not implemented because per-iteration CPU-GPU transfers
        would dominate runtime (preconditioner must be applied each iteration).

Author: PyPATools Team
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
from scipy.sparse.linalg import gmres, LinearOperator
import pyamg
import time
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from enum import IntEnum
import numba as nb
from .particles import ParticleDistribution
from py_electrodes.py_electrodes import PyElectrodeAssembly

try:
    import cupy as cp
    import cupyx.scipy.sparse as cu_sparse

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Import Field class from the project
from .field import Field


# ============================================================================
# Configuration and Enums
# ============================================================================

@dataclass
class PyAMGSolverConfig:
    """Configuration for PyAMG Poisson solver"""

    # Domain
    domain_extent: Tuple[float, float, float]  # (Lx, Ly, Lz) in m
    mesh_cells: Tuple[int, int, int]  # (nx, ny, nz) - number of cells

    # AMG parameters
    amg_strength: float = 0.25
    amg_max_levels: int = 10

    # Solver parameters
    solver_tol: float = 1e-6
    max_iterations: int = 500
    gmres_restart: int = 50
    amg_precond_tol: float = 1e-2
    amg_precond_maxiter: int = 1  # One V-cycle per GMRES iteration

    # Boundary handling
    distance_threshold: float = 1e-4  # cm, for conductor detection

    # GPU options
    use_gpu: bool = True

    # Field output
    interpolator_backend: str = 'auto'  # 'scipy', 'cupy', or 'auto'


class CellType(IntEnum):
    """Classification of mesh cells"""
    INTERIOR = 0
    BOUNDARY = 1
    CONDUCTOR = 2


# ============================================================================
# PyAMG Poisson Solver
# ============================================================================

@nb.jit(nopython=True, parallel=True, cache=True)
def compute_field_from_potential_numba(phi_3d, hx, hy, hz):
    """
    Compute E = -∇φ via finite differences.
    Parallelized across all interior grid points.

    Parameters
    ----------
    phi_3d : np.ndarray(nx, ny, nz)
        Potential on grid
    hx, hy, hz : float
        Grid spacings

    Returns
    -------
    Ex, Ey, Ez : np.ndarray(nx, ny, nz)
        Electric field components
    """

    nx, ny, nz = phi_3d.shape

    Ex = np.zeros((nx, ny, nz), dtype=np.float64)
    Ey = np.zeros((nx, ny, nz), dtype=np.float64)
    Ez = np.zeros((nx, ny, nz), dtype=np.float64)

    # Parallel loop over interior points
    for i in nb.prange(1, nx - 1):
        for j in range(1, ny - 1):
            for k in range(1, nz - 1):
                Ex[i, j, k] = -(phi_3d[i + 1, j, k] - phi_3d[i - 1, j, k]) / (2 * hx)
                Ey[i, j, k] = -(phi_3d[i, j + 1, k] - phi_3d[i, j - 1, k]) / (2 * hy)
                Ez[i, j, k] = -(phi_3d[i, j, k + 1] - phi_3d[i, j, k - 1]) / (2 * hz)

    return Ex, Ey, Ez


class PyAMGPoissonSolver:
    """
    3D Poisson solver for cyclotron space-charge using PyAMG.

    Features:
    - Proper Dirichlet BC on conductors via ray-casting
    - Shortley-Weller treatment for irregular boundaries
    - GPU acceleration for binning and field interpolation
    - Returns Field object with trilinear interpolators

    Parameters
    ----------
    config : PyAMGSolverConfig
        Solver configuration
    electrode_assembly : PyElectrodeAssembly
        Contains conductor geometry and ray-casting interface

    Attributes
    ----------
    A : scipy.sparse.csr_matrix
        System matrix (reused for all solves)
    amg : pyamg.multilevel_solver
        AMG hierarchy (reused for all solves)
    mesh_nodes : np.ndarray
        Cell center positions (N_dofs, 3)
    cell_type : np.ndarray
        Classification of each cell (N_dofs,)
    turn_count : int
        Number of solves performed
    """

    def __init__(self,
                 config: PyAMGSolverConfig,
                 electrode_assembly: PyElectrodeAssembly):

        self.config = config
        self.electrodes = electrode_assembly
        self.use_gpu = config.use_gpu and CUPY_AVAILABLE

        if self.use_gpu:
            logging.info("GPU acceleration enabled (CuPy)")
        else:
            if config.use_gpu:
                logging.warning("GPU requested but CuPy not available, using CPU")

        # Unpack domain
        self.Lx, self.Ly, self.Lz = config.domain_extent
        self.nx, self.ny, self.nz = config.mesh_cells
        self.hx = self.Lx / self.nx
        self.hy = self.Ly / self.ny
        self.hz = self.Lz / self.nz

        self.n_dofs = self.nx * self.ny * self.nz

        print(f"\n{'=' * 70}")
        print(f"PyAMG Poisson Solver Initialization")
        print(f"{'=' * 70}")
        print(f"Domain: {self.Lx:.1f} × {self.Ly:.1f} × {self.Lz:.1f} m")
        print(f"Mesh: {self.nx} × {self.ny} × {self.nz} = {self.n_dofs:,d} DOFs")
        print(f"Spacing: Δx={self.hx:.4f}, Δy={self.hy:.4f}, Δz={self.hz:.4f} m")
        print(f"GPU: {self.use_gpu}")

        # ONE-TIME SETUP
        t0 = time.time()

        self._generate_mesh()
        print(f"[OK] Generated mesh ({time.time() - t0:.2f}s)")

        t0 = time.time()
        self._classify_cells()
        print(f"[OK] Classified cells ({time.time() - t0:.2f}s)")

        t0 = time.time()
        self._assemble_system_matrix()
        print(f"[OK] Assembled matrix ({time.time() - t0:.2f}s)")

        t0 = time.time()
        self._build_amg_hierarchy()
        print(f"[OK] Built AMG hierarchy ({time.time() - t0:.2f}s)")

        # if self.use_gpu:
        #     t0 = time.time()
        #     self._transfer_matrix_to_gpu()
        #     print(f"[OK] Transferred matrix to GPU ({time.time() - t0:.2f}s)")

        self.turn_count = 0
        self.solve_times = []

        print(f"{'=' * 70}\n")

    # ====================================================================
    # Initialization Methods
    # ====================================================================

    def _generate_mesh(self):
        """Generate cell center coordinates"""
        x = np.linspace(-self.Lx / 2 + self.hx / 2, self.Lx / 2 - self.hx / 2, self.nx)
        y = np.linspace(-self.Ly / 2 + self.hy / 2, self.Ly / 2 - self.hy / 2, self.ny)
        z = np.linspace(-self.Lz / 2 + self.hz / 2, self.Lz / 2 - self.hz / 2, self.nz)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

        self.mesh_nodes = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        self.mesh_limits = [-self.Lx / 2 + self.hx / 2, self.Lx / 2 - self.hx / 2,
                            -self.Ly / 2 + self.hy / 2, self.Ly / 2 - self.hy / 2,
                            -self.Lz / 2 + self.hz / 2, self.Lz / 2 - self.hz / 2]

    def _classify_cells(self):
        """
        Classify cells as CONDUCTOR, BOUNDARY, or INTERIOR.
        Saves precise distances to boundaries for Shortley-Weller matrix assembly.
        """
        self.cell_type = np.full(self.n_dofs, CellType.INTERIOR, dtype=np.int32)

        # Store exact distances to boundary: [x+, x-, y+, y-, z+, z-]
        self.boundary_distances = np.full((self.n_dofs, 6), np.inf, dtype=np.float64)

        print("  Querying electrode assembly for surface intersections...")
        try:
            intersections = self.electrodes.compute_axis_aligned_surface_intersections(
                self.mesh_nodes, axes='all', use_gpu=self.use_gpu)
            print("  Done!")
        except Exception as e:
            print(f"Error: {e}")
            raise

        print("  Intersections calculated. Now classifying cells...")

        # Predefine sizes and keys to avoid recreating them in the loop
        mesh_sizes = [self.hx, self.hx, self.hy, self.hy, self.hz, self.hz]
        directions = ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']

        # Iterate over each node
        for node_idx, node_data in enumerate(intersections):
            is_conductor = False
            is_boundary = False

            for dir_idx, direction in enumerate(directions):
                dir_data = node_data[direction]
                min_dist = np.inf

                for elec_id, elec_data in dir_data.items():
                    if not elec_data:
                        continue

                    dists = elec_data["distances"]

                    if len(dists) == 0:
                        continue

                    # Jordan Curve Theorem: An odd number of ray intersections
                    # means the ray origin is strictly INSIDE the closed volume.
                    if len(dists) % 2 != 0:
                        is_conductor = True

                    # Find closest boundary in this direction
                    current_min = np.min(dists)
                    if current_min < min_dist:
                        min_dist = current_min

                # Save the exact distance for Shortley-Weller stencil later
                self.boundary_distances[node_idx, dir_idx] = min_dist

                # If the nearest wall is within one cell step, it's a boundary node
                if min_dist <= mesh_sizes[dir_idx]:
                    is_boundary = True

            # Assign type (Conductor takes precedence over boundary)
            if is_conductor:
                self.cell_type[node_idx] = CellType.CONDUCTOR
            elif is_boundary:
                self.cell_type[node_idx] = CellType.BOUNDARY

        # Log statistics
        n_conductor = np.sum(self.cell_type == CellType.CONDUCTOR)
        n_boundary = np.sum(self.cell_type == CellType.BOUNDARY)
        n_interior = np.sum(self.cell_type == CellType.INTERIOR)

        print(f"  Conductor cells: {n_conductor:,d} ({100 * n_conductor / self.n_dofs:.1f}%)")
        print(f"  Boundary cells:  {n_boundary:,d} ({100 * n_boundary / self.n_dofs:.1f}%)")
        print(f"  Interior cells:  {n_interior:,d} ({100 * n_interior / self.n_dofs:.1f}%)")

    def _assemble_system_matrix(self):
        """
        Assemble sparse matrix for 3D Poisson using Shortley-Weller
        boundary treatment for precise conductor geometry.
        """
        print("  Assembling sparse matrix (Shortley-Weller formulation)...")

        # Step 1: Identify which DOFs to keep (interior + boundary only)
        dof_mask = (self.cell_type == CellType.INTERIOR) | (self.cell_type == CellType.BOUNDARY)
        self.keep_indices = np.where(dof_mask)[0]
        self.n_active_dofs = len(self.keep_indices)

        self.dof_mapping = np.full(self.n_dofs, -1, dtype=np.int32)
        self.dof_mapping[self.keep_indices] = np.arange(self.n_active_dofs)

        print(f"    Active DOFs (interior+boundary): {self.n_active_dofs:,d} / {self.n_dofs:,d}")

        # lil_matrix is fastest for incremental, random-access construction
        A_temp = lil_matrix((self.n_active_dofs, self.n_active_dofs), dtype=np.float64)

        print("  Building matrix stencil...")

        for i in range(self.nx):
            if i % 10 == 0:
                print(f"    Progress: {i}/{self.nx}")

            for j in range(self.ny):
                for k in range(self.nz):
                    idx_old = self._ijk_to_idx(i, j, k)

                    if self.cell_type[idx_old] == CellType.CONDUCTOR:
                        continue

                    idx_new = self.dof_mapping[idx_old]
                    diag_val = 0.0

                    # ----------------------------------------------------
                    # X - AXIS (i)
                    # ----------------------------------------------------

                    # +x neighbor
                    ni_pos = i + 1
                    is_pos_valid = ni_pos < self.nx
                    idx_pos = self._ijk_to_idx(ni_pos, j, k) if is_pos_valid else -1

                    if is_pos_valid and self.cell_type[idx_pos] == CellType.CONDUCTOR:
                        # Fetch true distance, clamp to 0.1% of h to prevent infinite diagonals
                        hx_pos = max(self.boundary_distances[idx_old, 0], self.hx * 1e-3)
                    else:
                        hx_pos = self.hx

                    # -x neighbor
                    ni_neg = i - 1
                    is_neg_valid = ni_neg >= 0
                    idx_neg = self._ijk_to_idx(ni_neg, j, k) if is_neg_valid else -1

                    if is_neg_valid and self.cell_type[idx_neg] == CellType.CONDUCTOR:
                        hx_neg = max(self.boundary_distances[idx_old, 1], self.hx * 1e-3)
                    else:
                        hx_neg = self.hx

                    # Add Shortley-Weller diagonal contribution
                    diag_val += 2.0 / (hx_pos * hx_neg)

                    # Add off-diagonal contributions (if neighbor is not a conductor)
                    if is_pos_valid and self.cell_type[idx_pos] != CellType.CONDUCTOR:
                        A_temp[idx_new, self.dof_mapping[idx_pos]] = -2.0 / (hx_pos * (hx_pos + hx_neg))

                    if is_neg_valid and self.cell_type[idx_neg] != CellType.CONDUCTOR:
                        A_temp[idx_new, self.dof_mapping[idx_neg]] = -2.0 / (hx_neg * (hx_pos + hx_neg))

                    # ----------------------------------------------------
                    # Y - AXIS (j)
                    # ----------------------------------------------------

                    nj_pos = j + 1
                    is_pos_valid = nj_pos < self.ny
                    idx_pos = self._ijk_to_idx(i, nj_pos, k) if is_pos_valid else -1
                    if is_pos_valid and self.cell_type[idx_pos] == CellType.CONDUCTOR:
                        hy_pos = max(self.boundary_distances[idx_old, 2], self.hy * 1e-3)
                    else:
                        hy_pos = self.hy

                    nj_neg = j - 1
                    is_neg_valid = nj_neg >= 0
                    idx_neg = self._ijk_to_idx(i, nj_neg, k) if is_neg_valid else -1
                    if is_neg_valid and self.cell_type[idx_neg] == CellType.CONDUCTOR:
                        hy_neg = max(self.boundary_distances[idx_old, 3], self.hy * 1e-3)
                    else:
                        hy_neg = self.hy

                    diag_val += 2.0 / (hy_pos * hy_neg)

                    if is_pos_valid and self.cell_type[idx_pos] != CellType.CONDUCTOR:
                        A_temp[idx_new, self.dof_mapping[idx_pos]] = -2.0 / (hy_pos * (hy_pos + hy_neg))
                    if is_neg_valid and self.cell_type[idx_neg] != CellType.CONDUCTOR:
                        A_temp[idx_new, self.dof_mapping[idx_neg]] = -2.0 / (hy_neg * (hy_pos + hy_neg))

                    # ----------------------------------------------------
                    # Z - AXIS (k)
                    # ----------------------------------------------------

                    nk_pos = k + 1
                    is_pos_valid = nk_pos < self.nz
                    idx_pos = self._ijk_to_idx(i, j, nk_pos) if is_pos_valid else -1
                    if is_pos_valid and self.cell_type[idx_pos] == CellType.CONDUCTOR:
                        hz_pos = max(self.boundary_distances[idx_old, 4], self.hz * 1e-3)
                    else:
                        hz_pos = self.hz

                    nk_neg = k - 1
                    is_neg_valid = nk_neg >= 0
                    idx_neg = self._ijk_to_idx(i, j, nk_neg) if is_neg_valid else -1
                    if is_neg_valid and self.cell_type[idx_neg] == CellType.CONDUCTOR:
                        hz_neg = max(self.boundary_distances[idx_old, 5], self.hz * 1e-3)
                    else:
                        hz_neg = self.hz

                    diag_val += 2.0 / (hz_pos * hz_neg)

                    if is_pos_valid and self.cell_type[idx_pos] != CellType.CONDUCTOR:
                        A_temp[idx_new, self.dof_mapping[idx_pos]] = -2.0 / (hz_pos * (hz_pos + hz_neg))
                    if is_neg_valid and self.cell_type[idx_neg] != CellType.CONDUCTOR:
                        A_temp[idx_new, self.dof_mapping[idx_neg]] = -2.0 / (hz_neg * (hz_pos + hz_neg))

                    # ----------------------------------------------------
                    # Set Final Diagonal
                    # ----------------------------------------------------
                    A_temp[idx_new, idx_new] = diag_val

        # Convert to CSR format
        self.A = A_temp.tocsr()
        self.A.eliminate_zeros()

        print(
            f"    Reduced matrix: {self.A.nnz:,d} nonzeros ({100 * self.A.nnz / (self.n_active_dofs ** 2):.3f}% dense)")

        self._check_matrix_health()

    def _build_amg_hierarchy(self):
        """
        Build smoothed aggregation AMG hierarchy (one-time, reused for all solves).
        Tuned specifically for the asymmetric Shortley-Weller Poisson matrix.
        """
        print("  Building AMG hierarchy (this may take 30-60 seconds)...")

        # PyAMG parameters optimized for asymmetric 3D Poisson
        amg_kwargs = {
            'max_levels': self.config.amg_max_levels,
            # Use 'evolution' or 'symmetric' depending on matrix symmetry.
            # Shortley-Weller makes the matrix structurally symmetric but numerically
            # asymmetric. 'evolution' is safer for strongly asymmetric matrices,
            # but 'symmetric' (applied to A + A^T) is usually faster for Poisson.
            'strength': ('symmetric', {'theta': self.config.amg_strength}),

            # Standard aggregation
            'aggregate': 'standard',

            # Kaczmarz or Gauss-Seidel are good smoothers here
            'smooth': ('energy', {'krylov': 'gmres', 'degree': 2}),

            # Pre/Post smoothers (Jacobi is fast, Gauss-Seidel is stronger)
            'presmoother': ('gauss_seidel', {'sweep': 'symmetric'}),
            'postsmoother': ('gauss_seidel', {'sweep': 'symmetric'}),

            # Direct solver on the coarsest level
            'coarse_solver': 'pinv',

            'keep': True
        }

        try:
            self.amg = pyamg.smoothed_aggregation_solver(self.A, **amg_kwargs)

            print(f"    Levels: {len(self.amg.levels)}")
            print(f"    Operator complexity: {self.amg.operator_complexity():.2f}")
            print(f"    Grid complexity: {self.amg.grid_complexity():.2f}")

        except Exception as e:
            logging.warning(f"    Primary AMG construction failed: {e}")
            logging.warning("    Attempting fallback to Root-Node AMG (better for strong asymmetry)...")

            try:
                # Root-node solver is often much more robust for highly skewed/asymmetric meshes
                # like those created by extreme Shortley-Weller cutoffs.
                fallback_kwargs = amg_kwargs.copy()
                fallback_kwargs['strength'] = ('evolution', {'epsilon': 4.0})

                self.amg = pyamg.rootnode_solver(self.A, **fallback_kwargs)

                print(f"    Levels: {len(self.amg.levels)}")
                print(f"    Operator complexity: {self.amg.operator_complexity:.2f}")
                logging.info("    Successfully built Root-Node AMG hierarchy")

            except Exception as e2:
                logging.error(f"    Fallback AMG also failed: {e2}")
                logging.warning("    Attempting ultra-safe (but slow) ILU smoothing fallback...")

                try:
                    # Final safety net: Weaker strength, ILU smoother
                    safe_kwargs = amg_kwargs.copy()
                    safe_kwargs['strength'] = ('symmetric', {'theta': 0.05})  # Very aggressive coarsening
                    safe_kwargs['presmoother'] = 'ilu'
                    safe_kwargs['postsmoother'] = 'ilu'

                    self.amg = pyamg.smoothed_aggregation_solver(self.A, **safe_kwargs)
                    print(f"    Levels: {len(self.amg.levels)}")
                    logging.info("    Successfully built safe ILU-AMG hierarchy")

                except Exception as e3:
                    print("    Matrix has severe structural issues. See diagnostics.")
                    raise RuntimeError(f"Failed to build AMG hierarchy after 3 attempts. Last error: {e3}") from e3

    def _check_matrix_health(self):
        """Perform diagnostic checks on system matrix."""

        print(f"\n  Matrix Diagnostics:")
        print(f"    Shape: {self.A.shape}")
        print(f"    Nonzeros: {self.A.nnz:,d}")
        print(f"    Density: {100 * self.A.nnz / (self.A.shape[0] * self.A.shape[1]):.4f}%")

        # Check for empty rows/columns
        row_sums = np.array(self.A.sum(axis=1)).flatten()
        col_sums = np.array(self.A.sum(axis=0)).flatten()

        zero_rows = np.sum(row_sums == 0)
        zero_cols = np.sum(col_sums == 0)

        if zero_rows > 0:
            print(f"    ! Found {zero_rows} rows with zero sum")
        if zero_cols > 0:
            print(f"    ! Found {zero_cols} columns with zero sum")

        # Check for NaN or Inf
        if np.any(np.isnan(self.A.data)):
            print(f"    X Matrix contains NaN values!")
        if np.any(np.isinf(self.A.data)):
            print(f"    X Matrix contains Inf values!")

        # Diagonal dominance
        diag = np.abs(self.A.diagonal())
        off_diag = np.abs(self.A.sum(axis=1).A1) - diag
        strictly_diag_dom = np.sum(diag > off_diag)
        print(f"    Diag. dominant rows: {strictly_diag_dom}/{self.n_dofs}")
        print(f"    Diagonal range: [{diag.min():.2e}, {diag.max():.2e}]")
        print(f"    Data range: [{self.A.data.min():.2e}, {self.A.data.max():.2e}]")

    def _debug_boundary_matrix(self):
        """Print information about boundary and conductor rows"""

        conductor_indices = np.where(self.cell_type == CellType.CONDUCTOR)[0]
        boundary_indices = np.where(self.cell_type == CellType.BOUNDARY)[0]

        print(f"  Conductor row check (first 10):")
        for idx in conductor_indices[:10]:
            row = self.A.getrow(idx)
            print(f"    Row {idx}: nnz={row.nnz}, sum={row.sum():.2e}")

        logging.info(f"  Boundary row check (first 10):")
        for idx in boundary_indices[:10]:
            row = self.A.getrow(idx)
            print(f"    Row {idx}: nnz={row.nnz}, sum={row.sum():.2e}")

    def _transfer_matrix_to_gpu(self):
        """Convert sparse matrix to GPU format (cuSPARSE)"""

        # Convert scipy CSR to CuPy CSR
        A_coo = self.A.tocoo()
        self.A_gpu = cu_sparse.coo_matrix(
            (cp.asarray(A_coo.data),
             (cp.asarray(A_coo.row), cp.asarray(A_coo.col))),
            shape=A_coo.shape,
            dtype=cp.float64
        ).tocsr()

        logging.info(f"    Matrix on GPU: {self.A_gpu.nnz:,d} nonzeros")

    # ====================================================================
    # Index Conversion
    # ====================================================================

    def _ijk_to_idx(self, i: int, j: int, k: int) -> int:
        """Convert (i,j,k) to linear index"""
        return i * (self.ny * self.nz) + j * self.nz + k

    def _idx_to_ijk(self, idx: int) -> Tuple[int, int, int]:
        """Convert linear index to (i,j,k)"""
        i = idx // (self.ny * self.nz)
        remainder = idx % (self.ny * self.nz)
        j = remainder // self.nz
        k = remainder % self.nz
        return i, j, k

    # ====================================================================
    # Per-Solve Operations (called repeatedly)
    # ====================================================================

    def solve(self,
              particles: np.ndarray,
              charges: np.ndarray) -> Tuple[np.ndarray, Field]:
        """
        Solve Poisson equation for space-charge field.

        Returns Field object with trilinear interpolators for E field components.

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape (N_particles, 3)
        charges : np.ndarray
            Particle charges, shape (N_particles,)

        Returns
        -------
        phi_3d : np.ndarray(nx, ny, nz)
            Potential on mesh grid (in Volts).
            Useful for direct visualization, slicing, or custom analysis.

        E_field : Field
            Electric field object with trilinear interpolators (in V/m).
            Call E_field(positions_in_meters) to evaluate at arbitrary positions.

        Note: E is computed as E = -∇φ via central differences on interior points.
        Boundary values are zero (can be improved with one-sided differences if needed).

        Examples
        --------
        > solver = PyAMGPoissonSolver(config, electrode_assembly)
        > E = solver.solve(particles, charges)
        > E_at_origin = E(np.array([[0, 0, 0]]))  # Returns (1, 3) array
        """

        t_start = time.time()

        # ================================================================
        # Step 1: Bin particles (GPU)
        # ================================================================

        if self.use_gpu:
            rho_gpu = self._bin_particles_cic_gpu(particles, charges)
            b_gpu = self._assemble_rhs_gpu(rho_gpu)

            # Transfer to CPU for scipy GMRES
            b_cpu = cp.asnumpy(b_gpu)
        else:
            rho_cpu = self._bin_particles_cic_cpu(particles, charges)
            b_cpu = self._assemble_rhs_cpu(rho_cpu)

        t_bin = time.time()

        # ================================================================
        # Step 2: Solve with GMRES + AMG preconditioner (CPU)
        # ================================================================

        residuals = []
        iteration_count = [0]

        def callback(rk):
            """GMRES convergence callback"""
            iteration_count[0] += 1
            residuals.append(float(rk))
            if iteration_count[0] % 5 == 0:
                logging.info(f"    GMRES iter {iteration_count[0]:3d}: residual={rk:.2e}")

        def preconditioner_matvec(r):
            """Apply one AMG V-cycle"""
            return self.amg.solve(
                r,
                tol=self.config.amg_precond_tol,
                maxiter=self.config.amg_precond_maxiter,
                acycle='V'
            )

        M = LinearOperator((self.n_dofs, self.n_dofs), matvec=preconditioner_matvec)

        logging.info("  Solving with GMRES (scipy) + AMG preconditioner...")

        x_cpu, gmres_info = gmres(
            self.A,
            b_cpu,
            M=M,
            tol=self.config.solver_tol,
            restart=self.config.gmres_restart,
            maxiter=self.config.max_iterations,
            callback=callback,
        )

        if gmres_info != 0:
            logging.warning(f"  GMRES warning: info={gmres_info}")

        t_solve = time.time()

        # ================================================================
        # Step 3: Compute E field = -∇φ on grid
        # ================================================================

        logging.info("  Computing electric field from potential...")

        # x_cpu is solution to REDUCED system (n_active_dofs,)
        # Expand back to full grid (n_dofs,)

        x_full = np.zeros(self.n_dofs, dtype=np.float64)
        x_full[self.keep_indices] = x_cpu
        # Conductor cells remain 0 (as enforced by Dirichlet BC)

        phi_3d = x_full.reshape((self.nx, self.ny, self.nz))

        # Numba-accelerated finite difference
        Ex, Ey, Ez = compute_field_from_potential_numba(
            phi_3d, self.hx, self.hy, self.hz
        )

        t_field = time.time()

        # ================================================================
        # Step 4: Create Field object with interpolators
        # ================================================================

        logging.info("  Creating Field object with interpolators...")

        # Grid coordinates
        x_grid = np.linspace(-self.Lx/2 + self.hx/2, self.Lx/2 - self.hx/2, self.nx)
        y_grid = np.linspace(-self.Ly/2 + self.hy/2, self.Ly/2 - self.hy/2, self.ny)
        z_grid = np.linspace(-self.Lz/2 + self.hz/2, self.Lz/2 - self.hz/2, self.nz)

        grid_dict = {
            'x': x_grid,
            'y': y_grid,
            'z': z_grid,
        }

        values_dict = {
            'x': Ex,
            'y': Ey,
            'z': Ez,
        }

        E_field = Field.from_arrays(
            grid=grid_dict,
            values=values_dict,
            label=f"Space-charge E-field (turn {self.turn_count})",
            dim=3,
            scaling=1.0,
            units='m',
            method='linear',
            interpolator_backend=self.config.interpolator_backend,
        )

        t_interp = time.time()

        # ================================================================
        # Bookkeeping
        # ================================================================

        solve_time = t_interp - t_start
        self.solve_times.append(solve_time)
        self.turn_count += 1

        logging.info(f"\n  Solve complete (turn {self.turn_count})")
        logging.info(f"    Binning:      {(t_bin - t_start) * 1000:6.1f} ms")
        logging.info(f"    Solve:        {(t_solve - t_bin) * 1000:6.1f} ms ({iteration_count[0]} iterations)")
        logging.info(f"    Field comp.:  {(t_field - t_solve) * 1000:6.1f} ms")
        logging.info(f"    Interp. create: {(t_interp - t_field) * 1000:6.1f} ms")
        logging.info(f"    Total:        {solve_time * 1000:6.1f} ms")
        logging.info(f"    Residual:     {np.linalg.norm(self.A @ x_cpu - b_cpu):.2e}\n")

        return (phi_3d, E_field)

    # ====================================================================
    # Binning (CIC deposition)
    # ====================================================================

    def _bin_particles_cic_gpu(self,
                               particles: np.ndarray,
                               charges: np.ndarray) -> cp.ndarray:
        """Cloud-in-cell deposition on GPU"""

        particles_gpu = cp.asarray(particles, dtype=cp.float32)
        charges_gpu = cp.asarray(charges, dtype=cp.float64)

        rho_gpu = cp.zeros(self.n_dofs, dtype=cp.float64)

        # Grid indices
        px_grid = (particles_gpu[:, 0] + self.Lx / 2) / self.hx
        py_grid = (particles_gpu[:, 1] + self.Ly / 2) / self.hy
        pz_grid = (particles_gpu[:, 2] + self.Lz / 2) / self.hz

        # CIC deposition
        ix = cp.floor(px_grid).astype(cp.int32)
        iy = cp.floor(py_grid).astype(cp.int32)
        iz = cp.floor(pz_grid).astype(cp.int32)

        ix = cp.clip(ix, 0, self.nx - 2)
        iy = cp.clip(iy, 0, self.ny - 2)
        iz = cp.clip(iz, 0, self.nz - 2)

        fx = px_grid - ix
        fy = py_grid - iy
        fz = pz_grid - iz

        # Deposit to 8 corners
        for dix in [0, 1]:
            for diy in [0, 1]:
                for diz in [0, 1]:
                    jx = (ix + dix) % self.nx
                    jy = (iy + diy) % self.ny
                    jz = (iz + diz) % self.nz

                    idx = jx * (self.ny * self.nz) + jy * self.nz + jz

                    w = ((1.0 - fx) if dix == 0 else fx) * \
                        ((1.0 - fy) if diy == 0 else fy) * \
                        ((1.0 - fz) if diz == 0 else fz)

                    cp.add.at(rho_gpu, idx, charges_gpu * w)

        # Normalize by cell volume
        cell_volume = self.hx * self.hy * self.hz
        rho_gpu /= cell_volume

        return rho_gpu

    def _bin_particles_cic_cpu(self,
                               particles: np.ndarray,
                               charges: np.ndarray) -> np.ndarray:
        """Cloud-in-cell deposition on CPU (Numba)"""

        rho = np.zeros(self.n_dofs, dtype=np.float64)

        px_grid = (particles[:, 0] + self.Lx / 2) / self.hx
        py_grid = (particles[:, 1] + self.Ly / 2) / self.hy
        pz_grid = (particles[:, 2] + self.Lz / 2) / self.hz

        # Clamp to valid range
        px_grid = np.clip(px_grid, 0, self.nx - 1)
        py_grid = np.clip(py_grid, 0, self.ny - 1)
        pz_grid = np.clip(pz_grid, 0, self.nz - 1)

        # CIC deposition (Numba)
        self._cic_deposit_numba(px_grid, py_grid, pz_grid, charges, rho,
                                self.nx, self.ny, self.nz)

        # Normalize
        cell_volume = self.hx * self.hy * self.hz
        rho /= cell_volume

        return rho

    @staticmethod
    @nb.jit(nopython=True, parallel=True)
    def _cic_deposit_numba(px, py, pz, charges, rho, nx, ny, nz):
        """Numba-accelerated CIC deposition"""

        for pid in nb.prange(len(charges)):
            ix = int(np.floor(px[pid]))
            iy = int(np.floor(py[pid]))
            iz = int(np.floor(pz[pid]))

            fx = px[pid] - ix
            fy = py[pid] - iy
            fz = pz[pid] - iz

            ix = np.clip(ix, 0, nx - 2)
            iy = np.clip(iy, 0, ny - 2)
            iz = np.clip(iz, 0, nz - 2)

            # Deposit to 8 corners
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

    # ====================================================================
    # RHS Assembly
    # ====================================================================

    def _assemble_rhs_cpu(self, rho: np.ndarray) -> np.ndarray:
        """Assemble RHS vector for reduced system (interior+boundary DOFs only)"""

        # Full RHS
        b_full = -rho.copy()

        # Extract only active DOFs
        b = b_full[self.keep_indices]

        return b

    def _assemble_rhs_gpu(self, rho_gpu: cp.ndarray) -> cp.ndarray:
        """Assemble RHS vector on GPU for reduced system"""

        b_full_gpu = -rho_gpu.copy()

        # Extract only active DOFs
        keep_indices_gpu = cp.asarray(self.keep_indices, dtype=cp.int32)
        b_gpu = b_full_gpu[keep_indices_gpu]

        return b_gpu

    # ====================================================================
    # Diagnostics and Summary
    # ====================================================================

    def print_summary(self):
        """Print summary of all solves"""

        if len(self.solve_times) == 0:
            print("No solves performed yet")
            return

        print(f"\n{'=' * 70}")
        print(f"PyAMG Solver Summary")
        print(f"{'=' * 70}")
        print(f"Total solves: {self.turn_count}")
        print(f"Total time: {np.sum(self.solve_times):.1f} sec")
        print(f"Average time per solve: {np.mean(self.solve_times) * 1000:.1f} ms")
        print(f"Min/Max: {np.min(self.solve_times) * 1000:.1f} / {np.max(self.solve_times) * 1000:.1f} ms")
        print(f"Std dev: {np.std(self.solve_times) * 1000:.1f} ms")
        print(f"{'=' * 70}\n")


    def debug_visualize_cell_classification(self, output_file: Optional[str] = None):
        """
        Visualize cell classification (INTERIOR, BOUNDARY, CONDUCTOR).

        Creates 4-panel plot showing:
        - 3D scatter of all classified cells
        - XY slice at z=0
        - XZ slice at y=0
        - YZ slice at x=0

        Useful for debugging ray-casting and boundary detection.

        Parameters
        ----------
        output_file : str, optional
            If provided, save figure to this path. Otherwise just display.

        Examples
        --------
        > solver = PyAMGPoissonSolver(config, electrode_assembly)
        > solver.debug_visualize_cell_classification(output_file='cell_classification.png')
        """

        try:
            import matplotlib.pyplot as plt
            from matplotlib import patches
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            logging.error("Matplotlib required for visualization")
            return

        print(f"\n{'=' * 70}")
        print(f"Generating Cell Classification Debug Visualization")
        print(f"{'=' * 70}")

        # Grid coordinates
        x_grid = np.linspace(-self.Lx / 2 + self.hx / 2, self.Lx / 2 - self.hx / 2, self.nx)
        y_grid = np.linspace(-self.Ly / 2 + self.hy / 2, self.Ly / 2 - self.hy / 2, self.ny)
        z_grid = np.linspace(-self.Lz / 2 + self.hz / 2, self.Lz / 2 - self.hz / 2, self.nz)

        # Color mapping for cell types
        color_map = {
            CellType.INTERIOR: 'blue',
            CellType.BOUNDARY: 'orange',
            CellType.CONDUCTOR: 'red',
        }

        labels = {
            CellType.INTERIOR: 'Interior',
            CellType.BOUNDARY: 'Boundary',
            CellType.CONDUCTOR: 'Conductor',
        }

        # Create figure with 4 subplots
        fig = plt.figure(figsize=(16, 14))

        # ====================================================================
        # Plot 1: 3D view of all cells
        # ====================================================================

        print("  Creating 3D view...")

        ax1 = fig.add_subplot(221, projection='3d')

        for cell_type in [CellType.INTERIOR, CellType.BOUNDARY, CellType.CONDUCTOR]:
            mask = self.cell_type == cell_type

            if np.sum(mask) > 0:
                pts = self.mesh_nodes[mask]
                ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                            c=color_map[cell_type], s=10, alpha=0.4,
                            label=f"{labels[cell_type]} ({np.sum(mask):,d})")

        ax1.set_xlabel('X (m)', fontsize=10)
        ax1.set_ylabel('Y (m)', fontsize=10)
        ax1.set_zlabel('Z (m)', fontsize=10)
        ax1.set_title('3D Cell Classification', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.set_xlim(-self.Lx / 2, self.Lx / 2)
        ax1.set_ylim(-self.Ly / 2, self.Ly / 2)
        ax1.set_zlim(-self.Lz / 2, self.Lz / 2)

        # ====================================================================
        # Plot 2: XY slice at z=0 (middle z)
        # ====================================================================

        print("  Creating XY slice (z=0)...")

        ax2 = fig.add_subplot(222)

        mid_z_idx = self.nz // 2
        z_target = z_grid[mid_z_idx]

        # Find points in this slice (within tolerance)
        z_slice_mask = np.abs(self.mesh_nodes[:, 2] - z_target) < self.hz / 2.1
        slice_indices = np.where(z_slice_mask)[0]

        for cell_type in [CellType.INTERIOR, CellType.BOUNDARY, CellType.CONDUCTOR]:
            mask = self.cell_type[slice_indices] == cell_type

            if np.sum(mask) > 0:
                pts = self.mesh_nodes[slice_indices[mask]]
                ax2.scatter(pts[:, 0] * 100, pts[:, 1] * 100,
                            c=color_map[cell_type], s=30, alpha=0.6,
                            label=f"{labels[cell_type]} ({np.sum(mask)})")

        ax2.set_xlabel('X (cm)', fontsize=10)
        ax2.set_ylabel('Y (cm)', fontsize=10)
        ax2.set_title(f'XY Slice (z={z_target * 100:.2f} cm)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        ax2.legend(fontsize=9)
        ax2.set_xlim(-self.Lx / 2 * 100, self.Lx / 2 * 100)
        ax2.set_ylim(-self.Ly / 2 * 100, self.Ly / 2 * 100)

        # ====================================================================
        # Plot 3: XZ slice at y=0 (middle y)
        # ====================================================================

        print("  Creating XZ slice (y=0)...")

        ax3 = fig.add_subplot(223)

        mid_y_idx = self.ny // 2
        y_target = y_grid[mid_y_idx]

        # Find points in this slice
        y_slice_mask = np.abs(self.mesh_nodes[:, 1] - y_target) < self.hy / 2.1
        slice_indices = np.where(y_slice_mask)[0]

        for cell_type in [CellType.INTERIOR, CellType.BOUNDARY, CellType.CONDUCTOR]:
            mask = self.cell_type[slice_indices] == cell_type

            if np.sum(mask) > 0:
                pts = self.mesh_nodes[slice_indices[mask]]
                ax3.scatter(pts[:, 0] * 100, pts[:, 2] * 100,
                            c=color_map[cell_type], s=30, alpha=0.6,
                            label=f"{labels[cell_type]} ({np.sum(mask)})")

        ax3.set_xlabel('X (cm)', fontsize=10)
        ax3.set_ylabel('Z (cm)', fontsize=10)
        ax3.set_title(f'XZ Slice (y={y_target * 100:.2f} cm)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        ax3.legend(fontsize=9)
        ax3.set_xlim(-self.Lx / 2 * 100, self.Lx / 2 * 100)
        ax3.set_ylim(-self.Lz / 2 * 100, self.Lz / 2 * 100)

        # ====================================================================
        # Plot 4: YZ slice at x=0 (middle x)
        # ====================================================================

        print("  Creating YZ slice (x=0)...")

        ax4 = fig.add_subplot(224)

        mid_x_idx = self.nx // 2
        x_target = x_grid[mid_x_idx]

        # Find points in this slice
        x_slice_mask = np.abs(self.mesh_nodes[:, 0] - x_target) < self.hx / 2.1
        slice_indices = np.where(x_slice_mask)[0]

        for cell_type in [CellType.INTERIOR, CellType.BOUNDARY, CellType.CONDUCTOR]:
            mask = self.cell_type[slice_indices] == cell_type

            if np.sum(mask) > 0:
                pts = self.mesh_nodes[slice_indices[mask]]
                ax4.scatter(pts[:, 1] * 100, pts[:, 2] * 100,
                            c=color_map[cell_type], s=30, alpha=0.6,
                            label=f"{labels[cell_type]} ({np.sum(mask)})")

        ax4.set_xlabel('Y (cm)', fontsize=10)
        ax4.set_ylabel('Z (cm)', fontsize=10)
        ax4.set_title(f'YZ Slice (x={x_target * 100:.2f} cm)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
        ax4.legend(fontsize=9)
        ax4.set_xlim(-self.Ly / 2 * 100, self.Ly / 2 * 100)
        ax4.set_ylim(-self.Lz / 2 * 100, self.Lz / 2 * 100)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\n✓ Saved visualization to: {output_file}")

        plt.show()

        # Print statistics
        n_interior = np.sum(self.cell_type == CellType.INTERIOR)
        n_boundary = np.sum(self.cell_type == CellType.BOUNDARY)
        n_conductor = np.sum(self.cell_type == CellType.CONDUCTOR)

        print(f"\n  Cell Classification Summary:")
        print(f"    Interior:  {n_interior:,d} ({100 * n_interior / self.n_dofs:.1f}%)")
        print(f"    Boundary:  {n_boundary:,d} ({100 * n_boundary / self.n_dofs:.1f}%)")
        print(f"    Conductor: {n_conductor:,d} ({100 * n_conductor / self.n_dofs:.1f}%)")
        print(f"{'=' * 70}\n")


# ============================================================================
# Utility Functions
# ============================================================================

def create_solver(domain_extent: Tuple[float, float, float],
                  mesh_cells: Tuple[int, int, int],
                  electrode_assembly: 'PyElectrodeAssembly',
                  **config_kwargs) -> PyAMGPoissonSolver:
    """
    Factory function to create PyAMG solver.

    Parameters
    ----------
    domain_extent : tuple of float
        (Lx, Ly, Lz) domain size in cm
    mesh_cells : tuple of int
        (nx, ny, nz) number of cells in each dimension
    electrode_assembly : PyElectrodeAssembly
        Conductor geometry
    **config_kwargs
        Additional PyAMGSolverConfig parameters

    Returns
    -------
    PyAMGPoissonSolver
        Initialized solver

    Examples
    --------
    > solver = create_solver(
    ...     domain_extent=(40.0, 40.0, 30.0),
    ...     mesh_cells=(64, 64, 64),
    ...     electrode_assembly=electrodes,
    ...     solver_tol=1e-6,
    ...     use_gpu=True
    ... )
    """
    config = PyAMGSolverConfig(
        domain_extent=domain_extent,
        mesh_cells=mesh_cells,
        **config_kwargs
    )

    return PyAMGPoissonSolver(config, electrode_assembly)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Example: Use PyAMG solver in a cyclotron tracking loop
    """

    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Mock electrode assembly (replace with actual)
    class MockElectrodeAssembly:
        def compute_axis_aligned_surface_intersections(self, mesh_nodes, axes='all',
                                                       use_gpu=None, chunk_size=None):
            """Mock: no electrodes, all open space"""
            N = len(mesh_nodes)
            return {
                'x+': {'hit_mask': np.zeros(N, dtype=bool), 'hit_points': np.zeros((N, 3)),
                       'distances': np.full(N, np.inf)},
                'x-': {'hit_mask': np.zeros(N, dtype=bool), 'hit_points': np.zeros((N, 3)),
                       'distances': np.full(N, np.inf)},
                'y+': {'hit_mask': np.zeros(N, dtype=bool), 'hit_points': np.zeros((N, 3)),
                       'distances': np.full(N, np.inf)},
                'y-': {'hit_mask': np.zeros(N, dtype=bool), 'hit_points': np.zeros((N, 3)),
                       'distances': np.full(N, np.inf)},
                'z+': {'hit_mask': np.zeros(N, dtype=bool), 'hit_points': np.zeros((N, 3)),
                       'distances': np.full(N, np.inf)},
                'z-': {'hit_mask': np.zeros(N, dtype=bool), 'hit_points': np.zeros((N, 3)),
                       'distances': np.full(N, np.inf)},
                'electrode_ids': np.full((N, 6), -1, dtype=np.int32),
                'electrode_index_map': {},
            }

    # Create solver
    config = PyAMGSolverConfig(
        domain_extent=(40.0, 40.0, 30.0),
        mesh_cells=(32, 32, 32),  # Small for testing
        solver_tol=1e-5,
        use_gpu=CUPY_AVAILABLE,
    )

    solver = PyAMGPoissonSolver(config, MockElectrodeAssembly())

    # Generate test particles
    np.random.seed(42)
    n_particles = 100000
    particles = np.random.uniform(-5, 5, (n_particles, 3))
    charges = np.ones(n_particles) / n_particles

    print("\n" + "=" * 70)
    print("EXAMPLE: Solving space-charge field")
    print("=" * 70 + "\n")

    # Solve
    E_field = solver.solve(particles, charges)

    print(f"\n✓ Field computed successfully!")
    print(f"Field object: {E_field}")
    print(f"Field dim: {E_field.dim}")

    # Evaluate field at some test points
    test_points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [-2.0, 2.0, -1.0],
    ])

    E_at_test = E_field(test_points)
    print(f"\nField values at test points:")
    print(f"  E(0, 0, 0) = {E_at_test[0]}")
    print(f"  E(1, 1, 1) = {E_at_test[1]}")
    print(f"  E(-2, 2, -1) = {E_at_test[2]}")

    # Summary
    solver.print_summary()
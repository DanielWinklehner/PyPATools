"""PyAMGPoissonSolver on a domain that is not centred on the origin.

Added 2026-09-04 with config.domain_origin. The grid was previously hard-coded to
span [-L/2, +L/2] on every axis, in six separate places (mesh generation, the Field
returned by solve(), both CIC deposition paths, and the diagnostic plots), so a
geometry that did not straddle the origin -- e.g. a spiral inflector spanning
z = -0.12 .. 0.05 m -- could not be solved without shifting the particles by hand.

domain_origin is the CENTRE of the box and defaults to (0, 0, 0), which reproduces
the original grid exactly; test_default_origin_matches_legacy_grid pins that down.

The physics checks use a mock assembly with no conductors, so the only boundary
condition is the solver's own outer box. That keeps these tests about the grid and
the solve rather than about geometry queries.

Also covers, as a regression: _cic_deposit_numba used to be decorated
parallel=True while doing "rho[idx] += ..." with a computed index. That is a data
race, not a reduction numba can recognise, and it lost ~1% of the total charge
nondeterministically -- two runs on identical input disagreed. See
test_cic_conserves_total_charge.

Run: python tests/test_poisson_origin.py   (or via pytest)
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyPATools.global_variables import EPS0                               # noqa: E402

# poisson_amg needs pyamg, which is an optional dependency. Skip rather than
# error when it is absent; unittest.SkipTest is understood by pytest too, and
# keeps this module runnable without pytest installed (as the other tests are).
try:
    from PyPATools.poisson_amg import PyAMGSolverConfig, PyAMGPoissonSolver
    HAVE_PYAMG = True
except ImportError:                                                       # pragma: no cover
    PyAMGSolverConfig = PyAMGPoissonSolver = None
    HAVE_PYAMG = False


def _require_pyamg():
    if not HAVE_PYAMG:
        raise unittest.SkipTest("poisson_amg requires pyamg")

EXTENT = (0.20, 0.20, 0.20)
CELLS = (32, 32, 32)
OFFSET = np.array([0.05, -0.03, 0.11])

LX, LY, LZ = EXTENT
NX, NY, NZ = CELLS
HX, HY, HZ = LX / NX, LY / NY, LZ / NZ
CELL_VOL = HX * HY * HZ


class _NoConductors(object):
    """Stand-in for PyElectrodeAssembly with empty space everywhere.

    Matches the real signature: returns (min_distances, hit_counts), both (N, 6).
    A large finite distance keeps every cell INTERIOR without introducing infinities
    into the Shortley-Weller distances.
    """

    def compute_axis_aligned_surface_intersections(self, mesh_nodes, axes="all",
                                                   use_gpu=None, chunk_size=None):
        n = len(mesh_nodes)
        return (np.full((n, 6), 1.0e6, dtype=np.float32),
                np.zeros((n, 6), dtype=np.int32))


def _build(origin):
    _require_pyamg()

    config = PyAMGSolverConfig(domain_extent=EXTENT,
                               mesh_cells=CELLS,
                               domain_origin=tuple(origin),
                               use_gpu=False,
                               solver_tol=1e-10,
                               max_iterations=400)

    return PyAMGPoissonSolver(config, _NoConductors())


def _legacy_axes():
    """The axis arrays the solver produced before domain_origin existed."""
    return (np.linspace(-LX / 2 + HX / 2, LX / 2 - HX / 2, NX),
            np.linspace(-LY / 2 + HY / 2, LY / 2 - HY / 2, NY),
            np.linspace(-LZ / 2 + HZ / 2, LZ / 2 - HZ / 2, NZ))


def _blob(n_particles=20000, sigma=0.012, charge=1.0e-15, seed=1234):
    rng = np.random.default_rng(seed)
    return (rng.normal(scale=sigma, size=(n_particles, 3)),
            np.full(n_particles, charge))


# ============================================================================
# Grid geometry
# ============================================================================
def test_default_origin_matches_legacy_grid():
    solver = _build((0.0, 0.0, 0.0))
    old_x, old_y, old_z = _legacy_axes()

    assert np.array_equal(solver.x_grid, old_x)
    assert np.array_equal(solver.y_grid, old_y)
    assert np.array_equal(solver.z_grid, old_z)

    assert np.allclose(solver.mesh_limits,
                       [old_x[0], old_x[-1], old_y[0], old_y[-1], old_z[0], old_z[-1]])


def test_offset_grid_is_the_legacy_grid_shifted():
    solver = _build(OFFSET)
    old_x, old_y, old_z = _legacy_axes()

    assert np.allclose(solver.x_grid, old_x + OFFSET[0])
    assert np.allclose(solver.y_grid, old_y + OFFSET[1])
    assert np.allclose(solver.z_grid, old_z + OFFSET[2])

    # domain_origin is the box centre, so the node centroid lands on it
    assert np.allclose(solver.mesh_nodes.mean(axis=0), OFFSET, atol=1e-12)


# ============================================================================
# Charge deposition
# ============================================================================
def test_cic_conserves_total_charge():
    solver = _build((0.0, 0.0, 0.0))
    pos, q = _blob()

    # _bin_particles_cic_cpu divides by the cell volume, so rho is a density
    rho = solver._bin_particles_cic_cpu(pos, q)

    assert abs(rho.sum() * CELL_VOL - q.sum()) < 1e-12 * q.sum()


def test_cic_is_translation_invariant():
    s0 = _build((0.0, 0.0, 0.0))
    s1 = _build(OFFSET)
    pos, q = _blob()

    rho0 = s0._bin_particles_cic_cpu(pos, q)
    rho1 = s1._bin_particles_cic_cpu(pos + OFFSET, q)

    assert np.allclose(rho0, rho1, rtol=0, atol=1e-15 * rho0.max())


# ============================================================================
# The solve itself
# ============================================================================
def test_solve_is_translation_invariant():
    s0 = _build((0.0, 0.0, 0.0))
    s1 = _build(OFFSET)
    pos, q = _blob()

    phi0, e0 = s0.solve(pos, q)
    phi1, e1 = s1.solve(pos + OFFSET, q)

    assert np.abs(phi0 - phi1).max() < 1e-10 * np.abs(phi0).max()

    probe = np.array([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.0, -0.02, 0.015]])
    assert np.allclose(e0(probe), e1(probe + OFFSET), rtol=1e-7, atol=1e-9)


def test_potential_satisfies_discrete_poisson():
    solver = _build((0.0, 0.0, 0.0))
    pos, q = _blob()

    rho = solver._bin_particles_cic_cpu(pos, q)
    phi, _ = solver.solve(pos, q)

    lap = np.zeros_like(phi)
    lap[1:-1, :, :] += (phi[2:, :, :] - 2 * phi[1:-1, :, :] + phi[:-2, :, :]) / HX ** 2
    lap[:, 1:-1, :] += (phi[:, 2:, :] - 2 * phi[:, 1:-1, :] + phi[:, :-2, :]) / HY ** 2
    lap[:, :, 1:-1] += (phi[:, :, 2:] - 2 * phi[:, :, 1:-1] + phi[:, :, :-2]) / HZ ** 2

    rhs = -rho.reshape(CELLS) / EPS0

    core = (slice(2, -2), slice(2, -2), slice(2, -2))
    residual = np.abs(lap[core] - rhs[core]).max() / np.abs(rhs[core]).max()

    assert residual < 1e-8, residual


# ============================================================================
# Analytic reference
# ============================================================================
def test_far_field_matches_point_charge():
    """Outside a compact blob the field must approach Q / (4 pi eps0 r^2).

    The blob is Gaussian with sigma = 0.012 m, so r = 0.04 m is 3.3 sigma and
    encloses essentially all the charge. Closer in only part of it is enclosed,
    which is physics rather than solver error, so this is a far-field check.

    Checked at the origin and on the offset domain: the analytic reference does
    not move, so this catches an origin error that translation invariance alone
    would not (both domains could be wrong in the same way).
    """
    r = 0.04
    pts = np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r]])

    for origin in [(0.0, 0.0, 0.0), tuple(OFFSET)]:
        solver = _build(origin)
        pos, q = _blob()

        shift = np.asarray(origin)
        _, e_field = solver.solve(pos + shift, q)

        radial = np.einsum("ij,ij->i", e_field(pts + shift), pts) / r
        analytic = q.sum() / (4.0 * np.pi * EPS0 * r ** 2)

        assert np.all(radial > 0.0), (origin, radial)   # positive charge pushes outward
        assert abs(radial.mean() / analytic - 1.0) < 0.05, (origin, radial.mean(), analytic)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            fn()
            print("[ok]   {}".format(name))
        except unittest.SkipTest as exc:
            print("[skip] {}: {}".format(name, exc))
        except AssertionError as exc:
            print("[FAIL] {}: {}".format(name, exc))
            fails += 1
    sys.exit(1 if fails else 0)

"""FFTPoissonSolver (poisson_fft.py): open-boundary space-charge solver.

Analytic references
-------------------
* the integrated Green's function: self-cell constant of a unit cube,
  3 ln(2 + sqrt 3) - pi/2 = 2.380077..., the 1/r far field, and an off-centre
  anisotropic cell against direct quadrature;
* a uniformly charged sphere: E = Q r / (4 pi eps0 R^3) inside, Q / (4 pi eps0 r^2)
  outside (expect < 2 % rms away from the surface);
* a uniformly charged ellipsoid on ANISOTROPIC cells: E_i = rho n_i x_i / eps0
  inside with the depolarisation factors n_i (elliptic integrals);
* a Gaussian bunch against the closed-form radial field (at 5 cells / sigma);
* the PyAMG Shortley-Weller solver on the same Gaussian bunch in a large empty
  box (agreement to a few % in the core). NOTE: PyAMGPoissonSolver deposits its
  CIC charge at ``x0 + i h`` but puts its potential nodes at ``x0 + (i + 1/2) h``
  (poisson_amg._bin_particles_cic_* vs _generate_mesh), i.e. its solution is
  that of the bunch shifted by +h/2 on every axis. The comparison accepts
  either the shifted or the unshifted AMG field (whichever agrees), so it
  keeps passing once that offset is fixed, and reports which one matched.

Also checked: the GPU path reproduces the CPU path, the Green's function
spectrum is cached across solves and recomputed when the window changes, the
kernels are byte-identical copies of poisson_amg's, charge conservation, and
that zero charge gives exactly zero field.

Sources are deterministic sub-lattices (no shot noise), so the tolerances
measure the method, not the sampling.

py_electrodes (imported by poisson_amg) initialises MPI on import; on machines
where MPI cannot start that aborts the interpreter instead of raising, so the
AMG comparison blocks mpi4py first (py_electrodes then falls back to its
single-process mode). Set PYPATOOLS_TEST_ALLOW_MPI=1 to skip that block.

Run: python tests/test_poisson_fft.py   (or via pytest)
"""

import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyPATools.global_variables import EPS0                                    # noqa: E402
from PyPATools.poisson_fft import (FFTPoissonSolver, integrated_green_1_over_r,   # noqa: E402
                                   hockney_green_function, CUPY_AVAILABLE)
from PyPATools import poisson_fft                                              # noqa: E402

K = 1.0 / (4.0 * np.pi * EPS0)


# ============================================================================
# helpers
# ============================================================================
def _lattice(bounds_lo, bounds_hi, sub, inside):
    """Deterministic sub-lattice of spacing ``sub`` (3-vector) inside a shape."""
    axes = [np.arange(lo + 0.5 * s, hi, s) for lo, hi, s in zip(bounds_lo, bounds_hi, sub)]
    X, Y, Z = np.meshgrid(*axes, indexing='ij')
    P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return P[inside(P)]


def _radial(E, P):
    r = np.linalg.norm(P, axis=1)
    r = np.where(r < 1e-15, 1e-15, r)
    return np.einsum('ij,ij->i', E, P) / r, r


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def _import_poisson_amg():
    """poisson_amg with py_electrodes forced into single-process mode."""
    if os.environ.get("PYPATOOLS_TEST_ALLOW_MPI", "0") != "1" and "mpi4py" not in sys.modules:
        sys.modules["mpi4py"] = None        # 'from mpi4py import MPI' -> ImportError -> fallback
    try:
        from PyPATools import poisson_amg
    except ImportError as exc:
        raise unittest.SkipTest(f"poisson_amg not importable: {exc}")
    return poisson_amg


# ============================================================================
# Green's function
# ============================================================================
def test_green_self_cell_constant():
    g = integrated_green_1_over_r(np.array(0.0), np.array(0.0), np.array(0.0), 1.0, 1.0, 1.0)
    assert abs(float(g) - (3.0 * np.log(2.0 + np.sqrt(3.0)) - 0.5 * np.pi)) < 1e-12
    # scales as 1/h
    g2 = integrated_green_1_over_r(np.array(0.0), np.array(0.0), np.array(0.0), 2.0, 2.0, 2.0)
    assert abs(float(g2) - 0.5 * float(g)) < 1e-12


def test_green_far_field_is_one_over_r():
    # the quadrupole moment of a cube vanishes: <1/r> = 1/r (1 + O((h/r)^4))
    for pt in [(10.0, 3.0, 2.0), (5.0, 0.0, 0.0), (4.0, 4.0, 4.0)]:
        x, y, z = pt
        g = float(integrated_green_1_over_r(np.array(x), np.array(y), np.array(z), 1.0, 1.0, 1.0))
        r = np.sqrt(x * x + y * y + z * z)
        assert abs(g * r - 1.0) < 3.0 * (1.0 / r) ** 4, (pt, g * r - 1.0)


def test_green_anisotropic_cell_matches_quadrature():
    from scipy.integrate import tplquad
    hx, hy, hz = 1.0, 0.5, 2.0
    x0, y0, z0 = 2.0, 1.0, 4.0

    def f(z, y, x):
        return 1.0 / np.sqrt((x0 - x) ** 2 + (y0 - y) ** 2 + (z0 - z) ** 2)

    ref, _ = tplquad(f, -hx / 2, hx / 2, -hy / 2, hy / 2, -hz / 2, hz / 2, epsabs=1e-10, epsrel=1e-10)
    ref /= hx * hy * hz
    g = float(integrated_green_1_over_r(np.array(x0), np.array(y0), np.array(z0), hx, hy, hz))
    assert abs(g - ref) < 1e-8 * abs(ref), (g, ref)


def test_hockney_grid_is_mirror_symmetric():
    G = hockney_green_function((6, 5, 4), (1.0, 2.0, 0.5))
    assert G.shape == (12, 10, 8)
    assert np.allclose(G, G[::-1, :, :][np.r_[-1, 0:11], :, :])       # x-mirror about 0
    assert np.allclose(G[0, 0, 0], integrated_green_1_over_r(np.array(0.0), np.array(0.0), np.array(0.0), 1.0, 2.0, 0.5))
    assert np.allclose(G[3, 0, 0], integrated_green_1_over_r(np.array(3.0), np.array(0.0), np.array(0.0), 1.0, 2.0, 0.5))
    assert np.allclose(G[9, 0, 0], G[3, 0, 0])                          # 12 - 9 = 3


# ============================================================================
# Uniformly charged sphere
# ============================================================================
def test_uniform_sphere():
    R, h, Q = 0.010, 0.001, 1.0e-10
    P = _lattice([-R] * 3, [R] * 3, [h / 4] * 3, lambda p: np.linalg.norm(p, axis=1) <= R)
    q = np.full(len(P), Q / len(P))

    solver = FFTPoissonSolver(h=h, pad_cells=12, use_gpu=False)
    E = solver.solve(P, q)
    E_R = K * Q / R ** 2                                    # field at the surface (normalisation)

    Er, r = _radial(E, P)
    inside = r < 0.8 * R
    err_in = (Er - K * Q * r / R ** 3)[inside] / E_R
    assert _rms(err_in) < 0.02, _rms(err_in)
    # no spurious transverse field
    E_t = E[inside] - (Er[inside] / r[inside])[:, None] * P[inside]
    assert _rms(np.linalg.norm(E_t, axis=1) / E_R) < 0.01

    # outside, on probes not coincident with the sources
    rng = np.random.default_rng(3)
    d = rng.normal(size=(60, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    rr = np.linspace(1.3 * R, 2.0 * R, 6)
    probes = (rr[:, None, None] * d[None, :, :]).reshape(-1, 3)
    Eo = solver.gather(probes)
    Ero, ro = _radial(Eo, probes)
    rel = Ero / (K * Q / ro ** 2) - 1.0
    assert _rms(rel) < 0.02, _rms(rel)
    assert np.abs(rel).max() < 0.03, np.abs(rel).max()

    # potential at the centre: 3 Q / (8 pi eps0 R); the window is centred on the
    # bunch, so the node nearest the origin is within h/2 of it
    x, y, z = solver.grid_axes
    c = tuple(int(np.argmin(np.abs(ax))) for ax in (x, y, z))
    assert abs(x[c[0]]) <= h / 2 + 1e-12 and abs(y[c[1]]) <= h / 2 + 1e-12 and abs(z[c[2]]) <= h / 2 + 1e-12
    assert abs(solver.phi[c] / (1.5 * K * Q / R) - 1.0) < 0.01


# ============================================================================
# Uniformly charged ellipsoid on anisotropic cells
# ============================================================================
def _depolarisation_factors(a, b, c):
    from scipy.integrate import quad
    axes = np.array([a, b, c])

    def n_i(i):
        def integrand(s):
            return 1.0 / ((axes[i] ** 2 + s) * np.sqrt(np.prod(axes ** 2 + s)))
        val, _ = quad(integrand, 0.0, np.inf, epsabs=0, epsrel=1e-10, limit=200)
        return 0.5 * a * b * c * val

    n = np.array([n_i(i) for i in range(3)])
    assert abs(n.sum() - 1.0) < 1e-8
    return n


def test_uniform_ellipsoid_anisotropic_cells():
    a, b, c = 0.010, 0.006, 0.015
    h = np.array([1.0e-3, 0.75e-3, 1.5e-3])                 # anisotropic cells, ~10 per semi-axis
    Q = 1.0e-10
    P = _lattice([-a, -b, -c], [a, b, c], h / 4,
                 lambda p: (p[:, 0] / a) ** 2 + (p[:, 1] / b) ** 2 + (p[:, 2] / c) ** 2 <= 1.0)
    q = np.full(len(P), Q / len(P))
    rho = Q / (4.0 / 3.0 * np.pi * a * b * c)

    solver = FFTPoissonSolver(h=h, pad_cells=4, use_gpu=False)
    E = solver.solve(P, q)
    assert solver.shape[0] != solver.shape[1] or solver.shape[1] != solver.shape[2]   # really anisotropic

    n = _depolarisation_factors(a, b, c)
    E_ana = (rho / EPS0) * n[None, :] * P
    core = (P[:, 0] / a) ** 2 + (P[:, 1] / b) ** 2 + (P[:, 2] / c) ** 2 <= 0.8 ** 2
    norm = np.abs(E_ana[core]).max()
    err = np.linalg.norm((E - E_ana)[core], axis=1) / norm
    assert _rms(err) < 0.02, _rms(err)
    # the anisotropy itself is resolved: the three depolarisation factors differ by > 2x
    for i in range(3):
        m = core & (np.abs(P[:, i]) > 0.3 * (a, b, c)[i])
        fit = np.sum(E[m, i] * P[m, i]) / np.sum(P[m, i] ** 2)
        assert abs(fit / (rho / EPS0 * n[i]) - 1.0) < 0.02, (i, fit, rho / EPS0 * n[i])


# ============================================================================
# Gaussian bunch vs closed form
# ============================================================================
def _gaussian_field(P, sigma, Q):
    from scipy.special import erf
    r = np.linalg.norm(P, axis=1)
    r = np.where(r < 1e-12, 1e-12, r)
    enclosed = erf(r / (np.sqrt(2.0) * sigma)) - np.sqrt(2.0 / np.pi) * (r / sigma) * np.exp(-r ** 2 / (2 * sigma ** 2))
    return (K * Q * enclosed / r ** 3)[:, None] * P


def test_gaussian_matches_closed_form():
    sigma, Q = 0.010, 2.4e-10
    h = sigma / 5.0
    P = _lattice([-4 * sigma] * 3, [4 * sigma] * 3, [h / 3] * 3, lambda p: np.linalg.norm(p, axis=1) <= 4 * sigma)
    w = np.exp(-np.sum(P ** 2, axis=1) / (2 * sigma ** 2))
    q = Q * w / w.sum()
    solver = FFTPoissonSolver(h=h, pad_cells=3)
    E = solver.solve(P, q)
    E_ana = _gaussian_field(P, sigma, Q)
    core = np.linalg.norm(P, axis=1) < 2.0 * sigma
    err = np.linalg.norm((E - E_ana)[core], axis=1) / np.abs(E_ana).max()
    assert _rms(err) < 0.02, _rms(err)


# ============================================================================
# Bookkeeping: charge conservation, zero charge, caching, window, GPU parity
# ============================================================================
def test_deposit_conserves_charge_and_zero_charge_gives_zero_field():
    rng = np.random.default_rng(11)
    P = rng.normal(scale=0.004, size=(5000, 3))
    q = rng.uniform(0.5, 1.5, size=len(P)) * 1e-15
    solver = FFTPoissonSolver(h=1.5e-3)
    rho = solver._deposit(P, q, *solver._fit_window(P.min(0), P.max(0))[::-1][::-1][1:2],
                          *(solver._fit_window(P.min(0), P.max(0))[2:3] + solver._fit_window(P.min(0), P.max(0))[0:1]))
    assert abs(rho.sum() - q.sum()) < 1e-12 * q.sum()
    E0 = solver.solve(P, np.zeros(len(P)))
    assert np.all(E0 == 0.0)
    assert np.all(solver.phi == 0.0)
    E1 = solver.solve(P, q)
    assert np.all(np.isfinite(E1)) and np.abs(E1).max() > 0.0
    E2 = solver.solve(P, 2.0 * q)                             # linear in the charge
    assert np.allclose(E2, 2.0 * E1, rtol=1e-12, atol=0.0)


def test_green_spectrum_cached_and_window_follows_the_bunch():
    rng = np.random.default_rng(5)
    P = rng.normal(scale=0.004, size=(3000, 3))
    q = np.full(len(P), 1e-15)
    solver = FFTPoissonSolver(h=1.5e-3, pad_cells=2)
    solver.solve(P, q)
    assert solver.green_recomputes == 1
    shape0, origin0 = solver.shape, solver.origin.copy()
    lo, hi = solver.window
    assert np.all(lo <= P.min(0) - 2 * solver.h + 1e-12) and np.all(hi >= P.max(0) + 2 * solver.h - 1e-12)

    solver.solve(P + 0.05, q)                                # same bunch, moved: same spectrum
    assert solver.green_recomputes == 1
    assert solver.shape == shape0
    assert np.allclose(solver.origin - origin0, 0.05, atol=1e-9)

    solver.solve(P * np.array([3.0, 1.0, 1.0]), q)           # stretched: new window shape
    assert solver.green_recomputes == 2
    assert solver.shape[0] > shape0[0]
    assert solver.n_solves == 3 and len(solver.solve_times) == 3

    solver.solve(P, q)                                       # back to the first window: cache hit
    assert solver.green_recomputes == 2
    assert solver.last_timing['green_s'] == 0.0


def test_translation_invariance():
    rng = np.random.default_rng(9)
    P = rng.normal(scale=0.003, size=(4000, 3))
    q = np.full(len(P), 1e-15)
    solver = FFTPoissonSolver(h=1.0e-3)
    E0 = solver.solve(P, q)
    shift = np.array([0.0731, -0.0209, 0.0117])
    E1 = solver.solve(P + shift, q)
    assert np.allclose(E0, E1, rtol=1e-8, atol=1e-8 * np.abs(E0).max())


def test_solve_eb_returns_zero_b_and_same_e():
    rng = np.random.default_rng(2)
    P = rng.normal(scale=0.003, size=(2000, 3))
    q = np.full(len(P), 1e-15)
    solver = FFTPoissonSolver(h=1.0e-3)
    E = solver.solve(P, q)
    E2, B = solver.solve_eb(P, q, velocities=np.tile([0.0, 2.4e6, 0.0], (len(P), 1)))
    assert np.array_equal(E, E2)
    assert B.shape == E.shape and np.all(B == 0.0)
    try:
        FFTPoissonSolver(relativistic=True)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("relativistic=True must raise until the boost is implemented")


def test_gpu_matches_cpu():
    if not CUPY_AVAILABLE:
        raise unittest.SkipTest("cupy not available")
    R, h, Q = 0.010, 0.001, 1.0e-10
    P = _lattice([-R] * 3, [R] * 3, [h / 3] * 3, lambda p: np.linalg.norm(p, axis=1) <= R)
    q = np.full(len(P), Q / len(P))
    cpu = FFTPoissonSolver(h=h, pad_cells=4, use_gpu=False)
    gpu = FFTPoissonSolver(h=h, pad_cells=4, use_gpu=True)
    assert gpu.use_gpu
    Ec, Eg = cpu.solve(P, q), gpu.solve(P, q)
    scale = np.abs(Ec).max()
    assert np.allclose(Ec, Eg, rtol=0, atol=1e-9 * scale)
    assert np.allclose(cpu.phi, gpu.phi, rtol=0, atol=1e-9 * np.abs(cpu.phi).max())
    probes = np.array([[0.005, 0.0, 0.0], [0.0, -0.007, 0.002], [0.012, 0.0, 0.0]])
    assert np.allclose(cpu.gather(probes), gpu.gather(probes), rtol=0, atol=1e-9 * scale)


def test_to_field_interpolates_like_gather():
    R, h, Q = 0.010, 0.001, 1.0e-10
    P = _lattice([-R] * 3, [R] * 3, [h / 3] * 3, lambda p: np.linalg.norm(p, axis=1) <= R)
    q = np.full(len(P), Q / len(P))
    solver = FFTPoissonSolver(h=h, pad_cells=4)
    solver.solve(P, q)
    field = solver.to_field()
    probes = np.array([[0.005, 0.001, 0.0], [0.0, -0.007, 0.002], [0.011, 0.0, -0.003]])
    assert np.allclose(field(probes), solver.gather(probes), rtol=1e-9, atol=1e-9 * K * Q / R ** 2)


# ============================================================================
# Against the PyAMG solver
# ============================================================================
def test_kernels_identical_to_poisson_amg():
    """The deposit / gradient kernels are copies of poisson_amg's (see the
    module comment for why they are not imported): keep them in sync."""
    import ast
    import inspect
    import textwrap
    amg = _import_poisson_amg()

    def body(fn):
        """AST of the function body without its docstring (names, ops, structure)."""
        src = textwrap.dedent(inspect.getsource(fn.py_func if hasattr(fn, 'py_func') else fn))
        node = ast.parse(src).body[0]
        stmts = node.body
        if stmts and isinstance(stmts[0], ast.Expr) and isinstance(getattr(stmts[0], 'value', None), ast.Constant) \
                and isinstance(stmts[0].value.value, str):
            stmts = stmts[1:]
        return [ast.dump(s) for s in stmts]

    assert body(poisson_fft._cic_deposit_numba) == body(amg.PyAMGPoissonSolver._cic_deposit_numba)
    assert body(poisson_fft.compute_field_from_potential_numba) == body(amg.compute_field_from_potential_numba)


class _NoConductors(object):
    """PyElectrodeAssembly stand-in: empty space (see test_poisson_origin.py)."""

    def compute_axis_aligned_surface_intersections(self, mesh_nodes, axes="all", use_gpu=None, chunk_size=None):
        n = len(mesh_nodes)
        return (np.full((n, 6), 1.0e6, dtype=np.float32), np.zeros((n, 6), dtype=np.int32))


def test_matches_pyamg_on_gaussian_in_large_box():
    """Same Gaussian bunch, same cell size: the FFT (open) and AMG (grounded
    box 15 sigma wide, image field ~1e-3 of the direct one) solutions must
    agree to a few % of the peak field in the core."""
    amg = _import_poisson_amg()

    L, N = 0.30, 64
    h = L / N                                                 # 4.7 mm
    sigma, Q = 0.020, 2.4e-10                                 # 4.3 cells per sigma
    P = _lattice([-3.5 * sigma] * 3, [3.5 * sigma] * 3, [h / 2] * 3,
                 lambda p: np.linalg.norm(p, axis=1) <= 3.5 * sigma)
    w = np.exp(-np.sum(P ** 2, axis=1) / (2 * sigma ** 2))
    q = Q * w / w.sum()

    cfg = amg.PyAMGSolverConfig(domain_extent=(L, L, L), mesh_cells=(N, N, N), use_gpu=False,
                                solver_tol=1e-9, max_iterations=500)
    t0 = time.time()
    solver_amg = amg.PyAMGPoissonSolver(cfg, _NoConductors())
    _, e_amg = solver_amg.solve(P, q)
    t_amg = time.time() - t0

    solver_fft = FFTPoissonSolver(h=h, pad_cells=3)
    t0 = time.time()
    solver_fft.solve(P, q)
    t_fft = time.time() - t0

    rng = np.random.default_rng(17)
    probes = rng.normal(scale=sigma, size=(400, 3))
    probes = probes[np.linalg.norm(probes, axis=1) < 1.5 * sigma]
    e_fft = solver_fft.gather(probes)
    scale = np.abs(e_fft).max()

    results = {}
    for name, shift in (("unshifted", 0.0), ("shifted by h/2", 0.5 * h)):
        e_a = e_amg(probes + shift)
        results[name] = _rms(np.linalg.norm(e_a - e_fft, axis=1)) / scale
    best = min(results, key=results.get)
    print(f"\n  FFT vs PyAMG (Gaussian, {N}^3, h = {1e3 * h:.2f} mm): rms |dE| / max|E| = "
          + ", ".join(f"{k}: {v:.4f}" for k, v in results.items())
          + f"  (AMG {t_amg:.1f} s incl. setup, FFT {1e3 * t_fft:.0f} ms)")
    assert results[best] < 0.05, results
    # and both are within the same few % of the closed form
    e_ana = _gaussian_field(probes, sigma, Q)
    assert _rms(np.linalg.norm(e_fft - e_ana, axis=1)) / np.abs(e_ana).max() < 0.06


# ============================================================================
# Beam in a pipe: FFT (free space) vs SA-AMG (pipe as conductor) vs 2D theory
# ============================================================================
class _Pipe(object):
    """Analytic PyElectrodeAssembly stand-in: a grounded cylindrical shell
    b <= r < b_out around the z axis, infinite along z. Per node: the minimum
    distance to metal along +x, -x, +y, -y, +z, -z and the number of surface
    crossings along each ray (odd = the node is inside the metal), which is
    what PyAMGPoissonSolver._classify_cells consumes."""

    def __init__(self, b, b_out):
        self.b, self.b_out = float(b), float(b_out)

    def compute_axis_aligned_surface_intersections(self, nodes, axes="all", use_gpu=None, chunk_size=None):
        nodes = np.asarray(nodes, dtype=float)
        n = len(nodes)
        dist = np.full((n, 6), 1.0e6, dtype=np.float32)
        hits = np.zeros((n, 6), dtype=np.int32)
        for axis, other in ((0, 1), (1, 0)):
            u, w = nodes[:, axis], nodes[:, other]
            for sign, col in ((+1.0, 2 * axis), (-1.0, 2 * axis + 1)):
                d_min = np.full(n, 1.0e6)
                cnt = np.zeros(n, dtype=np.int32)
                for radius in (self.b, self.b_out):
                    disc = radius ** 2 - w ** 2
                    ok = disc > 0.0
                    s = np.sqrt(np.where(ok, disc, 0.0))
                    for root in (-u + s, -u - s):            # (u + t)^2 + w^2 = R^2 along +u
                        t = sign * root                       # along -u the ray parameter flips sign
                        pos = ok & (t > 1e-12)
                        cnt += pos
                        d_min = np.where(pos, np.minimum(d_min, t), d_min)
                dist[:, col] = d_min
                hits[:, col] = cnt
        return dist, hits


def test_beam_in_pipe_fft_vs_saamg_vs_theory():
    """Long uniform cylindrical beam (radius a) coaxial in a grounded pipe
    (radius b). 2D theory: E_r = lambda r / (2 pi eps0 a^2) inside the beam,
    lambda / (2 pi eps0 r) in the gap; the coaxial pipe adds no field inside
    (its induced charge is a uniform shell), only the potential reference
    phi(b) = 0. So both solvers must reproduce the same E_r in the middle of
    the beam: the FFT solver in free space (finite-length deficit of ~2 % at
    the outer probes, the pipe is absent by construction) and the SA-AMG
    solver with the pipe as a Shortley-Weller conductor (end effects screened
    by the pipe). The pipe reference is checked on the AMG potential on axis:
    lambda / (2 pi eps0) (ln(b/a) + 1/2)."""
    amg = _import_poisson_amg()

    a, b, b_out, L_beam, h = 0.006, 0.015, 0.019, 0.120, 0.001
    Q = 2.4e-10
    lam = Q / L_beam
    E_a = lam / (2.0 * np.pi * EPS0 * a)                                # peak field (r = a)
    P = _lattice([-a, -a, -L_beam / 2], [a, a, L_beam / 2], [h / 3, h / 3, h / 2],
                 lambda p: np.hypot(p[:, 0], p[:, 1]) <= a)
    q = np.full(len(P), Q / len(P))

    def e_theory(r):
        return np.where(r < a, lam * r / (2.0 * np.pi * EPS0 * a ** 2), lam / (2.0 * np.pi * EPS0 * r))

    rng = np.random.default_rng(23)
    n_probe = 300
    r_p = rng.uniform(0.001, 0.8 * b, n_probe)
    th = rng.uniform(0.0, 2.0 * np.pi, n_probe)
    probes = np.column_stack([r_p * np.cos(th), r_p * np.sin(th), rng.uniform(-0.02, 0.02, n_probe)])

    def radial(E):
        return (E[:, 0] * probes[:, 0] + E[:, 1] * probes[:, 1]) / r_p

    # --- FFT, free space (pad wide enough to gather out to 0.8 b)
    fft = FFTPoissonSolver(h=h, pad_cells=8)
    fft.solve(P, q)
    er_fft = radial(fft.gather(probes))

    # --- SA-AMG with the pipe
    cfg = amg.PyAMGSolverConfig(domain_extent=(0.040, 0.040, 0.160), mesh_cells=(40, 40, 160),
                                use_gpu=False, solver_tol=1e-9, max_iterations=500)
    t0 = time.time()
    solver_amg = amg.PyAMGPoissonSolver(cfg, _Pipe(b, b_out))
    n_cond = int(np.sum(solver_amg.cell_type == amg.CellType.CONDUCTOR))
    assert n_cond > 0, "the pipe was not classified as conductor"
    phi_amg, e_amg = solver_amg.solve(P, q)
    t_amg = time.time() - t0
    res_amg = {}
    for name, shift in (("unshifted", 0.0), ("shifted by h/2", 0.5 * h)):
        res_amg[name] = radial(e_amg(probes + shift))
    dev = {k: _rms((v - e_theory(r_p)) / E_a) for k, v in res_amg.items()}
    best = min(dev, key=dev.get)
    er_amg = res_amg[best]

    err_fft = _rms((er_fft - e_theory(r_p)) / E_a)
    err_amg = dev[best]
    diff = _rms((er_fft - er_amg) / E_a)
    # pipe reference: AMG potential on the axis in the middle of the beam
    ax_pts = np.column_stack([np.zeros(9), np.zeros(9), np.linspace(-0.02, 0.02, 9)])
    ix, iy = np.argmin(np.abs(solver_amg.x_grid)), np.argmin(np.abs(solver_amg.y_grid))
    iz = [np.argmin(np.abs(solver_amg.z_grid - z)) for z in ax_pts[:, 2]]
    phi_axis = phi_amg[ix, iy, iz].mean()
    phi_theory = lam / (2.0 * np.pi * EPS0) * (np.log(b / a) + 0.5)
    print(f"\n  beam in pipe (a = {1e3 * a:.0f} mm, b = {1e3 * b:.0f} mm, h = {1e3 * h:.0f} mm): rms(E_r - theory)/E_r(a) "
          f"FFT {err_fft:.4f}, AMG {err_amg:.4f} ({best}; unshifted {dev['unshifted']:.4f}), FFT vs AMG {diff:.4f}; "
          f"AMG phi on axis {phi_axis:.1f} V vs theory {phi_theory:.1f} V; AMG {t_amg:.1f} s, "
          f"{n_cond} conductor cells")
    assert err_fft < 0.05, err_fft
    assert err_amg < 0.05, err_amg
    assert diff < 0.05, diff
    assert abs(phi_axis / phi_theory - 1.0) < 0.05, (phi_axis, phi_theory)


# ============================================================================
# Baseline: long uniform beam drifting in a pipe (examples/space_charge_beam_in_pipe.py)
# ============================================================================
def _envelope_theory(z, a0, K):
    """a(z) of the paraxial envelope equation a'' = K / a, a(0) = a0, a'(0) = 0."""
    from scipy.integrate import quad
    from scipy.optimize import brentq

    def z_of(ratio):
        # z(a) = a0 / sqrt(K) * int_1^R du / sqrt(2 ln u); u = 1 + s^2 removes the endpoint singularity
        if ratio <= 1.0:
            return 0.0
        f = lambda s: 2.0 * s / np.sqrt(2.0 * np.log1p(s * s)) if s > 1e-9 else np.sqrt(2.0)   # noqa: E731
        val, _ = quad(f, 0.0, np.sqrt(ratio - 1.0), limit=200)
        return a0 / np.sqrt(K) * val

    return np.array([a0 if zz <= 0 else a0 * brentq(lambda R: z_of(R) - zz, 1.0, 50.0) for zz in z])


def test_drifting_uniform_beam_expands_like_the_envelope_equation():
    """H2+ 60 keV, 2 mA, a0 = 5 mm, 0.8 m drift in free space: the FFT field
    re-solved every step and applied as a transverse kick makes the beam
    radius grow by ~2.3x; the envelope equation with the generalised
    perveance K = q I / (2 pi eps0 m (gamma beta c)^3) predicts it to a few %.
    A centred beam feels no field from a coaxial pipe, so this is also the
    pipe case of the example."""
    from PyPATools.species import IonSpecies
    from PyPATools.global_variables import CLIGHT
    ion = IonSpecies('H2_1+')
    gamma = 1.0 + 60.0e3 / (ion.mass_mev * 1e6)
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    v0 = beta * CLIGHT
    current, a0, l_beam, l_drift = 2.0e-3, 0.005, 0.30, 0.80
    K_perv = ion.charge * current / (2.0 * np.pi * EPS0 * ion.mass_kg * (gamma * beta * CLIGHT) ** 3)
    lam = current / v0
    rng = np.random.default_rng(7)
    n = 20000
    r = a0 * np.sqrt(rng.uniform(0.0, 1.0, n))
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    pos = np.column_stack([r * np.cos(th), r * np.sin(th), rng.uniform(-l_beam / 2, l_beam / 2, n)])
    vel = np.zeros_like(pos)
    q = np.full(n, lam * l_beam / n)
    solver = FFTPoissonSolver(h=(1.0e-3, 1.0e-3, 5.0e-3), pad_cells=2)
    mid = np.abs(pos[:, 2]) < l_beam / 4
    n_steps = 120
    dt = l_drift / v0 / n_steps
    q_over_m = ion.charge / ion.mass_kg

    def accel(p):
        E = solver.solve(p, q)
        E[:, 2] = 0.0
        return q_over_m * E / gamma

    a_sim = [np.sqrt(2.0 * np.mean(pos[mid, 0] ** 2 + pos[mid, 1] ** 2))]
    acc = accel(pos)
    for _ in range(n_steps):
        vel += 0.5 * dt * acc
        pos += dt * vel
        acc = accel(pos)
        vel += 0.5 * dt * acc
        a_sim.append(np.sqrt(2.0 * np.mean(pos[mid, 0] ** 2 + pos[mid, 1] ** 2)))
    a_sim = np.asarray(a_sim)
    z = np.arange(n_steps + 1) * l_drift / n_steps
    a_th = _envelope_theory(z, a0, K_perv)
    assert a_th[-1] / a0 > 2.0, a_th[-1] / a0                       # a visible change ...
    assert a_sim[-1] < 0.020 - 0.003                                 # ... but well inside a 20 mm pipe
    dev = np.sqrt(np.mean((a_sim / a_th - 1.0) ** 2))
    print(f"\n  drifting beam: a({l_drift} m) = {1e3 * a_sim[-1]:.2f} mm, theory {1e3 * a_th[-1]:.2f} mm, "
          f"rms deviation {100 * dev:.2f} %, {n_steps} solves of {1e3 * np.mean(solver.solve_times):.1f} ms")
    assert dev < 0.03, dev
    assert solver.green_recomputes < 12                              # the window grows in quantised steps


def test_offcentre_beam_in_pipe_image_force():
    """Beam moved towards the wall: the SA-AMG solver with the pipe as
    conductor reproduces the 2D image force lambda d / (2 pi eps0 (b^2 - d^2))
    at the beam centre, the free-space FFT solver gives none (by
    construction). Particles handed to the AMG solver are shifted by -h/2 to
    compensate its CIC offset (see the module doc)."""
    amg = _import_poisson_amg()
    a, b, b_out, h, l_beam = 0.006, 0.015, 0.019, 0.001, 0.120
    lam = 2.4e-10 / l_beam
    rng = np.random.default_rng(5)
    n = 30000
    r = a * np.sqrt(rng.uniform(0.0, 1.0, n))
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    base = np.column_stack([r * np.cos(th), r * np.sin(th), rng.uniform(-l_beam / 2, l_beam / 2, n)])
    q = np.full(n, lam * l_beam / n)
    cfg = amg.PyAMGSolverConfig(domain_extent=(0.040, 0.040, 0.160), mesh_cells=(40, 40, 160),
                                use_gpu=False, solver_tol=1e-9, max_iterations=500)
    solver_amg = amg.PyAMGPoissonSolver(cfg, _Pipe(b, b_out))
    fft = FFTPoissonSolver(h=h, pad_cells=6)
    res = []
    for f in (0.0, 0.5, 0.9):
        d = f * (b - a)
        pos = base + np.array([d, 0.0, 0.0])
        _, e_amg = solver_amg.solve(pos - 0.5 * h, q)
        fft.solve(pos, q)
        centre = np.array([[d, 0.0, 0.0]])
        e_th = lam * d / (2.0 * np.pi * EPS0 * (b ** 2 - d ** 2))
        res.append((d, float(e_amg(centre)[0, 0]), float(fft.gather(centre)[0, 0]), e_th))
    res = np.asarray(res)
    e_ref = lam / (2.0 * np.pi * EPS0 * a)                          # field at the beam edge, for scale
    print("\n  off-centre beam, E_x at the beam centre [V/m] (d, AMG, FFT, theory):\n   "
          + "\n   ".join(f"{1e3 * d:5.2f} mm  {ea:9.1f}  {ef:9.1f}  {et:9.1f}" for d, ea, ef, et in res))
    assert np.all(np.abs(res[:, 2]) < 0.03 * e_ref)                  # FFT: no image force
    assert abs(res[0, 1]) < 0.03 * e_ref                             # centred: none either
    for d, ea, ef, et in res[1:]:
        assert abs(ea - et) < 0.1 * et + 0.02 * e_ref, (d, ea, et)   # AMG: the image force, to ~10 %
    # and growing towards the wall like d / (b^2 - d^2) (theory ratio 2.3 between the two offsets)
    assert abs((res[2, 1] / res[1, 1]) / (res[2, 3] / res[1, 3]) - 1.0) < 0.15


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        t0 = time.time()
        try:
            fn()
            print("[ok]   {} ({:.1f} s)".format(name, time.time() - t0))
        except unittest.SkipTest as exc:
            print("[skip] {}: {}".format(name, exc))
        except AssertionError as exc:
            print("[FAIL] {}: {}".format(name, exc))
            fails += 1
    sys.exit(1 if fails else 0)

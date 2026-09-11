"""
examples/space_charge_beam_in_pipe.py - long uniform beam drifting in a grounded
pipe: FFT solver (free space) vs SA-AMG solver (pipe as conductor) vs theory.

A baseline for the two space-charge solvers in PyPATools:

Part A  Drift. A long uniform cylindrical H2+ beam (60 keV, radius a0, current I)
        drifts L_DRIFT metres in a pipe of radius b with no focusing. The
        space-charge field is re-solved every step with FFTPoissonSolver and
        applied as a transverse kick (velocity Verlet). The edge radius grows
        visibly but the beam never reaches the wall. Reference: the paraxial
        envelope equation a'' = K / a with the generalised perveance
        K = q I / (2 pi eps0 m gamma^3 beta^3 c^3), a(0) = a0, a'(0) = 0, whose
        solution is z(a) = (a0 / sqrt(K)) * int_1^{a/a0} du / sqrt(2 ln u).
        For a uniform laminar beam the field is linear inside, so the beam
        stays uniform and a = sqrt(2) * r_rms. The FFT solver knows nothing
        about the pipe - for a CENTRED beam that is exact in 2D (the induced
        wall charge is a uniform shell, no field inside).

Part B  Sideways. The same beam, static, moved off axis by d = 0 .. 0.95 (b - a)
        until it barely touches the wall. Now the pipe matters: the induced
        charge is no longer symmetric and pulls the beam towards the wall with
        the image field E = lambda d / (2 pi eps0 (b^2 - d^2)) at the beam
        centre (2D image line charge -lambda at b^2 / d). The SA-AMG solver
        (poisson_amg, pipe as Shortley-Weller conductor) reproduces it; the
        free-space FFT solver by construction gives zero net force.

Part C  Radial profiles. Potential and E_x along the offset axis (through the
        pipe axis and the beam centre) for the centred and the near-wall beam:
        FFT (free space, shifted to phi = 0 at the near wall), AMG (grounded
        pipe) and the 2D theory
            phi = lambda/(2 pi eps0) [ ln|x - x_img| + ln(d/b) - g(|x - d|) ],
            g(s) = ln s for s >= a, ln a + (s^2/a^2 - 1)/2 inside the beam.

The pipe is described analytically for the AMG solver (the ray-intersection
interface of PyElectrodeAssembly, see _Pipe), so this example needs pyamg but
no CAD file, gmsh or MPI; examples/uniform_beam_in_pipe.py shows the same
geometry from a BREP file through py_electrodes.

NOTE on poisson_amg: its CIC deposit puts the charge half a cell (+h/2 on every
axis) off its potential grid. The particles handed to the AMG solver are shifted
by -h/2 here so that the beam sits where the pipe geometry expects it.

Outputs: examples/results/space_charge_beam_in_pipe/{drift,sideways}.png + summary.json
Options: --no-gpu, --fast (fewer particles / steps), --no-amg (skip Part B/C).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# py_electrodes (imported by poisson_amg) initialises MPI on import; where MPI
# cannot start that aborts the interpreter. Nothing here needs MPI.
if "mpi4py" not in sys.modules:
    sys.modules["mpi4py"] = None

from PyPATools.global_variables import EPS0, CLIGHT                          # noqa: E402
from PyPATools.species import IonSpecies                                      # noqa: E402
from PyPATools.poisson_fft import FFTPoissonSolver                            # noqa: E402

K_COUL = 1.0 / (4.0 * np.pi * EPS0)

# ---------------------------------------------------------------- parameters
ION = IonSpecies('H2_1+')
E_KIN_EV = 60.0e3           # kinetic energy
A0 = 0.005                  # beam radius [m]
B_PIPE = 0.020              # pipe inner radius [m]
B_OUT = 0.024               # pipe outer radius (AMG classification needs a shell)
CURRENT_A = 2.0e-3          # beam current
L_DRIFT = 1.0               # drift length [m]
L_BEAM = 0.30               # macro-particle cylinder length [m] (>> a, b: 2D in the middle)
H_XY, H_Z = 0.75e-3, 5.0e-3  # FFT cells: fine transversely, coarse along the uniform beam


def beam_velocity(e_kin_ev, ion):
    gamma = 1.0 + e_kin_ev / (ion.mass_mev * 1e6)
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    return beta * CLIGHT, beta, gamma


def perveance(current_a, ion, beta, gamma):
    return ion.charge * current_a / (2.0 * np.pi * EPS0 * ion.mass_kg * (gamma * beta) ** 3 * CLIGHT ** 3)


def envelope_theory(z, a0, K):
    """a(z) of a'' = K/a, a(0) = a0, a'(0) = 0 (numerical inversion of z(a))."""
    from scipy.integrate import quad
    from scipy.optimize import brentq

    def z_of(ratio):
        # u = 1 + s^2 removes the integrable endpoint singularity of 1 / sqrt(2 ln u)
        if ratio <= 1.0:
            return 0.0
        f = lambda s: 2.0 * s / np.sqrt(2.0 * np.log1p(s * s)) if s > 1e-9 else np.sqrt(2.0)   # noqa: E731
        val, _ = quad(f, 0.0, np.sqrt(ratio - 1.0), limit=200)
        return a0 / np.sqrt(K) * val

    z = np.atleast_1d(np.asarray(z, dtype=float))
    out = np.empty_like(z)
    for i, zz in enumerate(z):
        out[i] = a0 if zz <= 0 else a0 * brentq(lambda R: z_of(R) - zz, 1.0, 50.0)
    return out


def uniform_cylinder(n, a, length, rng):
    r = a * np.sqrt(rng.uniform(0.0, 1.0, n))
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    z = rng.uniform(-length / 2, length / 2, n)
    return np.column_stack([r * np.cos(th), r * np.sin(th), z])


# ---------------------------------------------------------------- 2D theory (beam at (d, 0) in the pipe)
def phi_theory(x, d, a, b, lam):
    x = np.asarray(x, dtype=float)
    s = np.abs(x - d)
    g = np.where(s >= a, np.log(np.maximum(s, 1e-300)), np.log(a) + 0.5 * (s ** 2 / a ** 2 - 1.0))
    if d == 0.0:
        return lam / (2.0 * np.pi * EPS0) * (np.log(b) - g)
    x_img = b ** 2 / d
    return lam / (2.0 * np.pi * EPS0) * (np.log(np.abs(x - x_img)) + np.log(d / b) - g)


def ex_theory(x, d, a, b, lam):
    """E_x = -d phi / dx along the offset axis."""
    x = np.asarray(x, dtype=float)
    s = x - d
    beam = np.where(np.abs(s) >= a, 1.0 / np.where(s == 0, np.inf, s), s / a ** 2)    # -d g/dx * (-1) ...
    e = lam / (2.0 * np.pi * EPS0) * beam
    if d != 0.0:
        x_img = b ** 2 / d
        e = e - lam / (2.0 * np.pi * EPS0) / (x - x_img)
    return e


# ---------------------------------------------------------------- analytic pipe for the AMG solver
class _Pipe(object):
    """Grounded cylindrical shell b <= r < b_out around the z axis (infinite in z)
    through PyElectrodeAssembly's ray-intersection interface: per node the
    minimum distance to metal along +x, -x, +y, -y, +z, -z and the number of
    surface crossings along each ray (odd = inside the metal)."""

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
                    for root in (-u + s, -u - s):
                        t = sign * root
                        pos = ok & (t > 1e-12)
                        cnt += pos
                        d_min = np.where(pos, np.minimum(d_min, t), d_min)
                dist[:, col] = d_min
                hits[:, col] = cnt
        return dist, hits


def amg_pipe_solver(b, b_out, h, z_len):
    from PyPATools.poisson_amg import PyAMGSolverConfig, PyAMGPoissonSolver
    n_xy = int(np.ceil(2.0 * (b_out + 2 * h) / h))
    n_z = int(np.ceil(z_len / h))
    cfg = PyAMGSolverConfig(domain_extent=(n_xy * h, n_xy * h, n_z * h), mesh_cells=(n_xy, n_xy, n_z),
                            use_gpu=False, solver_tol=1e-8, max_iterations=500)
    return PyAMGPoissonSolver(cfg, _Pipe(b, b_out))


def amg_line(solver, phi, e_field, x, h):
    """phi and E_x on the line (x, 0, 0): the potential from the nearest node
    row in y, z (the AMG grid has cell-centre nodes), E from the Field."""
    iy = int(np.argmin(np.abs(solver.y_grid)))
    iz = int(np.argmin(np.abs(solver.z_grid)))
    phi_line = np.interp(x, solver.x_grid, phi[:, iy, iz])
    pts = np.column_stack([x, np.full_like(x, solver.y_grid[iy]), np.full_like(x, solver.z_grid[iz])])
    return phi_line, e_field(pts)[:, 0]


def fft_line(solver, x):
    """phi and E_x of the last FFT solve on the line (x, 0, 0)."""
    from scipy.interpolate import RegularGridInterpolator
    gx, gy, gz = solver.grid_axes
    rgi = RegularGridInterpolator((gx, gy, gz), solver.phi, bounds_error=False, fill_value=0.0)
    pts = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    return rgi(pts), solver.gather(pts)[:, 0]


# ================================================================ Part A: drift
def run_drift(n_particles, n_steps, use_gpu, rng, verbose=True):
    v0, beta, gamma = beam_velocity(E_KIN_EV, ION)
    K = perveance(CURRENT_A, ION, beta, gamma)
    lam = CURRENT_A / v0
    q_macro = lam * L_BEAM / n_particles
    pos = uniform_cylinder(n_particles, A0, L_BEAM, rng)
    vel = np.zeros_like(pos)
    vel[:, 2] = v0
    q_over_m = ION.charge / ION.mass_kg
    dt = L_DRIFT / v0 / n_steps
    solver = FFTPoissonSolver(h=(H_XY, H_XY, H_Z), pad_cells=2, use_gpu=use_gpu)
    mid = np.abs(pos[:, 2]) < L_BEAM / 4          # 2D region, away from the ends
    charges = np.full(n_particles, q_macro)

    def accel(p):
        E = solver.solve(p, charges)
        E[:, 2] = 0.0                              # transverse dynamics only (the ends push along z)
        return q_over_m * E / gamma                # transverse: dp_perp/dt = q E_perp, p = gamma m v

    z_hist = [0.0]
    rms_hist = [np.sqrt(np.mean(pos[mid, 0] ** 2 + pos[mid, 1] ** 2))]
    edge_hist = [np.hypot(pos[mid, 0], pos[mid, 1]).max()]
    t0 = time.time()
    a = accel(pos)
    for step in range(n_steps):
        vel += 0.5 * dt * a
        pos += dt * vel
        a = accel(pos)
        vel += 0.5 * dt * a
        z_hist.append((step + 1) * L_DRIFT / n_steps)
        rms_hist.append(np.sqrt(np.mean(pos[mid, 0] ** 2 + pos[mid, 1] ** 2)))
        edge_hist.append(np.hypot(pos[mid, 0], pos[mid, 1]).max())
    wall = time.time() - t0
    z_hist = np.asarray(z_hist)
    a_sim = np.sqrt(2.0) * np.asarray(rms_hist)
    a_th = envelope_theory(z_hist, A0, K)
    out = {'v0_m_s': v0, 'beta': beta, 'gamma': gamma, 'K': K, 'lambda_c_per_m': lam, 'dt_s': dt,
           'n_particles': n_particles, 'n_steps': n_steps, 'wall_s': wall,
           'solve_ms_mean': 1e3 * np.mean(solver.solve_times), 'gpu': solver.use_gpu,
           'window_last': solver.shape, 'green_recomputes': solver.green_recomputes,
           'a_final_sim_mm': 1e3 * a_sim[-1], 'a_final_theory_mm': 1e3 * a_th[-1],
           'edge_final_mm': 1e3 * edge_hist[-1],
           'rms_rel_dev': float(np.sqrt(np.mean((a_sim / a_th - 1.0) ** 2))),
           'z': z_hist, 'a_sim': a_sim, 'a_theory': a_th, 'edge': np.asarray(edge_hist), 'pos_final': pos}
    if verbose:
        print(f"[drift] {ION.name} {E_KIN_EV / 1e3:.0f} keV, v = {v0:.3e} m/s, I = {1e3 * CURRENT_A:.1f} mA, "
              f"K = {K:.3e}, lambda = {lam:.3e} C/m; {n_particles} particles, {n_steps} steps of {1e3 * L_DRIFT / n_steps:.1f} mm")
        print(f"[drift] a(z = {L_DRIFT} m): simulation {1e3 * a_sim[-1]:.2f} mm (edge {1e3 * edge_hist[-1]:.2f}), "
              f"theory {1e3 * a_th[-1]:.2f} mm, rms deviation along z {100 * out['rms_rel_dev']:.2f} %; "
              f"pipe {1e3 * B_PIPE:.0f} mm; {wall:.1f} s ({out['solve_ms_mean']:.1f} ms per solve, "
              f"window {solver.shape}, gpu {solver.use_gpu})")
    return out


# ================================================================ Part B/C: sideways + profiles
def run_sideways(n_particles, use_gpu, rng, fractions=(0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95), h_amg=1.0e-3,
                 l_static=0.20, verbose=True):
    v0, beta, gamma = beam_velocity(E_KIN_EV, ION)
    lam = CURRENT_A / v0
    q_macro = lam * l_static / n_particles
    charges = np.full(n_particles, q_macro)
    base = uniform_cylinder(n_particles, A0, l_static, rng)
    t0 = time.time()
    amg = amg_pipe_solver(B_PIPE, B_OUT, h_amg, l_static + 0.08)
    t_setup = time.time() - t0
    fft = FFTPoissonSolver(h=h_amg, pad_cells=int(np.ceil((B_PIPE + 2 * h_amg) / h_amg)) + 2, use_gpu=use_gpu)
    x = np.linspace(-B_PIPE + 0.5e-3, B_PIPE - 0.5e-3, 161)
    offsets = [f * (B_PIPE - A0) for f in fractions]
    rows, profiles, t_amg = [], {}, []
    for f, d in zip(fractions, offsets):
        pos = base + np.array([d, 0.0, 0.0])
        t1 = time.time()
        phi_amg, e_amg = amg.solve(pos - 0.5 * h_amg, charges)          # -h/2: poisson_amg CIC offset
        t_amg.append(time.time() - t1)
        fft.solve(pos, charges)
        centre = np.array([[d, 0.0, 0.0]])
        e_c_amg = float(e_amg(centre)[0, 0])
        e_c_fft = float(fft.gather(centre)[0, 0])
        e_c_th = lam * d / (2.0 * np.pi * EPS0 * (B_PIPE ** 2 - d ** 2)) if d > 0 else 0.0
        rows.append([f, d, e_c_amg, e_c_fft, e_c_th])
        if f == fractions[0] or f == fractions[-1]:
            phi_a, ex_a = amg_line(amg, phi_amg, e_amg, x, h_amg)
            phi_f, ex_f = fft_line(fft, x)
            phi_f = phi_f - np.interp(B_PIPE - 0.5e-3, x, phi_f)         # free space: zero at the near wall
            profiles[f] = {'d': d, 'x': x, 'phi_amg': phi_a, 'ex_amg': ex_a, 'phi_fft': phi_f, 'ex_fft': ex_f,
                           'phi_th': phi_theory(x, d, A0, B_PIPE, lam), 'ex_th': ex_theory(x, d, A0, B_PIPE, lam)}
        if verbose:
            print(f"[side] d = {1e3 * d:5.2f} mm ({f:.2f} of b - a): E_x at the beam centre AMG {e_c_amg:9.1f}, "
                  f"theory {e_c_th:9.1f}, FFT {e_c_fft:9.1f} V/m  ({t_amg[-1]:.1f} s)")
    rows = np.asarray(rows)
    out = {'fractions': list(fractions), 'offsets_m': offsets, 'rows': rows, 'profiles': profiles, 'lambda': lam,
           'amg_setup_s': t_setup, 'amg_solve_s_mean': float(np.mean(t_amg)), 'amg_dofs': int(amg.n_active_dofs),
           'amg_cells': (amg.nx, amg.ny, amg.nz), 'h_amg': h_amg,
           'image_rel_dev': float(np.sqrt(np.mean(((rows[1:, 2] - rows[1:, 4]) / rows[1:, 4]) ** 2)))}
    if verbose:
        print(f"[side] AMG: {amg.nx}x{amg.ny}x{amg.nz} cells, {amg.n_active_dofs:,d} active DOFs, setup {t_setup:.1f} s, "
              f"{np.mean(t_amg):.1f} s per solve; image field rms deviation from theory {100 * out['image_rel_dev']:.1f} %")
    return out


# ================================================================ plots
def plot_drift(dr, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax = axes[0]
    ax.plot(dr['z'], 1e3 * dr['a_sim'], 'b-', lw=2, label=r'simulation: $\sqrt{2}\,r_{rms}$ (middle slice)')
    ax.plot(dr['z'], 1e3 * dr['edge'], 'c:', lw=1, label='simulation: outermost particle')
    ax.plot(dr['z'], 1e3 * dr['a_theory'], 'r--', lw=2, label=r"envelope $a'' = K/a$")
    ax.axhline(1e3 * B_PIPE, color='k', lw=1.5, label='pipe wall')
    ax.set_xlabel('drift z [m]')
    ax.set_ylabel('beam radius [mm]')
    ax.set_ylim(0, 1e3 * B_PIPE * 1.1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_title(f"{ION.name} {E_KIN_EV / 1e3:.0f} keV, I = {1e3 * CURRENT_A:.0f} mA, a0 = {1e3 * A0:.0f} mm: "
                 f"K = {dr['K']:.2e}\nfinal a = {dr['a_final_sim_mm']:.2f} mm (theory {dr['a_final_theory_mm']:.2f}), "
                 f"rms dev {100 * dr['rms_rel_dev']:.2f} %", fontsize=10)
    for ax, (pos, lab, col) in zip(axes[1:], ((dr['pos_initial'], 'z = 0', 'tab:blue'),
                                              (dr['pos_final'], f'z = {L_DRIFT:.1f} m', 'tab:red'))):
        mid = np.abs(pos[:, 2] - pos[:, 2].mean()) < L_BEAM / 4        # the beam has moved along z
        ax.plot(1e3 * pos[mid, 0], 1e3 * pos[mid, 1], '.', color=col, ms=1.5, alpha=0.5)
        ax.add_patch(plt.Circle((0, 0), 1e3 * B_PIPE, fill=False, color='k', lw=2))
        ax.add_patch(plt.Circle((0, 0), 1e3 * np.sqrt(2) * np.sqrt(np.mean(pos[mid, 0] ** 2 + pos[mid, 1] ** 2)),
                                fill=False, color='g', lw=1.5, ls='--'))
        ax.set_aspect('equal')
        ax.set_xlim(-1e3 * B_OUT, 1e3 * B_OUT)
        ax.set_ylim(-1e3 * B_OUT, 1e3 * B_OUT)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_title(f"cross-section of the middle slice at {lab} (green: $\\sqrt{{2}}\\,r_{{rms}}$)", fontsize=10)
    fig.suptitle('Part A - uniform beam drifting in a grounded pipe, FFT solver (free space) vs envelope theory', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / 'drift.png', dpi=130)
    plt.close(fig)


def plot_sideways(sd, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = sd['rows']
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax = axes[0, 0]
    ax.add_patch(plt.Circle((0, 0), 1e3 * B_PIPE, fill=False, color='k', lw=2.5, label='pipe wall'))
    cmap = plt.get_cmap('viridis')
    for i, (f, d) in enumerate(zip(sd['fractions'], sd['offsets_m'])):
        ax.add_patch(plt.Circle((1e3 * d, 0), 1e3 * A0, fill=True, alpha=0.35, color=cmap(i / max(len(rows) - 1, 1)),
                                ec='k', lw=0.8))
        row = i % 3                                                  # three label rows: no overlaps
        y_lab = (1e3 * (A0 + 1.0e-3) + 2.0 * row) if i % 2 == 0 else (-1e3 * (A0 + 1.0e-3) - 1.8 - 2.0 * row)
        ax.annotate(f"{f:.2f}", (1e3 * d, y_lab), ha='center', fontsize=8)
    ax.set_aspect('equal')
    ax.set_xlim(-1e3 * B_OUT, 1e3 * B_OUT)
    ax.set_ylim(-1e3 * B_OUT, 1e3 * B_OUT)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title(f"beam moved sideways by d = f (b - a), f = {sd['fractions'][0]:.2f} .. {sd['fractions'][-1]:.2f}", fontsize=10)
    ax.legend(loc='upper left', fontsize=9)

    ax = axes[0, 1]
    ax.plot(1e3 * rows[:, 1], rows[:, 4], 'r--', lw=2, label=r'theory: $\lambda d\,/\,2\pi\epsilon_0 (b^2 - d^2)$')
    ax.plot(1e3 * rows[:, 1], rows[:, 2], 'ko', ms=6, label='SA-AMG, pipe as conductor')
    ax.plot(1e3 * rows[:, 1], rows[:, 3], 'bs', ms=5, label='FFT, free space (no images)')
    ax.set_xlabel('beam offset d [mm]')
    ax.set_ylabel('$E_x$ at the beam centre [V/m]')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f"image force on the beam (rms deviation AMG vs theory {100 * sd['image_rel_dev']:.1f} %)", fontsize=10)

    for ax, key, what in ((axes[1, 0], 'phi', 'potential [V]'), (axes[1, 1], 'ex', '$E_x$ [V/m]')):
        for f, ls in ((sd['fractions'][0], '-'), (sd['fractions'][-1], '--')):
            p = sd['profiles'][f]
            lab = f"d = {1e3 * p['d']:.1f} mm"
            ax.plot(1e3 * p['x'], p[key + '_th'], color='r', ls=ls, lw=2.5, alpha=0.6, label=f'theory, {lab}')
            ax.plot(1e3 * p['x'], p[key + '_amg'], color='k', ls=ls, lw=1.2, label=f'SA-AMG, {lab}')
            ax.plot(1e3 * p['x'], p[key + '_fft'], color='b', ls=ls, lw=1.2, label=f'FFT, {lab}')
        for xx in (-1e3 * B_PIPE, 1e3 * B_PIPE):
            ax.axvline(xx, color='k', lw=1.5)
        ax.set_xlabel('x along the offset axis [mm]')
        ax.set_ylabel(what)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    axes[1, 0].set_title('potential along the offset axis (FFT: free space, set to 0 at the near wall)', fontsize=10)
    axes[1, 1].set_title('field along the offset axis', fontsize=10)
    fig.suptitle(f"Part B/C - beam (a = {1e3 * A0:.0f} mm, {1e3 * CURRENT_A:.0f} mA) in a grounded pipe (b = {1e3 * B_PIPE:.0f} mm): "
                 f"FFT (free space) vs SA-AMG (pipe) vs 2D image theory", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / 'sideways.png', dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--no-gpu', action='store_true')
    ap.add_argument('--fast', action='store_true', help='fewer particles and steps')
    ap.add_argument('--no-amg', action='store_true', help='skip the pipe (AMG) parts')
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'results' / 'space_charge_beam_in_pipe'))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    n_part, n_steps = (20000, 150) if args.fast else (60000, 400)

    dr = run_drift(n_part, n_steps, not args.no_gpu, rng)
    dr['pos_initial'] = uniform_cylinder(n_part, A0, L_BEAM, np.random.default_rng(7))
    plot_drift(dr, out_dir)
    summary = {'drift': {k: (v.tolist() if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) < 2000 else v)
                         for k, v in dr.items() if k not in ('pos_final', 'pos_initial')}}
    summary['drift']['window_last'] = list(dr['window_last'])
    if not args.no_amg:
        sd = run_sideways(20000 if args.fast else 40000, not args.no_gpu, rng)
        plot_sideways(sd, out_dir)
        summary['sideways'] = {'fractions': sd['fractions'], 'offsets_m': sd['offsets_m'],
                               'rows_f_d_Eamg_Efft_Eth': sd['rows'].tolist(), 'image_rel_dev': sd['image_rel_dev'],
                               'amg_setup_s': sd['amg_setup_s'], 'amg_solve_s_mean': sd['amg_solve_s_mean'],
                               'amg_dofs': sd['amg_dofs'], 'amg_cells': list(sd['amg_cells']), 'h_amg': sd['h_amg']}
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    print(f"wrote {out_dir / 'drift.png'}, {out_dir / 'sideways.png'}, {out_dir / 'summary.json'}")


if __name__ == '__main__':
    main()

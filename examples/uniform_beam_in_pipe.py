"""
examples/uniform_beam_in_pipe.py

Example: Space-charge field solver for uniform cylindrical beam in conducting pipe.

This example demonstrates:
1. Creating a uniform cylindrical particle beam
2. Loading conductor geometry from STL file
3. Solving the Poisson equation for space-charge
4. Visualizing the potential distribution

Beam parameters:
- Cylindrical, 30 cm long, 5 cm diameter
- Total charge: 10 pC
- Uniform density

Conductor:
- Conducting pipe from beam_pipe.stl
- Grounded (φ = 0 on surface)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Adjust imports based on your project structure
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from PyPATools.poisson_amg import PyAMGPoissonSolver, PyAMGSolverConfig, create_solver
from py_electrodes.py_electrodes import PyElectrode, PyElectrodeAssembly


# ============================================================================
# Step 1: Create uniform cylindrical beam
# ============================================================================

def create_uniform_cylindrical_beam(n_particles=100000, length=0.3, radius=0.025,
                                    total_charge=10e-12):
    """
    Create uniform cylindrical particle beam.

    Parameters
    ----------
    n_particles : int
        Number of macro-particles
    length : float
        Beam length in meters (z-direction)
    radius : float
        Beam radius in meters
    total_charge : float
        Total charge in Coulombs

    Returns
    -------
    particles : np.ndarray(n_particles, 3)
        Particle positions in meters
    charges : np.ndarray(n_particles,)
        Charge per macro-particle in Coulombs
    """

    print(f"\n{'=' * 70}")
    print(f"Creating Uniform Cylindrical Beam")
    print(f"{'=' * 70}")
    print(f"Number of particles: {n_particles:,d}")
    print(f"Length: {length * 100:.1f} cm")
    print(f"Diameter: {2 * radius * 100:.1f} cm")
    print(f"Total charge: {total_charge * 1e12:.1f} pC")

    # Random sampling in cylindrical volume
    # Use rejection sampling or direct cylindrical coordinates
    particles = []

    # Cylindrical coordinates: r, theta, z
    np.random.seed(42)

    # Direct sampling in cylindrical coordinates
    r_samples = np.sqrt(np.random.uniform(0, 1, n_particles)) * radius  # sqrt for uniform area
    theta_samples = np.random.uniform(0, 2 * np.pi, n_particles)
    z_samples = np.random.uniform(-length / 2, length / 2, n_particles)

    # Convert to Cartesian
    x_samples = r_samples * np.cos(theta_samples)
    y_samples = r_samples * np.sin(theta_samples)
    z_samples = z_samples

    particles = np.column_stack([x_samples, y_samples, z_samples])

    # Charge per macro-particle
    charge_per_particle = total_charge / n_particles
    charges = np.full(n_particles, charge_per_particle)

    print(f"Charge per macro-particle: {charge_per_particle:.2e} C")
    print(f"Beam volume: {np.pi * radius ** 2 * length:.3e} m³")
    print(f"Charge density: {total_charge / (np.pi * radius ** 2 * length):.3e} C/m³")

    return particles, charges


# ============================================================================
# Step 2: Create electrode assembly from STL
# ============================================================================

def create_electrode_assembly(filename="beam_pipe.brep"):
    """
    Load conducting pipe from BREP file and create electrode assembly.

    Parameters
    ----------
    filename : str
        Path to BREP file (relative to examples/ directory)

    Returns
    -------
    PyElectrodeAssembly
        Assembly containing the conducting pipe
    """

    print(f"\n{'=' * 70}")
    print(f"Loading Conductor Geometry")
    print(f"{'=' * 70}")

    # Create PyElectrode from BREP
    electrode = PyElectrode(name="Beam Pipe",
                            voltage=0)
    electrode.brep_h = 0.01
    electrode.generate_from_file(filename)

    print(f"Electrode loaded: {electrode.name}")
    print(f"Potential: {electrode.voltage} V")

    # Create assembly
    assembly = PyElectrodeAssembly()
    assembly.add_electrode(electrode)
    assembly.show(show_screen=True)

    print(f"Assembly contains {len(assembly.electrodes)} electrode(s)")

    return assembly


# ============================================================================
# Step 3: Solve for space-charge field
# ============================================================================

def solve_space_charge_field(particles, charges, electrode_assembly):
    """
    Solve Poisson equation for space-charge field.

    Parameters
    ----------
    particles : np.ndarray(N, 3)
        Particle positions in meters
    charges : np.ndarray(N,)
        Particle charges in Coulombs
    electrode_assembly : PyElectrodeAssembly
        Conductor geometry

    Returns
    -------
    phi_3d : np.ndarray(nx, ny, nz)
        Potential on mesh grid
    E_field : Field
        Electric field object with interpolators
    """

    print(f"\n{'=' * 70}")
    print(f"Setting Up Field Solver")
    print(f"{'=' * 70}")

    # Domain: ±0.25 m in z, ±0.065 m in x and y
    config = PyAMGSolverConfig(
        domain_extent=(0.13, 0.13, 0.5),  # (Lx, Ly, Lz) in meters
        mesh_cells=(64, 64, 128),  # Resolution
        amg_strength=0.25,
        solver_tol=1e-6,
        use_gpu=True,
    )

    solver = PyAMGPoissonSolver(config,
                                electrode_assembly=electrode_assembly)

    solver.debug_visualize_cell_classification()
    exit()

    print(f"\n{'=' * 70}")
    print(f"Solving Poisson Equation")
    print(f"{'=' * 70}")

    # Solve
    phi_3d, E_field = solver.solve(particles, charges)

    return phi_3d, E_field, solver


# ============================================================================
# Step 4: Visualization
# ============================================================================

def plot_potential_slices(phi_3d, solver, output_dir=None):
    """
    Plot potential along x-axis and z-axis.

    Parameters
    ----------
    phi_3d : np.ndarray(nx, ny, nz)
        Potential on mesh grid
    solver : PyAMGPoissonSolver
        Solver (contains grid info)
    output_dir : str, optional
        Directory to save plots (if None, just display)
    """

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Grid coordinates (in meters)
    x_grid = np.linspace(-solver.Lx / 2, solver.Lx / 2, solver.nx)
    y_grid = np.linspace(-solver.Ly / 2, solver.Ly / 2, solver.ny)
    z_grid = np.linspace(-solver.Lz / 2, solver.Lz / 2, solver.nz)

    # Center indices
    ix_center = solver.nx // 2
    iy_center = solver.ny // 2
    iz_center = solver.nz // 2

    # ====================================================================
    # Plot 1: Potential along x-axis (y=0, z=0)
    # ====================================================================

    phi_x = phi_3d[:, iy_center, iz_center]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_grid * 1000, phi_x, 'b-', linewidth=2, label='φ(x, y=0, z=0)')
    ax.set_xlabel('x (mm)', fontsize=12)
    ax.set_ylabel('Potential (V)', fontsize=12)
    ax.set_title('Space-Charge Potential Along X-Axis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    if output_dir:
        fig.savefig(output_dir / 'potential_x_axis.png', dpi=150, bbox_inches='tight')

    print(f"Potential along x-axis:")
    print(f"  Min: {phi_x.min():.3e} V")
    print(f"  Max: {phi_x.max():.3e} V")

    # ====================================================================
    # Plot 2: Potential along z-axis (x=0, y=0)
    # ====================================================================

    phi_z = phi_3d[ix_center, iy_center, :]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(z_grid * 1000, phi_z, 'r-', linewidth=2, label='φ(x=0, y=0, z)')
    ax.set_xlabel('z (mm)', fontsize=12)
    ax.set_ylabel('Potential (V)', fontsize=12)
    ax.set_title('Space-Charge Potential Along Z-Axis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    if output_dir:
        fig.savefig(output_dir / 'potential_z_axis.png', dpi=150, bbox_inches='tight')

    print(f"Potential along z-axis:")
    print(f"  Min: {phi_z.min():.3e} V")
    print(f"  Max: {phi_z.max():.3e} V")

    # ====================================================================
    # Plot 3: 2D slice in x-y plane (z=0)
    # ====================================================================

    phi_xy = phi_3d[:, :, iz_center]

    fig, ax = plt.subplots(figsize=(10, 9))

    x_mesh, y_mesh = np.meshgrid(x_grid * 1000, y_grid * 1000, indexing='ij')
    contour = ax.contourf(x_mesh, y_mesh, phi_xy, levels=20, cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax, label='Potential (V)')

    # Add circle to show beam outline
    circle = plt.Circle((0, 0), 2.5, fill=False, edgecolor='red', linewidth=2,
                        linestyle='--', label='Beam boundary (2.5 cm radius)')
    ax.add_patch(circle)

    ax.set_xlabel('x (mm)', fontsize=12)
    ax.set_ylabel('y (mm)', fontsize=12)
    ax.set_title('Space-Charge Potential in X-Y Plane (z=0)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(fontsize=11)

    if output_dir:
        fig.savefig(output_dir / 'potential_xy_plane.png', dpi=150, bbox_inches='tight')

    # ====================================================================
    # Plot 4: 2D slice in r-z plane (y=0)
    # ====================================================================

    phi_xz = phi_3d[:, iy_center, :]

    fig, ax = plt.subplots(figsize=(12, 8))

    x_mesh, z_mesh = np.meshgrid(x_grid * 1000, z_grid * 1000, indexing='ij')
    contour = ax.contourf(x_mesh, z_mesh, phi_xz, levels=20, cmap='plasma')
    cbar = plt.colorbar(contour, ax=ax, label='Potential (V)')

    # Add rectangle to show beam outline
    beam_rect = plt.Rectangle((-2.5, -15), 5, 30, fill=False, edgecolor='cyan',
                              linewidth=2, linestyle='--', label='Beam boundary')
    ax.add_patch(beam_rect)

    ax.set_xlabel('x (mm)', fontsize=12)
    ax.set_ylabel('z (mm)', fontsize=12)
    ax.set_title('Space-Charge Potential in X-Z Plane (y=0)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)

    if output_dir:
        fig.savefig(output_dir / 'potential_xz_plane.png', dpi=150, bbox_inches='tight')

    plt.show()


# ============================================================================
# Main
# ============================================================================

def main():
    """Run complete example"""

    print("\n" + "=" * 70)
    print("UNIFORM BEAM IN CONDUCTING PIPE - SPACE-CHARGE EXAMPLE")
    print("=" * 70)

    # Step 1: Create beam
    particles, charges = create_uniform_cylindrical_beam(
        n_particles=100000,
        length=0.3,  # m
        radius=0.025,  # m
        total_charge=10e-12,  # 10 pC
    )

    # Step 2: Load conductor
    electrode_assembly = create_electrode_assembly(filename=r"D:\Dropbox (Personal)\Code\Python\PyPATools\examples\beam_pipe.brep")

    # Step 3: Solve
    phi_3d, E_field, solver = solve_space_charge_field(
        particles, charges, electrode_assembly
    )

    # Step 4: Plot
    print(f"\n{'=' * 70}")
    print(f"Generating Plots")
    print(f"{'=' * 70}")

    output_dir = Path(__file__).parent / "results" / "uniform_beam_in_pipe"
    plot_potential_slices(phi_3d, solver, output_dir=output_dir)

    print(f"\nPlots saved to: {output_dir}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"Example Complete")
    print(f"{'=' * 70}")
    print(f"Field object: {E_field}")
    print(f"You can now use E_field to evaluate the field at any position:")
    print(f"  E_at_point = E_field(np.array([[0.01, 0.01, 0.01]]))")

    return phi_3d, E_field, solver, particles, charges


if __name__ == "__main__":
    phi_3d, E_field, solver, particles, charges = main()
"""
example_muon_trajectory.py - Single Particle Trajectory Visualization

Demonstrates tracking a muon through a magnetic field and visualizing
the trajectory overlaid on the field map.

Usage:
    python example_muon_trajectory.py
"""

import numpy as np
import matplotlib.pyplot as plt
from PyPATools.field import Field
from PyPATools.pusher import Pusher
from PyPATools.species import IonSpecies
from PyPATools.field_src.field_visualization import plot_field_magnitude


def calculate_velocity_from_energy(species, energy_kev):
    """
    Calculate particle velocity from kinetic energy.

    Parameters
    ----------
    species : IonSpecies
        Ion species object
    energy_kev : float
        Kinetic energy in keV

    Returns
    -------
    velocity : float
        Particle velocity in m/s
    """
    # Energy in MeV
    energy_mev = energy_kev / 1000.0

    # Rest mass in MeV
    mass_mev = species.mass_mev

    # Relativistic calculation: E_kin = (gamma - 1) * m_0 * c^2
    gamma = energy_mev / mass_mev + 1.0
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)

    # Speed of light
    c = 299792458.0  # m/s

    velocity = beta * c

    return velocity


def track_muon_trajectory(field_filename='magnetic_field.pickle',
                          energy_kev=230.0,
                          x0_mm=75.0,
                          y0_mm=0.0,
                          nsteps=50000,
                          dt=1e-12,
                          show=True,
                          save=None):
    """
    Track a muon through a magnetic field and visualize.

    Parameters
    ----------
    field_filename : str
        Path to magnetic field file
    energy_kev : float
        Muon kinetic energy in keV (default: 30.0)
    x0_mm : float
        Initial x position in mm (default: 75.0)
    y0_mm : float
        Initial y position in mm (default: 0.0)
    nsteps : int
        Number of tracking steps (default: 10000)
    dt : float
        Time step in seconds (default: 1e-12)
    show : bool
        Display plot (default: True)
    save : str, optional
        Save filename

    Returns
    -------
    r, v : np.ndarray
        Trajectory positions and velocities
    """

    print("=" * 70)
    print("MUON TRAJECTORY CALCULATION")
    print("=" * 70)

    # Create muon species
    print("\n1. Creating muon species...")
    muon = IonSpecies('muon')
    print(muon)

    # Calculate velocity from energy
    print(f"\n2. Calculating velocity from kinetic energy ({energy_kev} keV)...")
    velocity = calculate_velocity_from_energy(muon, energy_kev)
    print(f"   Velocity: {velocity:.3e} m/s ({velocity / 299792458.0 * 100:.2f}% of c)")

    # Calculate magnetic rigidity
    gamma = energy_kev / (muon.mass_mev * 1000.0) + 1.0
    beta = velocity / 299792458.0
    p_betagamma = beta * gamma
    b_rho = p_betagamma * muon.mass_mev * 1e6 / (abs(muon.q) * 299792458.0)
    print(f"   Magnetic rigidity (Brho): {b_rho * 1000:.3f} Gauss-cm")

    # Load magnetic field
    print(f"\n3. Loading magnetic field from '{field_filename}'...")
    bfield = Field.from_file(field_filename)
    print(f"   Field: {bfield}")

    # Get field value at starting position
    x0 = x0_mm / 1000.0  # mm to m
    y0 = y0_mm / 1000.0
    _, _, bz_start = bfield([[x0, y0, 0.0]])
    print(f"   Field at start position: Bz = {bz_start:.4f} T")

    # Calculate expected Larmor radius
    if bz_start != 0:
        r_larmor = b_rho / abs(bz_start) / 100.0  # Gauss-cm / T to m
        print(f"   Expected Larmor radius: {r_larmor * 1000:.2f} mm")

    # Set initial conditions
    print("\n4. Setting initial conditions...")
    r0 = np.array([x0, y0, 0.0])  # m
    v0 = np.array([0.0, velocity, 0.0])  # m/s, entirely y-directed

    print(f"   Initial position: x={x0 * 1000:.1f} mm, y={y0 * 1000:.1f} mm, z=0.0 mm")
    print(f"   Initial velocity: vx=0, vy={velocity:.3e} m/s, vz=0")

    # Create pusher
    print("\n5. Creating particle pusher...")
    pusher = Pusher(muon, algorithm='boris')
    print(f"   Pusher: {pusher}")

    # Track particle
    print(f"\n6. Tracking for {nsteps} steps (dt={dt * 1e12:.1f} ps)...")
    print(f"   Total time: {nsteps * dt * 1e9:.2f} ns")

    # Zero electric field
    efield_zero = Field.zero()

    r, v = pusher.track(
        r0, v0,
        efield_func=lambda pts: efield_zero(pts),
        bfield_func=lambda pts: bfield(pts),
        nsteps=nsteps,
        dt=dt
    )

    print(f"   Tracked {len(r)} points")
    print(f"   Final position: x={r[-1, 0] * 1000:.2f} mm, y={r[-1, 1] * 1000:.2f} mm, z={r[-1, 2] * 1000:.2f} mm")

    # Analyze trajectory
    print("\n7. Analyzing trajectory...")

    # Check if particle stayed in xy-plane
    z_max = np.max(np.abs(r[:, 2]))
    print(f"   Max |z| deviation: {z_max * 1e6:.2f} um")

    # Calculate actual path
    distances = np.sqrt(np.sum(np.diff(r, axis=0) ** 2, axis=1))
    total_distance = np.sum(distances)
    print(f"   Total path length: {total_distance * 1000:.2f} mm")

    # Energy conservation check
    v_mag_initial = np.linalg.norm(v[0])
    v_mag_final = np.linalg.norm(v[-1])
    energy_change = abs(v_mag_final - v_mag_initial) / v_mag_initial * 100
    print(f"   Energy conservation: {100 - energy_change:.4f}% (change: {energy_change:.2e}%)")

    # Visualize
    print("\n8. Creating visualization...")

    # Create figure
    fig = plt.figure(figsize=(12, 10))

    # Plot 1: Field magnitude with trajectory
    ax1 = plt.subplot(2, 2, 1)

    # Determine plot limits from trajectory
    x_min, x_max = np.min(r[:, 0]), np.max(r[:, 0])
    y_min, y_max = np.min(r[:, 1]), np.max(r[:, 1])

    # Add margin
    margin = 0.01  # 1 cm
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)

    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2

    plot_limits = (
        (x_center - max_range / 2 - margin, x_center + max_range / 2 + margin),
        (y_center - max_range / 2 - margin, y_center + max_range / 2 + margin)
    )

    # Plot field magnitude
    coord1 = np.linspace(plot_limits[0][0], plot_limits[0][1], 100)
    coord2 = np.linspace(plot_limits[1][0], plot_limits[1][1], 100)
    C1, C2 = np.meshgrid(coord1, coord2, indexing='ij')

    pts = np.column_stack([C1.ravel(), C2.ravel(), np.zeros(len(C1.ravel()))])
    fx, fy, fz = bfield(pts)
    B_mag = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2).reshape(100, 100)

    contour = ax1.contourf(C1, C2, B_mag, levels=20, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, ax=ax1, label='|B| (T)')

    # Overlay trajectory
    ax1.plot(r[:, 0], r[:, 1], 'r-', linewidth=2, label='Muon trajectory')
    ax1.plot(r[0, 0], r[0, 1], 'go', markersize=10, label='Start', zorder=5)
    ax1.plot(r[-1, 0], r[-1, 1], 'rs', markersize=10, label='End', zorder=5)

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title(f'Trajectory on |B| Field Map')
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Trajectory in xy-plane (no field)
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(r[:, 0] * 1000, r[:, 1] * 1000, 'b-', linewidth=2)
    ax2.plot(r[0, 0] * 1000, r[0, 1] * 1000, 'go', markersize=10, label='Start')
    ax2.plot(r[-1, 0] * 1000, r[-1, 1] * 1000, 'rs', markersize=10, label='End')
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('y (mm)')
    ax2.set_title('Trajectory (xy-plane)')
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Position vs time
    ax3 = plt.subplot(2, 2, 3)
    time_ns = np.arange(len(r)) * dt * 1e9
    ax3.plot(time_ns, r[:, 0] * 1000, label='x')
    ax3.plot(time_ns, r[:, 1] * 1000, label='y')
    ax3.plot(time_ns, r[:, 2] * 1e6, label='z (um scale)')
    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel('Position (mm)')
    ax3.set_title('Position vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Velocity magnitude vs time
    ax4 = plt.subplot(2, 2, 4)
    v_mag = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2)
    ax4.plot(time_ns, v_mag / 1e6, 'k-', linewidth=1.5)
    ax4.set_xlabel('Time (ns)')
    ax4.set_ylabel('Speed (Mm/s)')
    ax4.set_title('Speed vs Time (Energy Conservation Check)')
    ax4.grid(True, alpha=0.3)

    # Add text with parameters
    param_text = (
        f"Species: Muon ($\\mu^-$)\n"
        f"Energy: {energy_kev:.1f} keV\n"
        f"Start: ({x0_mm:.1f}, {y0_mm:.1f}) mm\n"
        f"Steps: {nsteps}, dt={dt * 1e12:.1f} ps"
    )
    ax4.text(0.02, 0.98, param_text, transform=ax4.transAxes,
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save is not None:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"\n   Saved figure to {save}")

    if show:
        plt.show()

    print("\n" + "=" * 70)
    print("TRACKING COMPLETE")
    print("=" * 70)

    return r, v


if __name__ == "__main__":
    import sys

    # Check if field file provided as argument
    if len(sys.argv) > 1:
        field_file = sys.argv[1]
    else:
        # Try to use a fixture file
        field_file = r"../backup/uCyclo_v2_Midplane_Res0.5mm_400x400mm.comsol"

    print(f"Using field file: {field_file}")
    print()

    # Run tracking
    try:
        r, v = track_muon_trajectory(
            field_filename=field_file,
            energy_kev=230.0,
            x0_mm=75.0,
            y0_mm=0.0,
            nsteps=50000,
            dt=1e-12,
            save='muon_trajectory.png'
        )
    except FileNotFoundError:
        print(f"\nError: Field file '{field_file}' not found!")
        print("\nPlease provide a field file, for example:")
        print("  python example_muon_trajectory.py path/to/field.pickle")
        print("\nOr generate fixtures first:")
        print("  python generate_fixtures.py")
        print("  python example_muon_trajectory.py fixtures/solenoid_field_3d.pickle")
        sys.exit(1)

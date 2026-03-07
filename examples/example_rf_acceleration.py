"""
example_rf_acceleration.py - Demonstrate RF Cavity Acceleration

Shows a particle gaining energy by crossing RF cavities in a magnetic field.
Uses manual tracking loop to check cavity crossings without modifying pusher.py

Usage:
    python example_rf_acceleration.py
"""

import numpy as np
import matplotlib.pyplot as plt
from PyPATools.field import Field
from PyPATools.pusher import Pusher
from PyPATools.species import IonSpecies
from rf_cavity import RFCavity


def track_with_rf_cavities(pusher, r0, v0, efield_func, bfield_func,
                           nsteps, dt, rf_cavities=None,
                           boundary_check=None, record_every=1):
    """
    Track particle with RF cavity support (manual implementation).

    This is a standalone version that doesn't require modifications to pusher.py

    Parameters
    ----------
    pusher : Pusher
        Pusher object with particle species
    r0, v0 : np.ndarray
        Initial position and velocity
    efield_func, bfield_func : callable
        Field functions
    nsteps : int
        Number of steps
    dt : float
        Time step
    rf_cavities : list of RFCavity, optional
        RF cavities to apply
    boundary_check : callable, optional
        Boundary checking function
    record_every : int
        Recording interval

    Returns
    -------
    r_array, v_array : np.ndarray
        Trajectory arrays
    """
    # Storage
    n_records = nsteps // record_every + 1
    r_array = np.zeros((n_records, 3))
    v_array = np.zeros((n_records, 3))

    # Initialize
    r = r0.copy()
    v = v0.copy()
    r_prev = r0.copy()
    t = 0.0

    # For Boris: initialize velocity at half-step back
    if pusher.algorithm == 'boris':
        ef = pusher._ensure_field_array(efield_func(r.reshape(1, 3)))
        bf = pusher._ensure_field_array(bfield_func(r.reshape(1, 3)))
        _, v = pusher.push(r, v, ef, bf, -0.5 * dt)

    # Store initial
    r_array[0] = r
    v_array[0] = v
    record_idx = 1

    # Main tracking loop
    for step in range(nsteps):
        r_prev = r.copy()

        # Get fields
        ef = pusher._ensure_field_array(efield_func(r.reshape(1, 3)))
        bf = pusher._ensure_field_array(bfield_func(r.reshape(1, 3)))

        # Push particle
        r, v = pusher.push(r, v, ef, bf, dt)
        t += dt

        # Check RF cavities
        if rf_cavities is not None:
            for cavity in rf_cavities:
                v, crossed, dE = cavity.apply_kick_if_crossing(
                    r_prev, r, v, t,
                    pusher.ion.charge, pusher.ion.mass_kg
                )

        # Check boundary
        if boundary_check is not None:
            if boundary_check(r):
                r_array = r_array[:record_idx]
                v_array = v_array[:record_idx]
                break

        # Record
        if (step + 1) % record_every == 0:
            if record_idx < n_records:
                r_array[record_idx] = r
                v_array[record_idx] = v
                record_idx += 1

    # For Boris: final half-step
    if pusher.algorithm == 'boris' and (boundary_check is None or record_idx == n_records):
        ef = pusher._ensure_field_array(efield_func(r.reshape(1, 3)))
        bf = pusher._ensure_field_array(bfield_func(r.reshape(1, 3)))
        _, v = pusher.push(r, v, ef, bf, 0.5 * dt)
        v_array[-1] = v

    return r_array, v_array


def example_rf_acceleration(field_filename='fixtures/constant_field_3d.pickle',
                            initial_energy_kev=30.0,
                            show=True,
                            save='rf_acceleration_example.png'):
    """
    Demonstrate particle acceleration with RF cavities.
    """

    print("=" * 70)
    print("RF CAVITY ACCELERATION EXAMPLE")
    print("=" * 70)

    # Create muon
    muon = IonSpecies('muon')
    print(f"\n1. Particle: {muon.name}")
    print(f"   Mass: {muon.mass_mev:.3f} MeV/c^2")
    print(f"   Charge: {muon.q}e")

    # Load field
    print(f"\n2. Loading field from {field_filename}...")
    bfield = Field.from_file(field_filename)

    # Create RF cavities
    print("\n3. Creating RF cavities...")
    cavities = []

    init_phi = -24*4+180

    for i, ang in enumerate([24, 66, 24+90, 66+90, 24+180, 66+180, 24+270, 66+270]):

        x = 0.4 * np.cos(np.deg2rad(ang))
        y = 0.4 * np.sin(np.deg2rad(ang))

        phase = init_phi if i%2==0 else init_phi+180

        cavities.append(RFCavity(
            p1=np.array([0.0, 0.0, 0.0]),  # Cavity line from y=-10cm
            p2=np.array([x, y, 0.0]),  # to y=+10cm at x=0
            voltage=60000.0,  # 60 kV
            frequency=168e6,  # 168 MHz
            phase=phase,  # 0 degrees
            transmission=1.0  # 100% transmission
        ))
    print(cavities)

    # Calculate initial velocity
    energy_mev = initial_energy_kev / 1000.0
    gamma = energy_mev / muon.mass_mev + 1.0
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    velocity = beta * 299792458.0

    print(f"\n4. Initial conditions:")
    print(f"   Energy: {initial_energy_kev:.1f} keV")
    print(f"   Velocity: {velocity:.3e} m/s ({beta * 100:.2f}% of c)")

    # Initial position and velocity
    r0 = np.array([0.075, 0.0, 0.0])  # Start at x=75mm
    v0 = np.array([0.0, velocity, 0.0])  # y-directed

    # Create pusher
    pusher = Pusher(muon, algorithm='boris')

    # Track with RF cavities
    print(f"\n5. Tracking particle with RF acceleration...")
    nsteps = 100000
    dt = 1e-12

    efield_zero = Field.zero()

    r, v = track_with_rf_cavities(
        pusher, r0, v0,
        efield_func=lambda pts: efield_zero(pts),
        bfield_func=lambda pts: bfield(pts),
        nsteps=nsteps,
        dt=dt,
        rf_cavities=cavities
    )

    print(f"   Tracked {len(r)} steps")
    print(f"   Total time: {len(r) * dt * 1e9:.2f} ns")

    # Calculate energy vs time
    v_mag = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2)
    gamma_array = 1.0 / np.sqrt(1.0 - (v_mag / 299792458.0) ** 2)
    energy_kev = (gamma_array - 1.0) * muon.mass_mev * 1000.0

    # Get cavity statistics
    print(f"\n6. RF Cavity Statistics:")
    stats = cavities[0].get_statistics()
    print(f"   Crossings: {stats['n_crossings']}")
    print(f"   Total energy gain: {stats['total_energy_gain_keV']:.2f} keV")
    if stats['n_crossings'] > 0:
        print(f"   Average energy per crossing: {stats['average_energy_gain_keV']:.2f} keV")
    print(f"   Final energy: {energy_kev[-1]:.2f} keV")
    print(f"   Energy increase: {energy_kev[-1] - energy_kev[0]:.2f} keV")

    # Visualize
    print(f"\n7. Creating visualization...")

    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Trajectory with field and cavity
    ax1 = plt.subplot(2, 3, 1)

    # Plot field magnitude
    extent = 0.4
    x_plot = np.linspace(-extent, extent, 100)
    y_plot = np.linspace(-extent, extent, 100)
    X, Y = np.meshgrid(x_plot, y_plot, indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(len(X.ravel()))])
    fx, fy, fz = bfield(pts)
    B_mag = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2).reshape(100, 100)

    contour = ax1.contourf(X, Y, B_mag, levels=20, cmap='viridis', alpha=0.5)
    plt.colorbar(contour, ax=ax1, label='|B| (T)')

    # Plot cavity
    for cavity in cavities:
        ax1.plot([cavity.p1[0], cavity.p2[0]], [cavity.p1[1], cavity.p2[1]],
                 'r-', linewidth=4, label='RF Cavity', zorder=5)

    # Plot trajectory
    ax1.plot(r[:, 0], r[:, 1], 'b-', linewidth=1, alpha=0.7, label='Trajectory')
    ax1.plot(r[0, 0], r[0, 1], 'go', markersize=10, label='Start', zorder=10)
    ax1.plot(r[-1, 0], r[-1, 1], 'rs', markersize=10, label='End', zorder=10)

    # Mark cavity crossings
    # crossing_indices = []
    # for i in range(1, len(r)):
    #     crossed, _ = cavity.check_crossing(r[i - 1], r[i])
    #     if crossed:
    #         crossing_indices.append(i)
    #
    # if len(crossing_indices) > 0:
    #     crossing_pts = r[crossing_indices]
    #     ax1.plot(crossing_pts[:, 0], crossing_pts[:, 1], 'mo',
    #              markersize=6, label=f'Crossings ({len(crossing_indices)})', zorder=8)

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Trajectory with RF Cavity')
    ax1.set_aspect('equal')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Energy vs time
    ax2 = plt.subplot(2, 3, 2)
    time_ns = np.arange(len(energy_kev)) * dt * 1e9
    ax2.plot(time_ns, energy_kev, 'b-', linewidth=2)
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Energy (keV)')
    ax2.set_title('Energy Gain from RF Acceleration')
    ax2.grid(True, alpha=0.3)

    # Mark cavity crossings
    # if len(crossing_indices) > 0:
    #     crossing_times = np.array(crossing_indices) * dt * 1e9
    #     crossing_energies = energy_kev[crossing_indices]
    #     ax2.plot(crossing_times, crossing_energies, 'ro',
    #              markersize=4, label='Cavity crossings')
    #     ax2.legend()

    # Plot 3: Speed vs time
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(time_ns, v_mag / 1e6, 'k-', linewidth=1.5)
    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel('Speed (Mm/s)')
    ax3.set_title('Particle Speed vs Time')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Radius vs time
    ax4 = plt.subplot(2, 3, 4)
    radius = np.sqrt(r[:, 0] ** 2 + r[:, 1] ** 2)
    ax4.plot(time_ns, radius * 1000, 'g-', linewidth=1.5)
    ax4.set_xlabel('Time (ns)')
    ax4.set_ylabel('Radius (mm)')
    ax4.set_title('Orbit Radius vs Time')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Energy gain per crossing
    ax5 = plt.subplot(2, 3, 5)
    # if len(crossing_indices) > 1:
    #     energy_gains = np.diff(energy_kev[crossing_indices])
    #     crossing_numbers = np.arange(1, len(energy_gains) + 1)
    #     ax5.plot(crossing_numbers, energy_gains, 'ro-', linewidth=2, markersize=6)
    #     ax5.set_xlabel('Crossing Number')
    #     ax5.set_ylabel('Energy Gain (keV)')
    #     ax5.set_title('Energy Gain per RF Crossing')
    #     ax5.grid(True, alpha=0.3)
    #     ax5.axhline(y=cavity.voltage * abs(muon.q) / 1000.0,
    #                 color='r', linestyle='--', alpha=0.5,
    #                 label=f'Peak voltage: {cavity.voltage / 1000:.0f} kV')
    #     ax5.legend()

    # Plot 6: Phase space (x-vx)
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(r[:, 0] * 1000, v[:, 0] / 1000, 'b-', linewidth=0.5, alpha=0.7)
    ax6.set_xlabel('x (mm)')
    ax6.set_ylabel('vx (km/s)')
    ax6.set_title('Phase Space (x-vx)')
    ax6.grid(True, alpha=0.3)

    # Add overall title
    fig.suptitle(f'RF Cavity Acceleration: {muon.name}, {initial_energy_kev:.0f} keV -> {energy_kev[-1]:.0f} keV',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"\n   Saved figure to {save}")

    if show:
        plt.show()

    print("\n" + "=" * 70)
    print("RF ACCELERATION EXAMPLE COMPLETE")
    print("=" * 70)

    return r, v, energy_kev


if __name__ == "__main__":
    import sys

    # Check if field file provided
    if len(sys.argv) > 1:
        field_file = sys.argv[1]
    else:
        field_file = r"../backup/uCyclo_v2_Midplane_Res0.5mm_400x400mm.comsol"

    print(f"Using field file: {field_file}\n")

    try:
        r, v, energy = example_rf_acceleration(
            field_filename=field_file,
            initial_energy_kev=230.0,
            save='rf_acceleration_example.png'
        )

        print("\nExample completed successfully!")
        print(f"Final energy: {energy[-1]:.2f} keV")
        print(f"Energy gain: {energy[-1] - energy[0]:.2f} keV")

    except FileNotFoundError:
        print(f"\nError: Field file '{field_file}' not found!")
        print("\nPlease generate fixtures first:")
        print("  python generate_fixtures.py")
        print("  python example_rf_acceleration.py fixtures/constant_field_3d.pickle")
        sys.exit(1)

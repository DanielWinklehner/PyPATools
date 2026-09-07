# Spiral inflector to central region hand-off: openPMD particle files

Spec version 1 (2026-09-07). Producer: `BunchTrack.py` / `BunchTrackSC.py` in the HCHC-60
spiral inflector deck (`--save-openpmd PATH --save-mode plane|lab6d|both`), writing through
`PyPATools.particles_src.particle_io.save_openpmd`. Consumer: PyCentralRegion.

Sample files (pg5L geometry, 43,969-particle core beam, no space charge):

```
D:\MIT Dropbox\Daniel Winklehner\Projects\IsoDAR\60 MeV Cyclotron\Spiral_inflector\Results\handoff\pg5L_vacuum_handoff.h5         (plane-crossing mode)
D:\MIT Dropbox\Daniel Winklehner\Projects\IsoDAR\60 MeV Cyclotron\Spiral_inflector\Results\handoff\pg5L_vacuum_handoff_lab6d.h5   (6-D snapshot mode)
```

## 1. The two modes

**plane_crossing** (the hand-off format). Every transmitted particle is recorded at the
moment it crosses a fixed plane (the *hand-off plane*), with its position on the plane, its
lab-frame momentum and its crossing time. The longitudinal coordinate is therefore the
crossing time (and the RF phase derived from it), not z. All particles lie exactly on the
plane; they are at *different times*.

**lab6d_snapshot** (the old way, kept as an option). Every transmitted particle at one
common time, wherever it is along its orbit (spread over roughly 15 mm of path). Same
records without the phase / plane extras.

Both are one HDF5 file, openPMD 2.0.0 base standard with the `BeamPhysics` and
`SpeciesType` extensions, one species group. They are readable with plain `h5py`, with
`PyPATools.particles_src.particle_io.load_openpmd`, and with openPMD-beamphysics
(`ParticleGroup(h5=path)`).

## 2. File layout

```
/                                   attrs: openPMD "2.0.0", openPMDextension "BeamPhysics;SpeciesType",
                                           basePath "/", particlesPath "particles", dataType "openPMD"
/particles/H2+/                     the species group (name = openPMD-beamphysics species name)
    attrs (standard):  numParticles, totalCharge [C], chargeUnitSI 1.0, speciesType "H2+"
    attrs (ours):      PyPATools:<key>  (see section 4)
    position/x, /y, /z              datasets (N,) float64, m           unitSI 1
    momentum/x, /y, /z              datasets (N,) float64, eV/c        unitSI 5.344286e-28 kg m/s
    time                            plane mode: dataset (N,) float64, s
                                    lab6d mode: constant record (group, attr value = the snapshot time)
    weight                          constant record: charge per macro-particle [C]
    particleStatus                  constant record: 1 (all alive)
    phase                           plane mode only: dataset (N,), rad, in (-pi, pi]
    t_rel                           plane mode only: dataset (N,), s   (= time - phase_reference_time_s)
    u, v                            plane mode only: datasets (N,), m  (in-plane coordinates)
```

"Constant record" is the openPMD idiom for a record whose value is the same for every
particle: a group carrying the attributes `value` and `shape` instead of a dataset. `time`
in lab6d mode, `weight` and `particleStatus` are written that way. The extra datasets
`phase`, `t_rel`, `u`, `v` carry `unitSI`, `unitDimension`, `timeOffset`, `weightingPower`
and `macroWeighted` attributes like any openPMD record.

## 3. Definitions

**Frame.** The spiral inflector deck frame: origin on the cyclotron axis in the median
plane, z along the axis. The beam travels in +z through the inflector and arrives at the
median plane z = 0; B_z is negative at the centre. In the deck's plots -z is drawn upwards
("the beam enters from the top"). Right-handed x, y, z. SI units throughout.

**Momentum.** Lab-frame momentum components in eV/c (openPMD-beamphysics convention):
p = beta*gamma * m c. Kinetic energy = sqrt(p^2 + m^2) - m with m = `PyPATools:species_mass_mev`
(1877.268 MeV for H2+). Velocity = p c^2 / E, direction = p / |p|. For the core beam the mean
kinetic energy is about 68.4 keV.

**Hand-off plane.** Origin o = `plane_origin_m`, unit normal n = `plane_normal`. The design
particle is tracked on from its exit point through the same fields; o is the point where its
path length past the exit reaches `handoff_distance_m` (30 mm by default, outside the electric
fringe of the electrodes) and n is its velocity direction there, so the plane is perpendicular
to the design orbit at the hand-off point. (The orbit bends by about 35 deg over those 30 mm, so
the exit direction itself, `design_exit_velocity_mps`, is not the normal.) Every particle
satisfies (r - o) . n = 0 to machine precision. The bunch centroid is within about 1 mm of o
and its mean momentum within about 2 deg of n (the tuning levelled the bunch, not the design
particle: in the sample n is tilted 1.7 deg towards -z, so the level bunch reads +1.7 deg in
the plane's vertical angle atan2(p . v, p . n)).

**In-plane axes.** u = (z_hat x n) / |z_hat x n|: horizontal, lying in the median plane.
v = n x u: within 1 deg of +z_hat, i.e. pointing along the beam's original direction of
travel, which is *downwards* in the -z-up pictures. (u, v, n) is right-handed. The stored
u, v datasets are (r - o) . u and (r - o) . v; a level, centred beam has mean v = 0.

**Time and phase.** `time` is the lab time of the crossing, counted from the start of the
inflector tracking (all particles started at t = 0 at the entrance, at about z = -0.275 m,
with their longitudinal spread expressed as a z spread). Only differences are meaningful.
`phase_reference_time_s` (t_ref) is the mean (or median, see `phase_reference`) crossing time
of the transmitted particles;

    t_rel = time - t_ref
    phase = wrap( 2 pi f_rf t_rel )  into (-pi, pi],   f_rf = rf_frequency_hz

so phase > 0 means the particle arrives *later* than the bunch centre. The offset between
this clock and the cyclotron RF is not known here: the RF phase of particle i at the plane is
phi_0 + phase_i, where phi_0 is a free parameter of the central region (the dee phase at
which the bunch centre crosses the plane).

**Charge.** `beam_current_ma` is the current of the injected core beam (8 mA), all of which
started at the inflector entrance. The file holds the *transmitted* particles only, so
`totalCharge` = (I / f_rf) x (N / N_tracked) = `bunch_charge_c`, and `weight` =
totalCharge / N is the charge per macro-particle. `transmitted_current_ma` is the
corresponding current, `injected_bunch_charge_c` the full I / f_rf.

**Species.** `speciesType` and the group name use the openPMD-beamphysics name (`H2+`);
`PyPATools:pypatools_species_name` keeps the PyPATools name (`H2_1+`), and
`PyPATools:species_mass_mev`, `species_a`, `species_charge_state` the exact numbers. beamphysics'
own H2+ mass differs from ours by 0.01 %, which changes a derived kinetic energy by a few eV.

## 4. `PyPATools:*` attributes on the species group

| attribute | type | meaning |
|---|---|---|
| `spec_version` | int | 1 |
| `mode` | str | `plane_crossing` or `lab6d_snapshot` |
| `frame`, `momentum_note` | str | the conventions above, in words |
| `plane_origin_m`, `plane_normal`, `plane_axis_u`, `plane_axis_v` | float[3] | hand-off plane (plane mode) |
| `handoff_distance_m` | float | design path length from the exit point to the plane (plane mode) |
| `handoff_definition` | str | how the plane was constructed, in words (plane mode) |
| `design_exit_point_m`, `design_exit_velocity_mps` | float[3] | the design particle where it leaves the electrodes |
| `rf_frequency_hz`, `bunch_freq_hz` | float | RF frequency used for the phase (both the same) |
| `phase_reference`, `phase_reference_time_s` | str, float | how t_ref was chosen and its value (plane mode) |
| `beam_current_ma`, `transmitted_current_ma` | float | injected core current; current carried by the particles in the file |
| `injected_bunch_charge_c`, `bunch_charge_c` | float | I / f_rf of the injected beam; charge in the file (= totalCharge) |
| `n_particles`, `n_tracked`, `transmission_through_housing` | int, int, float | particles in the file, particles started, ratio |
| `species_name`, `pypatools_species_name`, `species_mass_mev`, `species_a`, `species_charge_state` | | species |
| `source_tag`, `step_dir`, `space_charge`, `beam_rotation_deg`, `swap_xy` | | provenance: field tag, STEP folder of the geometry, space charge on/off, input beam rotation |
| `snapshot_step`, `snapshot_time_s` | int, float | lab6d mode only |

Array-valued attributes are HDF5 float arrays; strings are variable-length UTF-8 (h5py
returns `str`; some root attributes come back as `bytes`, decode them).

## 5. Reading

Plain h5py, no PyPATools needed:

```python
import h5py, numpy as np

def load_handoff(path):
    with h5py.File(path, "r") as f:
        sp = f[f.attrs["particlesPath"].decode() if isinstance(f.attrs["particlesPath"], bytes) else f.attrs["particlesPath"]]
        g = sp[list(sp.keys())[0]]                      # the single species group
        meta = {k[len("PyPATools:"):]: v for k, v in g.attrs.items() if k.startswith("PyPATools:")}
        def rec(name):                                   # dataset or constant record
            obj = g[name]
            if isinstance(obj, h5py.Dataset):
                return obj[...].astype(float)
            return np.full(int(np.asarray(obj.attrs["shape"]).ravel()[0]), float(obj.attrs["value"]))
        r = np.column_stack([rec("position/" + c) for c in "xyz"])       # m
        p = np.column_stack([rec("momentum/" + c) for c in "xyz"])       # eV/c
        t = rec("time")                                                  # s
        extra = {k: rec(k) for k in ("phase", "t_rel", "u", "v") if k in g}
        weight = rec("weight")                                           # C per macro-particle
    m = float(meta["species_mass_mev"]) * 1e6                           # eV
    ekin = np.sqrt(np.sum(p * p, axis=1) + m * m) - m                    # eV
    return r, p, t, extra, weight, ekin, meta
```

With PyPATools:

```python
from PyPATools.particles_src.particle_io import load_openpmd
r, betagamma, meta = load_openpmd(path)      # r (N,3) m, betagamma (N,3) dimensionless
# meta: "species" (IonSpecies), "time" (N,), "status" (N,), "extra_records" {"phase","t_rel","u","v"},
#       and every PyPATools:<key> attribute under <key> (arrays as numpy arrays)
```

With openPMD-beamphysics (energies from its own H2+ mass):

```python
from beamphysics import ParticleGroup
P = ParticleGroup(h5=path)     # P["mean_kinetic_energy"], P["sigma_t"], P.x, P.px, P.t, ...
```

## 6. Using the plane-crossing file in a tracker

Each particle i is a complete initial condition: position r_i on the plane, momentum p_i,
start time t_i (use `t_rel` so the bunch centre starts at 0). Two equivalent ways to
start:

1. Per-particle start times: create particle i at r_i, p_i at t = t_rel_i. If the tracker
   integrates everything on one clock, start the clock at min(t_rel) and keep particle i
   frozen (no force, no motion) until its own t_rel_i.
2. Common start time with phases: start every particle at the plane at t = 0 with RF phase
   phi_0 + phase_i. This is the same thing expressed through the RF; it is exact only if the
   fields the particle sees between t = 0 and t_rel_i are static (magnetic field only), which
   is the case 30 mm past the inflector where there is no electric field yet.

phi_0 (the dee phase at the bunch-centre crossing) is the knob the central region scans. The
lab6d file is for trackers that want all particles present at one instant; its particles
are spread along the orbit and the earliest ones may still see the last bit of the
inflector's fringe field.

Checks a loader can make: (r - o) . n = 0 within 1e-12 m; the stored u, v equal
(r - o) . u_hat, (r - o) . v_hat; mean kinetic energy 68 to 70 keV for the H2+ core beam;
phase inside (-pi, pi]; N = `numParticles` = `n_particles`.

## 7. Producer command lines

```
python BunchTrack.py   --tag d_direct --reload-dir <run dir> --step-dir <STEP dir> --phi 90 --n 50000 --particles <core file> \
                       --save-openpmd out.h5 --save-mode both [--handoff-distance 0.030] [--phase-reference mean|median] [--rf-mhz 32.8] [--current-ma 8]
python BunchTrackSC.py ... --sc --h 0.0015 --current-ma 8 --rf-mhz 32.8 --save-openpmd out_sc.h5 --save-mode plane
```

The hand-off plane sits `handoff_distance` of design path past the exit point, perpendicular
to the design orbit there; the crossing is interpolated inside the 0.1 ns tracking step (about
1 deg of RF phase), so the phase resolution is a small fraction of a degree. Particles are only recorded
if they crossed the inflector's exit plane first and did not hit the housing exit opening.
`--save-mode both` writes `<name>.h5` (plane) and `<name>_lab6d.h5` (snapshot at the recorded
step nearest the mean hand-off crossing).

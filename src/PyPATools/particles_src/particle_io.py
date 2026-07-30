"""
particle_io.py - Load and save particle distributions from various formats

Formats are held in a registry (FORMATS) so downstream packages can add their
own via register_format() without editing this file.

Implemented:
- openPMD standard, HDF5 backend (.h5) - the canonical inter-code handoff
  format of the suite. Uses the openPMD-beamphysics package (BeamPhysics
  extension records). Lossless for (positions [m], momenta [beta*gamma]).
- OPAL H5hut phase-space files (.h5), read only.
- Custom NumPy binary (.npz).

Deferred (stubs): TraceWin (.dst/.ini), AIMA (.lst), OPAL H5 writing.

Loader contract:  loader(filename, **kwargs) -> (positions (N,3) [m],
                  momenta (N,3) [beta*gamma], metadata dict)
Saver contract:   saver(filename, positions, momenta, species_data, **metadata)

Author: PyPATools Development Team
"""

import warnings
import numpy as np
import h5py
from typing import Tuple, Optional, Dict, Callable

# openPMD particleStatus convention: 1 = alive; anything else = lost.
# Downstream packages may assign negative codes to specific loss channels
# (e.g. septum foil, channel wall) and should document them in the file's
# attributes; readers must only rely on "== STATUS_ALIVE" for aliveness.
STATUS_ALIVE = 1
STATUS_LOST = 0

_ELEMENTARY_CHARGE = 1.602176634e-19  # C


# ------------------------------------------------------------------ registry

FORMATS: Dict[str, Dict] = {}


def register_format(name: str, loader: Optional[Callable] = None,
                    saver: Optional[Callable] = None,
                    extensions: Tuple[str, ...] = (),
                    sniffer: Optional[Callable] = None):
    """Register a particle file format.

    :param loader: loader(filename, **kwargs) -> (positions, momenta, metadata)
    :param saver: saver(filename, positions, momenta, species_data, **metadata)
    :param extensions: file extensions (with dot) claimed by this format
    :param sniffer: sniffer(filename) -> bool, content-based detection used to
        disambiguate shared extensions (checked before the extension map).
    """
    FORMATS[name] = {'loader': loader, 'saver': saver,
                     'extensions': tuple(extensions), 'sniffer': sniffer}


def detect_format(filename: str, default: str = 'npz') -> str:
    """Detect a registered format: content sniffers first, then extensions."""
    import os
    for name, spec in FORMATS.items():
        sniffer = spec['sniffer']
        try:
            if sniffer is not None and sniffer(filename):
                return name
        except (OSError, KeyError):
            continue
    ext = os.path.splitext(filename)[1].lower()
    for name, spec in FORMATS.items():
        if ext in spec['extensions']:
            return name
    return default


# ------------------------------------------------------------------- openPMD

def _is_openpmd_h5(filename: str) -> bool:
    if not h5py.is_hdf5(filename):
        return False
    with h5py.File(filename, 'r') as f:
        return 'openPMD' in f.attrs


def save_openpmd(filename: str, positions: np.ndarray, momenta: np.ndarray,
                 species_data: Dict, **metadata):
    """Save in openPMD (BeamPhysics extension) HDF5 format.

    Exact species parameters and any extra metadata survive as namespaced
    HDF5 attributes on the particle group, so custom ions (e.g. 'H2_1+')
    round-trip without degrading to the nearest standard speciesType.

    Recognized metadata keys: species (IonSpecies, ignored here - species_data
    is authoritative), bunch_charge [C], bunch_freq [Hz], status (N,) int,
    time (N,) or scalar [s]. Everything else is written as a
    'PyPATools:<key>' attribute if it is a scalar or string.
    """
    from beamphysics import ParticleGroup

    metadata.pop('species', None)
    n = len(positions)
    mass_ev = species_data['mass_mev'] * 1e6

    status = np.asarray(metadata.pop('status', np.full(n, STATUS_ALIVE)),
                        dtype=int)
    time = np.broadcast_to(np.asarray(metadata.pop('time', 0.0), dtype=float),
                           (n,)).copy()
    bunch_charge = float(metadata.pop('bunch_charge', 0.0))
    bunch_freq = float(metadata.pop('bunch_freq', 0.0))

    # weight = macro-charge per particle [C]; must be > 0 for beamphysics.
    if bunch_charge > 0.0:
        weight = np.full(n, bunch_charge / n)
    else:
        weight = np.full(n, abs(species_data.get('charge_state', 1))
                         * _ELEMENTARY_CHARGE)

    pg = ParticleGroup(data=dict(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
        px=momenta[:, 0] * mass_ev, py=momenta[:, 1] * mass_ev,
        pz=momenta[:, 2] * mass_ev,
        t=time, status=status, weight=weight,
        species=species_data.get('name', 'proton'),
    ))
    pg.write(filename)

    # Namespaced attributes for exact reconstruction + user metadata
    with h5py.File(filename, 'a') as f:
        grp = f[f.attrs['particlesPath'].decode()
                if isinstance(f.attrs['particlesPath'], bytes)
                else f.attrs['particlesPath']]
        (species_group,) = grp.values()
        for key, val in species_data.items():
            if val is not None:
                species_group.attrs[f'PyPATools:species_{key}'] = val
        species_group.attrs['PyPATools:bunch_freq_hz'] = bunch_freq
        species_group.attrs['PyPATools:bunch_charge_c'] = bunch_charge
        for key, val in metadata.items():
            if isinstance(val, (int, float, str, np.number)):
                species_group.attrs[f'PyPATools:{key}'] = val


def load_openpmd(filename: str, iteration: Optional[int] = None,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load from openPMD HDF5 (BeamPhysics extension layout)."""
    from beamphysics import ParticleGroup
    from ..species import IonSpecies

    pg = ParticleGroup(filename)

    # Recover exact species from our namespaced attributes if present
    metadata: Dict = {}
    with h5py.File(filename, 'r') as f:
        ppath = f.attrs['particlesPath']
        if isinstance(ppath, bytes):
            ppath = ppath.decode()
        (species_group,) = f[ppath].values()
        attrs = {k[len('PyPATools:'):]: v for k, v in species_group.attrs.items()
                 if k.startswith('PyPATools:')}

    name = attrs.pop('species_name', pg.species)
    try:
        species = IonSpecies(name,
                             a=attrs.pop('species_a', None),
                             z=attrs.pop('species_z', None),
                             q=attrs.pop('species_charge_state', None))
    except SystemExit:
        warnings.warn(f"Species '{name}' not reconstructible, using proton.")
        species = IonSpecies('proton')
    attrs.pop('species_mass_mev', None)

    mass_ev = species.mass_mev * 1e6
    positions = np.column_stack([pg.x, pg.y, pg.z])
    momenta = np.column_stack([pg.px, pg.py, pg.pz]) / mass_ev

    metadata['species'] = species
    metadata['bunch_charge'] = float(attrs.pop('bunch_charge_c', pg.charge))
    metadata['bunch_freq'] = float(attrs.pop('bunch_freq_hz', 0.0))
    metadata['status'] = pg.status
    metadata['time'] = pg.t
    metadata.update(attrs)  # remaining PyPATools:* user metadata
    return positions, momenta, metadata


# ------------------------------------------------------------------- OPAL H5

def _is_opal_h5(filename: str) -> bool:
    if not h5py.is_hdf5(filename):
        return False
    with h5py.File(filename, 'r') as f:
        return 'Step#0' in f and 'openPMD' not in f.attrs


def load_opal_h5(filename: str, step: int = -1, species=None,
                 frame: str = 'lab') -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load a phase-space snapshot from an OPAL H5hut file.

    OPAL stores momenta as beta*gamma already - no unit conversion needed.

    :param step: snapshot index. Negative indices count from the end
        (default -1 = last step). Uses the Step#N numbering in the file.
    :param species: IonSpecies or preset name. OPAL H5 files do not carry
        enough species information to reconstruct compound ions, so this
        should be supplied; defaults to 'proton' with a warning.
    :param frame: 'lab' returns coordinates as stored (x, y horizontal
        midplane, z vertical - OPAL-cycl global frame, the right choice for
        cyclotron tracking). 'paraxial' swaps y and z (and py, pz) for the
        linac convention used by ParticleDistribution's paraxial analysis
        (OPAL-cycl bunch-local frame has y longitudinal).
    """
    from ..species import IonSpecies

    if species is None:
        warnings.warn("OPAL H5 carries no full species definition; "
                      "defaulting to proton. Pass species= to override.")
        species = IonSpecies('proton')
    elif isinstance(species, str):
        species = IonSpecies(species)

    with h5py.File(filename, 'r') as f:
        step_ids = sorted(int(k.split('#')[1]) for k in f.keys()
                          if k.startswith('Step#'))
        if not step_ids:
            raise ValueError(f"No Step#N groups found in {filename}")
        step_id = step_ids[step] if step < 0 else step
        if step_id not in step_ids:
            raise ValueError(f"Step#{step_id} not in file "
                             f"(available: {step_ids[0]}..{step_ids[-1]})")
        g = f[f'Step#{step_id}']

        x, y, z = g['x'][:], g['y'][:], g['z'][:]
        px, py, pz = g['px'][:], g['py'][:], g['pz'][:]
        step_attrs = {k: v for k, v in g.attrs.items()}

    if frame == 'paraxial':
        y, z = z, y
        py, pz = pz, py
    elif frame != 'lab':
        raise ValueError(f"Unknown frame '{frame}' (use 'lab' or 'paraxial')")

    positions = np.column_stack([x, y, z])
    momenta = np.column_stack([px, py, pz])
    metadata = {'species': species, 'opal_step': step_id, 'frame': frame}
    for key in ('SPOS', 'TIME', 'ENERGY'):
        if key in step_attrs:
            metadata[key.lower()] = np.asarray(step_attrs[key]).item() \
                if np.size(step_attrs[key]) == 1 else step_attrs[key]
    return positions, momenta, metadata


def save_opal_h5(filename: str, positions: np.ndarray, momenta: np.ndarray,
                 species_data: Dict, **metadata):
    """Deferred: writing OPAL H5hut is not implemented (use openPMD)."""
    raise NotImplementedError(
        "Writing OPAL H5hut files is deferred - save as openPMD instead")


# ------------------------------------------------------------------ deferred

def load_tracewin(filename: str, **kwargs):
    raise NotImplementedError(
        "TraceWin reading deferred (no reference reader exists; "
        "see 'Quick and Dirty Scripts' for a partial ASCII writer)")


def save_tracewin(filename: str, positions, momenta, species_data, **metadata):
    raise NotImplementedError("TraceWin writing deferred")


def load_aima(filename: str, **kwargs):
    raise NotImplementedError("AIMA .lst reading deferred (no reference code)")


def save_aima(filename: str, positions, momenta, species_data, **metadata):
    raise NotImplementedError("AIMA .lst writing deferred")


# ----------------------------------------------------------------------- npz

def load_npz(filename: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load from custom NumPy compressed format (.npz).

    Fast and lossless, good for internal use.
    """
    data = np.load(filename, allow_pickle=True)  # metadata is an object array
    positions = data['positions']
    momenta = data['momenta']
    metadata = dict(data['metadata'].item()) if 'metadata' in data else {}
    return positions, momenta, metadata


def save_npz(filename: str, positions: np.ndarray, momenta: np.ndarray,
             species_data: Dict = None, **metadata):
    """Save in NumPy compressed format."""
    metadata.pop('species', None)  # IonSpecies object does not pickle cleanly
    np.savez_compressed(filename, positions=positions, momenta=momenta,
                        metadata=np.array(metadata, dtype=object))


# ------------------------------------------------------- built-in registry

register_format('openpmd', load_openpmd, save_openpmd,
                extensions=('.bp',), sniffer=_is_openpmd_h5)
register_format('opal', load_opal_h5, save_opal_h5,
                extensions=('.h5',), sniffer=_is_opal_h5)
register_format('tracewin', load_tracewin, save_tracewin,
                extensions=('.dst', '.ini'))
register_format('aima', load_aima, save_aima, extensions=('.lst',))
register_format('npz', load_npz,
                lambda fn, pos, mom, sd=None, **md: save_npz(fn, pos, mom, sd, **md),
                extensions=('.npz',))

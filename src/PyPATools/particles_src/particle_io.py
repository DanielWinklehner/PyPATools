"""
particle_io.py - Load and save particle distributions from various formats

Supported formats:
- OPAL H5 (.h5)
- OpenPMD standard (.h5, .bp)
- TraceWin (.dst, .ini)
- AIMA Agora (.lst) - already in particles.py
- Custom binary (.npz)

Author: PyPATools Development Team
"""

import numpy as np
import h5py
from typing import Tuple, Optional, Dict


def load_opal_h5(filename: str, step: int = -1) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load particle distribution from OPAL H5 format.

    Parameters
    ----------
    filename : str
        Path to OPAL .h5 file
    step : int
        Time step to load (default: -1 = last step)

    Returns
    -------
    positions : np.ndarray(n_particles, 3)
        [x, y, z] in meters
    momenta : np.ndarray(n_particles, 3)
        [px, py, pz] as β·γ
    metadata : dict
        Additional data (charge, mass, time, etc.)
    """
    raise NotImplementedError("Waiting for your existing code")


def save_opal_h5(filename: str, positions: np.ndarray, momenta: np.ndarray,
                 species_data: Dict, **metadata):
    """
    Save particle distribution in OPAL H5 format.

    Parameters
    ----------
    filename : str
        Output filename
    positions : np.ndarray(n_particles, 3)
        Particle positions [m]
    momenta : np.ndarray(n_particles, 3)
        Particle momenta [β·γ]
    species_data : dict
        Species info (mass, charge, etc.)
    **metadata
        Additional metadata (time, step, bunch_charge, etc.)
    """
    raise NotImplementedError("Waiting for your existing code")


def load_openpmd(filename: str, iteration: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load particle distribution from openPMD standard format.

    Parameters
    ----------
    filename : str
        Path to openPMD file (.h5 or .bp)
    iteration : int, optional
        Iteration to load (default: latest)

    Returns
    -------
    positions, momenta, metadata
    """
    raise NotImplementedError("Requires openpmd-api")


def save_openpmd(filename: str, positions: np.ndarray, momenta: np.ndarray,
                 species_data: Dict, **metadata):
    """Save in openPMD format."""
    raise NotImplementedError("Requires openpmd-api")


def load_tracewin(filename: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load particle distribution from TraceWin format (.dst or .ini).

    TraceWin uses: x[cm], x'[mrad], y[cm], y'[mrad], phi[deg], ΔE[keV]

    Parameters
    ----------
    filename : str
        Path to TraceWin file

    Returns
    -------
    positions, momenta, metadata
    """
    raise NotImplementedError("Waiting for your existing code")


def save_tracewin(filename: str, positions: np.ndarray, momenta: np.ndarray,
                  species_data: Dict, reference_energy_mev: float, frequency_mhz: float):
    """Save in TraceWin format."""
    raise NotImplementedError("Waiting for your existing code")


def load_npz(filename: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load from custom NumPy compressed format (.npz).

    Fast and lossless, good for internal use.
    """
    data = np.load(filename)
    positions = data['positions']
    momenta = data['momenta']
    metadata = dict(data['metadata'].item()) if 'metadata' in data else {}
    return positions, momenta, metadata


def save_npz(filename: str, positions: np.ndarray, momenta: np.ndarray, **metadata):
    """Save in NumPy compressed format."""
    np.savez_compressed(filename, positions=positions, momenta=momenta,
                        metadata=np.array(metadata, dtype=object))
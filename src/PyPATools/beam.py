"""
beam.py
=======
Handles particle beam data storage and management for accelerator simulations.

Key design considerations:
- Memory efficiency: Use NumPy views where safe, explicit copies where needed
- Performance: Pre-allocated arrays for all timesteps
- API clarity: Distinguish between views (internal) and copies (external)
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field
from .species import IonSpecies
from .particles import ParticleDistribution


@dataclass
class Beam:
    """
    Manages particle beam data across simulation timesteps.

    Stores particle positions, velocities, and alive status at regular intervals
    determined by save_freq.

    Attributes
    ----------
    species : IonSpecies
        The ion species being simulated
    n_particles : int
        Number of particles
    n_steps : int
        Maximum total number of simulation timesteps
    save_freq : int
        Save particle data every n_steps timesteps
    n_saves : int
        Calculated number of save points: ceil(n_steps / save_freq)

    Internal Arrays
    ---------------
    t : ndarray, shape (n_saves,)
        Time at each save point [seconds]
    x_vec : ndarray, shape (n_saves, n_particles, 3)
        Position vectors [meters]
    v_vec : ndarray, shape (n_saves, n_particles, 3)
        Velocity vectors [m/s]
    alive : ndarray, shape (n_saves, n_particles), dtype=bool
        Alive status (True if particle still in simulation)
    """

    species: IonSpecies
    n_particles: int
    n_steps: int
    save_freq: int

    # Derived parameters (computed during post_init)
    n_saves: int = field(init=False)

    # Data arrays (computed during post_init)
    t: np.ndarray = field(init=False)
    x_vec: np.ndarray = field(init=False)
    v_vec: np.ndarray = field(init=False)
    alive: np.ndarray = field(init=False)

    # Pre-initialized output object (avoids repeated allocation)
    _output_pd: ParticleDistribution = field(init=False, default=None)

    # Track which save index we're currently at (for convenience)
    _current_save_idx: int = field(init=False, default=0)

    def __post_init__(self):
        """Initialize arrays after dataclass initialization."""
        # Calculate number of save points
        # ceil(n_steps / save_freq)
        self.n_saves = (self.n_steps + self.save_freq - 1) // self.save_freq

        # Initialize data arrays
        self.t = np.full(self.n_saves, np.nan, dtype=np.float64)
        self.x_vec = np.full((self.n_saves, self.n_particles, 3), np.nan, dtype=np.float64)
        self.v_vec = np.full((self.n_saves, self.n_particles, 3), np.nan, dtype=np.float64)

        # alive should be False (no particles "alive" until set)
        self.alive = np.zeros((self.n_saves, self.n_particles), dtype=bool)

        # Pre-initialize output ParticleDistribution
        # This avoids repeated allocation when repeatedly calling get methods
        self._output_pd = ParticleDistribution(
            species=self.species,
            n_particles=self.n_particles
        )

        self._current_save_idx = 0

    def set_pd_at_step(self, pd: ParticleDistribution, step: int, time: float) -> None:
        """
        Set particle data at a specific simulation timestep.

        Parameters
        ----------
        pd : ParticleDistribution
            Source particle distribution to copy data from
        step : int
            Simulation timestep (0 to n_steps-1). Must be a multiple of save_freq.
        time : float
            Physical time [seconds] at this timestep

        Raises
        ------
        ValueError
            If step is not a multiple of save_freq
        IndexError
            If save index exceeds n_saves
        """
        # Check if this step should be saved
        if step % self.save_freq != 0:
            raise ValueError(
                f"Timestep {step} is not a save point. "
                f"Only multiples of save_freq={self.save_freq} are saved."
            )

        # Calculate save index from step
        save_idx = step // self.save_freq

        if save_idx >= self.n_saves:
            raise IndexError(
                f"Timestep {step} → save index {save_idx}, exceeds n_saves={self.n_saves}"
            )

        # Copy data into internal arrays
        self.t[save_idx] = time
        self.x_vec[save_idx, :, :] = pd.x_vec[:]
        self.v_vec[save_idx, :, :] = pd.v_vec[:]
        self.alive[save_idx, :] = pd.alive[:]

    def get_pd_at_step(self, step: int) -> ParticleDistribution:
        """
        Retrieve particle distribution at a specific simulation timestep.

        Parameters
        ----------
        step : int
            Simulation timestep (0 to n_steps-1). Must be a multiple of save_freq.

        Returns
        -------
        ParticleDistribution
            A NEW ParticleDistribution object (deep copy of internal data)

        Raises
        ------
        ValueError
            If step is not a multiple of save_freq
        """
        if step % self.save_freq != 0:
            raise ValueError(
                f"Timestep {step} is not a save point. "
                f"Only multiples of save_freq={self.save_freq} are saved."
            )

        save_idx = step // self.save_freq
        return self.get_pd_at_index(save_idx)

    def get_pd_at_time(self, time: float) -> ParticleDistribution:
        """
        Retrieve particle distribution at a specific physical time.

        WARNING: Not recommended due to floating-point precision issues.
        Use get_pd_at_step(step) or find_nearest_step(time) instead.

        Parameters
        ----------
        time : float
            Physical time [seconds] to retrieve (must match exactly)

        Returns
        -------
        ParticleDistribution
            A NEW ParticleDistribution object (deep copy of internal data)

        Raises
        ------
        ValueError
            If exact time is not found
        """
        try:
            idx = np.where(self.t == time)[0][0]
        except IndexError:
            raise ValueError(
                f"Time {time} not found in saved times. "
                f"Use find_nearest_step() for approximate matches."
            )

        return self.get_pd_at_index(idx)

    def find_nearest_step(self, target_time: float) -> Tuple[int, float]:
        """
        Find the nearest saved timestep to a given target time.

        Parameters
        ----------
        target_time : float
            Desired physical time [seconds]

        Returns
        -------
        step : int
            Actual saved timestep
        actual_time : float
            Actual physical time of that timestep

        Example
        -------
        > step, time = beam.find_nearest_step(0.00523)
        > pd = beam.get_pd_at_step(step)
        """
        idx = np.argmin(np.abs(self.t - target_time))
        step = idx * self.save_freq
        return step, self.t[idx]

    def get_nearest_pd(self, target_time: float) -> ParticleDistribution:
        """
        Convenience: get ParticleDistribution at nearest saved time.

        Equivalent to:
            step, _ = beam.find_nearest_step(target_time)
            pd = beam.get_pd_at_step(step)

        Parameters
        ----------
        target_time : float
            Desired physical time [seconds]

        Returns
        -------
        ParticleDistribution
            At the nearest saved timestep
        """
        step, _ = self.find_nearest_step(target_time)
        return self.get_pd_at_step(step)

    def get_pd_at_index(self, index: int) -> ParticleDistribution:
        """
        Retrieve particle distribution at a specific save index.

        Parameters
        ----------
        index : int
            Save index (0 to n_saves-1). Negative indices supported
            (e.g., -1 returns last saved state)

        Returns
        -------
        ParticleDistribution
            A NEW ParticleDistribution object (deep copy of internal data)

        Raises
        ------
        IndexError
            If index out of range

        Notes
        -----
        Returns an explicit .copy() to ensure caller cannot accidentally
        modify internal beam state. The pre-initialized _output_pd object
        is reused for efficiency (only data arrays are updated).
        """
        # Handle negative indices
        if index < 0:
            index = self.n_saves + index

        if not 0 <= index < self.n_saves:
            raise IndexError(
                f"Save index {index} out of range [0, {self.n_saves})"
            )

        # Update the pre-allocated output object with data from this timestep
        # Using view assignment (: slicing) which preserves reference
        self._output_pd.t = self.t[index]
        self._output_pd.x[:] = self.x_vec[index, :, :]
        self._output_pd.v[:] = self.v_vec[index, :, :]
        self._output_pd.alive[:] = self.alive[index, :]

        # Return a deep copy to prevent external modification of internal state
        return self._output_pd.copy()

    def get_time_array(self) -> np.ndarray:
        """Return a copy of all saved times."""
        return self.t.copy()

    def get_n_saves(self) -> int:
        """Return number of save points."""
        return self.n_saves

    def reset_save_counter(self) -> None:
        """Reset internal save counter for sequential writes."""
        self._current_save_idx = 0
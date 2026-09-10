"""
trackers.py - Generic, hook-driven particle tracking loop for PyPATools.

This is the single tracking core: ONE integration loop, with Boris half-step
handling in exactly one place, that drives ``PyPATools.pusher.Pusher`` and
delegates all problem-specific behaviour to three kinds of hook objects:

  * ``Interaction`` - modifies (r, v) mid-step
                      (e.g. RF-cavity kicks, electrode backtrack).
  * ``Terminator``  - updates the alive mask, i.e. marks particles dead
                      (e.g. radial loss, aperture, electrode collision).
  * ``Recorder``    - observes the state and may request the run to stop
                      (e.g. Poincare sections, beam statistics, turn/energy targets).

The loop itself is GEOMETRY-AGNOSTIC: it knows nothing about radius, azimuth, RF
or any particular machine. All cylindrical / accelerator-specific assumptions
live inside the hook objects, so the same ``Tracker`` works for cyclotron central
regions, beam lines or spiral inflectors - only the hooks differ.

Every hook receives both the pre-step state ``(r_prev, v_prev)`` and the post-step
state ``(r, v)`` so that crossing interpolation (re-pushing from the previous
state) can be done inside a hook.

The canonical "alive" mask is owned here: it is read from
``ParticleDistribution.alive`` at the start, threaded through every hook, and
(optionally) written back at the end. Terminators are the only things that flip a
particle dead.

Boris note: velocity is staggered half a step (v-only); positions are left at the
on-step grid. This is the standard Boris leapfrog staggering.

Part of: PyPATools. This is an *additive* module - it does not modify ``Pusher``
or ``ParticleDistribution``.
"""

from dataclasses import dataclass
from typing import List, Sequence
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# ============================================================================
# Hook protocols (subclass these, or duck-type the same methods)
# ============================================================================
class Interaction:
    """Modifies particle state mid-step.

    Operates on the full ``(N, 3)`` arrays using the ``active`` mask and returns
    the (possibly updated) ``(r, v, active)``.
    """

    def apply(self, step, r_prev, v_prev, r, v, active, t, dt):
        return r, v, active


class Terminator:
    """Updates the alive mask - the ONLY place particles are marked dead.

    Returns the new ``active`` boolean mask (shape ``(N,)``).
    """

    def update(self, step, r_prev, v_prev, r, v, active, t):
        return active


class Recorder:
    """Observes state; return a truthy value to request the run to stop.

    Optionally set ``self.stop_reason``. ``finalize`` runs once after the loop.
    """

    stop_reason = "recorder_stop"

    def record(self, step, r_prev, v_prev, r, v, active, t):
        return None

    def finalize(self, r, v, active, t):
        pass


@dataclass
class TrackerResult:
    """Final state + metadata from a tracking run."""
    r: np.ndarray            # (N, 3) final positions [m]
    v: np.ndarray            # (N, 3) final velocities [m/s]
    active: np.ndarray       # (N,) alive mask
    n_steps: int             # steps actually executed
    stopped: bool            # True if stopped early (recorder stop or all-lost)
    stop_reason: str = ""
    t: float = 0.0


class Tracker:
    """Generic hook-driven tracking loop.

    Parameters
    ----------
    pusher : PyPATools.pusher.Pusher
        Configured integrator (single algorithm; Boris staggering handled here).
    efield, bfield : callable
        Field query callables ``f(pts(N,3)) -> (N,3)``.
    interactions, terminators, recorders : sequence of hook objects
        See module docstring. Empty by default (pure field tracking).
    """

    def __init__(self, pusher, efield, bfield,
                 interactions: Sequence[Interaction] = (),
                 terminators: Sequence[Terminator] = (),
                 recorders: Sequence[Recorder] = ()):
        self.pusher = pusher
        self.efield = efield
        self.bfield = bfield
        self.interactions: List[Interaction] = list(interactions)
        self.terminators: List[Terminator] = list(terminators)
        self.recorders: List[Recorder] = list(recorders)
        self._boris = str(getattr(pusher, "algorithm", "")).lower() == "boris"

    def run(self, pd, dt, n_steps, *, t0: float = 0.0, record_every: int = 1,
            show_progress: bool = False, sync_back: bool = True,
            stop_on_all_lost: bool = True) -> TrackerResult:
        """Track ``pd`` for ``n_steps`` of ``dt``.

        ``pd`` is a ``ParticleDistribution``; its ``x_vec``/``v_vec``/``alive``
        provide the initial state. With ``sync_back=True`` the final state is
        written back into ``pd`` (positions, momenta, alive mask).

        ``stop_on_all_lost=False`` keeps stepping when no particle is alive, for
        interactions that inject particles later (they start with ``alive`` False
        and are switched on at their injection step).
        """
        ef, bf, pusher = self.efield, self.bfield, self.pusher

        # Time-dependent fields (e.g. TimedField) expose set_time(); the time is
        # frozen at the STEP MIDPOINT for each push (second-order accurate).
        # Fields without set_time take the exact pre-existing code path.
        set_time = getattr(ef, "set_time", None)

        r = np.array(pd.x_vec, dtype=float, copy=True)
        v = np.array(pd.v_vec, dtype=float, copy=True)
        active = np.array(pd.alive, dtype=bool, copy=True)
        t = float(t0)

        # Boris: stagger velocity half a step back (v-only; positions stay on-grid).
        if self._boris and np.any(active):
            if set_time is not None:
                set_time(t)
            _, v[active] = pusher.push_batch(r[active], v[active], ef, bf, -0.5 * dt)

        pbar = tqdm(total=n_steps, ncols=120, desc="Tracking") if (show_progress and tqdm) else None

        stopped = False
        stop_reason = ""
        steps_done = n_steps

        for step in range(n_steps):
            r_prev = r.copy()
            v_prev = v.copy()

            # 1) advance active particles
            if set_time is not None:
                set_time(t + 0.5 * dt)
            r[active], v[active] = pusher.push_batch(r[active], v[active], ef, bf, dt)

            # 2) interactions (e.g. RF kicks) - pre-increment t
            for ix in self.interactions:
                r, v, active = ix.apply(step, r_prev, v_prev, r, v, active, t, dt)

            # 3) terminators (loss / aperture) - update alive mask
            for tm in self.terminators:
                active = tm.update(step, r_prev, v_prev, r, v, active, t)

            t += dt

            # 4) recorders (diagnostics) - post-increment t; may request stop
            if step % record_every == 0:
                for rc in self.recorders:
                    if rc.record(step, r_prev, v_prev, r, v, active, t):
                        stopped = True
                        stop_reason = getattr(rc, "stop_reason", rc.__class__.__name__)

            if stopped:
                steps_done = step + 1
                break

            if stop_on_all_lost and not np.any(active):
                stopped = True
                stop_reason = "all_lost"
                steps_done = step + 1
                break

            if pbar is not None:
                pbar.update(1)

        # Boris: forward half-step (v-only), only on normal completion - matching the
        # legacy loops, whose early returns leave the staggered velocity as-is.
        if self._boris and not stopped and np.any(active):
            if set_time is not None:
                set_time(t)
            _, v[active] = pusher.push_batch(r[active], v[active], ef, bf, 0.5 * dt)

        if pbar is not None:
            pbar.close()

        for rc in self.recorders:
            rc.finalize(r, v, active, t)

        if sync_back:
            pd.x_vec = r
            pd.set_p_from_v_vec(v)
            pd.alive = active

        return TrackerResult(r=r, v=v, active=active, n_steps=steps_done,
                             stopped=stopped, stop_reason=stop_reason, t=t)


# ============================================================================
# Self-test: parity with a manual Pusher.push_batch loop + alive bookkeeping
# ============================================================================
if __name__ == "__main__":
    from PyPATools.species import IonSpecies
    from PyPATools.particles import ParticleDistribution
    from PyPATools.pusher import Pusher

    ion = IonSpecies("proton")

    def bfield(pts):
        out = np.zeros_like(pts)
        out[:, 2] = 1.0
        return out

    def efield(pts):
        return np.zeros_like(pts)

    r0 = np.array([[0.05, 0.0, 0.0], [0.06, 0.0, 0.0]])
    v0 = np.array([[0.0, 1e6, 0.0], [0.0, 1.1e6, 0.0]])
    dt = 1e-11
    nsteps = 200

    pusher = Pusher(ion, algorithm="rk4_rel")
    r_ref, v_ref = r0.copy(), v0.copy()
    for _ in range(nsteps):
        r_ref, v_ref = pusher.push_batch(r_ref, v_ref, efield, bfield, dt)

    pd = ParticleDistribution(species=ion, x_vec=r0.copy(), p_vec=np.zeros_like(r0))
    pd.set_p_from_v_vec(v0.copy())
    res = Tracker(pusher, efield, bfield).run(pd, dt, nsteps)
    err = np.max(np.abs(res.r - r_ref)) + np.max(np.abs(res.v - v_ref))
    print(f"[parity] max abs diff Tracker vs manual loop: {err:.3e}")
    assert err < 1e-9, "Tracker does not match manual push_batch loop"

    class KillOne(Terminator):
        def update(self, step, r_prev, v_prev, r, v, active, t):
            if step == 100:
                active = active.copy()
                active[1] = False
            return active

    pd2 = ParticleDistribution(species=ion, x_vec=r0.copy(), p_vec=np.zeros_like(r0))
    pd2.set_p_from_v_vec(v0.copy())
    res2 = Tracker(pusher, efield, bfield, terminators=[KillOne()]).run(pd2, dt, nsteps)
    assert res2.active.tolist() == [True, False], "alive mask not updated by terminator"
    assert pd2.alive.tolist() == [True, False], "alive not synced back to pd"
    print(f"[alive] final active mask = {res2.active.tolist()}, pd2.alive = {pd2.alive.tolist()}")
    print("[OK] trackers.py self-test passed")

"""Regression: velocity <-> beta*gamma conversion must be NORM-AWARE.

Bug fixed 2026-07-06 (particles.py). The conversions were per-component:
    beta*gamma_j = v_j / sqrt(c^2 - v_j^2)          (v->P)
    v_j          = c*P_j / sqrt(P_j^2 + 1)          (P->v)
which is exact ONLY for purely axial motion. For in-plane (cyclotron)
motion the momentum NORM -- and hence ekin_mev (which is derived from |P|)
-- was wrong. Correct form (matches OPAL, src/Classic/Utilities/Util.h,
getGamma(p) = sqrt(dot(p,p)+1)): a SHARED Lorentz factor from the full
3-vector, beta*gamma_j = gamma*v_j/c with gamma = 1/sqrt(1 - |v|^2/c^2).

The v->P->v round trip was self-consistent before AND after the fix (so
tracking, which stays in velocity, is unaffected either way); the bug was
only in quantities read off the momentum norm.

Run: python tests/test_betagamma.py   (or via pytest)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyPATools.particles import (_momenta_from_velocities_batch,   # noqa: E402
                                 _velocities_from_momenta_batch)
from PyPATools.particles import ParticleDistribution               # noqa: E402
from PyPATools.species import IonSpecies                           # noqa: E402
from PyPATools import global_variables as gv                       # noqa: E402

CLIGHT = 299792458.0


def _bg_norm_and_gamma(beta):
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    return beta * gamma, gamma


def test_kernel_inplane_norm_matches_closed_form():
    """45-deg in-plane velocity: |P| must equal the true beta*gamma."""
    beta = 0.5
    v0 = beta * CLIGHT
    v = np.array([[v0 / np.sqrt(2.0), v0 / np.sqrt(2.0), 0.0]])
    P = _momenta_from_velocities_batch(v, True, CLIGHT)
    bg, _ = _bg_norm_and_gamma(beta)
    assert abs(np.linalg.norm(P[0]) - bg) < 1e-9, (np.linalg.norm(P[0]), bg)


def test_kernel_norm_is_direction_independent():
    """Same speed, different directions -> same |P| (was FALSE before fix:
    axial gave beta*gamma, 45-deg in-plane gave a smaller wrong value)."""
    beta = 0.5
    v0 = beta * CLIGHT
    dirs = {
        "axial": [0.0, 0.0, 1.0],
        "diag3d": [1.0, 1.0, 1.0],
        "inplane45": [1.0, 1.0, 0.0],
    }
    bg, _ = _bg_norm_and_gamma(beta)
    for name, d in dirs.items():
        d = np.array(d) / np.linalg.norm(d)
        v = (v0 * d)[None, :]
        P = _momenta_from_velocities_batch(v, True, CLIGHT)
        assert abs(np.linalg.norm(P[0]) - bg) < 1e-9, (name, np.linalg.norm(P[0]), bg)


def test_kernel_roundtrip_v_p_v():
    """v -> P -> v recovers the original velocity (all directions)."""
    rng = np.random.default_rng(0)
    d = rng.uniform(-1.0, 1.0, (200, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    beta = rng.uniform(0.01, 0.85, (200, 1))
    v = d * beta * CLIGHT
    P = _momenta_from_velocities_batch(v, True, CLIGHT)
    v_back = _velocities_from_momenta_batch(P, True, CLIGHT)
    assert np.allclose(v_back, v, rtol=1e-9, atol=1e-3)


def test_kernel_pnorm_is_gamma_beta_consistent():
    """gamma recovered from |P| (sqrt(1+|P|^2)) matches the input beta."""
    for beta in (0.1, 0.3, 0.5, 0.7, 0.9):
        v = np.array([[0.6, -0.5, 0.62]])
        v = v / np.linalg.norm(v) * beta * CLIGHT
        P = _momenta_from_velocities_batch(v, True, CLIGHT)
        gamma_from_P = np.sqrt(1.0 + np.dot(P[0], P[0]))
        _, gamma = _bg_norm_and_gamma(beta)
        assert abs(gamma_from_P - gamma) < 1e-9, (beta, gamma_from_P, gamma)


def test_distribution_energy_direction_independent():
    """Full ParticleDistribution path: ekin is the same for axial and
    in-plane motion at equal speed (the physically correct behavior)."""
    if not gv.RELATIVISTIC or gv.Z_ENERGY:
        print("[skip] test_distribution_energy_direction_independent "
              f"(RELATIVISTIC={gv.RELATIVISTIC}, Z_ENERGY={gv.Z_ENERGY})")
        return
    mu = IonSpecies("muon")
    m = mu.mass_mev
    beta = 0.5
    v0 = beta * CLIGHT
    _, gamma = _bg_norm_and_gamma(beta)
    ekin_true = m * (gamma - 1.0)

    for d in ([0, 0, 1], [1, 1, 0], [1, 1, 1]):
        d = np.array(d, dtype=float)
        d /= np.linalg.norm(d)
        dist = ParticleDistribution(species=mu)
        dist.set_p_from_v(np.array([v0 * d[0]]), np.array([v0 * d[1]]),
                          np.array([v0 * d[2]]))
        assert abs(dist.ekin_mev[0] - ekin_true) < 1e-6 * ekin_true, \
            (d.tolist(), dist.ekin_mev[0], ekin_true)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[ok]   {name}")
            except AssertionError as exc:
                print(f"[FAIL] {name}: {exc}")
                fails += 1
    sys.exit(1 if fails else 0)

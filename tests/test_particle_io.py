"""Tests for the particle I/O registry: openPMD round-trip, sniffing, status.

The openPMD cases need openPMD-beamphysics, which is an optional extra, and the
OPAL H5hut case needs the reference file from the IsoDAR runs. Both SKIP when
unavailable rather than failing -- only poisson_amg is allowed to hard-require
its extras (see the note in pyproject.toml).
"""
import os
import sys
import tempfile
import unittest
import warnings
import numpy as np
import h5py

from PyPATools.particles import ParticleDistribution
from PyPATools.species import IonSpecies


def _require_beamphysics():
    """Skip unless openPMD-beamphysics is importable.

    Note the import name is `beamphysics`, not `pmd_beamphysics`, and the package
    is openpmd-beamphysics -- not openpmd-api, which is a different dependency.
    """
    try:
        import beamphysics  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("openPMD cases need openpmd-beamphysics")


OPAL_REFERENCE_H5 = (r"D:\Dropbox (Personal)\Code\Python"
                     r"\Quick and Dirty Scripts\NNCollim.h5")


def _make_pd(n=500, seed=7):
    rng = np.random.default_rng(seed)
    return ParticleDistribution(
        species=IonSpecies('H2_1+'),
        x_vec=rng.normal([0.25, 0, 0], 1e-3, (n, 3)),
        p_vec=rng.normal([0, 0.046, 0], 5e-4, (n, 3)),
        q=2e-12, f=32.8e6, recalculate=True)


def test_openpmd_roundtrip_custom_ion():
    _require_beamphysics()

    pd = _make_pd()
    with tempfile.TemporaryDirectory() as d:
        fn = os.path.join(d, 'bunch.h5')
        pd.save_to_file(fn, turn=42, plane_azimuth_deg=120.0)
        pd2 = ParticleDistribution.from_file(fn)  # sniffed as openPMD
        assert pd2.species.name == 'H2_1+'
        assert abs(pd2.species.mass_mev - pd.species.mass_mev) < 1e-9
        assert np.abs(pd2.x_vec - pd.x_vec).max() < 1e-12
        assert np.abs(pd2.p_vec - pd.p_vec).max() < 1e-12
        assert abs(pd2.q - 2e-12) < 1e-24 and abs(pd2.f - 32.8e6) < 1e-3
        with h5py.File(fn, 'r') as f:
            assert f.attrs['openPMD'].startswith(b'2') or \
                str(f.attrs['openPMD']).startswith('2')
            (grp,) = f['particles'].values()
            assert grp.attrs['PyPATools:turn'] == 42


def test_status_roundtrip():
    _require_beamphysics()

    pd = _make_pd(n=200)
    pd.alive = np.arange(200) % 5 != 0
    with tempfile.TemporaryDirectory() as d:
        fn = os.path.join(d, 'losses.h5')
        pd.save_to_file(fn)
        with h5py.File(fn, 'r') as f:
            (grp,) = f['particles'].values()
            status = grp['particleStatus'][:]
        assert (status == 1).sum() == pd.alive.sum()
        assert (status != 1).sum() == 40


def test_npz_roundtrip_with_metadata():
    pd = _make_pd(n=100)
    with tempfile.TemporaryDirectory() as d:
        fn = os.path.join(d, 'bunch.npz')
        pd.save_to_file(fn, comment='hello')
        pd2 = ParticleDistribution.from_file(fn)
        assert np.abs(pd2.x_vec - pd.x_vec).max() < 1e-12


def test_deferred_formats_raise():
    for name in ('x.dst', 'x.lst'):
        try:
            ParticleDistribution.from_file(name)
        except NotImplementedError:
            pass
        else:
            raise AssertionError(f"{name} should raise NotImplementedError")


def test_opal_h5_load():
    if not os.path.exists(OPAL_REFERENCE_H5):
        raise unittest.SkipTest("OPAL reference file not present on this machine")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pd = ParticleDistribution.from_file(OPAL_REFERENCE_H5,
                                            species='H2_1+', step=-1)
        pd_par = ParticleDistribution.from_file(OPAL_REFERENCE_H5,
                                                species='H2_1+', step=-1,
                                                frame='paraxial')
    assert pd.numpart > 0
    assert np.allclose(pd_par.x_vec[:, 1], pd.x_vec[:, 2])
    assert np.allclose(pd_par.p_vec[:, 2], pd.p_vec[:, 1])


if __name__ == '__main__':
    fails = 0
    for fn_test in (test_openpmd_roundtrip_custom_ion, test_status_roundtrip,
                    test_npz_roundtrip_with_metadata,
                    test_deferred_formats_raise, test_opal_h5_load):
        try:
            fn_test()
            print(f"[ok]   {fn_test.__name__}")
        except unittest.SkipTest as exc:
            print(f"[skip] {fn_test.__name__}: {exc}")
        except AssertionError as exc:
            print(f"[FAIL] {fn_test.__name__}: {exc}")
            fails += 1
    sys.exit(1 if fails else 0)

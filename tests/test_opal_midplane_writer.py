"""Round-trip tests for the OPAL CARBONCYCL midplane writer.

field_writers.write_opal_midplane() takes a CARTESIAN midplane in SI
(meters, Tesla) and emits the POLAR map OPAL wants, in OPAL's midplane
units -- millimeters and kGauss. field_loaders.load_opal_midplane() reads
it back and converts kGauss -> Tesla, so the pair must close the loop.

The load side divides by 10, the write side multiplies by 10, and the
header carries r/theta in mm/degrees; a regression in either direction
shows up as a factor of 10 or a transposed (r, theta) block.

Run: python tests/test_opal_midplane_writer.py   (or via pytest)
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyPATools.field_src.field_loaders import load_opal_midplane  # noqa: E402
from PyPATools.field_src.field_writers import write_opal_midplane  # noqa: E402

# '%.5e' keeps 6 significant figures, so the text round-trip is good to
# ~5e-6 relative. Anything worse is a real bug, not formatting.
TEXT_TOL = 2e-5


def _cartesian_midplane(n=81, half_extent=0.2):
    """A smooth, non-symmetric analytic Bz(x, y) in Tesla on a square grid."""
    x = np.linspace(-half_extent, half_extent, n)
    y = np.linspace(-half_extent, half_extent, n)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    # Something with real azimuthal structure so a transpose would show up.
    bz = 1.0 + 0.3 * xx + 0.15 * yy + 0.5 * (xx ** 2 - yy ** 2)
    return {"x": x, "y": y, "z": np.array([0.0])}, {"z": bz}


def test_header_and_shape():
    grid, values = _cartesian_midplane()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mid.dat")
        write_opal_midplane(path, grid, values, r_max=100.0, dr=1.0, dtheta=0.5)
        with open(path) as fh:
            r_min = float(fh.readline())
            dr = float(fh.readline())
            th_min = float(fh.readline())
            dth = float(fh.readline())
            n_th = int(fh.readline())
            n_r = int(fh.readline())
    assert (r_min, dr, th_min, dth) == (0.0, 1.0, 0.0, 0.5), (r_min, dr, th_min, dth)
    assert n_r == 101, n_r                      # 0..100 mm inclusive
    assert n_th == 720, n_th                    # full turn, endpoint dropped


def test_full_turn_drops_duplicate_endpoint():
    """theta must span 0..360-dtheta, so 0 deg is not written twice."""
    grid, values = _cartesian_midplane()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mid.dat")
        write_opal_midplane(path, grid, values, r_max=20.0, dr=10.0, dtheta=90.0)
        d = load_opal_midplane(path, cartesian_grid=False)
    th = d["grid"]["theta"]
    assert len(th) == 4, th                     # 0, 90, 180, 270
    assert np.allclose(th, [0.0, 90.0, 180.0, 270.0]), th


def test_roundtrip_values_and_kgauss_scaling():
    """write -> load must return the source Tesla values at each (r, theta)."""
    grid, values = _cartesian_midplane()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mid.dat")
        write_opal_midplane(path, grid, values, r_max=100.0, dr=5.0, dtheta=5.0)
        d = load_opal_midplane(path, cartesian_grid=False)

    bz = d["values"]["z"]                       # Tesla, (n_r, n_theta)
    r = d["grid"]["r"]                          # meters
    th = np.deg2rad(d["grid"]["theta"])
    assert bz.shape == (len(r), len(th)), bz.shape

    # Compare against the analytic source at the same physical points.
    rr = r[:, None]
    xx = rr * np.cos(th)[None, :]
    yy = rr * np.sin(th)[None, :]
    expect = 1.0 + 0.3 * xx + 0.15 * yy + 0.5 * (xx ** 2 - yy ** 2)
    rel = np.max(np.abs(bz - expect) / np.maximum(1e-12, np.abs(expect)))
    assert rel < TEXT_TOL, "worst relative round-trip error %.3e" % rel


def test_theta_varies_fastest():
    """A transposed data block would swap these two lookups."""
    grid, values = _cartesian_midplane()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mid.dat")
        write_opal_midplane(path, grid, values, r_max=100.0, dr=50.0, dtheta=90.0)
        d = load_opal_midplane(path, cartesian_grid=False)
    bz = d["values"]["z"]
    # r = 0.1 m, theta = 0 -> x = +0.1, y = 0 -> 1 + 0.03 + 0.005 = 1.035
    # r = 0.1 m, theta = 90 -> x = 0, y = +0.1 -> 1 + 0.015 - 0.005 = 1.010
    assert abs(bz[2, 0] - 1.035) < 1e-4, bz[2, 0]
    assert abs(bz[2, 1] - 1.010) < 1e-4, bz[2, 1]


def test_scaling_one_writes_tesla():
    """scaling=1.0 writes Tesla; the loader still divides by 10, so x10 back."""
    grid, values = _cartesian_midplane()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mid.dat")
        write_opal_midplane(path, grid, values, r_max=10.0, dr=10.0,
                            dtheta=180.0, scaling=1.0)
        d = load_opal_midplane(path, cartesian_grid=False)
    # centre value is 1.0 T; written as 1.0, read back as 0.1 -> x10 = 1.0
    assert abs(d["values"]["z"][0, 0] * 10.0 - 1.0) < 1e-4, d["values"]["z"][0, 0]


def test_txt_extension_accepted():
    """The OPAL toolchain names these '*_CARBONCYCL.txt' as often as '.dat'."""
    grid, values = _cartesian_midplane()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mid_CARBONCYCL.txt")
        write_opal_midplane(path, grid, values, r_max=10.0, dr=10.0, dtheta=180.0)
        d = load_opal_midplane(path, cartesian_grid=False)
    assert d["dim"] == 2


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

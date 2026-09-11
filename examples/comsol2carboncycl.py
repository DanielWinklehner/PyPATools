"""
comsol2carboncycl.py -- Convert a COMSOL 2D midplane export to an OPAL
CARBONCYCL midplane map (polar grid, mm + kGauss).

Uses the PyPATools field tools end to end:
    PyPATools.field.Field.from_file()            -> field_loaders.load_comsol()
    field_src.field_writers.write_opal_midplane() -> CARBONCYCL text

The COMSOL export is a CARTESIAN midplane (x, y, Bz) in SI. CARBONCYCL is a
POLAR (r, theta) map in OPAL's midplane units -- millimeters and kGauss -- so
the field is resampled onto the polar grid and scaled by 10 on the way out.
See PyPATools/examples/comsol2h5hut.py for the units story in full.

Defaults reproduce the grid of the existing HCHC-60 map
(20250723_HCHC60_2D_FullCyclo_Res1mm_xy220cm_corrected_CARBONCYCL.txt):
r = 0:1:2200 mm (2201 points), theta = 0:0.5:359.5 deg (720 points).

Usage:
    python comsol2carboncycl.py <input.comsol> [output.txt]
                                [--dr MM] [--r-max MM] [--dtheta DEG]
                                [--units kgauss|tesla] [--no-verify]
"""

import argparse
import os
import sys
import time

import numpy as np

from PyPATools.field import Field
from PyPATools.field_src.field_writers import write_opal_midplane

B_UNIT_FACTOR = {"kgauss": 10.0, "tesla": 1.0}      # Tesla -> ...


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024.0:
            return "%.1f %s" % (nbytes, unit)
        nbytes /= 1024.0
    return "%.1f PB" % nbytes


def verify(out_path, field, bfac, samples=8, seed=0):
    """Re-read the written map and compare against the source interpolator."""
    from PyPATools.field_src.field_loaders import load_opal_midplane

    print("")
    print("--- verification ---------------------------------------------")
    with open(out_path) as fh:
        hdr = [float(fh.readline()) for _ in range(4)]
        n_theta = int(fh.readline())
        n_r = int(fh.readline())
    r_min, dr, th_min, dth = hdr
    print("header : r_min=%g mm  dr=%g mm  th_min=%g deg  dth=%g deg"
          % (r_min, dr, th_min, dth))
    print("         n_theta=%d  n_r=%d  -> %d values" % (n_theta, n_r, n_theta * n_r))
    print("         r  : %g .. %g mm" % (r_min, r_min + (n_r - 1) * dr))
    print("         th : %g .. %g deg" % (th_min, th_min + (n_theta - 1) * dth))

    # load_opal_midplane divides by 10 (kGauss -> T); undo that to get the
    # numbers actually in the file, then compare to the source field.
    got = load_opal_midplane(out_path, cartesian_grid=False)
    bz_cyl = got["values"]["z"] * 10.0 / bfac      # -> Tesla, source convention
    r_grid = got["grid"]["r"]                      # meters
    th_grid = np.deg2rad(got["grid"]["theta"])

    rng = np.random.default_rng(seed)
    print("")
    print("spot-check %d random (r, theta) nodes against the source field:" % samples)
    worst = 0.0
    for _ in range(samples):
        i = int(rng.integers(0, len(r_grid)))
        j = int(rng.integers(0, len(th_grid)))
        xx = r_grid[i] * np.cos(th_grid[j])
        yy = r_grid[i] * np.sin(th_grid[j])
        src = float(field(np.array([[xx, yy, 0.0]]))[0, 2])
        out = float(bz_cyl[i, j])
        den = max(1e-12, abs(src))
        worst = max(worst, abs(src - out) / den)
        print("  r=%7.1f mm th=%6.1f deg  (x=%7.4f, y=%7.4f) m  "
              "src=% .6e T  file=% .6e T  rel=%.1e"
              % (r_grid[i] * 1000.0, np.rad2deg(th_grid[j]), xx, yy, src, out,
                 abs(src - out) / den))
    # The data block is text at '%.5e' -- 6 significant figures -- so the
    # round-trip can only be as good as that format's quantum (half-ulp of a
    # 1.00000 mantissa is 5e-6 relative). Anything materially above that is a
    # real conversion error, not formatting.
    tol = 2e-5
    print("")
    print("worst relative deviation: %.3e   (tolerance %.0e, set by the '%%.5e' "
          "text format, not by the conversion)" % (worst, tol))
    print("Bz at r=0 : %.5f kGauss  (= %.6f T)"
          % (bz_cyl[0, 0] * bfac, bz_cyl[0, 0]))
    print("|Bz| max  : %.5f kGauss  (= %.6f T)"
          % (np.max(np.abs(bz_cyl)) * bfac, np.max(np.abs(bz_cyl))))
    ok = worst < tol
    print("verification: %s" % ("PASSED" if ok else "*** FAILED ***"))
    print("--------------------------------------------------------------")
    return ok


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("infile")
    ap.add_argument("outfile", nargs="?", default=None)
    ap.add_argument("--dr", type=float, default=1.0, help="radial step, mm (default 1)")
    ap.add_argument("--r-min", type=float, default=0.0, help="start radius, mm (default 0)")
    ap.add_argument("--r-max", type=float, default=None,
                    help="end radius, mm (default: largest circle inside the data)")
    ap.add_argument("--dtheta", type=float, default=0.5,
                    help="azimuthal step, deg (default 0.5)")
    ap.add_argument("--units", choices=sorted(B_UNIT_FACTOR), default="kgauss",
                    help="B units in the file (default kgauss, what OPAL midplane maps use)")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()

    infile = args.infile
    outfile = args.outfile or os.path.splitext(infile)[0] + "_CARBONCYCL.txt"
    bfac = B_UNIT_FACTOR[args.units]

    print("=" * 70)
    print("COMSOL midplane -> OPAL CARBONCYCL (PyPATools field tools)")
    print("=" * 70)
    print("input  : %s" % infile)
    print("         %s" % human(os.path.getsize(infile)))
    print("output : %s" % outfile)
    print("units  : B in %s (x%g), r in mm, theta in deg" % (args.units, bfac))
    print("")

    t0 = time.time()
    print("[1/2] Field.from_file()")
    field = Field.from_file(infile, label=os.path.basename(infile), debug=True)
    t1 = time.time()
    g = field.grid
    print("      dim=%d   nx,ny = %d, %d" % (field.dim, len(g["x"]), len(g["y"])))
    print("      x: [%g, %g] m   y: [%g, %g] m"
          % (g["x"][0], g["x"][-1], g["y"][0], g["y"][-1]))
    print("      Bz range: [%g, %g] T"
          % (np.min(field.grid_values["z"]), np.max(field.grid_values["z"])))
    print("      loaded in %.1f s" % (t1 - t0))

    print("")
    print("[2/2] write_opal_midplane()")
    write_opal_midplane(outfile, field.grid, field.grid_values,
                        r_min=args.r_min, r_max=args.r_max, dr=args.dr,
                        theta_min=0.0, theta_max=360.0, dtheta=args.dtheta,
                        scaling=bfac)
    t2 = time.time()
    print("      written in %.1f s   (%s)" % (t2 - t1, human(os.path.getsize(outfile))))
    print("")
    print("total: %.1f s" % (t2 - t0))

    if not args.no_verify:
        return 0 if verify(outfile, field, bfac, args.samples) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

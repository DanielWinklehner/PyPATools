"""
comsol2h5hut.py -- Convert COMSOL 3D field exports to OPAL H5hut/H5Block format.

Uses the PyPATools field tools end to end:
    PyPATools.field.Field.from_file()      -> field_src.field_loaders.load_comsol()
    PyPATools.field.Field.save_to_h5part() -> Step#0/Block/{Efield,Hfield}/{0,1,2}

UNITS -- the one thing PyPATools does NOT do for you
----------------------------------------------------
load_comsol() returns SI (Tesla, meters) and save_to_h5part() dumps the arrays
verbatim, so the raw round-trip writes Tesla + meters. Which units OPAL wants
depends on the vintage of the build:

    older OPAL : B in kGauss (1 T = 10 kG), lengths in mm
    newer OPAL : B in Tesla, lengths in m
    spyral_inflector / PyPATools : always SI -- Tesla, V/m, meters

so this script takes --units and --length-unit and you produce whichever pair
your build expects. Evidence for the old convention lives in the OPAL
toolchain: ascii2h5block.py writes `self.data["hz"] = tesla * 10.0` and notes
"COMSOL exported in the OPAL units of MV/m"; OPAL_Midplane_FieldConverter.py
says "OPAL units here are mm and kGauss".

B scaling rides on Field.scaling, which save_to_h5part() multiplies through.
The length unit is applied by rescaling the Field's grid axes before saving,
so that __Origin__ / __Spacing__ come out in the requested unit -- see the
comment in main(); passing rescaled r_min/r_max alone would silently drop
save_to_h5part() off its fast raw-dump path.

NOTE: if an efield= dict is ever passed to save_to_h5part it is written
verbatim, so it must already be in the units OPAL expects (MV/m for the old
convention).

Usage:
    python comsol2h5hut.py <input.comsol> [output.h5part]
                           [--units kgauss|tesla] [--length-unit m|mm]
                           [--freq HZ] [--no-verify]

Examples:
    # old-style OPAL: kGauss + mm
    python comsol2h5hut.py field.comsol out_kGauss_mm.h5part --units kgauss --length-unit mm

    # new-style OPAL, and what spyral_inflector / PyPATools want: T + m
    python comsol2h5hut.py field.comsol out_Tesla_m.h5part --units tesla --length-unit m
"""

import argparse
import gc
import os
import sys
import time

import numpy as np

from PyPATools.field import Field

AXES = ("x", "y", "z")

# Multiplier applied to the SI values returned by load_comsol().
B_UNIT_FACTOR = {"kgauss": 10.0, "tesla": 1.0}      # Tesla -> ...
LENGTH_UNIT_FACTOR = {"m": 1.0, "mm": 1000.0}       # meters -> ...


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024.0:
            return "%.1f %s" % (nbytes, unit)
        nbytes /= 1024.0
    return "%.1f PB" % nbytes


def sample_source_rows(filename, n_samples=12, seed=0):
    """Pull random full data lines out of the .comsol text by byte-seeking."""
    size = os.path.getsize(filename)
    rng = np.random.default_rng(seed)
    rows = []
    with open(filename, "rb") as fh:
        for _ in range(9):
            fh.readline()
        header_end = fh.tell()
        for off in rng.integers(header_end, size - 512, size=n_samples * 4):
            fh.seek(int(off))
            fh.readline()                 # discard the partial line
            parts = fh.readline().decode("ascii", "ignore").split()
            if len(parts) == 6:
                try:
                    rows.append([float(v) for v in parts])
                except ValueError:
                    continue
            if len(rows) >= n_samples:
                break
    return np.asarray(rows)


def verify(h5_path, comsol_path, n_samples=12, units="kgauss", length_unit="m"):
    """Re-open the written file and check geometry + sampled values.

    Source rows are SI (Tesla, meters); the file should hold
    Tesla * B_UNIT_FACTOR on a grid described in LENGTH_UNIT_FACTOR units.
    """
    import h5py

    bfac = B_UNIT_FACTOR[units]
    lfac = LENGTH_UNIT_FACTOR[length_unit]

    print("")
    print("--- verification ---------------------------------------------")
    ok = True
    with h5py.File(h5_path, "r") as h:
        freq = h.attrs["Resonance Frequency(Hz)"]
        hf = h["Step#0/Block/Hfield"]
        ef = h["Step#0/Block/Efield"]
        origin = np.asarray(hf.attrs["__Origin__"], dtype=float)
        spacing = np.asarray(hf.attrs["__Spacing__"], dtype=float)
        nz, ny, nx = hf["0"].shape
        print("Resonance Frequency(Hz) : %s" % freq)
        print("Hfield shape (nz,ny,nx) : (%d, %d, %d)" % (nz, ny, nx))
        print("__Origin__  [%-2s]        : %s" % (length_unit, origin))
        print("__Spacing__ [%-2s]        : %s" % (length_unit, spacing))
        print("Efield shape            : %s  (mid-plane all zero: %s)"
              % (ef["0"].shape, not bool(np.any(ef["0"][nz // 2]))))
        print("extent x/y/z [%s]        : [%g, %g] / [%g, %g] / [%g, %g]"
              % (length_unit,
                 origin[0], origin[0] + (nx - 1) * spacing[0],
                 origin[1], origin[1] + (ny - 1) * spacing[1],
                 origin[2], origin[2] + (nz - 1) * spacing[2]))

        rows = sample_source_rows(comsol_path, n_samples)
        print("")
        print("spot-check %d random rows read straight out of the .comsol text"
              % len(rows))
        print("(expecting B_h5 == B_src[T] * %g on a grid in %s)" % (bfac, length_unit))
        hf0, hf1, hf2 = hf["0"], hf["1"], hf["2"]
        worst = 0.0
        peak = 0.0
        for r in rows:
            x, y, z, bx, by, bz = r
            # source coords are meters -> convert into the file's length unit
            ix = int(round((x * lfac - origin[0]) / spacing[0]))
            iy = int(round((y * lfac - origin[1]) / spacing[1]))
            iz = int(round((z * lfac - origin[2]) / spacing[2]))
            if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
                print("  OUT OF RANGE: %g %g %g -> %d,%d,%d" % (x, y, z, ix, iy, iz))
                ok = False
                continue
            got = (float(hf0[iz, iy, ix]), float(hf1[iz, iy, ix]), float(hf2[iz, iy, ix]))
            exp = (bx * bfac, by * bfac, bz * bfac)
            err = max(abs(a - b) for a, b in zip(exp, got))
            scale = max(1e-30, max(abs(v) for v in exp))
            worst = max(worst, err / scale)
            peak = max(peak, max(abs(v) for v in got))
            flag = "" if err / scale < 1e-14 else "   <-- MISMATCH"
            print("  (%8.3f,%8.3f,%8.3f) m  B_src=(% .5e,% .5e,% .5e) T -> "
                  "B_h5=(% .5e,% .5e,% .5e) rel=%.1e%s"
                  % (x, y, z, bx, by, bz, got[0], got[1], got[2], err / scale, flag))
            if err / scale >= 1e-14:
                ok = False
        print("")
        print("worst relative |B_expected - B_h5| over samples: %.3e" % worst)
        print("peak |B| seen in samples: %.4g %s  (= %.4g T)" % (peak, units, peak / bfac))
    print("verification: %s" % ("PASSED" if ok else "*** FAILED ***"))
    print("--------------------------------------------------------------")
    return ok


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("infile")
    ap.add_argument("outfile", nargs="?", default=None)
    ap.add_argument("--units", choices=sorted(B_UNIT_FACTOR), default="tesla",
                    help="B units written to Hfield (default: tesla). Newer OPAL "
                         "and PyPATools/spyral_inflector use tesla; older OPAL kgauss.")
    ap.add_argument("--length-unit", choices=sorted(LENGTH_UNIT_FACTOR), default="m",
                    help="Unit for __Origin__/__Spacing__ (default: m). Older OPAL mm.")
    ap.add_argument("--freq", type=float, default=32800000.0,
                    help="Resonance Frequency(Hz) file attribute (default 32.8e6)")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--samples", type=int, default=12)
    args = ap.parse_args()

    infile = args.infile
    outfile = args.outfile or os.path.splitext(infile)[0] + ".h5part"
    bfac = B_UNIT_FACTOR[args.units]
    lfac = LENGTH_UNIT_FACTOR[args.length_unit]

    print("=" * 70)
    print("COMSOL -> OPAL H5hut conversion (PyPATools field tools)")
    print("=" * 70)
    print("input  : %s" % infile)
    print("         %s" % human(os.path.getsize(infile)))
    print("output : %s" % outfile)
    print("units  : B in %s (x%g), lengths in %s (x%g)"
          % (args.units, bfac, args.length_unit, lfac))
    print("")

    t0 = time.time()
    print("[1/2] Field.from_file()  (load_comsol: parse -> lexsort -> reshape)")
    field = Field.from_file(infile, label=os.path.basename(infile), debug=True)
    t1 = time.time()
    g = field.grid
    print("      dim=%d   nx,ny,nz = %d, %d, %d"
          % (field.dim, len(g["x"]), len(g["y"]), len(g["z"])))
    print("      x: [%g, %g]   y: [%g, %g]   z: [%g, %g]  (m)"
          % (g["x"][0], g["x"][-1], g["y"][0], g["y"][-1], g["z"][0], g["z"][-1]))
    print("      loaded in %.1f s" % (t1 - t0))

    # Grid axes come from np.unique() over the exported text, so the step
    # carries float noise (e.g. 0.0019999999999996). Recover the clean step
    # from the full span (exact to ~1e-16 relative) and snap it, so
    # __Spacing__ / __Origin__ hold nominal values.
    spacing_m, r_min_m, r_max_m = [], [], []
    for k in AXES:
        ax = np.asarray(g[k], dtype=float)
        step = (ax[-1] - ax[0]) / (len(ax) - 1) if len(ax) > 1 else 1.0
        step = round(step, 9)          # COMSOL grids are whole mm / um
        spacing_m.append(step)
        r_min_m.append(round(float(ax[0]), 9))
        r_max_m.append(round(float(ax[0]) + step * (len(ax) - 1), 9))

    devs = [float(np.max(np.abs(np.asarray(g[k], dtype=float)
                                - (r_min_m[i] + spacing_m[i] * np.arange(len(g[k]))))))
            for i, k in enumerate(AXES) if len(g[k]) > 1]
    dev = max(devs) if devs else 0.0
    print("      max |axis - uniform grid| = %.3e m  (%s)"
          % (dev, "uniform" if dev < 1e-9 else "NON-UNIFORM -- CHECK!"))

    # Express the grid in the requested length unit. save_to_h5part() only
    # takes its fast raw-array dump when the r_min/r_max it is given match the
    # Field's own grid axes; handing it mm while the Field still holds meters
    # would fail that test, fall back to evaluating the interpolators on a
    # mm-spaced meshgrid, land entirely out of bounds and write zeros. So
    # rescale the axes too. The interpolators are left holding the old meter
    # grid, which is fine here because the raw path never consults them (and
    # the verification below would catch it if it did).
    spacing = [round(s * lfac, 9) for s in spacing_m]
    r_min = [round(v * lfac, 9) for v in r_min_m]
    r_max = [round(v * lfac, 9) for v in r_max_m]
    if lfac != 1.0:
        field._grid = {k: np.asarray(v, dtype=float) * lfac
                       for k, v in field._grid.items()}
        print("      grid axes rescaled m -> %s" % args.length_unit)
    print("      spacing [%s] : %s" % (args.length_unit, spacing))
    print("      origin  [%s] : %s" % (args.length_unit, r_min))

    # Replicate save_to_h5part()'s own use_raw test, so a silent fall-back to
    # the interpolator path is reported rather than discovered in the output.
    nr = np.rint((np.asarray(r_max) - np.asarray(r_min)) / np.asarray(spacing) + 1).astype(int)
    raw_ok = (tuple(len(field._grid[k]) for k in AXES) == tuple(nr)
              and all(np.allclose([field._grid[k][0], field._grid[k][-1]],
                                  [r_min[i], r_max[i]])
                      for i, k in enumerate(AXES)))
    print("      raw-dump path: %s" % ("yes" if raw_ok else "NO -- would interpolate!"))
    if not raw_ok:
        print("      ABORTING: refusing to write via the interpolator path.")
        return 2

    print("")
    print("[2/2] Field.save_to_h5part()")
    # save_to_h5part() writes self._scaling * values, so the B unit conversion
    # rides along on the Field's scaling property -- no array copy of our own.
    field.scaling = bfac
    print("      Field.scaling = %g  (Tesla -> %s)" % (field.scaling, args.units))
    field.save_to_h5part(outfile, spacing=spacing, r_min=r_min, r_max=r_max,
                         resonance_frequency_hz=args.freq)
    t2 = time.time()
    print("      written in %.1f s   (%s)" % (t2 - t1, human(os.path.getsize(outfile))))
    print("")
    print("total: %.1f s" % (t2 - t0))

    if not args.no_verify:
        del field
        gc.collect()
        return 0 if verify(outfile, infile, args.samples,
                           args.units, args.length_unit) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

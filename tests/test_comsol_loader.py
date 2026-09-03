"""Regression tests for the COMSOL export loader (field_loaders.load_comsol).

The recurring failure mode: symmetry-unfolded COMSOL exports use MATH
EXPRESSIONS as column headers (the unfolding formulas, with spaces and
nested parentheses), e.g.

  -(mf.Bx * mir1side - mf.By * (mir1side - 1)) * (2 * mir2side - 1) * ...

The loader must (a) split such headers into columns, (b) identify each
field expression's component from its LEADING mf.B*/es.E* reference, and
(c) load the values VERBATIM -- no sign interpretation (field orientation
is project/species-dependent; consumers align orientation themselves).

Run: python tests/test_comsol_loader.py   (or via pytest)
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyPATools.field_src.field_loaders import (_comsol_component_of,  # noqa: E402
                                               _split_comsol_header_columns,
                                               load_comsol)

MIRROR4_HEADER = (
    "% mir4x                   mir4y                    mir4z                 "
    "   -(mf.Bx * mir1side - mf.By * (mir1side - 1)) * (2 * mir2side - 1) * "
    "(2 * mir4side - 1) -(mf.By * mir1side - mf.Bx * (mir1side - 1)) * "
    "(2 * mir3side - 1) * (2 * mir4side - 1) -mf.Bz (T)")


def _write(path, header, rows, dim, n_expr):
    lines = ["% Model: test", "% Version: test", "% Date: today",
             f"% Dimension: {dim}", f"% Nodes: {len(rows)}",
             f"% Expressions: {n_expr}", "% Description: ",
             "% Length unit: m", header]
    lines += [" ".join(f"{v}" for v in row) for row in rows]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def test_split_mirror4_header():
    cols, units = _split_comsol_header_columns(MIRROR4_HEADER)
    assert len(cols) == 6, cols
    assert cols[:3] == ["mir4x", "mir4y", "mir4z"]
    assert cols[5] == "-mf.Bz"
    assert "(T)" in units
    assert _comsol_component_of(cols[3]) == ("B", "x")
    assert _comsol_component_of(cols[4]) == ("B", "y")
    assert _comsol_component_of(cols[5]) == ("B", "z")


def test_split_plain_header():
    cols, _ = _split_comsol_header_columns(
        "% x y z mf.Bx (T) mf.By (T) mf.Bz (T)")
    assert cols == ["x", "y", "z", "mf.Bx", "mf.By", "mf.Bz"]


def test_load_mirror4_values_verbatim():
    rows = []
    k = 0
    for z in (0.0, 1.0):
        for y in (0.0, 1.0):
            for x in (0.0, 1.0):
                rows.append([x, y, z, 10 + k, 20 + k, -(30 + k)])
                k += 1
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m4.comsol")
        _write(path, MIRROR4_HEADER, rows, dim=3, n_expr=3)
        d = load_comsol(path)
    assert d["field_type"] == "magnetic"
    v = d["values"]
    assert v["x"][0, 0, 0] == 10.0
    assert v["y"][1, 0, 0] == 21.0
    # sign preserved VERBATIM (orientation-agnostic load)
    assert v["z"][0, 0, 0] == -30.0
    assert np.allclose(d["grid"]["x"], [0.0, 1.0])


def test_load_2d_signed_single_expression():
    # midplane export with prefixed coords + "-mf.Bz" expression
    header = "% cpl1x                   cpl1y                    -mf.Bz (T)"
    rows = []
    for y in (0.0, 1.0, 2.0):
        for x in (0.0, 1.0):
            rows.append([x, y, -(x + 10 * y)])
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mid.comsol")
        _write(path, header, rows, dim=2, n_expr=1)
        d = load_comsol(path)
    v = d["values"]
    bz = v["z"][:, :, 0] if v["z"].ndim == 3 else v["z"]
    assert bz[1, 2] == -(1 + 20)   # verbatim, x=1, y=2
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

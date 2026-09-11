"""Make the tests importable without installing the package.

PyPATools lives under src/, which is not on sys.path when pytest is run
from the repo root. Each test module also inserts it itself so that
`python tests/test_foo.py` works standalone (conftest.py is not loaded in
that case); this file covers the pytest path, where module import order
would otherwise decide whether a given test file happens to find the
package.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# poisson_amg imports py_electrodes, which initialises MPI at import time. On
# machines where MPI cannot start (or with MPI4PY_RC_INITIALIZE=false) that
# aborts the interpreter instead of raising, taking the whole pytest session
# down while test_poisson_origin.py is being collected. Blocking mpi4py makes
# py_electrodes fall back to its single-process mode; nothing in these tests
# needs MPI. Set PYPATOOLS_TEST_ALLOW_MPI=1 to keep MPI available.
if os.environ.get("PYPATOOLS_TEST_ALLOW_MPI", "0") != "1" and "mpi4py" not in sys.modules:
    sys.modules["mpi4py"] = None

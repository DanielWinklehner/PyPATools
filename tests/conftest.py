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

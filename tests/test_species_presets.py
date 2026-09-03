"""Consistency of the IonSpecies presets: `a` is the stored quantity.

IonSpecies derives rest mass from the mass number alone:

    self._mass_mev = a * AMU_MEV

so PRESETS[...]["mass_mev"] is a REFERENCE value that is never read at
runtime -- the assignment that used to read it was dead code. These tests
pin that invariant: edit a preset's `mass_mev` without editing its `a` (or
the other way round) and the table and the class silently disagree, with
nothing else in the package to notice.

examples/species_example.py does the same comparison by eye; this is the
version that fails CI.

Run: python tests/test_species_presets.py   (or via pytest)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PyPATools.species import PRESETS, IonSpecies    # noqa: E402
from PyPATools.global_variables import AMU_MEV       # noqa: E402

# Tight enough to catch a mistyped digit, loose enough to absorb the float
# round trip through a * AMU_MEV. Every preset is exact to <1e-15.
RTOL = 1e-7


def test_preset_table_is_internally_consistent():
    """Each preset's mass number must reproduce its reference rest mass."""
    for name, preset in sorted(PRESETS.items()):
        reference = preset["mass_mev"]
        derived = preset["a"] * AMU_MEV
        assert abs(derived - reference) <= RTOL * reference,             (name, derived, reference)


def test_ionspecies_reproduces_preset_mass():
    """A constructed species must agree with the table it came from."""
    for name, preset in sorted(PRESETS.items()):
        reference = preset["mass_mev"]
        mass_mev = IonSpecies(name=name).mass_mev
        assert abs(mass_mev - reference) <= RTOL * reference,             (name, mass_mev, reference)


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

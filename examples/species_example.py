from PyPATools.species import IonSpecies, PRESETS
import numpy as np

print("Testing IonSpecies class:")
for ion_name in PRESETS.keys():
    ion = IonSpecies(name=ion_name)
    print()
    print(ion)
    print("Mass comparison (MeV/c^2): Preset {}, {} {}, close? {}".format(PRESETS[ion_name]["mass_mev"],
                                                                          ion.label, ion.mass_mev,
                                                                          np.isclose(PRESETS[ion_name]["mass_mev"],
                                                                                     ion.mass_mev)
                                                                          ))

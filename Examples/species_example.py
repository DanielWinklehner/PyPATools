from PyPATools.species import IonSpecies, PRESETS

print("Testing IonSpecies class:")
for ion_name in PRESETS.keys():
    ion = IonSpecies(name=ion_name)
    print(ion)

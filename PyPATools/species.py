from scipy import constants as const

__author__ = "Daniel Winklehner, Philip Weigel"
__doc__ = "Simple class to hold and calculate particle data like mass, charge, etc."

# Initialize some global constants
amu = const.value("atomic mass constant energy equivalent in MeV")
echarge = const.value("elementary charge")
emass_mev = const.value("electron mass energy equivalent in MeV")
clight = const.value("speed of light in vacuum")

presets = {'proton': {'label': r"$\mathrm{p}^+$",
                      'a': 1.00727646687991,
                      'z': 1.0,
                      'q': 1.0},
           'electron': {'label': r"$\mathrm{e}^-$",
                        'a': const.value('electron mass energy equivalent in MeV')/amu,
                        'z': 0.0,
                        'q': -1.0},
           'H2_1+': {'label': r"$\mathrm{H}_2^+$",
                     'a': 2.015133,
                     'z': 2.0,
                     'q': 1.0},
           '4He_2+': {'label': r"$^4\mathrm{He}^{2+}$",
                      'a': 4.001506466,
                      'z': 2.0,
                      'q': 2.0}}


class IonSpecies(object):

    def __init__(self,
                 name,
                 label=None,
                 a=None,
                 z=None,
                 q=None,
                 debug=False):

        """
        Simple ion species class that holds data and can calculate some basic values like rigidity and energy.
        :param name: Name of the species, can be one of the presets:
            'protons'
            'electrons'
            'H2_1+'
            '4He_2+'
        if it is not a preset, a, z and q have to be defined as well:
        :param label: A text label for plotting, can be in latex shorthand, defaults to name.
        :param a: atomic (molecular) mass in amu
        :param z: number of protons
        :param q: charge state
        """

        # Check if user wants a preset ion species:
        if name in presets.keys():

            species = presets[name]

            if label is None:
                label = species["label"]
            z = species["z"]
            a = species["a"]
            q = species["q"]

            if debug:
                print("Using preset ion species '{}' with label '{}':".format(name, label))

        # If not, check for missing initial values
        else:

            init_values = [a, z, q]

            if None in init_values:

                print("Sorry, ion species {} was initialized with missing values!".format(name))
                print("a = {}, z = {}, q = {}". format(a, z, q))
                exit(1)

            else:

                if debug:
                    print("Uing user defined ion species {}:".format(name))

        # Initialize values (default for a proton)
        self._name = name
        if label is None:
            self._label = name
        else:
            self._label = label        # A label for this species
        self._mass_mev = a * amu       # Rest Mass (MeV/c^2)
        self._a = a                    # Mass number A of the ion (amu)
        self._z = z                    # Proton number Z of the ion (unitless)
        self._q = q                    # charge state

        # Calculate mass of the particle in kg
        self._mass_kg = self._mass_mev * echarge * 1.0e6 / clight**2.0

    def __str__(self):
        return "Ion Species {} with label {}:\n" \
               "A = {}, Z = {}, q = {}\n" \
               "M_0 = {} MeV/c^2 = {} kg, Q = {} C\n" \
               "(For OPERA: M_0 = {} * M_electron)".format(self._name,
                                                           self._label,
                                                           self._a, self._z, self._q,
                                                           self._mass_mev, self._mass_kg, self._q * echarge,
                                                           self._mass_mev / emass_mev)

    @property
    def mass_mev(self):
        return self._mass_mev

    @property
    def mass_kg(self):
        return self._mass_kg

    @property
    def mass_opera(self):
        return self._mass_mev / emass_mev

    @property
    def q_over_a(self):
        """
        :return: charge state - to - mass number ratio (unitless) 
        """
        return self._q / self._a

    @property
    def q_over_m(self):
        """
        :return: charge - to - mass ratio (C/kg) 
        """
        return self._q * echarge / self._mass_kg

    @property
    def label(self):
        return self._label

    @property
    def name(self):
        return self._name

    @property
    def q(self):
        return self._q

    @property
    def charge(self):
        return self._q * echarge

    @property
    def a(self):
        return self._a


if __name__ == '__main__':
    pass

from scipy import constants as const
import numpy as np
from .species import IonSpecies
from .particles import ParticleDistribution

__author__ = "Daniel Winklehner, Philip Weigel"
__doc__ = "Simple class to ion beam data. A beam can consist of several distributions with different ion species."

# Initialize some global constants
amu = const.value("atomic mass constant energy equivalent in MeV")
echarge = const.value("elementary charge")
emass_mev = const.value("electron mass energy equivalent in MeV")
clight = const.value("speed of light in vacuum")


class IonBeam(object):
    def __init__(self):
        pass

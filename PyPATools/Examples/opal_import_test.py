from PyPATools.particles import ParticleDistribution, RELATIVISTIC, Z_ENERGY
from PyPATools.species import IonSpecies
import h5py
import numpy as np
import matplotlib.pyplot as plt


def get_bunch_in_local_frame(datasource, step):
    x = np.array(datasource["Step#{}".format(step)]["x"])
    y = np.array(datasource["Step#{}".format(step)]["y"])
    px = np.array(datasource["Step#{}".format(step)]["px"])
    py = np.array(datasource["Step#{}".format(step)]["py"])

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    px_mean = np.mean(np.array(px))
    py_mean = np.mean(np.array(py))

    theta = np.arccos(py_mean / np.sqrt(np.square(px_mean) + np.square(py_mean)))

    if px_mean < 0:
        theta = -theta

    # Center the beam
    x -= x_mean
    y -= y_mean

    # Rotate the beam and return
    return [x * np.cos(theta) - y * np.sin(theta), x * np.sin(theta) + y * np.cos(theta),
            px * np.cos(theta) - py * np.sin(theta), px * np.sin(theta) + py * np.cos(theta)]


def get_data(fn):

    datasource = h5py.File(fn, "r+")
    print("Number of timesteps: {}".format(len(datasource)))
    step0_data = datasource["Step#{}".format(0)]

    x, y, px, py = get_bunch_in_local_frame(datasource, 0)
    z = step0_data["z"][:]
    pz = step0_data["pz"][:]

    # _pd = ParticleDistribution(species=IonSpecies("H2_1+"),
    #                            x=step0_data["x"][:],
    #                            y=step0_data["y"][:],
    #                            z=step0_data["z"][:],
    #                            px=step0_data["px"][:],
    #                            py=step0_data["py"][:],
    #                            pz=step0_data["pz"][:])

    _pd = ParticleDistribution(species=IonSpecies("H2_1+"),
                               x=x,
                               y=z,
                               z=y,
                               px=px,
                               py=pz,
                               pz=py)

    return _pd


fn0 = r"D:\Dropbox (MIT)\Projects\IsoDAR\60 MeV Cyclotron\harmonic4\v20200326_1 (5 Collimators)" \
      r"\Stage1 (102 turns - no ESC)\OPAL run\NNCollim.h5"

print("Importing Data from OPAL h5 file.")
pd = get_data(fn0)

fig, ax = plt.subplots(1, 3)

ax[0].plot(pd.x * 1e3, pd.y * 1e3, 'ko', ms=0.1, alpha=0.8)
ax[0].set_aspect(1)
ax[0].set_xlabel("radial (mm)")
ax[0].set_ylabel("vertical (mm)")
ax[0].set_xlim(-20, 20)
ax[0].set_ylim(-20, 20)
ax[1].plot(pd.z * 1e3, pd.y * 1e3, 'ko', ms=0.1, alpha=0.8)
ax[1].set_aspect(1)
ax[1].set_xlabel("longitudinal (mm)")
ax[1].set_ylabel("vertical (mm)")
ax[1].set_xlim(-20, 20)
ax[1].set_ylim(-20, 20)
ax[2].plot(pd.z * 1e3, pd.x * 1e3, 'ko', ms=0.1, alpha=0.8)
ax[2].set_aspect(1)
ax[2].set_xlabel("longitudinal (mm)")
ax[2].set_ylabel("radial (mm)")
ax[2].set_xlim(-20, 20)
ax[2].set_ylim(-20, 20)

plt.tight_layout()
plt.show()

print("Calculating relativistically: {} and assuming paraxial: {}".format(RELATIVISTIC, Z_ENERGY))
print("Mean Energy = {} keV/amu".format(pd.mean_energy_mev_per_amu * 1e3))
print("Emittances (1-rms, norm.):", pd.get_emittances() * 1e6)
print("Twiss Parameters:", pd.get_twiss_parameters())

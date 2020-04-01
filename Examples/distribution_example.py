from PyPATools.particles import ParticleDistribution
from scipy import constants as const

clight = const.value("speed of light in vacuum")
numpart = 20000

print("Testing ParticleDistribution and emittance calculation with {} particles.".format(numpart))
pd = ParticleDistribution(debug=True)
pd.set_numpart(numpart)
pd.gaussian_sphere(sigma=np.array([0.001, 0.001, 0.001]))
pd.boltzmann_velocity(2.0)
pd.add_directed_velocity(vz=0.1 * clight)
pd.calculate_emittances()
pd.plot_positions_3d()


def plot_positions_3d(self):
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(1000.0 * self.x, 1000.0 * self.y, 1000.0 * self.z,
               s=4.0, edgecolor='none')

    ax.set_aspect('equal')

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")

    plt.show()
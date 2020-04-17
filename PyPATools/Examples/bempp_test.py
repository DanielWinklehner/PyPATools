import bempp.api
import numpy as np
import matplotlib
from matplotlib import pylab as plt

matplotlib.rcParams['figure.figsize'] = (5.0, 4.0)

# Create a sphere
grid = bempp.api.shapes.sphere(h=0.1)
bempp.api.PLOT_BACKEND = "gmsh"
grid.plot()

dp0_space = bempp.api.function_space(grid, "DP", 0)
p1_space = bempp.api.function_space(grid, "P", 1)

identity = bempp.api.operators.boundary.sparse.identity(p1_space, p1_space, dp0_space)
dlp = bempp.api.operators.boundary.laplace.double_layer(p1_space, p1_space, dp0_space)
slp = bempp.api.operators.boundary.laplace.single_layer(dp0_space, p1_space, dp0_space)


@bempp.api.real_callable
def dirichlet_data(x, n, domain_index, result):
    result[0] = 1. / (4 * np.pi * ((x[0] - .9) ** 2 + x[1] ** 2 + x[2] ** 2) ** 0.5)


dirichlet_fun = bempp.api.GridFunction(p1_space, fun=dirichlet_data)
dirichlet_fun.plot()


rhs = (.5 * identity + dlp) * dirichlet_fun

neumann_fun, info = bempp.api.linalg.cg(slp, rhs, tol=1e-3)

n_grid_points = 300
plot_grid = np.mgrid[-1:1:n_grid_points * 1j, -1:1:n_grid_points * 1j]
points = np.vstack((plot_grid[0].ravel(),
                    plot_grid[1].ravel(),
                    np.zeros(plot_grid[0].size)))

slp_pot = bempp.api.operators.potential.laplace.single_layer(dp0_space, points)
dlp_pot = bempp.api.operators.potential.laplace.double_layer(p1_space, points)

u_evaluated = slp_pot * neumann_fun - dlp_pot * dirichlet_fun

# Filter out solution values that are associated with points outside the unit circle.
u_evaluated = u_evaluated.reshape((n_grid_points,n_grid_points))
radius = np.sqrt(plot_grid[0]**2 + plot_grid[1]**2)
u_evaluated[radius > 1] = np.nan

# Plot the image
plt.imshow(np.log(np.abs(u_evaluated.T)), extent=(-1, 1, -1, 1))
plt.title('Computed solution')
plt.colorbar()
plt.show()

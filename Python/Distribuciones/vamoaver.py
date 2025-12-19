import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import special_ortho_group

def von_mises_fisher(mu, kappa, size=1000):
    """
    Generate samples from the von Mises-Fisher distribution.

    Parameters:
        mu (array-like): The mean direction of the distribution.
        kappa (float): The concentration parameter of the distribution.
        size (int): Number of samples to generate.

    Returns:
        numpy.ndarray: Samples from the von Mises-Fisher distribution.
    """
    dim = len(mu)
    random_directions = special_ortho_group.rvs(dim)  # Random orthogonal matrix
    normal_samples = np.random.normal(size=(size, dim))
    samples = np.dot(normal_samples, random_directions)  # Samples in the standard normal coordinate system

    norm = np.linalg.norm(samples, axis=1)
    uniform_samples = np.random.uniform(size=size) ** (1 / dim)
    uniform_samples_reshaped = np.arccos(uniform_samples).reshape(-1, 1)
    scaled_samples = (samples.T / norm).T * uniform_samples_reshaped

    return scaled_samples * kappa + mu

def project_on_sphere(samples):
    """
    Project samples onto the surface of a unit sphere.

    Parameters:
        samples (numpy.ndarray): Samples to project.

    Returns:
        numpy.ndarray: Projected samples on the surface of the unit sphere.
    """
    norms = np.linalg.norm(samples, axis=1)
    return samples / norms[:, None]

# Parameters for the von Mises-Fisher distribution
mu = np.array([0, 0, 1])  # Mean direction
kappa = 500  # Concentration parameter

# Generate samples from the von Mises-Fisher distribution
samples = von_mises_fisher(mu, kappa)

# Project samples onto the surface of a unit sphere
projected_samples = project_on_sphere(samples)

# Plotting on a sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(projected_samples[:, 0], projected_samples[:, 1], projected_samples[:, 2])
ax.set_title('von Mises-Fisher Distribution on a Sphere')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_aspect('equal')
plt.show()

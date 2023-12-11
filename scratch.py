import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Set random seed for reproducibility
np.random.seed(42)

# Number of Gaussians
num_gaussians = 25
grid_size = 5

# Generate 2D grid coordinates
x, y = np.meshgrid(np.linspace(0, grid_size - 1, grid_size), np.linspace(0, grid_size - 1, grid_size))
grid_coordinates = np.column_stack((x.ravel(), y.ravel()))

# Create means for each Gaussian in the grid
means = grid_coordinates

# Create covariances (identity matrices for simplicity)
covariances = np.tile(np.eye(2), (num_gaussians, 1, 1))

# Create equal weights for each Gaussian
weights = np.ones(num_gaussians) / num_gaussians

# Create and fit the Gaussian Mixture Model
gmm = GaussianMixture(n_components=num_gaussians, means_init=means, weights_init=weights)
gmm.fit(grid_coordinates)

# Generate samples from the GMM
samples, _ = gmm.sample(500)

print(samples)

# Plot the GMM components and generated samples
plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5, label='Generated Samples')
plt.scatter(means[:, 0], means[:, 1], c='red', marker='x', label='GMM Means')
plt.title('Gaussian Mixture Model with 25 Gaussians in a 5x5 Grid')
plt.legend()
plt.show()
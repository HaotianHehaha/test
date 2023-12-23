import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from deep_bsde import bsde

# Set random seed for reproducibility
np.random.seed(30)

# Mean and covariance matrices for three 2D Gaussian distributions
mean1 = [2, 2]
cov1 = [[1.2, 0.5],
        [0.5, 1.2]]

mean2 = [-1, -3]
cov2 = [[1, -0.8],
        [-0.8, 1]]

mean3 = [4, -1]
cov3 = [[0.8, 0],
        [0, 0.8]]

# Generate random samples from the three Gaussian distributions
samples1 = np.random.multivariate_normal(mean1, cov1, 3)
samples2 = np.random.multivariate_normal(mean2, cov2, 2)
samples3 = np.random.multivariate_normal(mean3, cov3, 3)

sample_num = 100
mean_initial = [0,0]
cov_initial = [[1, 0],
        [0, 1]]
samples = np.random.multivariate_normal(mean_initial, cov_initial, sample_num)



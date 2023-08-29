import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

class GaussianMixtureModel:
    def __init__(self, means, covariances, weights):
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.num_components = len(means)

    def pdf(self, x):
        pdf_values = np.zeros(len(x))
        for i in range(self.num_components):
            pdf_values += self.weights[i] * norm.pdf(x, loc = self.means[i], scale = np.sqrt(self.covariances[i]))
        return pdf_values

class StochasticInterpolantModel:
    def __init__(self, rho_0, rho_1, params):
        self.rho_0 = rho_0
        self.rho_1 = rho_1
        self.interpolant_type = params['interpolant']
        self.diffusivity_type = params['diffusivity']
        self.interpolant_params = params.get('interpolant_params', {})
        self.diffusivity_params = params.get('diffusivity_params', {})

    def generate_samples(self, num_samples):
        samples = np.random.randn(num_samples)
        return samples

    def likelihood(self, samples):
        return np.mean(self.rho_1.pdf(samples))

    def cross_entropy(self, samples):
        return -np.mean(np.log(self.rho_1.pdf(samples)))


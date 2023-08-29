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

means_0 = [-2, 2]
covariances_0 = [0.5, 0.3]
weights_0 = [0.4, 0.6]
rho_0 = GaussianMixtureModel(means_0, covariances_0, weights_0)

means_1 = [0, 3]
covariances_1 = [0.2, 0.4]
weights_1 = [0.6, 0.4]
rho_1 = GaussianMixtureModel(means_1, covariances_1, weights_1)

params = {
    'interpolant': 'linear',
    'diffusivity': 'constant',
    'interpolant_params': {'alpha': 0.7},  
    'diffusivity_params': {'D': 0.1}  
}

model = StochasticInterpolantModel(rho_0, rho_1, params)
samples = model.generate_samples(num_samples = 1000)

likelihood = model.likelihood(samples)
cross_entropy = model.cross_entropy(samples)

print('Likelihood of SIM for Gaussian Mixture: ', likelihood)
print('Cross-Entropy of SIM for Gaussian Mixture: ', cross_entropy)
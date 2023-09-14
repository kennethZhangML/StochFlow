import numpy as np
from scipy.stats import multivariate_normal


class GaussianMixtureModel:
    def __init__(self, means, covariances, weights):
        self.means = means
        self.covariances = covariances
        self.weights = weights

    def probability_density(self, x):
        return np.sum([w * multivariate_normal.pdf(x, mean=mu, cov=cov)
                       for w, mu, cov in zip(self.weights, self.means, self.covariances)])


def likelihood(prob_density, samples):
    return np.mean([prob_density(sample) for sample in samples])


def cross_entropy(prob_density_p, prob_density_q, samples):
    return np.mean([-np.log(prob_density_q(sample)) for sample in samples])

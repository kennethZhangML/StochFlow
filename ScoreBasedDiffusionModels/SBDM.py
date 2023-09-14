import numpy as np
from scipy.stats import norm


class StochasticInterpolantModel:
    def __init__(self, initial_density, final_density):
        self.initial_density = initial_density
        self.final_density = final_density

    def get_initial_llh(self, samples):
        return self.initial_density.pdf(samples)

    def get_final_llh(self, samples):
        return self.final_density.pdf(samples)

    @staticmethod
    def generate_samples(num_samples):
        return np.random.normal(size=num_samples)

    def estimate_likelihood(self, samples):
        return self.get_final_llh(samples) / self.get_initial_llh(samples)

    def compute_cross_entropy(self):
        x = np.linspace(-5, 5, 1000)
        initial_pdf = self.get_initial_llh(x)
        final_pdf = self.get_final_llh(x)
        cross_entropy = -np.sum(initial_pdf * np.log(final_pdf + 1e-10))
        return cross_entropy


class ScoreBasedDiffusionModel:
    def __init__(self, score_estimator):
        self.score_estimator = score_estimator

    @staticmethod
    def generate_samples(num_samples):
        return np.random.normal(size=num_samples)

    def estimate_likelihood(self, samples):
        return np.exp(self.score_estimator(samples).sum())

    @staticmethod
    def compute_cross_entropy():
        return 0.0


def score_estimator(samples):
    return -2 * samples


if __name__ == "__main__":
    initial_density = norm(0, 1)
    final_density = norm(2, 1)
    stochastic_interpolant = StochasticInterpolantModel(initial_density, final_density)

    score_based_model = ScoreBasedDiffusionModel(score_estimator)
    num_samples = 1000

    samples_si = stochastic_interpolant.generate_samples(num_samples)
    likelihood_si = stochastic_interpolant.estimate_likelihood(samples_si)
    print(f"Likelihood SI: {likelihood_si}")
    cross_entropy_si = stochastic_interpolant.compute_cross_entropy()
    print(f"Cross-Entropy SI: {cross_entropy_si}")

    samples_sbdm = score_based_model.generate_samples(num_samples)
    likelihood_sbdm = score_based_model.estimate_likelihood(samples_sbdm)
    print(f"Likelihood SBDM: {likelihood_sbdm}")
    cross_entropy_sbdm = score_based_model.compute_cross_entropy()
    print(f"Cross-Entropy SBDM: {cross_entropy_sbdm}")

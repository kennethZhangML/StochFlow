import numpy as np
import scipy.stats

class StochasticInterpolantModel:
    def __init__(self, initial_density, final_density):
        self.initial_density = initial_density
        self.final_density = final_density
    
    def generate_samples(self, num_samples):
        return np.random.normal(size = num_samples) 
    
    def estimate_likelihood(self, samples):
        initial_likelihood = self.initial_density.pdf(samples)
        final_likelihood = self.final_density.pdf(samples)
        return final_likelihood / initial_likelihood
    
    def compute_cross_entropy(self):
        x = np.linspace(-5, 5, 1000)
        initial_pdf = self.initial_density.pdf(x)
        final_pdf = self.final_density.pdf(x)
        cross_entropy = -np.sum(initial_pdf * np.log(final_pdf + 1e-10))  
        return cross_entropy

class ScoreBasedDiffusionModel:
    def __init__(self, score_estimator):
        self.score_estimator = score_estimator
    
    def generate_samples(self, num_samples):
        return np.random.normal(size = num_samples)  
    
    def estimate_likelihood(self, samples):
        likelihood = np.exp(self.score_estimator(samples).sum())
        return likelihood
    
    def compute_cross_entropy(self):
        return 0.0  

# Example score estimator
def score_estimator(samples):
    return -2 * samples

if __name__ == "__main__":
    initial_density = scipy.stats.norm(0, 1)
    final_density = scipy.stats.norm(2, 1)
    stochastic_interpolant = StochasticInterpolantModel(initial_density, final_density)

    score_based_model = ScoreBasedDiffusionModel(score_estimator)

    num_samples = 1000
    samples_si = stochastic_interpolant.generate_samples(num_samples)
    samples_sbdm = score_based_model.generate_samples(num_samples)

    likelihood_si = stochastic_interpolant.estimate_likelihood(samples_si)
    likelihood_sbdm = score_based_model.estimate_likelihood(samples_sbdm)

    cross_entropy_si = stochastic_interpolant.compute_cross_entropy()
    cross_entropy_sbdm = score_based_model.compute_cross_entropy()

    print("Likelihood SI:", likelihood_si)
    print("Likelihood SBDM:", likelihood_sbdm)
    print("Cross-Entropy SI:", cross_entropy_si)
    print("Cross-Entropy SBDM:", cross_entropy_sbdm)
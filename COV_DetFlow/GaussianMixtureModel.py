import numpy as np 

class GaussianMixtureModel:
    def __init__(self, means, variances, weights):
        self.means = means 
        self.variances = variances 
        self.weights = weights 
    
    def get_density(self, x):
        density = 0
        for mean, variance, weight in zip(self.means, self.variances, self.weights):
            density += weight * np.exp(-0.5 * ((x - mean) / variance) ** 2)
        return density 

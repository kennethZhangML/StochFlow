from TD_VelocityFields.optimization import optimize_velocity_field

import numpy as np

class StochasticProcess:
    def __init__(self, initial_velocity, forward_drift, diffusivity):
        self.velocity_field = initial_velocity
        self.forward_drift = forward_drift
        self.diffusivity = diffusivity
    
    def evolve_density(self, density, time_step):
        new_density = density.get_density() - time_step * self.velocity_field * density.get_density_derivative()
        new_density += time_step * self.forward_drift * density.get_density() + self.diffusivity * density.get_density_second_derivative()
        density.update_density(new_density)
    
    def optimize_velocity(self, initial_density, regularization_lambda):
        optimized_velocity = optimize_velocity_field(self.velocity_field, initial_density, regularization_lambda)
        self.velocity_field = optimized_velocity

class Likelihood:
    def __call__(self, density, samples):
        total_likelihood = 0
        for sample in samples:
            total_likelihood += np.log(density.get_density(sample))
        return total_likelihood / len(samples)

class CrossEntropy:
    def __call__(self, initial_density, final_density, samples):
        total_cross_entropy = 0
        for sample in samples:
            initial_density_value = initial_density.get_density(sample)
            final_density_value = final_density.get_density(sample)
            total_cross_entropy += np.log(final_density_value / initial_density_value)
        return total_cross_entropy / len(samples)
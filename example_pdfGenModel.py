from COV_DetFlow.GaussianMixtureModel import GaussianMixtureModel
from COV_DetFlow.StochasticProcess import StochasticProcess 
from COV_DetFlow.StochasticProcess import Likelihood, CrossEntropy
from FokkerPlanckModels.forward_fokker_planck import ForwardFokkerPlanck, BackwardFokkerPlanck

import numpy as np

initial_density = GaussianMixtureModel(means = [-2, 2], variances = [0.5, 0.5], weights = [0.5, 0.5])
final_density = GaussianMixtureModel(means = [0, 3], variances = [0.7, 0.8], weights = [0.7, 0.3])

def fokker_planck_equation(density):
    return density.get_density_derivative() + density.get_density_second_derivative()

time_step = 0.01
num_steps = 100
regularization_lambda = 0.1  

forward_drift = np.array([0.1, 0.2, -0.1, 0.0, 0.1])
initial_velocity = np.array([0.1, 0.2, 0.0, -0.1, -0.2])

diffusivity = 0.01

stochastic_process = StochasticProcess(initial_velocity, forward_drift, diffusivity)

stochastic_process.optimize_velocity(initial_density, regularization_lambda)

forward_solver = ForwardFokkerPlanck(fokker_planck_equation, initial_density, time_step, num_steps)
forward_solver.solve()
forward_samples = forward_solver.get_solution()

likelihood_forward = Likelihood()(final_density, forward_samples)

cross_entropy_estimate = CrossEntropy()(initial_density, final_density, forward_samples)

print("Likelihood of the forward generative model:", likelihood_forward)
print("Cross-entropy between initial and final densities:", cross_entropy_estimate)

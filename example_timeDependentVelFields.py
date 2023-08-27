import numpy as np
from TD_VelocityFields.quadratic_objective import quadratic_objective
from TD_VelocityFields.optimization import optimize_velocity_field

from FokkerPlanckModels.forward_fokker_planck import ForwardFokkerPlanck, BackwardFokkerPlanck

forward_drift = np.array([0.1, 0.2, -0.1, 0.0, 0.1])
diffusivity = 0.01

class StochasticProcess:
    def __init__(self, initial_velocity, forward_drift, diffusivity):
        self.velocity_field = initial_velocity
        self.forward_drift = forward_drift
        self.diffusivity = diffusivity
    
    def evolve_density(self, density, time_step):
        new_density = density.get_density() - time_step * np.gradient(self.velocity_field * density.get_density())
        new_density += time_step * np.gradient(self.forward_drift * density.get_density()) + self.diffusivity * np.gradient(np.gradient(density.get_density()))
        density.update_density(new_density)
    
    def optimize_velocity(self, initial_density, regularization_lambda):
        optimized_velocity = optimize_velocity_field(self.velocity_field, initial_density, regularization_lambda)
        self.velocity_field = optimized_velocity

def fokker_planck_equation(density):
    drift_term = forward_drift * np.gradient(density)
    diffusion_term = diffusivity * np.gradient(np.gradient(density))
    return drift_term + diffusion_term

initial_density = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
initial_velocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  
final_density = np.array([0.2, 0.1, 0.3, 0.1, 0.2])

time_step = 0.01
num_steps = 100
regularization_lambda = 0.1  


stochastic_process = StochasticProcess(initial_velocity, forward_drift, diffusivity)
stochastic_process.optimize_velocity(initial_density, regularization_lambda)

forward_solver = ForwardFokkerPlanck(fokker_planck_equation, initial_density, time_step, num_steps)
forward_solver.solve()
forward_samples = forward_solver.get_solution()

backward_solver = BackwardFokkerPlanck(fokker_planck_equation, final_density, time_step, num_steps)
backward_solver.solve()
backward_samples = backward_solver.get_solution()

print("Backward Samples: ", backward_samples)
print("Forward Samples: ", forward_samples)



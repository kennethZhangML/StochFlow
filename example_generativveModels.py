
import numpy as np
from FokkerPlanckModels.numerical_solver import NumericalSolver
from FokkerPlanckModels.forward_fokker_planck import ForwardFokkerPlanck, BackwardFokkerPlanck

initial_density = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
final_density = np.array([0.2, 0.1, 0.3, 0.1, 0.2])

def forward_equation(density):
    return -0.1 * density  # Example equation for demonstration

def backward_equation(density):
    return 0.1 * density  # Example equation for demonstration

time_step = 0.01
num_steps = 100

# Forward generative model
forward_solver = ForwardFokkerPlanck(forward_equation, initial_density, time_step, num_steps)
forward_solver.solve()
forward_samples = forward_solver.get_solution()

# Backward generative model
backward_solver = BackwardFokkerPlanck(backward_equation, final_density, time_step, num_steps)
backward_solver.solve()
backward_samples = backward_solver.get_solution()

print("Forward Generative Model Samples:")
for sample in forward_samples:
    print(sample)

print("\nBackward Generative Model Samples:")
for sample in backward_samples:
    print(sample)

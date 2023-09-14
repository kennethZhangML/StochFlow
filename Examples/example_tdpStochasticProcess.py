import numpy as np
import TemporalDependencyFunctions.time_dependent_pdf as tdp
import TemporalDependencyFunctions.stoch_process as sp
from Utils.util import calculate_total_variation

initial_density = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
time_dependent_pdf = tdp.TimeDependentPDF(initial_density)

velocity_field = np.array([0.1, 0.0, -0.1, 0.0, 0.1])
forward_drift = velocity_field + 0.05
diffusivity = 0.01

stochastic_process = sp.StochasticProcess(velocity_field, forward_drift, diffusivity)

time_step = 0.01
num_iterations = 100

previous_density = initial_density.copy()
total_variation_sum = 0.0

for _ in range(num_iterations):
    stochastic_process.evolve_density(time_dependent_pdf, time_step)
    current_density = time_dependent_pdf.get_density()

    total_variation = calculate_total_variation(previous_density, current_density)
    total_variation_sum += total_variation

    previous_density = current_density.copy()

    print(f"Total Variation: {total_variation:.6f}")

average_total_variation = total_variation_sum / num_iterations
print(f"Average Total Variation: {average_total_variation:.6f}")

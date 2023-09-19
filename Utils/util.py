"""
I think it might be worth creating a util file to store functions that would
otherwise be duplicated; but, I didn't remove the repeat function from the
code in SIM_Text because I wasn't sure how it might affect the behavior of
solv_ivp() and quad() since they take the functions as arguments.
"""
import numpy as np


def time_derivative(t, y, interpolant_func, diffusivity_func):
    interpolant = interpolant_func(t)
    diffusivity = diffusivity_func(t)
    return interpolant(y) * diffusivity(y)  # dydt


def integrand(x, initial_model, final_model):
    initial_density = initial_model.likelihood(x.reshape(-1, 1))
    final_density = final_model.likelihood(x.reshape(-1, 1))
    return initial_density * np.log(initial_density / final_density)


def interpolant_func(t):
    return lambda x: x * (1 - t) + t * np.sin(x)


def diffusivity_func(t):
    return lambda x: 1.0 + t * np.cos(x)


def calculate_total_variation(density1, density2):
    return np.sum(np.abs(density1 - density2))


def forward_equation(density):
    """
    Example equation for demonstration
    """
    return -0.1 * density


def backward_equation(density):
    """
    Example equation for demonstration
    """
    return 0.1 * density

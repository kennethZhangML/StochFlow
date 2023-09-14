import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import quad
import matplotlib.pyplot as plt
from Utils.util import interpolant_func, diffusivity_func


class StochasticInterpolantModel:
    def __init__(self, initial_density, final_density, interpolant_func,
                 diffusivity_func, time_interval, time_step):
        self.initial_density = initial_density
        self.final_density = final_density
        self.interpolant_func = interpolant_func
        self.diffusivity_func = diffusivity_func
        self.time_interval = time_interval
        self.time_step = time_step

    def generate_samples(self, num_samples):
        initial_samples = self.initial_density.sample(num_samples)
        return self._integrate_samples(initial_samples)

    def _integrate_samples(self, samples):
        def time_derivative(t, y):
            interpolant = self.interpolant_func(t)
            diffusivity = self.diffusivity_func(t)
            dydt = interpolant(y) * diffusivity(y)
            return dydt

        samples_integrated = []
        for sample in samples:
            sol = solve_ivp(time_derivative, [0, self.time_interval], sample,
                            t_eval=[self.time_interval], method='RK45')
            samples_integrated.append(sol.y[:, -1])

        return np.array(samples_integrated)

    def likelihood(self, samples):
        return self.final_density.likelihood(samples)

    def cross_entropy(self):
        def integrand(x):
            initial_density_value = self.initial_density.likelihood(x.reshape(-1, 1))
            final_density_value = self.final_density.likelihood(x.reshape(-1, 1))
            return initial_density_value * np.log(initial_density_value / final_density_value)

        integral_result, _ = quad(integrand, -np.inf, np.inf)
        return integral_result


class DensityFunction:
    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance

    def sample(self, num_samples):
        return np.random.multivariate_normal(self.mean, self.covariance, num_samples)

    def likelihood(self, samples):
        inv_covariance = np.linalg.inv(self.covariance)
        diff = samples - self.mean
        exponent = -0.5 * np.sum(diff.dot(inv_covariance) * diff, axis=1)
        return np.exp(exponent) / np.sqrt(np.linalg.det(self.covariance) * (2 * np.pi) ** self.mean.shape[0])


if __name__ == "__main__":
    initial_mean = np.array([0.0])
    initial_covariance = np.array([[1.0]])
    final_mean = np.array([3.0])
    final_covariance = np.array([[0.5]])

    initial_density = DensityFunction(initial_mean, initial_covariance)
    final_density = DensityFunction(final_mean, final_covariance)

    time_interval = 1.0
    time_step = 0.01

    model = StochasticInterpolantModel(initial_density, final_density, interpolant_func,
                                       diffusivity_func, time_interval, time_step)
    samples = model.generate_samples(num_samples=1000)
    likelihood = model.likelihood(samples)

    print(f"Likelihood given samples: {likelihood}")
    print(f"Samples: {samples}")

    fig = plt.figure()

    plt.hist(samples, bins=30, density=True, alpha=0.7, color='blue', label='Generated Samples')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Generated Samples')
    plt.legend()
    plt.show()

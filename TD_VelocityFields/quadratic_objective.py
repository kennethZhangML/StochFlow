import numpy as np


class QuadraticObjective:

    def __init__(self, b, rho, regularization_lambda):
        self.b, self.rho = b, rho
        self.regularization_lambda = regularization_lambda

    def b_norm(self):
        gradient_norm_term = np.sum(np.square(np.gradient(self.b) * self.rho))
        b_norm_term = np.sum(np.square(self.b) * self.rho)
        return gradient_norm_term + self.regularization_lambda * b_norm_term

    def velocity_magnitude(self):
        gradient_norm_term = np.sum(np.square(np.gradient(self.b) * self.rho.get_density_derivative()))
        velocity_magnitude_term = np.sum(np.square(self.b * self.rho.get_density()))
        return gradient_norm_term + self.regularization_lambda * velocity_magnitude_term

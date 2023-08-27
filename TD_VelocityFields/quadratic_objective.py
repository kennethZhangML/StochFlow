import numpy as np

def quadratic_objective(b, rho, regularization_lambda):
    gradient_norm_term = np.sum(np.square(np.gradient(b) * rho))
    b_norm_term = np.sum(np.square(b) * rho)
    
    return gradient_norm_term + regularization_lambda * b_norm_term

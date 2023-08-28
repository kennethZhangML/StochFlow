from scipy.optimize import minimize
from TD_VelocityFields.quadratic_objective import quadratic_objective

def optimize_velocity_field(initial_b, rho, regularization_lambda):
    def objective_function(b):
        return quadratic_objective(b, rho, regularization_lambda)
    
    result = minimize(objective_function, initial_b, method = 'L-BFGS-B')
    optimized_b = result.x
    return optimized_b

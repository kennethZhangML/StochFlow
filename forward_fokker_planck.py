from numerical_solve import NumericalSolver

class ForwardFokkerPlanck(NumericalSolver):
    def __init__(self, equation, initial_condition, time_step, num_steps):
        super().__init__(equation, initial_condition, time_step, num_steps)
    
    def step(self):
        current_solution = self.solution[-1]
        new_solution = current_solution + self.time_step * self.equation(current_solution)
        return new_solution

class NumericalSolver:
    def __init__(self, equation, initial_condition, time_step, num_steps):
        self.equation = equation
        self.time_step = time_step
        self.num_steps = num_steps
        self.solution = [initial_condition]
    
    def solve(self):
        for _ in range(self.num_steps):
            new_solution = self.step()
            self.solution.append(new_solution)
    
    def step(self):
        current_soln = self.soln[-1]
        new_soln = current_soln + self.time_step * self.equation(current_soln)
        return new_soln

    def get_solution(self):
        return self.solution

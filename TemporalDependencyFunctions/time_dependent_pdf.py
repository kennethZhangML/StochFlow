# time_dependent_pdf.py

import numpy as np

class TimeDependentPDF:
    def __init__(self, initial_density):
        self.density = initial_density
    
    def update_density(self, new_density):
        self.density = new_density
    
    def get_density(self):
        return self.density
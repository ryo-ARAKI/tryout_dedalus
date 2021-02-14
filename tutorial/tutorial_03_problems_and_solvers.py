"""
Tutorial 3: Problems and Solvers
https://dedalus-project.readthedocs.io/en/latest/notebooks/dedalus_tutorial_problems_solvers.html
"""

import numpy as np
import matplotlib.pyplot as plt
from dedalus import public as de

# Build bases and domain
x_basis = de.Chebyshev('x', 1024, interval=(0, 300), dealias=2)
domain = de.Domain([x_basis], grid_dtype=np.complex128)

# Build problem
problem = de.IVP(  # IVP: Initial Value Problem
    domain,  # 1D domain
    variables=['u', 'ux']  # First-order formulation
)


# Variable metadata
# problem.meta['u']['x']['parity'] = +1  # Even parity along x axis of the quantity u
# problem.meta['ux']['x']['parity'] = -1  # Odd parity along x axis of the quantity ux


# Parameters and non-constant coefficients
problem.parameters['b'] = 0.5
problem.parameters['c'] = -1.76

# Define Non-Constant Coefficients (NCC)
# ncc = Field(domain, name='c')
# ncc['g'] = z**2
# ncc.meta['x', 'y']['constant'] = True
# problem.parameters['c'] = ncc


# Substitutions
# Function-like substitution using dummy variables
problem.substitutions["mag_sq(A)"] = "A * conj(A)"


# Equation entry
# Add main equation, with linear terms on the LHS and nonlinear terms on the RHS
# One-dimensional Complex Ginzburg-Landau Equation (1D-CGLE)
problem.add_equation("dt(u) - u - (1 + 1j*b) * dx(ux) = - (1 + 1j*c) * mag_sq(u) * u")

# Add auxiliary equation defining the first-order reduction
problem.add_equation("ux - dx(u) = 0")  # Define first-order spatial derivative

# Add boundary conditions
# Dirichlet b.c. for both ends
problem.add_equation("left(u) = 0")
problem.add_equation("right(u) = 0")

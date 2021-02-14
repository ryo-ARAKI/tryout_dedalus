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


# Building a solver
solver = problem.build_solver(
    'RK222'  # 2nd-order 2-stage Diagonally Implicit Runge–Kutta (DIRK) + Explicit Runge–Kutta ERK scheme
)


# Setting stop criteria
solver.stop_sim_time = 500
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf


# Setting initial conditions
# Reference local grid and state fields
x = domain.grid(0)
u = solver.state['u']
ux = solver.state['ux']

# Setup a sine wave
u.set_scales(1)
u['g'] = 1e-3 * np.sin(5 * np.pi * x / 300)
u.differentiate('x', out=ux)


# Solving/iterating a problem
# Setup storage
u.set_scales(1)
u_list = [np.copy(u['g'])]
t_list = [solver.sim_time]

# Main loop
dt = 0.05
while solver.ok:
    solver.step(dt)
    if solver.iteration % 10 == 0:
        u.set_scales(1)
        u_list.append(np.copy(u['g']))
        t_list.append(solver.sim_time)
    if solver.iteration % 1000 == 0:
        print('Completed iteration {}'.format(solver.iteration))

# Convert storage lists to arrays
t_array = np.array(t_list)
u_array = np.array(u_list)


# Plot solution
plt.close()
plt.figure(figsize=(6, 7), dpi=100)
plt.pcolormesh(
    x, t_array, np.abs(u_array),
    cmap=plt.get_cmap('plasma'),
    shading='nearest'
)
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title('Hole-defect chaos in the CGLE: |u|')
plt.tight_layout()
plt.savefig("fig/Hole-defect_chaos_in_the_CGLE_u.png")

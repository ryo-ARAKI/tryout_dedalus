import pathlib
import subprocess
import h5py
import numpy as np
import matplotlib.pyplot as plt
from dedalus import public as de

# Clean up any old files
import shutil
shutil.rmtree('analysis', ignore_errors=True)

# Build bases and domain
x_basis = de.Chebyshev('x', 1024, interval=(0, 300), dealias=2)
domain = de.Domain([x_basis], grid_dtype=np.complex128)

# Build problem
problem = de.IVP(domain, variables=['u', 'ux'])
problem.parameters['b'] = 0.5
problem.parameters['c'] = -1.76
problem.substitutions["mag_sq(A)"] = "A * conj(A)"
problem.add_equation(
    "dt(u) - u - (1 + 1j*b)*dx(ux) = - (1 + 1j*c) * mag_sq(u) * u"
)
problem.add_equation("ux - dx(u) = 0")
problem.add_equation("left(u) = 0")
problem.add_equation("right(u) = 0")

# Build solver
solver = problem.build_solver('RK222')
solver.stop_sim_time = 500
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# Reference local grid and state fields
x = domain.grid(0)
u = solver.state['u']
ux = solver.state['ux']

# Setup a sine wave
u.set_scales(1)
u['g'] = 1e-3 * np.sin(5 * np.pi * x / 300)
u.differentiate('x', out=ux)


# Analysis handlers
analysis = solver.evaluator.add_file_handler(
    'analysis',  # Output directory name
    iter=10,  # Interval to conduct the analysis
    max_writes=200  # number of writes in each "sets" of divided file handler
)


# Analysis tasks
analysis.add_task(
    "integ(sqrt(mag_sq(u)), 'x') / 300",  # Operation
    layout='g',  # Conducted in grid space?
    name='<|u|>'
)

analysis.add_system(solver.state, layout='g')

# Main loop
dt = 0.05
while solver.ok:
    solver.step(dt)
    if solver.iteration % 1000 == 0:
        print('Completed iteration {}'.format(solver.iteration))

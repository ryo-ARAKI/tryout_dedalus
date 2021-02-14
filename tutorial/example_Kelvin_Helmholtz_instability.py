"""
Kelvin-Helmholtz Instability
https://dedalus-project.readthedocs.io/en/latest/notebooks/KelvinHelmholtzInstability.html
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
from IPython import display

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Problem domain
Lx, Ly = (2., 1.)  # Aspect ratio = 2
nx, ny = (192, 96)

# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Chebyshev('y', ny, interval=(-Ly/2, Ly/2), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# Parameters
Reynolds = 1e4
Schmidt = 1.

# Build problem
problem = de.IVP(
    domain,
    variables=['p', 'u', 'v', 'uy', 'vy', 'S', 'Sy']
)
problem.parameters['Re'] = Reynolds
problem.parameters['Sc'] = Schmidt
problem.add_equation("dt(u) + dx(p) - 1/Re * (dx(dx(u)) + dy(uy)) = - u*dx(u) - v*uy")
problem.add_equation("dt(v) + dy(p) - 1/Re * (dx(dx(v)) + dy(vy)) = - u*dx(v) - v*vy")
problem.add_equation("dt(S) - 1/(Re*Sc) * (dx(dx(S)) + dy(Sy)) = - u*dx(S) - v*Sy")
problem.add_equation("dx(u) + vy = 0")
# Auxiliary variables for y-derivatives
problem.add_equation("uy - dy(u) = 0")
problem.add_equation("vy - dy(v) = 0")
problem.add_equation("Sy - dy(S) = 0")

# Set boundary condisions
problem.add_bc("left(u) = 0.5")
problem.add_bc("right(u) = -0.5")
problem.add_bc("left(v) = 0")
problem.add_bc("right(v) = 0", condition="(nx != 0)")
problem.add_bc("left(p) = 0", condition="(nx == 0)")  # Gauge choice for the pressure, originates from kx=0 b.c. of v
problem.add_bc("left(S) = 0")
problem.add_bc("right(S) = 1")


# Timestepping
ts = de.timesteppers.RK443


# Initial Value Problem
solver = problem.build_solver(ts)

# Set coordinates/variables
x = domain.grid(0)
y = domain.grid(1)
u = solver.state['u']
uy = solver.state['uy']
v = solver.state['v']
vy = solver.state['vy']
S = solver.state['S']
Sy = solver.state['Sy']

# Set initial conditions
a = 0.05
sigma = 0.2
flow = -0.5
amp = -0.2
u['g'] = flow * np.tanh(y/a)
v['g'] = amp * np.sin(2.0*np.pi*x/Lx) * np.exp(-(y*y)/(sigma*sigma))
S['g'] = 0.5 * (1 + np.tanh(y/a))
u.differentiate('y', out=uy)
v.differentiate('y', out=vy)
S.differentiate('y', out=Sy)

# Set parameters and CFL condition
solver.stop_sim_time = 2.01
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

initial_dt = 0.2*Lx/nx
cfl = flow_tools.CFL(solver, initial_dt, safety=0.8)
cfl.add_velocities(('u', 'v'))


# Analysis
analysis = solver.evaluator.add_file_handler(
    'analysis_tasks',
    sim_dt=0.1,
    max_writes=50
)
analysis.add_task('S')
analysis.add_task('u')
solver.evaluator.vars['Lx'] = Lx
analysis.add_task("integ(S, 'x')/Lx", name='S profile')


# Main loop
# Make plot of scalar field
x = domain.grid(0, scales=domain.dealias)
y = domain.grid(1, scales=domain.dealias)
xm, ym = np.meshgrid(x, y)
fig, ax = plt.subplots(figsize=(10, 5))
p = ax.pcolormesh(
    xm, ym, S['g'].T,
    cmap='bwr'
)
ax.set_xlim([0, 2.])
ax.set_ylim([-0.5, 0.5])

logger.info('Starting loop')
start_time = time.time()
while solver.ok:
    dt = cfl.compute_dt()
    solver.step(dt)
    if solver.iteration % 10 == 0:
        # Update plot of scalar field
        p.set_array(np.ravel(S['g'][:-1, :-1].T))
        display.clear_output()
        display.display(plt.gcf())
        logger.info('Iteration: %i, Time: %e, dt: %e' % (solver.iteration, solver.sim_time, dt))
        plt.savefig("fig/s_itr={:d}.png".format(solver.iteration))

plt.close()
end_time = time.time()

p.set_array(np.ravel(S['g'][:-1, :-1].T))
display.clear_output()

# Print statistics
logger.info('Run time: %f' % (end_time - start_time))
logger.info('Iterations: %i' % solver.iteration)


# Analysis
# Read in the data
f = h5py.File(
    'analysis_tasks/analysis_tasks_s1/analysis_tasks_s1_p0.h5',
    'r'
)
y = f['/scales/y/1.0'][:]
t = f['/scales']['sim_time'][:]
S_ave = f['tasks']['S profile'][:]
f.close()

S_ave = S_ave[:,0,:]  # Remove length=1 x dimension

for i in range(0, 21, 5):
    plt.plot(
        S_ave[i, :], y,
        label='t=%4.2f' %t[i]
    )

plt.ylim([-0.5, 0.5])
plt.xlim([0, 1])
plt.xlabel(r'$\frac{\int S dx}{L_x}$', fontsize=24)
plt.ylabel(r'$y$', fontsize=24)
plt.legend(loc='lower right').draw_frame(False)
plt.tight_layout()
plt.savefig("fig/s_profile.png")

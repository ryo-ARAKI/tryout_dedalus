"""
Turotial 1: Bases and Domains
https://dedalus-project.readthedocs.io/en/latest/notebooks/dedalus_tutorial_bases_domains.html
"""

import numpy as np
import matplotlib.pyplot as plt
from dedalus import public as de


# Creating basis
xbasis = de.Chebyshev(
    'x',  # name of the spatial domain
    32,  # number of modes
    interval=(0, 1),  # range of the domain
    dealias=3/2  # dealialing scale factor
)


# Basis grids and scale factors
grid_normal = xbasis.grid(scale=1)
grid_dealias = xbasis.grid(scale=3/2)

plt.close()
plt.figure(figsize=(6, 1.5), dpi=100)
plt.plot(grid_normal, 0*grid_normal+1, 'o', markersize=5)
plt.plot(grid_dealias, 0*grid_dealias-1, 'o', markersize=5)
plt.xlabel('x')
plt.title('Chebyshev grid with scales 1 and 3/2')
plt.ylim([-2, 2])
plt.gca().yaxis.set_ticks([]);
plt.tight_layout()
plt.savefig("fig/Chebyshev_grid_with_scales_1_and_1.5.png")


# Compound bases
xb1 = de.Chebyshev('x1', 16, interval=(0, 2))
xb2 = de.Chebyshev('x2', 32, interval=(2, 8))
xb3 = de.Chebyshev('x3', 16, interval=(8, 10))
xbasis = de.Compound('x', (xb1, xb2, xb3))  # Define compound bases by three individual bases

compound_grid = xbasis.grid(scale=1)

plt.close()
plt.figure(figsize=(6, 1.5), dpi=100)
plt.plot(compound_grid, 0*compound_grid, 'o', markersize=5)
plt.xlabel('x')
plt.title('Compound Chebyshev grid')
plt.gca().yaxis.set_ticks([]);
plt.tight_layout()
plt.savefig("fig/Compound_Chebyshev_grid.png")

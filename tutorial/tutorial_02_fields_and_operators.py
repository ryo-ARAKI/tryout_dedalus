"""
Tutorial 2: Fields and Operators
https://dedalus-project.readthedocs.io/en/latest/notebooks/dedalus_tutorial_fields_operators.html
"""

import numpy as np
import matplotlib.pyplot as plt
from dedalus import public as de
from dedalus.extras.plot_tools import plot_bot_2d
figkw = {'figsize':(6, 4), 'dpi':100}


# Creating a field
xbasis = de.Fourier('x', 64, interval=(-np.pi, np.pi), dealias=3/2)
ybasis = de.Chebyshev('y', 64, interval=(-1, 1), dealias=3/2)
domain = de.Domain([xbasis, ybasis], grid_dtype=np.float64)
f = de.Field(domain, name='f')
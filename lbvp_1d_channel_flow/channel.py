import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Parameters
Ny = 64
dtype = np.float64

# Bases
coord = d3.Coordinate('y')
dist = d3.Distributor(coord, dtype=dtype)
basis = d3.Chebyshev(coord, size=Ny, bounds=(-1, 1))

# Fields
u = dist.Field(name='u', bases=basis)
tau_u1 = dist.Field(name='tau_u1') # y = -1
tau_u2 = dist.Field(name='tau_u2') # y = +1

# Substitutions
y = dist.local_grid(basis)
dy = lambda A: d3.Differentiate(A, coord)
lift_basis = basis.derivative_basis(2)
lift = lambda A, n: d3.Lift(A, lift_basis, n)

# Problem
problem = d3.LBVP([u, tau_u1, tau_u2], namespace=locals())
problem.add_equation("dy(dy(u)) + lift(tau_u1, -1) + lift(tau_u2, -2) = -2")
problem.add_equation("u(y=-1) = 0")
problem.add_equation("u(y=1) = 0")

# Solver
solver = problem.build_solver()
solver.solve()

fig, ax = plt.subplots()
yy = np.linspace(-1, 1, 1001)
ax.plot(y, u['g'], 'ko', label='Dedalus')
ax.plot(yy, 1 - yy**2, 'k-', label=r'$u=1-y^2$')
ax.legend()
fig.savefig('channel.png')

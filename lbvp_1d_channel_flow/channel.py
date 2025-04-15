import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Parameters
Nz = 64
dtype = np.float64

# Bases
zcoord = d3.Coordinate('z')
dist = d3.Distributor(zcoord, dtype=dtype)
basis = d3.Chebyshev(zcoord, size=Nz, bounds=(-1, 1))

# Fields
u = dist.Field(name='u', bases=basis)
tau_u1 = dist.Field(name='tau_u1') # z = -1
tau_u2 = dist.Field(name='tau_u2') # z = +1

# Substitutions
z = dist.local_grid(basis)
dz = lambda A: d3.Differentiate(A, zcoord)
lift_basis = basis.derivative_basis(2)
lift = lambda A, n: d3.Lift(A, lift_basis, n)

# Problem
problem = d3.LBVP([u, tau_u1, tau_u2], namespace=locals())
problem.add_equation("dz(dz(u)) + lift(tau_u1, -1) + lift(tau_u2, -2) = -2")
problem.add_equation("u(z=-1) = 0")
problem.add_equation("u(z=1) = 0")

# Solver
solver = problem.build_solver()
solver.solve()

fig, ax = plt.subplots()
zz = np.linspace(-1, 1, 1001)
ax.plot(z, u['g'], 'ko', label='Dedalus')
ax.plot(zz, 1 - zz**2, 'k-', label=r'$u=1-y^2$')
ax.legend()
fig.savefig('channel.png')

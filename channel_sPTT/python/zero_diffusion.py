import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Parameters
Ny = 1024
dtype = np.float64
beta = 0.8
Wi = 80.0
epsilon = 1e-3 # sPTT

# Bases
ycoord = d3.Coordinate('y')
dist = d3.Distributor(ycoord, dtype=dtype)
basis = d3.Chebyshev(ycoord, size=Ny, bounds=(-1, 1))

# Fields
U = dist.Field(name='U', bases=basis)
tau_u1 = dist.Field(name='tau_u1') # z = -1
tau_u2 = dist.Field(name='tau_u2') # z = +1

# know stuff
a_xx = dist.Field(name='a_xx', bases=basis)
a_xy = dist.Field(name='a_xy', bases=basis)


# Substitutions
y = dist.local_grid(basis)
dy = lambda A: d3.Differentiate(A, ycoord) 
A = 1/(6*beta*epsilon); B = Wi/(2*beta*epsilon)
Fm = lambda y: np.cbrt(-B*y - np.sqrt(A**3 + (B*y)**2))
Fp = lambda y: np.cbrt(-B*y + np.sqrt(A**3 + (B*y)**2))
a_xy['g'] = Fm(y) + Fp(y)
a_xx['g'] = 2*(a_xy['g'])**2 + 1
da_xy = dy(a_xy)
# tau terms
lift_basis = basis.derivative_basis(2)
lift = lambda A, n: d3.Lift(A, lift_basis, n)

# Problem
problem = d3.LBVP([U, tau_u1, tau_u2], namespace=locals())
problem.add_equation("dy(dy(U)) + lift(tau_u1, -1) + lift(tau_u2, -2) = -2/beta - (1 - beta)/(beta*Wi)*da_xy")
problem.add_equation("U(y=-1) = 0")
problem.add_equation("U(y=1) = 0")

# Solver
solver = problem.build_solver()
solver.evaluate
solver.solve()

data = np.vstack((y, U['g'], a_xx['g'], a_xy['g']))
np.savetxt('zero_diffusion.txt', data.T)
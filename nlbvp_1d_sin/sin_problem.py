import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Parameters
N = 12
ncc_cutoff = 1e-6
tolerance = 1e-6
dealias = 1.5
dtype = np.float64
a = 0
b = 0

# Basis
coord = d3.Coordinate('x')
dist = d3.Distributor(coord, dtype=dtype)
basis = d3.Jacobi(coord, size=N, a = a, b = b, bounds=(0, 1), dealias=dealias)
x = dist.local_grid(basis, scale=1)

# Fields
u = dist.Field(bases=basis)
tau = dist.Field()

# Substitutions
dx = lambda A: d3.Differentiate(A, coord)
lift_basis = basis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)

# Problem
problem = d3.NLBVP([u, tau], namespace=locals())
problem.add_equation("dx(u)**2 + u**2 + lift(tau) = 1")
problem.add_equation("u(x=0) = 1")

# Solver
solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
u['g'] = 1 - x/2
error = np.inf
while error > tolerance:
    solver.newton_iteration()
    error = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
    print(error)
    if solver.iteration > 20:
        assert False

# Check solution
u_true = np.cos(x)
u.change_scales(1)
print(np.allclose(u['g'], u_true))


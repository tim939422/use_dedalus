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
kappa = 5e-5 # conformation diffusion
dealias = 1.5
ncc_cutoff = 1e-6
tolerance = 1e-10

# Bases
ycoord = d3.Coordinate('y')
dist = d3.Distributor(ycoord, dtype=dtype)
basis = d3.Chebyshev(ycoord, size=Ny, bounds=(-1, 1), dealias=dealias)

'''Fields'''
U = dist.Field(name='U', bases=basis)
a_xx = dist.Field(name='a_xx', bases=basis)
a_xy = dist.Field(name='a_xy', bases=basis)
# z = -1
tau_u1 = dist.Field(name='tau_u1')
tau_a_xx1 = dist.Field(name="tau_a_xx1")
tau_a_xy1 = dist.Field(name="tau_a_xy1") 
# z = +1
tau_u2 = dist.Field(name='tau_u2')
tau_a_xx2 = dist.Field(name="tau_a_xx2")
tau_a_xy2 = dist.Field(name="tau_a_xy2") # z = +1

'''Substitutions'''
# differential operator and tau terms
dy = lambda A: d3.Differentiate(A, ycoord)
lift_basis = basis.derivative_basis(2)
lift = lambda A, n: d3.Lift(A, lift_basis, n)

# analytical solution of a_xx and a_xy (kappa = 0)
y = dist.local_grid(basis, scale=1)
A = 1/(6*beta*epsilon); B = Wi/(2*beta*epsilon)
Fm = lambda y: np.cbrt(-B*y - np.sqrt(A**3 + (B*y)**2))
Fp = lambda y: np.cbrt(-B*y + np.sqrt(A**3 + (B*y)**2))
a_xy['g'] = Fm(y) + Fp(y)
a_xx['g'] = 2*(a_xy['g'])**2 + 1
a_xy_lower = Fm(-1) + Fp(-1); a_xx_lower = 2*a_xy_lower**2 + 1
a_xy_upper = Fm( 1) + Fp( 1); a_xx_upper = 2*a_xy_upper**2 + 1

# Problem
lbvp_problem = d3.LBVP([U, tau_u1, tau_u2], namespace=locals())
lbvp_problem.add_equation("dy(dy(U)) + lift(tau_u1, -1) + lift(tau_u2, -2) = -2/beta - (1 - beta)/(beta*Wi)*dy(a_xy)")
lbvp_problem.add_equation("U(y=-1) = 0")
lbvp_problem.add_equation("U(y= 1) = 0")

# Solver
lbvp_solver = lbvp_problem.build_solver()
lbvp_solver.solve()

data_ref = np.loadtxt('prl_2022_Morozov_Fig2b.txt')
U.change_scales(1)
a_xx.change_scales(1)
a_xy.change_scales(1)

fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
ax[0].plot(data_ref[:, 0], data_ref[:, 1], 'ko', label='Ref', markerfacecolor='none')
ax[0].plot(y, U['g'], 'k--', label=r"$\kappa=0$")
ax[1].plot(y, a_xx['g'], 'k--')
ax[2].plot(y, a_xy['g'], 'k--')

# add diffusion
nlbvp_problem = d3.NLBVP([U, a_xx, a_xy, tau_u1, tau_u2, tau_a_xx1, tau_a_xx2, tau_a_xy1, tau_a_xy2], namespace=locals())
nlbvp_problem.add_equation("beta*dy(dy(U)) + lift(tau_u1, -1) + lift(tau_u2, -2) + (1 - beta)/Wi*dy(a_xy) + 2 = 0")
nlbvp_problem.add_equation("a_xy/Wi*(epsilon*(a_xx - 1) + 1) - kappa*dy(dy(a_xy)) + lift(tau_a_xy1, -1) + lift(tau_a_xy2, -2) = dy(U)")
nlbvp_problem.add_equation("(a_xx - 1)/Wi*(epsilon*(a_xx - 1) + 1) - kappa*dy(dy(a_xx)) + lift(tau_a_xx1, -1) + lift(tau_a_xx2, -2) = 2*a_xy*dy(U)")
nlbvp_problem.add_equation("U(y=-1) = 0")
nlbvp_problem.add_equation("U(y= 1) = 0")
nlbvp_problem.add_equation("a_xx(y=-1) = a_xx_lower")
nlbvp_problem.add_equation("a_xy(y=-1) = a_xy_lower")
nlbvp_problem.add_equation("a_xx(y= 1) = a_xx_upper")
nlbvp_problem.add_equation("a_xy(y= 1) = a_xy_upper")

solver = nlbvp_problem.build_solver(ncc_cutoff=ncc_cutoff)
error = np.inf
while error > tolerance:
    solver.newton_iteration()
    error = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
    logger.info(f'Perturbation norm: {error:.3e}')

U.change_scales(1)
a_xx.change_scales(1)
a_xy.change_scales(1)
ax[0].plot(y, U['g'], 'k-', label=rf"$\kappa={kappa:.3e}$")
ax[1].plot(y, a_xx['g'], 'k-')
ax[2].plot(y, a_xy['g'], 'k-')

ax[0].legend()
ax[0].set_ylabel(r'$U_{\text{lam}}$')
ax[1].set_ylabel(r'$a_{xx}$')
ax[2].set_ylabel(r'$a_{xy}$')
for i in range(3):
    ax[i].set_xlabel(r"y")

fig.savefig('laminar.png', dpi=1000)

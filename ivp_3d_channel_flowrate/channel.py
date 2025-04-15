'''
created on Apr 1, 2025

@author Duosi Fan
'''

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Parameters
Lx, Ly = 5, 2
Nx, Ny, Nz = 2, 2, 256
Reb = 100.0
dealias = 3/2
stop_sim_time = 100
timestepper = d3.RK222
max_timestep=0.1

# Basis
coords =d3.CartesianCoordinates('x','y', 'z')
dist = d3.Distributor(coords,dtype=np.float64)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)
zbasis=d3.Chebyshev(coords['z'], size=Nz, bounds=(-1, 1), dealias=dealias)


# Fields
p = dist.Field(name='p', bases=(xbasis, ybasis, zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis, ybasis, zbasis))
tau_p = dist.Field(name='tau_p') # for pressure gauge
tau_u1= dist.VectorField(coords, name='tau_u1', bases=(xbasis, ybasis)) # no slip BC in continuity equatio div u = 0
tau_u2= dist.VectorField(coords, name='tau_u2', bases=(xbasis, ybasis)) # no-slip BC in momentum equation
f_dp= dist.Field(name='f_dp') # time-dependent bodyforce to enforce constant flowrate

# substitute
nu = 1/Reb
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
ex, ey, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift= lambda A: d3.Lift(A,lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction

# problem
problem=d3.IVP([u, p, tau_p, tau_u1, tau_u2,f_dp], namespace=locals())
# governing equation: continuity + momentum
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(u) + grad(p) - nu*div(grad_u)  + lift(tau_u2) + f_dp*ex = -u@grad(u)")

# pressure gauge
problem.add_equation("integ(p) = 0")

# boundary condition
problem.add_equation("u(z = -1 ) = 0")
problem.add_equation("u(z = 1) = 0")

# fixed flow rate
problem.add_equation("integ(u@ex) = 2*Lx*Ly")


# solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# initial condition
u['g'][0] = 0
u['g'][1] = 0

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=20, max_writes=100)
snapshots.add_task(u, name='velocity')
snapshots.add_task(p, name='pressure')

checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=100, max_writes=1, mode='overwrite')
checkpoints.add_tasks(solver.state)

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10) # cadence a.k.a. compute this every cadence step
flow.add_property(u@ex, name='U')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            Uc = flow.max('U')
            logger.info(f'Iteration={solver.iteration:d}, Time={solver.sim_time:e}, dt={timestep:e}, Uc={Uc:f}')
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()


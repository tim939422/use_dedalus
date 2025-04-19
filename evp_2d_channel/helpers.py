import numpy as np
import dedalus.public as d3


def orr_sommerfeld_channel(params):
    # parameters
    N = params["N"]
    alpha = params["streamwise wavenumber"]
    R = params["Reynolds number"]

    # Basis
    coord = d3.Coordinate('y')
    dist = d3.Distributor(coord, dtype=np.complex128)
    basis  = d3.Chebyshev(coord, N, bounds=(-1, 1))

    # Fields
    v = dist.Field(name='v', bases=basis)
    c = dist.Field(name='c')
    tau_1 = dist.Field(name='tau_1')
    tau_2 = dist.Field(name='tau_2')
    tau_3 = dist.Field(name='tau_3')
    tau_4 = dist.Field(name='tau_4')
    U = dist.Field(name='U', bases=basis)
    Uyy = dist.Field(name='Uyy', bases=basis)

    # substitutions
    dy = lambda A: d3.Differentiate(A, coord)
    lift_basis = basis.derivative_basis(1)
    lift = lambda A: d3.lift(A, lift_basis, -1)
    vy = dy(v) + lift(tau_1)
    vyy = dy(vy) + lift(tau_2)
    vyyy = dy(vyy) + lift(tau_3)
    vyyyy = dy(vyyy) + lift(tau_4)

    # setup the base flow and parameters
    y = dist.local_grid(basis)
    U['g'] = 1 - y**2
    Uyy['g'] = -2

    # problem
    problem = d3.EVP([v, tau_1, tau_2, tau_3, tau_4], eigenvalue=c, namespace=locals())
    problem.add_equation("vyyyy - 2*alpha**2*vyy + alpha**4*v - 1j*alpha*R*((U - c)*(vyy - alpha**2*v) - Uyy*v) = 0")
    problem.add_equation("v(y = -1) = 0")
    problem.add_equation("v(y =  1) = 0")
    problem.add_equation("vy(y = -1) = 0")
    problem.add_equation("vy(y = 1) = 0")

    return problem

def primitive_channel(params):
    # parameters
    N = params["N"]
    alpha = params["streamwise wavenumber"]
    R = params["Reynolds number"]
    
    # Basis
    coord = d3.Coordinate('y')
    dist = d3.Distributor(coord, dtype=np.complex128)
    basis  = d3.Chebyshev(coord, N, bounds=(-1, 1))

    # Fields
    u = dist.Field(name='u', bases=basis)
    v = dist.Field(name='v', bases=basis)
    p = dist.Field(name='p', bases=basis)
    c = dist.Field(name='c')

    tau_u1 = dist.Field(name='tau_u1'); tau_u2 = dist.Field(name='tau_u2')
    tau_v1 = dist.Field(name='tau_v1'); tau_v2 = dist.Field(name='tau_v2')

    U = dist.Field(name='U', bases=basis)
    Uy = dist.Field(name='Uy', bases=basis)

    # substitutions
    dy = lambda A: d3.Differentiate(A, coord)
    lift_basis = basis.derivative_basis(1)
    lift = lambda A: d3.lift(A, lift_basis, -1)
    uy = dy(u) + lift(tau_u1); uyy = dy(uy) + lift(tau_u2)
    vy = dy(v) + lift(tau_v1); vyy = dy(vy) + lift(tau_v2)
    py = dy(p)

    # setup the base flow and parameters
    y = dist.local_grid(basis)
    U['g'] = 1 - y**2
    Uy['g'] = -2*y

    # problem
    problem = d3.EVP([u, v, p, tau_u1, tau_u2, tau_v1, tau_v2], eigenvalue=c, namespace=locals())
    problem.add_equation("1j*alpha*u + vy = 0")
    problem.add_equation("1j*alpha*U*u + v*Uy + 1/R*(alpha**2*u - uyy) + 1j*alpha*p - 1j*c*alpha*u = 0")
    problem.add_equation("1j*alpha*U*v + 1/R*(alpha**2*v - vyy) + py - 1j*c*alpha*v = 0")
    problem.add_equation("u(y = -1) = 0")
    problem.add_equation("u(y =  1) = 0")
    problem.add_equation("v(y = -1) = 0")
    problem.add_equation("v(y =  1) = 0")

    return problem


def reject_spurious(eigvals_low, eigvals_high, threshold=1e7):
    # use nearest drift see
    # see Boyd, J. P. (2001). Chebyshev and Fourier spectral methods. Courier Corporation.
    # equation 7.21 on page 138

    eigvals_low = eigvals_low[~np.isinf(eigvals_low) & ~np.isnan(eigvals_low)]
    eigvals_high = eigvals_high[~np.isinf(eigvals_high) & ~np.isnan(eigvals_high)]
    sorted_eigvals_low = eigvals_low[np.argsort(eigvals_low.real)]
    sorted_eigvals_high = eigvals_high[np.argsort(eigvals_high.real)]

    n_lower = len(sorted_eigvals_low)
    sigma = np.zeros(n_lower)
    sigma[0] = np.abs(sorted_eigvals_high[0] - sorted_eigvals_high[1])
    sigma[1:n_lower] = 0.5*(np.abs(sorted_eigvals_high[1:n_lower] - sorted_eigvals_high[:n_lower - 1])
                        + np.abs(sorted_eigvals_high[2:n_lower + 1] - sorted_eigvals_high[1:n_lower]))
    
    delta_nearest = np.zeros(n_lower)
    for i in range(n_lower):
        delta_nearest[i] = np.min(np.abs(sorted_eigvals_low[i] - sorted_eigvals_high))/sigma[i]
    inv_delta_nearest = 1/delta_nearest

    return inv_delta_nearest[inv_delta_nearest > threshold], eigvals_low[inv_delta_nearest > threshold]

    


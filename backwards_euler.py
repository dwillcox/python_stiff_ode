import numpy as np

SMALL = 1.e-100

def backwards_euler(neq, t, tmax, dt_init, y_init, rhs, jac,
                    tol=1.e-6, max_iter=100):
    """solve the system dy/dt = f(y), where f(y) is provided by the
    routine rhs(), and the Jacobian is provided by the routine jac().

      neq : the number of equations in the system
      t : the current time
      tmax : the ending time of integration
      dt_init : initial timestep
      y_init : the initial conditions

    """

    time = t
    dt = dt_init

    # starting point of integration of each step
    y_n = np.zeros(neq)
    y_n[:] = y_init[:]

    y_new = np.zeros(neq)

    total_be_solves = 0

    while time < tmax:

        converged = False

        # we want to solve for the updated y.  Set an initial guess to
        # the current solution.
        y_new[:] = y_n[:]

        err = 1.e30
        niter = 0
        while err > tol and niter < max_iter:

            # construct the matrix A = I - dt J
            A = np.eye(neq) - dt * jac(time, y_n)

            # construct the RHS
            b = y_n - y_new + dt * rhs(time, y_new)

            # solve the linear system A dy = b
            dy = np.linalg.solve(A, b)

            # correct our guess
            y_new += dy

            # check for convergence
            err = np.linalg.norm(dy)/max(abs(y_new) + SMALL)
            niter += 1

        total_be_solves += niter

        if time + dt > tmax:
            dt = tmax - time

        y_n[:] = y_new[:]
        time += dt

    return y_n, total_be_solves

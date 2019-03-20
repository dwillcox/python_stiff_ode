import numpy as np

SMALL = 1.e-100

def sdc4(neq, t, tmax, dt_init, y_init, rhs, jac,
         tol=1.e-6, max_iter=100):
    """solve the system dy/dt = f(y), using a 4th order SDC integration
    technique based on Simpson's rule.  f(y) is provided by the
    routine rhs(), and the Jacobian is provided by the routine jac().

      neq : the number of equations in the system
      t : the current time
      tmax : the ending time of integration
      dt_init : initial timestep
      y_init : the initial conditions

    """

    # for 4th order SDC, we will have 3 time nodes, corresponding to
    # n, n+1/2, and n+1.  We will use the integer m to denote the
    # node, with m = 0, 1, 2

    sdc_nodes = 3

    # we also need an new iteration (k+1) and old iteration (k).
    # We'll call these new and old.

    # technically, at m=0 the old and new are the same, since there is
    # no update, so we don't need storage for both.

    y_old = []
    y_new = []

    # m = 0
    y_old.append(np.zeros(neq))
    y_new.append(y_old[0])   # a view

    for m in range(1, sdc_nodes):
        y_old.append(np.zeros(neq))
        y_new.append(np.zeros(neq))

    # set the initial conditions
    y_old[0][:] = y_init[:]

    # storage for the rhs at time nodes
    r_old = []
    for m in range(sdc_nodes):
        r_old.append(np.zeros(neq))

    time = t
    dt = dt_init

    total_be_solves = 0

    while time < tmax:

        dt_m = dt/(sdc_nodes-1)

        # initially, we don't have the old solution at the m > 0 time
        # nodes, so just set it to m = 0
        for m in range(1, sdc_nodes):
            y_old[m][:] = y_old[0][:]

        # we also need to initialize R_old
        r_old[0][:] = rhs(time, y_old[0])
        for m in range(1, sdc_nodes):
            r_old[m][:] = r_old[0][:]

        # SDC iterations
        for kiter in range(4):

            # node iterations
            for m in range(1, sdc_nodes):

                # define C = -R(y_old) + I/dt_m
                C = -r_old[m][:] + (1/dt_m) * int_simps(m, dt_m, r_old[0], r_old[1], r_old[2])

                if kiter > 0:
                    # initial guess for time node m is m's solution in the previous iteration
                    y_new[m][:] = y_old[m][:]
                else:
                    # initial guess for time node m is m-1's solution
                    y_new[m][:] = y_new[m-1][:]

                # solve the nonlinear system for the updated y
                err = 1.e30
                niter = 0
                while err > tol and niter < max_iter:

                    # construct the Jacobian
                    J = np.eye(neq) - dt_m * jac(time, y_new[m])

                    # construct f for our guess
                    f = y_new[m][:] - y_new[m-1][:] - dt_m * rhs(time, y_new[m]) - dt_m * C

                    # solve the linear system J dy = - f
                    dy = np.linalg.solve(J, -f)

                    # correct our guess
                    y_new[m][:] += dy

                    # check for convergence
                    err = max(abs((dy/(y_new[m]) + SMALL)))
                    niter += 1

                total_be_solves += niter

            # save the solution as the old solution for iteration
            for m in range(1, sdc_nodes):
                y_old[m][:] = y_new[m][:]

        time += dt

        if time + dt > tmax:
            dt = tmax - time

        # set the starting point for the next timestep
        y_old[0][:] = y_new[sdc_nodes-1][:]

    return y_new[sdc_nodes-1][:], total_be_solves


def int_simps(m_end, dt_m, r0, r1, r2):

    if m_end == 1:
        # integral from m = 0 to m = 1
        return dt_m/12.0 * (5*r0 + 8*r1 - r2)
    else:
        # integral from m = 1 to m = 2
        return dt_m/12.0 * (-r0 + 8*r1 + 5*r2)

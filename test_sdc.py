import numpy as np
import matplotlib.pyplot as plt

import sdc
from model_system import rhs, jac

def doit():

    # setup the initial conditions
    neq = 3
    y_init = np.array([1.0, 0.0, 0.0])

    # like the vode driver, we will do the integration in a bunch of
    # pieces, increasing the stopping time by 10x each run
    tends = np.logspace(-6, 8, 15)

    time = 0.0
    y_old = y_init.copy()

    ys = []
    for y in y_init:
        ys.append([y])

    ts = [time]

    total_be_solves = 0

    for tmax in tends:

        y_new, num_be_solves = sdc.sdc4(neq, time, tmax, tmax/10,
                                        y_old, rhs, jac)

        time = tmax
        ts.append(time)
        for n, y in enumerate(y_new):
            ys[n].append(y)

        y_old[:] = y_new[:]

        total_be_solves += num_be_solves

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for n in range(neq):
        ax.plot(ts, ys[n], label="y[{}]".format(n))

    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.savefig("test_sdc.png")

    print("Total no. BE solves: {}".format(total_be_solves))

if __name__ == "__main__":
    doit()



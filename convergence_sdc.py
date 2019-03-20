import numpy as np
import matplotlib.pyplot as plt

import sdc
from model_system import rhs, jac

def doit():

    neq = 3

    y_init = np.array([1.0, 0.0, 0.0])

    tmax = 1.e-2

    time = 0.0

    nstep = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    y_final = []

    for ns in nstep:
        y_new, _ = sdc.sdc4(neq, time, tmax, tmax/ns,
                            y_init, rhs, jac, tol=1.e-14)

        y_final.append(y_new)

        print(ns, y_new[2])

    # compute the difference between adjacent runs as the error
    err = []
    diff = []
    for n in range(1, len(nstep)):
        print(y_final[n], y_final[n-1], np.linalg.norm(y_final[n] - y_final[n-1]))
        err.append(np.linalg.norm(y_final[n] - y_final[n-1]))
        diff.append("{} -> {}".format(nstep[n-1], nstep[n]))

    for n in range(1, len(diff)):
        print("{} to {}; order = {}".format(diff[n-1], diff[n], err[n-1]/err[n]))

if __name__ == "__main__":

    doit()



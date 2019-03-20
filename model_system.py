# example of a stiff non-linear system -- this is the example defined in
# the VODE source code
#
# y1' = -0.04 y1 + 1.e4 y2 y3
# y2' =  0.04 y1 - 1.e4 y2 y3 - 3.e7 y2**2
# y3' =                         3.e7 y2**2
#
# y1(0) = 1
# y2(0) = 0
# y3(0) = 0
#
# long term behavior: y1 -> 0, y2 -> 0, y3 -> 1

import numpy as np

def rhs(t, Y):
    """ RHS of the system -- using 0-based indexing """
    y1 = Y[0]
    y2 = Y[1]
    y3 = Y[2]

    dy1dt = -0.04*y1 + 1.e4*y2*y3
    dy2dt =  0.04*y1 - 1.e4*y2*y3 - 3.e7*y2**2
    dy3dt =                         3.e7*y2**2

    return np.array([dy1dt, dy2dt, dy3dt])


def jac(t, Y):
    """ J_{i,j} = df_i/dy_j """

    y1 = Y[0]
    y2 = Y[1]
    y3 = Y[2]

    df1dy1 = -0.04
    df1dy2 = 1.e4*y3
    df1dy3 = 1.e4*y2

    df2dy1 = 0.04
    df2dy2 = -1.e4*y3 - 6.e7*y2
    df2dy3 = -1.e4*y2

    df3dy1 = 0.0
    df3dy2 = 6.e7*y2
    df3dy3 = 0.0

    return np.array([[ df1dy1, df1dy2, df1dy3 ],
                     [ df2dy1, df2dy2, df2dy3 ],
                     [ df3dy1, df3dy2, df3dy3 ]])


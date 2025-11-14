import numpy as np


def harmfit(invariant, *args):
    # a0, Cn, and Dn are the estimated amplitudes and phases of the constituents
    # wn are the angular frequencies of the consistuents
    # t is the time vector
    # output = mean + sum_k=1^k=Nharm Cn*sin(wn(k)*t) + Dn*cos(wn(k)*t)

    if len(args) == 1:
        args = tuple(args[0])

    t = invariant["timesteps"]
    wn = invariant["wn"]

    in_args = np.array(args[1:])

    a0 = args[0]
    Cn, Dn = np.split(in_args, 2)

    ts = np.array(t)

    test = np.empty(len(t))
    test.fill(a0)

    for k in range(len(Cn)):
        test = test + Cn[k] * np.sin(wn[k] * ts) + Dn[k] * np.cos(wn[k] * ts)
    return test
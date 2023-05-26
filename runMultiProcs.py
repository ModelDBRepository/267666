import os
import numpy as np
import sys

from base import name_from_pars, runmodl


if __name__ == "__main__":

    ind = int(sys.argv[1])

    # read the correct line (given by ind) of the args file
    args_list = np.loadtxt('args.txt', delimiter=' ')
    if ind >= len(args_list):
        quit()
    print("=" * 10, ind + 1, "/", len(args_list), "=" * 10)
    from neuron import h

    args = args_list[ind]

    dur = 1
    folder = "sim_outputs"   # name of folder in which to save results
    b0 = 22
    if not os.path.exists(folder):
        os.mkdir(folder)

    q0, tau, cond, ca = args

    name = os.path.join(folder, name_from_pars(b0, q0, tau, dur, ca, cond, acti=False)[:-7])
    runmodl(h, asic_cond=cond, ca_ratio=ca, b0=b0, q0=q0, tau_pH=tau,
            duration=dur, name=name, save_pH=True,
            save_vars=((q0, tau) in [(1., 0.01), (0.05, 1.), (0.3, 0.1)]))   # only save the (heavy) evolution of
            # variables over time for the chosen model (and alternatives 1 and 2)

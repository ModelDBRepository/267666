import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import pickle as pk
import os


"""
This file simulates our models of ASIC as isolated channels (i.e. not embedded into a cell, as is done with the NEURON 
simulator).
This allows to explore the basic properties of the channel models, such as their pH-dependent inactivation, recovery 
from inactivation, etc.
The Supplementary figure 6 is mostly generated here.
"""


class ASICType1Chosen:
    """
    Homomeric channel model, based on the data from Baron et al., 2008.
    """
    type = "type1"

    def __init__(self, type="type1", pH05_m=6.46, n_m=1.5, pH05_h=7.3, n_h=4.6, state0=None, basepH=7.4, gmax=1, E_rev=50, Vm=-75):
        self.type = type
        self.pH05_m = pH05_m
        self.n_m = n_m
        self.pH05_h = pH05_h
        self.n_h = n_h
        self.a1_tau_m = 0.3487
        self.a2_tau_m = 234.3
        self.b1_tau_m = 10.14
        self.b2_tau_m = -5.553
        self.c1_tau_m = 6.392
        self.c2_tau_m = 8.595

        self.gamma = 0

        if not state0:
            state0 = [self.m_inf(basepH), self.h_inf(basepH)]
        self.state = state0
        self.gmax = gmax
        self.E_rev = E_rev
        self.Vm = Vm

    def m_inf(self, pH):
        return 1. / (1. + 10**(self.n_m*(pH - self.pH05_m)))

    def h_inf(self, pH):
        return 1.3 / (1 + 10**(-self.n_h*(pH - self.pH05_h)))

    def tau_h(self, pH):
        return 49.196 * np.exp(- 34.682 * (pH - 7.144) ** 2) + (pH - 5) * (4.78 - 0.98) / (9 - 5) + 0.98

    def tau_m(self, pH):
        """Same activation time constant as in Alijevic et al., 2020."""
        return  1 / ( self.a1_tau_m / (1 + np.exp(self.b1_tau_m * (pH - self.c1_tau_m))) + self.a2_tau_m / (1 + np.exp(self.b2_tau_m * (pH - self.c2_tau_m))) ) / 1000

    def set_pHstim(self, fun):
        self.pHstim = fun

    def der(self, state, time, pH=None):
        if type (time) != float:
            state, time = time, state
        m, h = state
        if pH is None:
            pH = self.pHstim(time)
        return np.array([(self.m_inf(pH) - m) / self.tau_m(pH),
                         (self.h_inf(pH) - h) / self.tau_h(pH)])

    def i(self, a, s):
        return self.gmax * a * s * (self.Vm - self.E_rev)

    def grel(self, a, s):
        return a * s

    def run_sim(self, times, out="i"):
        states = integrate(self.der, self.state, times)
        self.state = states[-1]
        if out == "i":
            i = [self.i(*state) for state in states]
        elif out == "grel" or out == "g_rel":
            i = [self.grel(*state) for state in states]
        else:
            raise ValueError
        return np.array(i)


class ASICType2Chosen:
    """
    Heteromeric channel model, based on the data from Baron et al., 2008.
    """
    type = "type2"
    def __init__(self, type="type2", pH05_m=6.03, n_m=1.94, pH05_h=6.74, n_h=3.82, state0=None, basepH=7.4, gmax=1,
                 E_rev=50, Vm=-75):
        self.type = type
        self.pH05_m = pH05_m
        self.n_m = n_m
        self.pH05_h = pH05_h
        self.n_h = n_h
        self.a1_tau_m = 0.3487
        self.a2_tau_m = 234.3
        self.b1_tau_m = 10.14
        self.b2_tau_m = -5.553
        self.c1_tau_m = 6.392
        self.c2_tau_m = 8.595

        self.gamma = 0

        if not state0:
            state0 = [self.m_inf(basepH), self.h_inf(basepH)]
        self.state = state0
        self.gmax = gmax
        self.E_rev = E_rev
        self.Vm = Vm

    def m_inf(self, pH):
        return 1. / (1. + 10**(self.n_m*(pH - self.pH05_m)))

    def h_inf(self, pH):
        return 1 / (1 + 10**(-self.n_h*(pH - self.pH05_h)))

    def tau_h(self, pH):
        a = 42.862
        b = 5.375
        c = 6.600
        d = 1.645
        return a * np.exp(- b * np.abs(pH - c)) + d

    def tau_m(self, pH):
        """Same activation time constant as in Alijevic et al., 2020."""
        return  1 / ( self.a1_tau_m / (1 + np.exp(self.b1_tau_m * (pH - self.c1_tau_m))) + self.a2_tau_m / (1 + np.exp(self.b2_tau_m * (pH - self.c2_tau_m))) ) / 1000

    def set_pHstim(self, fun):
        self.pHstim = fun

    def der(self, state, time, pH=None):
        if type (time) != float:
            state, time = time, state
        m, h = state
        if pH is None:
            pH = self.pHstim(time)
        return np.array([(self.m_inf(pH) - m) / self.tau_m(pH),
                         (self.h_inf(pH) - h) / self.tau_h(pH)])

    def i(self, m, h):
        return self.gmax * m * h * (self.Vm - self.E_rev)

    def grel(self, a, s):
        return a * s

    def run_sim(self, times, out="i"):
        states = integrate(self.der, self.state, times)
        self.state = states[-1]
        if out == "i":
            i = [self.i(*state) for state in states]
        elif out == "grel" or out == "g_rel":
            i = [self.grel(*state) for state in states]
        else:
            raise ValueError
        return np.array(i)


def load_channel(which=1):
    if which == 1:
        chan = ASICType1Chosen()
    elif which == 2:
        chan = ASICType2Chosen()
    return chan


def integrate(ode_fun, state0, time):
    out = solve_ivp(ode_fun, (time[0], time[-1]), state0, t_eval=time, max_step=1) # mxstep=100
    state = np.transpose(out["y"])
    return state


def general_pH(tags):
    """
    tags is a list of the form [(pH1, pH1_duration), (pH2, pH2_duration), ...]
    Generates a pH protocol in which pH1 is applied during pH1_duration, then pH2 during pH2_duration...
    Returns a function that, given a time t, returns the pH for this time in the protocol.
    """
    def fun(t):
        time = 0
        for pH, duration in tags:
            time += duration
            if t <= time:
                return pH
        return pH
    return fun


def get_figS6_protocols():
    return [reactivation_protocol(), inactivation_protocol(), single_drop_protocol()]


def reactivation_protocol():
    """
    Generates the recovery from inactivation protocol of Supplementary figure 6A.
    """
    reac_times = [.5, 1, 2, 4, 8]
    return ("Recovery from inactivation", general_pH,
            [[(7.4, 5), (5, 10), (7.4, t_reac), (5, 15-t_reac)] for t_reac in reac_times],
            ["{}".format(t_reac) for t_reac in reac_times])


def inactivation_protocol():
    """
    Generates the pH-dependent inactivation protocol of Supplementary figure 6B.
    """
    testpHs = [8, 7.6, 7.4, 7, 6.6, 6]
    return ("Activation/Inactivation/SSD", general_pH,
            [[(testpH, 40), (5, 5), (8, 20)] for testpH in testpHs],
            ["{}".format(testpH) for testpH in testpHs])


def single_drop_protocol():
    """
    Generates the pH-dependent inactivation protocol of Supplementary figure 6F.
    """
    testpHs = [6, 7.32]
    return ("Simple drop", general_pH,
            [[(7.4, 40), (testpH, 5), (7.4, 40)] for testpH in testpHs],
            ["{}".format(testpH) for testpH in testpHs])


def current_graphs(funs, folder):
    """
    funs must be a list.
    Each element of the list is a tuple that describes a pH protocol. (Each protocol will be shown in a different figure.)
    The tuple is of the form (protocol_name, genfun, params, labels) where:
        - protocol_name is the name/description of the protocol;
        - gen_fun is a function that, given a parameter set, returns a function t -> pH;
        - params is a list of the different parameter sets for this protocol, to be given to gen_fun (e.g. one parameter
            set describes the protocol for one reactivation time, for the 'reactivation' protocol);
        - labels is a list of labels corresponding to each parameter set.
    The pH protocol will be applied to both type1 and type2 models.
    Because the integration can be slow, the results of the simulation will be saved to pickle files, and the data will
    be loaded instead of simulated if the simulation data is already existing.
    folder is the folder in which the simulated data is to be saved.
    """
    os.makedirs(folder, exist_ok=True)
    for k, (title, genfun, params, labels) in enumerate(funs):
        name = f"rev2_try3_{title.replace(' ', '_').replace('/', '-')}"
        time = np.arange(0, sum(x[1] for x in params[0]), 1e-4)
        fig, axs = plt.subplots(3, 1, figsize=(6, 8), tight_layout=True, gridspec_kw={'height_ratios': [1, 2.5, 2.5]})
        for j, (param, label) in enumerate(zip(params, labels)):
            axs[0].plot(time, np.array([genfun(param)(t) for t in time])-0.015*j, label=label)

            for l, asic_class in enumerate([ASICType1Chosen, ASICType2Chosen]):
                filename = os.path.join(folder, name+f"_{asic_class}_{j}.pickle")
                if os.path.exists(filename):
                    print("Loading")
                    with open(filename, "rb") as f:
                        i = pk.load(f)
                else:
                    print("Simulating")
                    asic = asic_class()
                    asic.set_pHstim(genfun(param))
                    i = asic.run_sim(time, out="grel")
                    with open(filename, "wb") as f:
                        pk.dump(i, f)
                axs[l+1].plot(time, i, label=label)
        axs[0].set_ylabel("Applied pH")
        for ax in axs:
            ax.set_xlabel("Time (s)")
        axs[1].set_ylabel("Homomeric relative conductance m * h")
        axs[2].set_ylabel("Heteromeric relative conductance m * h")
    plt.show()


if __name__ == "__main__":
    current_graphs(get_figS6_protocols(), folder="figS6")



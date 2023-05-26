# general helpers
def name_from_pars(b0, q0, tau, dur, ca, cond, acti=False):
    return "q0{}_tau{}_cond{:.3g}nS_b0{:.3g}_dur{:.3g}_ca{:.3g}_{}spikes".format(float(q0), float(tau), float(cond), float(b0), float(dur), float(ca), "acti_" if acti else "")


def get_params_from_name(name, first=0):
    """
    Gets the simulation parameters from the file name, according to the naming convention used in runMultiProcs
    Modify this function if using a different naming convention.
    """
    q0 = float(name.split("_")[first+0][2:])
    tau = float(name.split("_")[first+1][3:])
    cond = float(name.split("_")[first+2][4:-2])
    b0 = float(name.split("_")[first+3][2:])
    dur = float(name.split("_")[first+4][3:])
    ca = float(name.split("_")[first+5][2:])

    # naming convention that the "ASIC full activation" simulations should have "acti" in their name:
    act = int("acti" in name)

    return (b0, q0, tau, dur), ca, cond, act


# main simulation function
def runmodl(h, asic_cond=0.1, ca_ratio=0.1, b0=22, q0=2.2, tau_pH=1, duration=1, name="noname", save_pH=True,
            save_vars=False):
    """
    :param h: from neuron import h
    :param asic_cond: (nS) maximal conductance of ASIC channels
    :param ca_ratio: proportion of calcium current through ASIC channels
    :param b0: (mM) buffer concentration parameter B0 for the synaptic cleft milieu
    :param q0: (mM/ms) proton current q flowing into the synaptic cleft at action potential
    :param tau_pH: (ms) synaptic cleft time constant tau of the pH model
    :param duration: (ms) duration of the proton current flowing into the synaptic cleft at action potential (kept fixed=1ms)
    :param name: name of the simulation, used to name the data files
    :param save_pH: whether to save the evolution of pH through time (a bit heavy). Spike times are saved anyways.
    :param save_vars: whether to save the evolution of all variables through time (very heavy). Spike times are saved anyways.
    """
    import sim_saving as rm

    h.load_file("nrngui.hoc")
    h.load_file("wdr-complete-model-without-interneuron.hoc")
    h.cvode.active(1)
    h.cvode.use_local_dt(1)
    h.cvode.atol(1e-4)

    set_noise(h)
    set_nb_stims(h, 15)
    construct_stim_times(h)

    for i in range(int(h.N_IC)):
        h.asic[i].gbar = asic_cond * 0.001
        h.asic[i].ca_ratio = ca_ratio

        h.pH[i].b0 = b0
        h.pH[i].q0 = q0
        h.pH[i].tau = tau_pH
        h.pH[i].duration = duration

    print("Running")
    h.run()
    print("Ran")

    h.store_wdr(name + "_spikes.dat")

    if save_vars:
        plots = [
            # potentials and calcium concentrations
            [("h.wdr_tvec", ["h.wdr_soma_v"]), ("h.wdr_tvec", ["h.wdr_dend_v"])],
            [("h.wdr_tvec", ["h.wdr_soma_cai"]), ("h.wdr_tvec", ["h.wdr_dend_cai"])],
            [("h.wdr_tvec", ["h.wdr_soma_ica"]), ("h.wdr_tvec", ["h.wdr_dend_ica"])],
            # WDR dendrite channels
            [("h.wdr_tvec", ["h.wdr_dend_icaan", "grel_dend_icaan()"]),
             ("h.wdr_tvec", ["h.wdr_dend_ikca", "grel_dend_ikca()"]),
             ("h.wdr_tvec", ["h.wdr_dend_ical", "-prel_dend_ical()"]), ("h.wdr_tvec", ["h.wdr_dend_ica"])],
            # etude de ical
            [("h.wdr_tvec", ["h.wdr_dend_ical"]), ("h.wdr_tvec", ["all_ca_currents()"]),
             ("h.wdr_tvec", ["h.wdr_dend_ica"])],
            [("h.wdr_tvec", ["np.array(h.wdr_dend_v)+65"]),
             ("h.wdr_tvec", ["np.array(h.wdr_dend_ical)", "h.m_dend_ical"]),
             ("h.wdr_tvec", ["prel_dend_ical()/prel_dend_ical()[0]*np.array(h.wdr_dend_ical)[0]"])],
            # WDR soma channels
            [("h.wdr_tvec", ["h.wdr_soma_ikhh", "grel_ikhh()"]), ("h.wdr_tvec", ["h.wdr_soma_inahh", "grel_inahh()"]),
             ("h.wdr_tvec", ["h.wdr_soma_ipas"])],
            [("h.wdr_tvec", ["h.wdr_soma_inap", "grel_inap()"]),
             ("h.wdr_tvec", ["h.wdr_soma_ical", "grel_soma_ical()"]), ("h.wdr_tvec", ["h.wdr_soma_ica"]),
             ("h.wdr_tvec", ["h.wdr_soma_ikca", "grel_soma_ikca()"])],
            # WDR dendrite variables
            [("h.wdr_tvec", ["np.array(h.tau_m_icaan)/4000"]), ("h.wdr_tvec", ["h.m_inf_icaan"]),
             ("h.wdr_tvec", ["h.m_icaan"])],
            [("h.wdr_tvec", ["np.array(h.tau_m_dend_ikca)/7."]), ("h.wdr_tvec", ["h.m_inf_dend_ikca"]),
             ("h.wdr_tvec", ["h.m_dend_ikca"])],
            [("h.wdr_tvec", ["h.tau_m_dend_ical"]), ("h.wdr_tvec", ["h.m_inf_dend_ical"]),
             ("h.wdr_tvec", ["h.m_dend_ical"]), ("h.wdr_tvec", ["np.array(h.ghk_dend_ical)/-1897.569"])],
            # WDR soma variables
            [("h.wdr_tvec", ["h.m_inf_inahh", "np.array(h.tau_m_inahh)"]),
             ("h.wdr_tvec", ["h.h_inf_inahh", "np.array(h.tau_h_inahh)"]), ("h.wdr_tvec", ["h.m_inahh"]),
             ("h.wdr_tvec", ["h.h_inahh"]), ("h.wdr_tvec", ["grel_inahh()"]),
             ("h.wdr_tvec", ["h.n_inf_ikhh", "np.array(h.tau_n_ikhh)"]), ("h.wdr_tvec", ["h.n_ikhh"])],
            [("h.wdr_tvec", ["h.m_inf_inap", "np.array(h.tau_m_inap)"]),
             ("h.wdr_tvec", ["h.h_inf_inap", "np.array(h.tau_h_inap)"]), ("h.wdr_tvec", ["h.m_inap"]),
             ("h.wdr_tvec", ["h.h_inap"]), ("h.wdr_tvec", ["grel_inap()"])],
            [("h.wdr_tvec", ["np.array(h.tau_m_soma_ikca)/7."]), ("h.wdr_tvec", ["h.m_inf_soma_ikca"]),
             ("h.wdr_tvec", ["h.m_soma_ikca"])],
            [("h.wdr_tvec", ["h.tau_m_soma_ical"]), ("h.wdr_tvec", ["h.m_inf_soma_ical"]),
             ("h.wdr_tvec", ["h.m_soma_ical"]), ("h.wdr_tvec", ["np.array(h.ghk_soma_ical)/-1897.569"])],
            [("h.wdr_tvec", ["h.wdr_dend_icaan", "h.m_icaan", "h.m_inf_icaan"])],
            [("h.wdr_tvec", ["h.wdr_dend_icaan", "h.tau_m_icaan"])],
            # C-fiber receptors
            [("h.Cr_tvec", ["h.C_ampa_i[0]"]), ("h.Cr_tvec", ["h.C_nmda_i[0]"]), ("h.Cr_tvec", ["h.C_nk1_i[0]"])],
            [("h.Cr_tvec", ["np.array(h.C_ampa_B[0])-np.array(h.C_ampa_A[0])"]),
             ("h.Cr_tvec", ["np.array(h.C_nmda_B[0])-np.array(h.C_nmda_A[0])"]),
             ("h.Cr_tvec", ["np.array(h.C_nk1_B[0])-np.array(h.C_nk1_A[0])"])],
            [("h.Cr_tvec", ["all_C_ampa_i()"]), ("h.Cr_tvec", ["all_C_nmda_i()"]), ("h.Cr_tvec", ["all_C_gaba_i()"]),
             ("h.Cr_tvec", ["all_nk1_i()"]), ("h.Cr_tvec", ["all_asic_i()"])],
            [("h.Cr_tvec", ["all_C_ampa_g()"]), ("h.Cr_tvec", ["all_C_nmda_g()"]), ("h.Cr_tvec", ["all_C_gaba_g()"]),
             ("h.Cr_tvec", ["all_asic_g()"])],
            [("h.Ad_tvec", ["all_Ad_ampa_i()"]), ("h.Ad_tvec", ["all_Ad_nmda_i()"])],
            # pH and ASICs
            [("h.wdr_tvec", ["pH_cleft()"])],
            [("h.wdr_tvec", ["all_asic_i()", "all_asic_g()"])],
            [("h.wdr_tvec", ["h.ASIC_m[0]"]), ("h.wdr_tvec", ["h.ASIC_m_inf[0]", "h.ASIC_tau_m[0]"]),
             ("h.wdr_tvec", ["h.ASIC_h[0]"]), ("h.wdr_tvec", ["h.ASIC_h_inf[0]", "h.ASIC_tau_h[0]"]),
             ("h.wdr_tvec", ["one_asic_grel()"])
             ],
        ]
        rm.save_vars(name + "_vars.pickle", plots)
    elif save_pH:
        plots = [[("h.wdr_tvec", ["pH_cleft()"])]]
        rm.save_vars(name + "_vars.pickle", plots)


# simulation helpers

def construct_stim_times(h):
    h.stim_times.resize(h.n_stim_sets*h.n_close_stims*h.n_burst_spikes)
    index = 0
    for i in range(int(h.n_stim_sets)):
        for j in range(int(h.n_close_stims)):
            for k in range(int(h.n_burst_spikes)):
                h.stim_times.x[index] = h.start_time + i * h.T2 + j * h.T1 + k * h.T0
                index += 1
    h.tstop = (h.n_stim_sets + 1) * h.T2 + h.start_time


def set_nb_stims(h, n=15):
    h.n_stim_sets = n


def set_noise(h, exc=0.0001, inh=0.0001):
    h.set_noise(exc, inh)

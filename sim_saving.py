"""
This file is used by run.py to save the simulation results.
"""

from neuron import h   # This may be called by the "eval" in save_vars. EXPECT BUGS IF REMOVED
import pickle
import numpy as np


def save_vars(name, plots):
    savefile = open(name, "wb")
    tvecs = {}
    saveplots = []
    for plot in plots:
        saveplot = []
        for t,ys in plot:
            if t not in tvecs:
                tvecs[t] = list(eval(t))[::5]
            saveplot.append((t, [(y, list(eval(y))[::5]) for y in ys]))
        saveplots.append(saveplot)
    pickle.dump((saveplots, tvecs, True), savefile)
    savefile.close()
    print("Saved vars to file")


# Here are functions used to fetch and aggregate the values of variables in the simulation
# Even if they are not obviously used in this file, they may be called by the "eval" in save_vars.
def all_ca_currents():
    dense = [h.wdr_dend_ical]
    ca_curs = sum(dense_from_receptor_ca())
    ca_curs += sum(np.array(d) for d in dense)
    return ca_curs

def dense_from_receptor_ca():
    lst = []
    for ica_rec in [h.dend_C_nk1_ica[j] for j in range(int(h.N_IC))] + [h.dend_nmda_ica[j] for j in range(int(h.N_IC))]:
        lst.append(np.array(ica_rec) / (np.pi * h.wdr.dendrite.diam * h.wdr.dendrite.L / h.wdr.dendrite.nseg) * 100)
    return lst

def pH_cleft():
    return -np.log10(np.array(h.cleft_he[0])*1e-3)

def prel_dend_ical():
    return np.array(h.m_dend_ical)**2 * np.array(h.ghk_dend_ical) / -1897.569

def prel_soma_ical():
    return np.array(h.m_soma_ical)**2 * np.array(h.ghk_soma_ical) / -1897.569

def grel_inap():
    return np.array(h.m_inap) * np.array(h.h_inap)

def grel_inahh():
    return np.array(h.m_inahh)**3 * np.array(h.h_inahh)

def grel_soma_ical():
    return np.array(h.m_soma_ical)**2

def grel_soma_ikca():
    return np.array(h.m_soma_ikca)**3

def grel_dend_ikca():
    return np.array(h.m_dend_ikca)**3

def grel_ikhh():
    return np.array(h.n_ikhh)**4

def grel_dend_icaan():
    return np.array(h.m_icaan)**2

def all_C_ampa_i():
    return sum([np.array(h.C_ampa_i[j]) for j in range(int(h.N_IC))])

def all_C_nmda_i():
    return sum([np.array(h.C_nmda_i[j]) for j in range(int(h.N_IC))])

def all_C_gaba_i():
    return sum([np.array(h.C_gaba_i[j]) for j in range(int(h.N_IC))])

def all_asic_i():
    return sum([np.array(h.ASIC_i[j]) for j in range(int(h.N_IC))])

def all_nk1_i():
    return sum([np.array(h.C_nk1_i[j]) for j in range(int(h.N_IC))])

def all_C_ampa_g():
    return sum([np.array(h.C_ampa_B[j])-np.array(h.C_ampa_A[j]) for j in range(int(h.N_IC))])

def all_C_nmda_g():
    return sum([(np.array(h.C_nmda_B[j]) - np.array(h.C_nmda_A[j]) * np.array(h.C_nmda_mgb[j])) for j in range(int(h.N_IC))])

def all_C_gaba_g():
    return sum([np.array(h.C_gaba_B[j])-np.array(h.C_gaba_A[j]) for j in range(int(h.N_IC))])

def all_asic_g():
    return sum([np.array(h.ASIC_m[j]) * np.array(h.ASIC_h[j]) * h.asic[j].gbar for j in range(int(h.N_IC))])

def one_asic_grel():
    return np.array(h.ASIC_m[0]) * np.array(h.ASIC_h[0])

def all_Ad_nmda_i():
    return sum([np.array(h.Ad_nmda_i[j]) for j in range(int(h.N_A))])

def all_Ad_ampa_i():
    return sum([np.array(h.Ad_ampa_i[j]) for j in range(int(h.N_A))])

def all_Ad_ampa_g():
    return sum([np.array(h.Ad_ampa_B[j])-np.array(h.Ad_ampa_A[j]) for j in range(int(h.N_A))])

def all_Ad_nmda_g():
    return sum([np.array(h.Ad_nmda_B[j])-np.array(h.Ad_nmda_A[j]) for j in range(int(h.N_A))])

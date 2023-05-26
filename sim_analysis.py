import numpy as np
from base import name_from_pars, get_params_from_name
import matplotlib.pyplot as plt
import os
from collections import defaultdict
# from windup_analysis import *
# from just_plot import intelligent_plotstyles, plotdata_from_file, asic_plotdata_from_file
import pickle
from matplotlib.colors import LogNorm
# from matplotlib.cm import get_cmap
import pandas as pd


t_ab = 0.02
t_ad = 0.09
t_c1 = 0.35
t_c2 = 1


def nb_pA_per_stim(stim, spikes):
    """
    Compute the number of action potentials per stimulation, and separate A-beta, A-delta, C1 and C2 spikes by timing.
    """
    n = len(stim)
    nb_tot_spikes = np.zeros(n, dtype=int)
    nb_spikes_AB = np.zeros(n, dtype=int)
    nb_spikes_AD = np.zeros(n, dtype=int)
    nb_spikes_C1 = np.zeros(n, dtype=int)
    nb_spikes_C2 = np.zeros(n, dtype=int)
    inter_stim = stim[1] - stim[0]
    for i in range(len(stim)):
        t = stim[i]

        nb_spikes_AB[i] = sum((t <= spikes) * (spikes <= t+t_ab))
        nb_spikes_AD[i] = sum((t+t_ab <= spikes) * (spikes <= t+t_ad))
        nb_spikes_C1[i] = sum((t+t_ad <= spikes) * (spikes <= t+t_c1))
        nb_spikes_C2[i] = sum((t+t_c1 <= spikes) * (spikes <= min(t+t_c2, t+inter_stim)))
        nb_tot_spikes[i] = nb_spikes_AB[i] + nb_spikes_AD[i] + nb_spikes_C1[i] + nb_spikes_C2[i]
    return nb_tot_spikes, nb_spikes_AB, nb_spikes_AD, nb_spikes_C1, nb_spikes_C2


def get_spikes_from_file(filepath, stims=None, start=1):
    """
    Gets the C-fiber-induced spikes in the simulation saved in filepath.
    """
    if stims is None:
        stims = np.arange(start, start + 15)
    spikes = []
    with open(filepath, "r") as f:
        for l in f.readlines():
            if l != "\n":
                spikes.append(float(l) / 1000)
    spikes = np.array(spikes)
    tot, Ab, Ad, spikesC1, spikesC2 = nb_pA_per_stim(stims, spikes)
    return spikesC1 + spikesC2


def get_windup_from_spikes(spikes):
    auc = int(sum(spikes) - spikes[0] * len(spikes))
    return auc


def name_from_pars_forpH(b0, q0, tau, dur, ca, cond):
    """
    Creates the key for the minpH dictionary
    """
    return "b0{:.3g}_q0{}_tau{}_dur{:.3g}".format(float(b0), float(q0), float(tau), float(dur))


def custom_tqdm(it):
    """
    it is an iterable.
    Prints progress while iterating.
    """
    try:
        n = len(it)
        if n > 100:
            print("Iterating over", n)
    except:
        n = 1
        pass
    i = 0
    for x in it:
        i += 1
        if not i%100:
            print("{:<5}    ----    {}%".format(i, int(100*i/n)), flush=True)
        yield x


def make_paper_plots(pars_lst, spikesC, data):
    """
    This function selects the chosen parameters given by pars_lst and makes the typical wind-up curve for our Figure 4, 
    Figure 5 and Figure 9.
    pars_lst is a list of the parameter sets to be chosen given as (q, tau).
    """
    # data[(b0, q0, tau, dur)] = [((ca, cond, act), auc, name), (), ...] (sorted list)
    restricted = {}
    for (q0, tau) in pars_lst:
        wanted = (22., q0, tau, 1)
        restricted.update({wanted: data[wanted]})

    paper_plots(spikesC, restricted)


def paper_plots(spikesC, data, small_conds=[0.01, 0.05, 0.1, 0.2], ca_conds=[0.2, 1, 1.4],
                mitx_cond=0.2):
    """
    This function makes the typical wind-up curve for our Figure 4, Figure 5 and Figure 9.
    small_conds is the list of conductances to be displayed in Figure 4C.
    ca_conds is the list of conductances to be displayed in Figure 4D which highlights the effect of calcium.
    mitx_cond is the conductance to be displayed in Figure 5C which demonstrates the effect of fully activating ASICs.
    """
    # shape with conductance:
    nb = 0
    for pars, lst in data.items():
        nb += 1
        fig, axs = plt.subplots(2, 2, figsize=(13,8))
        fig.suptitle("b0={:.2g}mM, q0={:.2g}mM/ms, tau={:.2g}ms, duration={:.2g}ms".format(*pars))

        # control with no ASIC:
        try:
            name = name_from_pars(*pars, 0., 0.)
            print(name)
            control_spikes = np.array(spikesC[name])
        except:
            print("cond 0 not available")
            control_spikes = []

        # Figure 4C: low to medium ASIc conductances
        axs[0][0].plot(control_spikes, label="0nS", color="k", marker=".")
        for i, cond in enumerate(small_conds):
            try:
                name = name_from_pars(*pars, 0.1, cond)
                axs[0][0].plot(np.array(spikesC[name]), label=str(cond)+"nS", color=plt.cm.Reds(i/len(small_conds)),
                               marker=".")
            except:
                print("cond {} not available".format(cond))
        axs[0][0].set_title("Fig 4C")
        axs[0][0].legend()

        # Fig 4D: higher ASIc conductances, with and without calcium
        axs[0][1].plot(control_spikes, label="0nS", color="k", marker=".")
        cas = [0, 0.1]
        for i, cond in enumerate(ca_conds):
            for ca in cas:
                try:
                    kwargs = {"linestyle": "-" if ca else "--", "marker": "."}
                    if ca:
                        kwargs["label"] = str(cond)+"nS"
                    name = name_from_pars(*pars, ca, cond)
                    axs[0][1].plot(np.array(spikesC[name.format(*pars, ca, cond)]), color=plt.cm.Reds(i/len(ca_conds)), **kwargs)
                except:
                    print("cond {}, ca {} not available".format(cond, ca))
        axs[0][1].set_title("Fig 4D, dashed=without calcium permeability")
        axs[0][1].legend()

        # Figure 5C
        axs[1][0].plot(control_spikes, label="0nS", color="k", marker=".")
        ca = 0.1
        name2 = name_from_pars(*pars, ca, mitx_cond, True)
        name = name_from_pars(*pars, ca, mitx_cond)
        try:
            spikes_acti = spikesC[name2]
        except KeyError:
            print(f"No 'full activation' data found for ca={ca} and cond={mitx_cond}.")
        else:
            axs[1][0].plot(np.array(spikes_acti), label=str(f"mitx {mitx_cond}nS"), marker=".")
        try:
            spikes = spikesC[name]
        except KeyError:
            print(f"No data found for ca={ca} and cond={mitx_cond}.")
        else:
            axs[1][0].plot(np.array(spikes), label=str(f"normal {mitx_cond}nS"), marker=".")
        axs[0][1].set_title("Fig 5C")
        axs[1][0].legend()

        axs[1][1].axis("off")
    plt.show()


def heatmap(spikesC, data, minpH, chose_cond=0.2, save=True, savefig=False):
    """
    Creates heatmaps of the variables (wind-up=area under curve, minimum pH reached, number of spikes at last 
    stimulation, number of spikes at first stimulation) of interest along parameters tau, q
    """
    # data[(b0, q0, tau, dur)] = [((ca, cond, act), auc, name), (), ...] (sorted list)

    qs = np.unique([q for (b0, q, tau, dur) in data.keys()])
    taus = np.unique([tau for (b0, q, tau, dur) in data.keys()])
    b0 = 22
    dur = 1

    if save:
        if isinstance(savefig, str):
            save_root = os.path.join(save, "heatmaps")
        else:   # save = True
            save_root = "heatmaps"
        os.makedirs(save_root, exist_ok=True)

    if savefig:
        if isinstance(savefig, str):
            save_fig_root = os.path.join(savefig, "heatmaps")
        else:   # savefig = True
            save_fig_root = save_root
        os.makedirs(save_fig_root, exist_ok=True)

    if not isinstance(chose_cond, list):
        chose_cond = [chose_cond]

    # Max wind-dup achieved for a given pH parameter set
    max_wu_grid = np.zeros((len(qs), len(taus))) * np.nan
    # Wind-up (for each pH parameter set, for each ASIC conductance)
    wu_grid = {c: np.zeros((len(qs), len(taus))) * np.nan for c in chose_cond}
    # Minimum pH reached over the simulation (for each pH parameter set, for each ASIC conductance, although it does
    # not depend on the ASIc conductance)
    ph_grid = {c: np.zeros((len(qs), len(taus))) * np.nan for c in chose_cond}
    # Number of spikes at last stimulation (for each pH parameter set, for each ASIC conductance)
    last_grid = {c: np.zeros((len(qs), len(taus))) * np.nan for c in chose_cond}
    # Number of spikes at first stimulation (for each pH parameter set, for each ASIC conductance)
    first_grid = {c: np.zeros((len(qs), len(taus))) * np.nan for c in chose_cond}

    available_conds = set()
    for i, q in enumerate(qs):
        for j, tau in enumerate(taus):
            if (b0, q, tau, dur) not in data:
                continue
            this_data = data[(b0, q, tau, dur)]

            max_auc = max([auc for ((ca, cond, act), auc, name) in this_data if ca and not act], default=np.nan)
            max_wu_grid[i,j] = max_auc
            if max_auc is np.nan:
                print(i, j)

            for ((ca, cond, act), auc, name) in this_data:
                if not ca:
                    continue
                available_conds.add(cond)
                if cond in chose_cond and not act:
                    wu_grid[cond][i,j] = auc
                    last_grid[cond][i,j] = spikesC[name][-1]
                    first_grid[cond][i,j] = spikesC[name][0]
                    ph_name = name_from_pars_forpH(b0, q, tau, dur, ca, cond)
                    ph_grid[cond][i, j] = minpH.get(ph_name, 0)

    levels_auc = [180, 100000]
    levels_last = [22, 32]
    levels_pH = [7, 10000]
    levels_first = [0, 6, 10]
    # draw_heatmap(max_wu_grid, qs, taus, "q0", "tau [ms]", "Maximum wind-up", log_col=True)
    # draw_heatmap(ph_grid[cond], qs, taus, "q0 [mM/ms]", "tau [ms]", "min pH", levels=levels_pH, log_col=False, vmin=6)
    for cond in wu_grid.keys():
        if save:
            name = os.path.join(save_root, "{}_"+"{}nS.csv".format(cond))
            save_heatmap(wu_grid[cond], qs, taus, name.format("auc"))
            save_heatmap(last_grid[cond], qs, taus, name.format("last_spike_count"))
            save_heatmap(wu_grid[cond], qs, taus, name.format("auc"))
        if savefig:
            savefig = os.path.join(save_fig_root, "superimposed{}nS.pdf".format(cond))
        # draw_heatmap(wu_grid[cond], qs, taus, "q0 [mM/ms]", "tau [ms]", f"Wind-up for {cond}nS", center_col=False, levels=levels_auc)
        # draw_heatmap(last_grid[cond], qs, taus, "q0 [mM/ms]", "tau [ms]", f"Last number of spikes for {cond}nS", center_col=22, levels=levels_last)
        # draw_heatmap(first_grid[cond], qs, taus, "q0 [mM/ms]", "tau [ms]", f"First number of spikes for {cond}nS", center_col=False, levels=levels_first)
        draw_superimposed_contours(wu_grid[cond], ph_grid[cond], last_grid[cond], qs, taus,
                                   r" \LARGE \textbf{Proton current} \boldmath $q$ \textbf{(mM/ms)}",    # \fontfamily{Arial}\selectfont
                                   r" \LARGE \textbf{Time constant} \boldmath $\tau$ \textbf{(ms)}",
                                   levels_auc, levels_pH, levels_last, savefig=savefig)
    if save:
        name = os.path.join(save_root, "pH.csv")
        save_heatmap(ph_grid[cond], qs, taus, name)   # any cond because pH does not depend on ASIC currents


def save_heatmap(vals, x, y, name):
    df = pd.DataFrame(vals, columns=y, index=x)
    df.to_csv(name, index=True)


def draw_heatmap(vals, x, y, xlabel, ylabel, zlabel, log_col=False, center_col=False, vmin=None, levels=None):
    """
    :param vals: values of the plotted variable (2D array)
    :param x: values of the variable of first dimension of array
    :param y: values of the variable of second dimension of array
    :param xlabel: variable of first dimension of array
    :param ylabel: variable of second dimension of array
    :param zlabel: plotted variable
    """
    plt.figure()
    if center_col is not False:
        if vmin is None:
            max_width = max(center_col - np.amin(vals), np.amax(vals)-center_col)
        else:
            max_width = center_col - vmin
        plt.imshow(vals, cmap="bwr", vmin=center_col-max_width if vmin is None else vmin, vmax=center_col+max_width)
    elif log_col:
        plt.imshow(vals, norm=LogNorm(vmin=np.nanmin(vals), vmax=np.nanmax(vals)))
    else:
        plt.imshow(vals, vmin=np.nanmin(vals) if vmin is None else vmin)
    # plt.pcolor(vals)
    plt.xticks(range(len(y)), y)
    plt.xlabel(ylabel)
    plt.yticks(range(len(x)), x)
    plt.ylabel(xlabel)
    # plt.plot([2.5, 2.5, 3.5, 3.5, 2.5], [4.5, 3.5, 3.5, 4.5, 4.5], color="red")
    cb = plt.colorbar()
    cb.set_label(zlabel)
    if levels is not None:
        plt.contour(vals, levels=levels, cmap="nipy_spectral")


def draw_superimposed_contours(wu_grid, ph_grid, last_grid, x, y, xlabel, ylabel, levels_auc, levels_pH, levels_last,
                               savefig=False):
    """

    :param x: values of the variable of first dimension of array
    :param y: values of the variable of second dimension of array
    :param xlabel: variable of first dimension of array
    :param ylabel: variable of second dimension of array
    :param zlabel: plotted variable
    :param savefig: if False, do not save figures. Otherwise, give full path (incl. filename)
    """

    plt.figure(figsize=(9, 7))
    r = [1.0, 0., 0.]
    yel = [1., 1., 0.]
    b = [0, 0.4, 1]

    plt.contourf(last_grid, levels=levels_last, colors=[b]*len(levels_last), alpha=0.4)
    plt.contour(last_grid, levels=levels_last, colors=[b]*len(levels_last))

    plt.contourf(wu_grid, levels=levels_auc, colors=[yel+[0.5], yel+[0.5]],   # darkblue, blue, lightblue
                 alpha=0.5)  # ["navy", "blue", "dodgerblue"]
    plt.contour(wu_grid, levels=levels_auc,
                colors=[yel, yel, yel])  # [darkblue, blue, lightblue]   # ["navy", "blue", "dodgerblue"]

    plt.contourf(ph_grid, levels=levels_pH, colors=[r] * len(levels_pH), alpha=0.3)
    plt.contour(ph_grid, levels=levels_pH, colors=[r] * len(levels_pH))

    ax = plt.gca()
    set_axes_props(ax)
    plt.xticks(range(len(y)), y, fontsize=18)
    plt.xlabel(ylabel, usetex=True, fontsize=18)   # ylabel
    plt.yticks(range(len(x)), x, fontsize=18)
    plt.ylabel(xlabel, usetex=True, fontsize=18)   # xlabel
    plt.grid(alpha=0.7, linewidth=1.5, color="grey")

    ax.invert_yaxis()

    if savefig:
        plt.savefig(savefig)


def analyse_asic_currents(source):
    """
    Function to create the plot of Supplementary figure 8.
    Represents all the dendritic currents during simulation with data saved in the source file (which should be as
    created by sim_saving.save_vars)
    """
    # first, load the data and collect the dendritic currents
    with open(source, "rb") as savefile:
        (plots, tvecs, styled) = pickle.load(savefile)
    for plot in plots:
        for tstr, ys in plot:
            t = np.array(tvecs[tstr])
            for ystr, y in ys:
                if ystr == "all_asic_i()":
                    asic_i = (t, np.array(y))
                elif "all_C_ampa_i()" == ystr:
                    ampa_i = (t, np.array(y))
                elif "all_C_nmda_i()" == ystr:
                    nmda_i = (t, np.array(y))
                elif "all_Ad_ampa_i()" == ystr:
                    Ad_ampa_i = (t, np.array(y))
                elif "all_Ad_nmda_i()" == ystr:
                    Ad_nmda_i = (t, np.array(y))
                elif "all_C_gaba_i()" == ystr:
                    gaba_i = (t, np.array(y))
                elif "all_nk1_i()" == ystr:
                    nk1_i = (t, np.array(y))
                elif "h.wdr_dend_ical" == ystr:
                    ical_i = (t, np.array(y))
                elif "h.wdr_dend_icaan" == ystr:
                    icaan_i = (t, np.array(y))
                elif "h.wdr_dend_ikca" == ystr:
                    ikca_dend_i = (t, np.array(y))
                else:
                    continue

    # now, plot
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    assert np.all(asic_i[0] == ampa_i[0]) and np.all(asic_i[0] == nmda_i[0])
    t = asic_i[0]

    ax2.plot(t, ampa_i[1]+nmda_i[1]+ical_i[1]+ikca_dend_i[1]+nk1_i[1]+gaba_i[1]+icaan_i[1] + Ad_ampa_i[1] + Ad_nmda_i[1],
             label="all except ASIC", color="k")
    ax2.plot([0], [0])
    ax2.plot(t, nk1_i[1], label="NK1 (C-fiber)")
    ax2.plot(t, gaba_i[1], label="GABAA (C-fiber)")
    ax2.plot(t, ampa_i[1], label="AMPA (C-fiber)")
    ax2.plot(t, nmda_i[1], label="NMDA (C-fiber)")
    ax2.plot(t, Ad_ampa_i[1], label="AMPA (Ad-fiber)")
    ax2.plot(t, Ad_nmda_i[1], label="NMDA (Ad-fiber)")
    ax2.plot(t, ikca_dend_i[1], label="iKCa")
    ax2.plot(t, ical_i[1], label="iCa,L")
    ax2.plot(t, icaan_i[1], label="iCaAN")
    ax2.plot(t, asic_i[1], label="ASIC", color="C0")
    fig2.legend()
    ax2.tick_params(labelsize=18, width=2)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(2)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Current (nA)")
    ax2.set_title("Currents in the dendrite")
    plt.show()


def analyse_calcium(source_potentiation, source_inhibition):
    """
    Function to create the plot of Supplementary figure 7.
    source_potentiation and source_inhibition should be two files as created by sim_saving.save_vars containing data for
    conductance 0.2nS and 1.4nS respectively.
    """
    for source, title_fill, name in [(source_potentiation, "0.2nS (wind-up potentiation)", "med"),
                                     (source_inhibition, "1.4nS (wind-up inhibition)", "high")]:
        analyse_calcium_(source, title_fill, name)
    plt.show()


def analyse_calcium_(source, title_fill, name):
    # first, load the data and collect the dendritic currents
    with open(source, "rb") as savefile:
        (plots, tvecs, styled) = pickle.load(savefile)
    for plot in plots:
        for tstr, ys in plot:
            t = np.array(tvecs[tstr])
            for ystr, y in ys:
                if ystr == "all_asic_i()":
                    asic_i = (t, np.array(y))
                elif "h.wdr_dend_icaan" == ystr:
                    icaan_dend_i = (t, np.array(y))

                elif "h.wdr_dend_ical" == ystr:
                    ical_dend_i = (t, np.array(y))
                elif "h.wdr_soma_ical" == ystr:
                    ical_soma_i = (t, np.array(y))
                elif "h.wdr_dend_ikca" == ystr:
                    ikca_dend_i = (t, np.array(y))
                elif "h.wdr_dend_cai" == ystr:
                    dend_ca = (t, np.array(y))
                elif "h.wdr_soma_cai" == ystr:
                    soma_ca = (t, np.array(y))
                elif "h.wdr_soma_v" == ystr:
                    soma_v = (t, np.array(y))
                elif "h.wdr_dend_v" == ystr:
                    dend_v = (t, np.array(y))
                else:
                    continue
                # add membrane potential, make a good zoom inset

    # now, plot
    fig2, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    assert np.all(asic_i[0] == ical_soma_i[0]) and np.all(asic_i[0] == soma_ca[0]) and np.all(asic_i[0] == soma_v[0])
    t = asic_i[0]

    ax2, ax1, ax0 = axs

    # the dendritic currents of interest
    ax2.plot(t, ikca_dend_i[1], label="iKCa", color="C9")
    ax2.plot(t, ical_dend_i[1], label="iCa,L", color="navy")
    ax2.plot(t, icaan_dend_i[1], label="iCaAN", color="C3")
    ax2.legend(loc=(0.81, 0.12))
    set_axes_props(ax2)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Current (nA)")
    ax2.set_ylim(-0.013, 0.12)

    # the dendritic calcium concentration
    set_axes_props(ax1)
    ax1.plot(t, dend_ca[1]*1000, label="dendrite", color="C0")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Calcium concentration (uM)")
    ax1.set_ylim(-0.25, 7.6)

    # the membrane potentials at soma and dendrite
    ax0.plot(t, soma_v[1], label="soma", color="gray")
    ax0.plot(t, dend_v[1], label="dendrite", color="mediumblue")
    ax0.legend()  # 'lower center'   # 0.38   # loc=(0.81, 0.12),
    set_axes_props(ax0)
    ax0.set_xlabel("Time (ms)")
    ax0.set_ylabel("Membrane potential (mV)")

    fig2.suptitle(f"Insights into the effect of ASIC calcium permeability in the dendrite.\nASIC maximal "
                  f"conductance {title_fill}")
    # option to plot the black square box representing the area of the close-up view:
    # ax2.plot([5000, 7000, 7000, 5000, 5000], [15e-5, 15e-5, -0.004, -0.004, 15e-5], color="k", linewidth=2)

    # option to zoom onto the close-up view:
    # ax2.set_ylim(-0.004, 15e-5)
    # ax2.set_xlim(5000, 7000)


def set_axes_props(ax):
    ax.tick_params(labelsize=18, width=2)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)


def plotASIC_grel(source_homomeric, source_heteromeric):
    """
    Function to create the plot of Supplementary figure 6E.
    source_homomeric and source_heteromeric should be two files as created by sim_saving.save_vars containing data for
    a homomeric and a heteromeric model respectively.
    """
    fig2, ax2 = plt.subplots(figsize=(12, 9))
    ax3 = plt.twinx(ax2)
    ax3.plot([0], [3e-3])
    for source, lab, ax in [(source_homomeric, "homomeric", ax2), (source_heteromeric, "heteromeric", ax3)]:
        plotASIC_grel_(source, ax, lab)
    ax2.tick_params(labelsize=18, width=2)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(2)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Rel. conductance m*h homomeric")
    ax3.set_ylabel("Rel. conductance m*h heteromeric")
    ax2.set_title("Relative conductance m*h of ASICs at one synapse")
    ax2.plot([0], [4e-2], label="heteromeric")
    ax2.legend(loc="lower right")
    set_axes_props(ax3)
    plt.show()


def plotASIC_grel_(source, ax2, label):
    savefile = open(source, "rb")
    (plots, tvecs, styled) = pickle.load(savefile)
    savefile.close()
    for plot in plots:
        for tstr, ys in plot:
            t = np.array(tvecs[tstr])
            for ystr, y in ys:
                if ystr == "one_asic_grel()":
                    asic_gr = (t, np.array(y))
                else:
                    continue

    t = asic_gr[0]
    ax2.plot(t, asic_gr[1], label=label)


def plot_chosen_different_pHs(folder, chosen_pars=[(1., 0.01), (0.05, 1.), (0.3, 0.1)],
                              labels=["alternative 1", "alternative 2", "chosen"], ca=0.1, cond=0.2):
    """
    Function to generate the graphs of Supplementary figure 9 A and E.
    chosen_pars is the list of (q, tau) for each parameter set
    labels is the list of corresponding names
    folder is where the pickle files as created by sim_saving.save_vars can be found
    If the pickle files were not generated using the standard name form given in the runMultiProcs.py file, you will
    need to modify accordingly the variable source
    ca and cond describe the conditions for which the pH was saved
    """
    fig, ax = plt.subplots(figsize=(10, 7.5))
    set_axes_props(ax)
    ax.set_title("Synaptic cleft acidification")

    for (q0, tau), lab in zip(chosen_pars, labels):
        name = name_from_pars(22, q0, tau, 1, 0.1, 0.2)[:-6]+"vars.pickle"
        source = os.path.join(folder, name)
        with open(source, "rb") as savefile:
            (plots, tvecs, styled) = pickle.load(savefile)
        for plot in plots:
            for tstr, ys in plot:
                t = tvecs[tstr]
                ystr, y = ys[0]
                if not "pH" in ystr:
                    continue
                ax.plot(t, y, label=lab, linewidth=2)
                break
    ax.legend()
    ax.set_xlim(0, 16000)
    ax.set_ylim(7.08, 7.415)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("pH")
    plt.show()


def plot_asic_currents_diff_chosen_pHs(source_dir, chosen_pH_pars=[(1., 0.01), (0.05, 1.)], cond=0.2):
    """
    Function to create the plot of Supplementary figure 9 D and H.
    source_dir is the folder in which the data files (as created by sim_saving.save_vars) of all parameter sets are to be found
    chosen_pH_pars is a list of the alternative parameter sets (q, tau)
    cond is the ASIc conductance to be displayed
    """
    for q0, tau in chosen_pH_pars:
        with open(os.path.join(source_dir, f"q0{q0}_tau{tau}_cond{cond}nS_b022_dur1_ca0.1_vars.pickle"), "rb") as savefile:   # here
            (plots, tvecs, styled) = pickle.load(savefile)
        for plot in plots:
            for tstr, ys in plot:
                t = np.array(tvecs[tstr])
                for ystr, y in ys:
                    if ystr == "all_asic_i()":
                        asic_i = (t, np.array(y))
                    elif ystr == "all_asic_g()":
                        asic_g = (t, np.array(y))
                    else:
                        continue
        fig, ax = plt.subplots(figsize=(10, 8))
        ax2 = plt.twinx(ax)
        assert np.all(asic_i[0] == asic_g[0])
        t = asic_i[0]
        ax.plot(t, asic_i[1]*1000, label="i", color="mediumblue")
        ax2.plot(t, asic_g[1]*1000, label="g", color="k", linewidth=2)

        fig.legend()   # 'lower center'   # 0.38   # loc=(0.64, 0.12),
        ax.tick_params(labelsize=18, width=2)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
            ax2.spines[axis].set_linewidth(2)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Total ASIC current i (pA)")
        ax2.set_ylabel("Total ASIC conductance g (pS)")
    plt.show()


def get_minph_from_file(filename):
    """
    Finds the minimum pH value reached during the simulation saved in file filename.
    """
    with open(filename, "rb") as savefile:
        (plots, tvecs, styled) = pickle.load(savefile)
    for plot in plots:
        for tstr, ys in plot:
            ystr, y = ys[0]
            if "pH" not in ystr:
                continue
            min_pH = np.amin(y)
            return min_pH
    return 0


def get_minph_data(directory):
    names = [name[:-7] for name in os.listdir(directory) if
             name.endswith(".pickle")]
    min_pHs = {}

    for name in custom_tqdm(names):
        ((b0, q0, tau, dur), ca, cond, act) = get_params_from_name(name)
        new_name = name_from_pars_forpH(b0, q0, tau, dur, ca, cond)
        mi = get_minph_from_file(os.path.join(directory, name + ".pickle"))
        if new_name in min_pHs:
            assert np.abs(min_pHs[new_name]-mi) < 0.01
        min_pHs[new_name] = mi
    with open(os.path.join(directory, "minpH.pk"), "wb") as f:
        pickle.dump(min_pHs, f)


def raw_analysis(directory, first=0):
    names = [name[:-4] for name in os.listdir(directory) if
             name.endswith(".dat")]
    spikesC = {}
    aucs = {}
    params = {}

    for name in custom_tqdm(names):
        spikesC[name] = get_spikes_from_file(os.path.join(directory, name+".dat"))
        aucs[name] = get_windup_from_spikes(spikesC[name])
        params[name] = get_params_from_name(name, first=first)   # ((b0, q0, tau, dur), ca, cond, act)

    with open(os.path.join(directory, "res.pk"), "wb") as f:
        pickle.dump((aucs, params, spikesC), f)

    data = defaultdict(list)
    for name, param in params.items():  # param: ((b0, q0, tau, dur), ca, cond, act)
        data[param[0]].append(
            ((param[1], param[2], param[3]), aucs[name], name))
    for lst in data.values():
        lst.sort()
    with open(os.path.join(directory, "data.pk"), "wb") as f:
        pickle.dump(data, f)


def prepare_spikes_and_pH_data(data_dir):
    raw_analysis(data_dir)
    get_minph_data(data_dir)


def load_data(data_dir):
    with open(os.path.join(data_dir, "res.pk"), "rb") as f:
        (aucs, params, spikesC) = pickle.load(f)
    with open(os.path.join(data_dir, "data.pk"), "rb") as f:
        data = pickle.load(f)
    try:
        with open(os.path.join(data_dir, "minpH.pk"), "rb") as f:
            minpH = pickle.load(f)
    except FileNotFoundError:
        print("No min pH data in this directory...")
        minpH = None
    # data[(b0, q0, tau, dur)] = [((ca, cond, act), auc, name), (), ...](sorted list)
    return data, spikesC, minpH


if __name__ == "__main__":
    data_dir = "sim_outputs"   # the directory containing the .dat and .pickle files
    prepare_spikes_and_pH_data(data_dir)
    if True:   # Figures 4, 5 and 9
        prepare_spikes_and_pH_data(data_dir)
        data, spikesC, _ = load_data(data_dir)
        make_paper_plots([(0.3, 0.1)], spikesC, data)   # (0.3, 0.2), , (0.1, 0.2)
        # The Figure 5C will not be complete unless you have also run the required simulations with ASIC fully activated
        # (see README for how to do this)

    if True:   # Supplementary figure 4
        data, spikesC, minpH = load_data(data_dir)
        heatmap(spikesC, data, minpH,
                chose_cond=([3] if "type2" in data_dir else [0.2]),
                save=data_dir)
        plt.show()

    if False:   # Supplementary figure 6E
        # you will need to run simulations for both heteromeric (type2) and homomeric (type1) models for this figure,
        # then change the paths below accordingly
        source1 = ".../type1/q00.3_tau0.1_cond0.2nS_b022_dur1_ca0.1_vars.pickle"
        source2 = ".../type2/q00.3_tau0.1_cond3.0nS_b022_dur1_ca0.1_vars.pickle"
        plotASIC_grel(source1, source2)

    if True:   # Supplementary figure 7
        source1 = os.path.join(data_dir, "q00.3_tau0.1_cond0.2nS_b022_dur1_ca0.1_vars.pickle")
        source2 = os.path.join(data_dir, "q00.3_tau0.1_cond1.4nS_b022_dur1_ca0.1_vars.pickle")
        analyse_calcium(source1, source2)

    if True:   # Supplementary figure 8
        analyse_asic_currents(os.path.join(data_dir, "q00.3_tau0.1_cond0.2nS_b022_dur1_ca0.1_vars.pickle"))
        

    if True:  # Supplementary figure 9 A and E together
        plot_chosen_different_pHs(folder=data_dir)

    if True:   # Supplementary figure 9 D and H
        plot_asic_currents_diff_chosen_pHs(source_dir=data_dir)

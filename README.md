This is the full code for the paper _Involvement of ASIC1a channels in the spinal processing of pain information by deep 
projection neurons_, Chafaï, M., Delrocq, A., Inquimbert, P., Pidoux, L., Delanoe, K., Lingueglia, E., Veltz, R., & Deval, E., 2023. doi: 10.1101/2021.08.02.454740.


It includes a NEURON model for the study of wind-up in wide-dynamic-range (WDR) neurons in the spinal cord, adapted from 
[[1]](#1), a NEURON model for Acid Sensing Ion Channels (ASIC) based on [[2]](#2) and [[3]](#3), a python model for 
ASICs, and python simulation and analysis code.

Written by Ariane Delrocq and Romain Veltz.



## Quick Start

(Optional: if you want, create and activate a python virtual environment.)

In the directory which contains the source code, please run

```bash
pip install neuron matplotlib pandas scipy
nrnivmodl
python write_args.py
./script.sh
python sim_analysis.py
python ASIC.py
```

The simulations may take quite some time, please be patient.

## More details
If you do not have NEURON installed, you must install it by running `pip install neuron`.
You also need the python packages scipy, matplotlib and pandas.

Before you run our model, you must compile the NEURON model with `nrnivmodl` (in this directory).
This needs to be done only once.

The script script.sh runs the simulation by calling `runMultiProcs.py` once for each parameter set in the `args.txt` file, 
which is written by `write_args.py`.
If you want to use multi-processing to reduce the duration of the total simulation time, 
set `num_processes` in `script_multiprocs.sh` to your desired number of processes and execute this file instead of `script.sh`.

The spike times are saved to .dat files, and the evolution of some variables over tine is saved in .pickle files.

You can then generate most of the paper simulation figures by running `sim_analysis.py`, which uses the saved data to create the graphs.

The file `ASIC.py` generates the figures for the simulations of isolated ASIC channels (Supplementary figure 6: pH drops...).


## Finding the models
The NEURON model for native homomeric ASIC1a channel is `ASICnative1.mod`. 
The alternative models for tau_h are commented out in the same file.

The NEURON model for native heteromeric ASIC channel is `ASIC_native2.mod`.

A python model of each type is implemented in `ASIC.py`.

Finally, the NEURON model for synaptic cleft proton dynamics with buffer is implemented in `PostProtCleftDyn.mod`.


## Understanding and modifying the code
Most of the run and analysis functions are well documented. Reading the comments and docstrings should help you 
understand the functions and modify them if you need.

The file `base.py` contains the basic functions used to run the simulation, as well as the functions that define the 
naming convention for the data files.
If you want to run simulations without the means of the `args.txt` file, you can call directly the function `runmodl` of this file,
with your desired arguments.
If you want to use a different naming convention, modify the `name_from_pars` and `get_params_from_name` functions. 
The parameter `acti` is whether the ASIC channels are structurally fully activated (effect of MiTx, see below for how to generate such data).

Because the hoc variables (of the NEURON simulation) are not properly reset at the end of the simulation, _never_ run 
two simulations in the same python session. _Always_ quit python and re-import neuron.

If you modify the content of a .mod file, do not forget to recompile with `nrnivmodl` (after deleting the old x86_64 folder).


To use the heteromeric/type 2 ASIC model, replace ASICnativeTone by ASICnativeTtwo in wdr-complete-model-without-interneuron.hoc.

To simulate full activation of the ASIC channels (effect of MiTx), go into the ASICnative1.mod file (or ASICnative2.mod) 
and replace the `m*h` factor by 1 in the expression of the current `i` in the `BREAKPOINT` procedure.
Then recompile with `nrnivmodl`.
to respect the naming convention, set `acti=True` when calling the `name_from_pars` function for these simulations.


### References
<a id="1">[1]</a> 
P. Aguiar, M. Sousa, and D. Lima, “NMDA Channels Together With L-Type Calcium Currents and Calcium-Activated Nonspecific Cationic Currents Are Sufficient to Generate Windup in WDR Neurons,” Journal of Neurophysiology, vol. 104, no. 2, pp. 1155–1166, Aug. 2010, doi: 10.1152/jn.00834.2009.

<a id="2">[2]</a>
O. Alijevic, O. Bignucolo, E. Hichri, Z. Peng, J. P. Kucera, and S. Kellenberger, “Slowing of the Time Course of Acidification Decreases the Acid-Sensing Ion Channel 1a Current Amplitude and Modulates Action Potential Firing in Neurons,” Front. Cell. Neurosci., vol. 14, 2020, doi: 10.3389/fncel.2020.00041.

<a id="3">[3]</a>
A. Baron, N. Voilley, M. Lazdunski, and E. Lingueglia, “Acid Sensing Ion Channels in Dorsal Spinal Cord Neurons,” Journal of Neuroscience, vol. 28, no. 6, pp. 1498–1508, Feb. 2008, doi: 10.1523/JNEUROSCI.4975-07.2008.

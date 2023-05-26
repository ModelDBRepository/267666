TITLE ASIC native

COMMENT
Acid-Sensing Ion Channel (ASIC) "Type 2" native current, as measured in Baron, A. et al. (2008) "Acid Sensing Ion
Channels in Dorsal Spinal Cord Neurons", Journal of Neuroscience, 28(6), pp. 1498–1508.
doi: 10.1523/JNEUROSCI.4975-07.2008.

Model built (based on the data from Baron et al., 2008) and implemented by Ariane Delrocq and Romain Veltz, 2019.
ENDCOMMENT

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
    POINT_PROCESS ASICnativeTtwo
    POINTER he
    USEION ca WRITE ica
    RANGE pH
	RANGE gbar, m, n, h
	RANGE m_inf, h_inf
	RANGE tau_m, tau_h
	RANGE i, g, e, ica, inon, ca_ratio
	GLOBAL a1_tau_m, a2_tau_m, b1_tau_m, b2_tau_m, c1_tau_m, c2_tau_m
	NONSPECIFIC_CURRENT inon
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
    (molar) = (1/liter)
    (mM) = (millimolar)
}

PARAMETER {
	gbar	= .1 	(mho/cm2)
	e = 50  (mV)
    a1_tau_m = 0.3487 (1)
    a2_tau_m = 234.3  (1)
    b1_tau_m = 10.14  (1)
    b2_tau_m = -5.553 (1)
    c1_tau_m = 6.392  (1)
    c2_tau_m = 8.595  (1)
    a_tau_h = 42.862
    b_tau_h = 5.375
    c_tau_h = 6.600
    d_tau_h = 1.645
	ca_ratio = 0.1 (1)
}

STATE {
	m   : activation variable
    h   : inactivation variable
}

ASSIGNED {
    v   (mV)
	i   (mA)
	ica (mA)
	inon (mA)
	g   (umho)
	m_inf
	h_inf
	tau_m (ms)
	tau_h (ms)
	pH
	he   (mM)
}


BREAKPOINT {
	SOLVE states
	i = gbar * m*h * (v - e)
	ica = ca_ratio * i
	inon = (1 - ca_ratio) * i
}


DERIVATIVE states {
	evaluate_fct(v)
	m' = (m_inf - m) / tau_m
	h' = (h_inf - h) / tau_h
}


UNITSOFF
INITIAL {

	evaluate_fct(v)
	m = m_inf
	h = h_inf
}


PROCEDURE evaluate_fct(v(mV)) { LOCAL a,b,v2,vh

    pH = -log(he * (0.001)) / log(10)

    tau_m = 1 / ( a1_tau_m / (1 + exp(b1_tau_m * (pH - c1_tau_m))) + a2_tau_m / (1 + exp(b2_tau_m * (pH - c2_tau_m))) )    : unchanged from homomeric model
    m_inf = 1 / (1+ 10^(1.94*(pH-6.03)))

    tau_h = (a_tau_h * exp(- b_tau_h * abs(pH - c_tau_h)) + d_tau_h) * 1000
    h_inf = 1 / (1+ 10^(-3.82*(pH-6.74)))
    
}

UNITSON

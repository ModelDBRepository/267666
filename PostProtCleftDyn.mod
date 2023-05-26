TITLE synaptic cleft H+ concentration



: INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	POINT_PROCESS PostProtCleftDyn
    RANGE tot_prot, he, q, h0, tau, pKd, pH0, b0, q0, duration
}

UNITS {
    (mV) = (millivolt)
	(molar) = (1/liter)		: moles do not appear in units
	(mM)	= (millimolar)
}


PARAMETER {
    tau = 1   (ms)
    pKd = 6.3    (1)
    b0 = 22      (mM)
    pH0 = 7.4    (1)
    q0 = 2.2     (M/s)
    Kd  (mM)
    duration = 1 (ms)
}

STATE {
	tot_prot  (mM)
	he  (mM)
}

INITIAL {
    h0 = (1000) * 10^(-pH0)
	he = h0
	Kd = (1000) * 10^(-pKd)
	tot_prot = h0 * (1 + b0 / (h0 + Kd))
	t0 = -2
}

ASSIGNED {
    v   (mV)
    q   (mM/ms)
    h0  (mM)
    t0  (ms)
}
	
BREAKPOINT {
	SOLVE state METHOD derivimplicit
}

DERIVATIVE state {
    evaluate_fct(v)
    tot_prot' = q - ((-(Kd + b0 - tot_prot) + sqrt((Kd + b0 - tot_prot)^2 + 4 * tot_prot * Kd)) / 2 - h0) / tau
    he = (-(Kd + b0 - tot_prot) + sqrt((Kd + b0 - tot_prot)^2 + 4 * tot_prot * Kd)) / 2      : pourquoi il faut le mettre la ?
}


PROCEDURE evaluate_fct(v(mV)) {

	if (t - t0 <= duration) {
	    q = q0
	} else {
	    q = 0
	}

}


NET_RECEIVE (weight){
    t0 = t
}

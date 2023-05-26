TITLE synaptic cleft H+ concentration



: INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
    : could be a POINTPROCESS ?
	SUFFIX ProtCleftDyn
    RANGE tot_prot, he, q, h0
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
    pH0 = 7.3    (1)
    q0 = 2.2     (M/s)
    Kd  (mM)
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
}

ASSIGNED {
    v   (mV)
    q   (mM/ms)
    h0  (mM)
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

	if (v >= -30) {
	    q = q0
	} else {
	    q = 0
	}

}

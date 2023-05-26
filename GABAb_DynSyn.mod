TITLE GABA_B receptor with pre-synaptic short-term plasticity 


COMMENT
GABA_B receptor conductance using a dual-exponential profile
Pre-synaptic short-term plasticity based on Fuhrmann et al, 2002

Written by Paulo Aguiar and Mafalda Sousa, IBMC, May 2008
pauloaguiar@fc.up.pt ; mafsousa@ibmc.up.pt
ENDCOMMENT



NEURON {
	POINT_PROCESS GABAb_DynSyn	
	RANGE tau_rise, tau_decay
	RANGE U1, tau_rec, tau_fac, stp
	RANGE i, g, e
	NONSPECIFIC_CURRENT i
}

PARAMETER {
	tau_rise  = 50   (ms)  : dual-exponential conductance profile
	tau_decay = 200   (ms)  : IMPORTANT: tau_rise < tau_decay
	U1        = 1.0   (1)   : The parameter U1, tau_rec and tau_fac define _
	tau_rec   = 0.1   (ms)  : the pre-synaptic SP short-term plasticity _
	tau_fac   = 0.1   (ms)  : mechanism (see Fuhrmann et al, 2002)
	e         = -95   (mV)  : GABAb synapse reversal potential
	stp       = 1.0   (1)   : boolean for synaptic plasticity
}
     

ASSIGNED {
	v (mV)
	i (nA)
	g (umho)
	factor
}

STATE {
	A	: state variable to construct the dual-exponential profile
	B	: 
}

INITIAL{
	LOCAL tp
	A = 0
	B = 0
	tp = (tau_rise*tau_decay)/(tau_decay-tau_rise)*log(tau_decay/tau_rise)
	factor = -exp(-tp/tau_rise)+exp(-tp/tau_decay)
	factor = 1/factor
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g = B-A
	i = g*(v-e)
}

DERIVATIVE state{
	A' = -A/tau_rise
	B' = -B/tau_decay
}

NET_RECEIVE (weight, Pv, P, Use, t0 (ms)){
	INITIAL{
		P=1
		Use=0
		t0=t
	    }

	if(stp){
        Use = Use * exp(-(t-t0)/tau_fac)
        Use = Use + U1*(1-Use)
        P   = 1-(1- P) * exp(-(t-t0)/tau_rec)
        Pv  = Use * P
        P   = P - Use * P

        t0 = t

        A = A + weight*factor*Pv
        B = B + weight*factor*Pv
    } else {
        A = A + weight*factor
        B = B + weight*factor
    }
}

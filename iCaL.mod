TITLE high threshold calcium current (L-current)

COMMENT
        *********************************************
        reference:      McCormick & Huguenard (1992) 
			J.Neurophysiology 68(4), 1384-1400
        found in:       hippocampal pyramidal cells
        *********************************************
	Assembled for MyFirstNEURON by Arthur Houweling
	
ENDCOMMENT




INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}


NEURON {
	SUFFIX iCaL
	USEION ca READ cai,cao WRITE ica
        RANGE pcabar, m_inf, tau_m, ica, ghkval
}

UNITS {
	(mA)	= (milliamp)
	(mV)	= (millivolt)
	(mM)	= (milli/liter)
        FARADAY = 96480 (coul)
        R       = 8.314 (volt-coul/degC)
}

PARAMETER {
	v			(mV)
	celsius			(degC)
        dt              	(ms)
	cai			(mM)
	cao			(mM)
	pcabar= 0.000276	(cm/s)		
}

STATE {
	m
}

ASSIGNED {
	ica		(mA/cm2)
	tau_m		(ms)
	m_inf 
	tadj
	ghkval
}

BREAKPOINT { 
	SOLVE states :METHOD euler
    ica = pcabar * m*m * ghk(v,cai,cao,2)
}

DERIVATIVE states {
       rates(v)

       m'= (m_inf-m) / tau_m
}
  
:PROCEDURE states() {
:        rates(v)
:
:        m= m + (1-exp(-dt/tau_m))*(m_inf-m)
:	:printf("\n iCaL tau_m=%g", tau_m)
:}

UNITSOFF

INITIAL {
	tadj = 3.0 ^ ((celsius-23.5)/10)
	rates(v)
	m = m_inf
}

FUNCTION ghk( v(mV), ci(mM), co(mM), z)  (millicoul/cm3) {
        LOCAL e, w
        w = v * (.001) * z*FARADAY / (R*(celsius+273.16))
        
	if (fabs(w)>1e-4) 
          { e = w / (exp(w)-1) }
    else
	: denominator is small -> Taylor series
        { e = 1-w/2 }

	ghkval = - (.001) * z*FARADAY * (co-ci*exp(w)) * e
    ghk = ghkval
}
UNITSOFF

PROCEDURE rates(v(mV)) { LOCAL a,b
	a = 1.6 / (1+ exp(-0.072*(v-5)))
	b = 0.02 * vtrap( -(v-1.31), 5.36)

	tau_m = 1/(a+b) / tadj
	m_inf = 1/(1+exp((v+10)/-10))
}

FUNCTION vtrap(x,c) { 
	: Traps for 0 in denominator of rate equations
        if (fabs(x/c) < 1e-6) {
          vtrap = c + x/2 }
        else {
          vtrap = x / (1-exp(-x/c)) }
}
UNITSON

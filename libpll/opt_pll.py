""" Calculate optimal PLL values from theory
"""
import numpy as np
K_B = 1.38064852e-23
TWOPI = 2*np.pi
BETA = 0.8947598824664404
##########################################################################
#       Theoretical PN limit for ring oscillator
##########################################################################

def _pn_min(p, temp, fosc, df):
    """ Ideal ring oscillator SSB Phase noise at df from carrer fosc
        args:
            p - DC power [W]
            temp - temperature [k]
            fosc - oscillator frequency
            df - phase noise offset from fosc
        returns
            phase noise density (carrier relative), linear scale
    """
    return (7.33*K_B*temp/p)*(fosc/df)**2
pn_min = np.vectorize(_pn_min, otypes=[float])

def _pn_min_db(p, temp, fosc, df):
    """ Ideal ring oscillator SSB Phase noise at df from carrer fosc
        args:
            p - DC power [W]
            temp - temperature [k]
            fosc - oscillator frequency
            df - phase noise offset from fosc
        returns
            phase noise density (carrier relative), dB
    """
    return 10*np.log10(pn_min(p, temp, fosc, df))
pn_min_db = np.vectorize(_pn_min_db, otypes=[float])

def _fom_min_ro(temp):
    """ Theoretical min ring oscillator FOM
    args:
        temp - temperature [k]
    returns:
        FOM phase noise, linear
    """
    return 7.33*1000*K_B*temp
fom_min_ro = np.vectorize(_fom_min_ro, otypes=[float])

def _fom_min_ro_db(temp):
    """ Theoretical min ring oscillator FOM
    args:
        temp - temperature [k]
    returns:
        FOM phase noise, linear
    """
    return(fom_min_ro(temp))

def _s0_osc(fom, p, fosc, mode="db"):
    if mode == "db":
        fom = 10**(fom/10.0)
    return 0.001*fom*fosc**2/p
s0_osc = np.vectorize(_s0_osc, otypes=[float])

##########################################################################
#   Derived values for PI-PLL with damping=1.0
##########################################################################



def _bw_pipll(arg):
    """ finds bandwidth of damping=1 PI-PLL
    args:
        - k = 2*pi*Kpd*Kdco*Ki
    returns
        bandwidth [Hz]
    """
    if type(arg) is dict:
        k = arg["k"]
    elif type(arg) in [float, int]:
        k = arg
    return (1/TWOPI)*np.sqrt(k)*np.sqrt(3 + np.sqrt(10))
bw_pipll = np.vectorize(_bw_pipll, otypes=[float])

def _k_fixed_bw(alpha, fref):
    """ find value of K for fixed BW, where BW = a*fref
    args:
        alpha - ratio of desired BW to fref
        fref - reference frequency
    """
    return (TWOPI*alpha*fref)**2/(3 + np.sqrt(10))
k_fixed_bw = np.vectorize(_k_fixed_bw, otypes=[float])

def _tsettle(k, ftol, df0):
    """ finds settling time
    args:
        k - 2*pi*Kpd*Kdco*Ki
        ftol - frequency tolerance from fosc that is "locked"
        df0 - initial frequency error
    """
    if df0 == 0:
        return 0
    else:
        return -np.log(abs(ftol/df0))/np.sqrt(k)
tsettle = np.vectorize(_tsettle, otypes=[float])

def tsettle_lf(lf, ftol, df0):
    return tsettle(lf["k"], ftol, df0)

##########################################################################
#   BBPD
##########################################################################

def _kbbpd(pn_rms):
    return np.sqrt(2/np.pi)/pn_rms
kbbpd = np.vectorize(_kbbpd, otypes=[float])

def _total_int_pn_bbpd_jit(s0_osc, rms_jit, k, fref, fosc):
    """ RMS jitter is the BBPD added jitter component [s]
    """
    int_pn_jit = (TWOPI*fosc*rms_jit)**2
    a = s0_osc*np.pi**2/np.sqrt(k)
    b = 5*np.pi*np.sqrt(k)*int_pn_jit/(8*fref)
    c = 1-5*np.sqrt(k)*(np.pi/2 - 1)/(4*fref)
    return (a+b)/c
total_int_pn_bbpd_jit = np.vectorize(_total_int_pn_bbpd_jit, otypes=[float])

# s0 = s0_osc(-157, 1e-4, 2.448e9)
# print(10*np.log10(total_int_pn_bbpd_jit(s0, 1.4e-12, 1e14, 16e6, 2.448e9)))

def _kbbpd_gain_reduction(s0_osc, rms_jit, k, fref, fosc):
    """ Computes fraction of noiseless kbbpd the detector operates at
    """
    tot_jit = total_int_pn_bbpd_jit(s0_osc, rms_jit, k, fref, fosc)
    tot_no_jit = total_int_pn_bbpd_jit(s0_osc, 0, k, fref, fosc)
    return tot_no_jit/tot_jit
kbbpd_gain_reduction = np.vectorize(_kbbpd_gain_reduction, otypes=[float])
# s0 = s0_osc(-157, 1e-4, 2.448e9)
# print(kbbpd_gain_reduction(s0, 1.4e-12, 1e14, 16e6, 2.448e9))

def _kopt_bbpd_jit(s0_osc, rms_jit, fref, fosc):
    """ determines optimal k for PLL with bbpd jitter as second/first most dominant source
        of noise
    """
    int_pn_jit = (TWOPI*fosc*rms_jit)**2
    a = s0_osc*np.pi*(np.pi-2)/int_pn_jit
    b = s0_osc*8*np.pi*fref/(5*int_pn_jit)
    return (a-np.sqrt(a**2 + b))**2
kopt_bbpd_jit = np.vectorize(_kopt_bbpd_jit, otypes=[float])
# s0 = s0_osc(-165.2, 1e-4, 2.448e9)
# k = kopt_bbpd_jit(s0, 1.4e-12, 16e6, 2.448e9)
# print("K=%E"%k)
# print("BW=%E"%bw_pipll(k))

def _bbpd_jit_limit(s0_osc, tot_int_pn, alpha, fref, fosc):
    a = np.sqrt(tot_int_pn)/(TWOPI*fosc)
    b = 4*np.sqrt(3 + np.sqrt(10))/(5*np.pi**2*alpha) + 2/np.pi -1
    c = 2*s0_osc*(3 + np.sqrt(10))/(5*np.pi*alpha**2*fref)
    return a*np.sqrt(b-c)
bbpd_jit_limit = np.vectorize(_bbpd_jit_limit, otypes=[float])
# s0 = s0_osc(-157.0, 80e-6, 2448e6)
# print(bbpd_jit_limit(s0, 0.01, 0.1, 16e6, 2448e6))

def calc_lf_bbpd(kdco, k, tot_int_pn):
    lf_params = {}
    lf_params["kdco"] = kdco
    lf_params["k"] = k
    lf_params["int_pn"] = tot_int_pn
    lf_params["fz"] = np.sqrt(k)/(2*TWOPI)
    lf_params["kp"] = np.sqrt(k)*np.sqrt(tot_int_pn)/(np.sqrt(TWOPI)*kdco)
    lf_params["ki"] = k*np.sqrt(tot_int_pn)/(2*np.sqrt(TWOPI)*kdco)
    lf_params["kpd"] = kbbpd(pn_rms=np.sqrt(tot_int_pn))
    lf_params["mode"] = "bbpd"
    return lf_params

# print(calc_lf_bbpd(5000, 1e14, 0.04))

##########################################################################
#   Emergent BB noise
##########################################################################

# def _total_int_pn_emerg_bb(s0_osc, k, fref):
    # """ OLD, innacurate
    # """
    # a = s0_osc*np.pi**2/np.sqrt(k)
    # b = 2*np.pi**3*np.sqrt(k)*s0_osc/(fref**2 - TWOPI*k)
    # return a + b
# total_int_pn_emerg_bb = np.vectorize(_total_int_pn_emerg_bb, otypes=[float])

def _total_int_pn(s0_osc, k, fref, beta=BETA):
    """ full model including bbpd noise, emergent bb noise, dco noise
    """
    a = s0_osc*np.pi**2/np.sqrt(k)
    b = beta**2*2*np.pi*k/fref**2
    c = 5*np.sqrt(k)*(np.pi/2 - 1)/(4*fref)
    return a/(1-b-c)
total_int_pn = np.vectorize(_total_int_pn, otypes=[float])

def _alpha_opt(beta=BETA):
    a = 1
    b = 5*np.sqrt(3+np.sqrt(10))*(np.pi/2-1)/(24*np.pi**2*beta**2)
    c = -(3+np.sqrt(10))/(24*np.pi**3*beta**2)
    # print(b, c)
    return -b/(2*a) + 0.5*np.sqrt(b**2 - 4*a*c)
alpha_opt = np.vectorize(_alpha_opt, otypes=[float])

def _kopt(beta, fref):
    return k_fixed_bw(alpha_opt(beta), fref)
kopt = np.vectorize(_kopt, otypes=[float])
# s0 = s0_osc(-160.0, 80e-6, 2448e6)
# print(total_int_pn_emerg_bb(s0, 1e13, 16e6))

# def _kopt_emerg_bb(fref):
    # """ OLD, innacurate
    # """
    # return fref**2/(6*np.pi)
# kopt_emerg_bb = np.vectorize(_kopt_emerg_bb, otypes=[float])

# kopt = kopt_emerg_bb(16e6)
# print(total_int_pn_emerg_bb(s0, kopt, 16e6))

##########################################################################
#   Osc
##########################################################################

def _int_pn_osc(s0_osc, k):
    return s0_osc*np.pi**2/np.sqrt(k)
int_pn_osc = np.vectorize(s0_osc, otypes=[float])

##########################################################################
#   Synchronous counter
##########################################################################

def calc_lf_sc(alpha, fref, fosc, kdco):
    lf = {}
    a = np.pi*alpha*fref
    b = 3 + np.sqrt(10)
    lf["kdco"] = kdco
    lf["k"] = k_fixed_bw(alpha, fref)
    lf["fz"] = a/(TWOPI*np.sqrt(b))
    lf["kp"] = 4*a/(np.sqrt(b)*kdco)
    lf["ki"] = (2*a)**2/(b*kdco)
    lf["kpd"] = 1/TWOPI
    lf["mode"] = "tdc"
    lf["m"] = int(round(fosc/fref))
    return lf

##########################################################################
#   Digitize loop filter
##########################################################################

def calc_discrete_lf(lf_params, fref):
    """ requires Kp, wz, fref
    """
    kp = lf_params["kp"]
    wz = TWOPI*lf_params["fz"]
    lf_params["fref"] = fref
    lf_params["b0"] = kp*(1+wz/fref)
    lf_params["b1"] = -kp
    lf_params["delay"] = 1/fref
    return lf_params

# lf = calc_lf_bbpd(5000, 1e14, 0.04)
# calc_lf_bbpd(lf, 16e6)
# print(lf_params)
##########################################################################
#   Estimate LF parameters
##########################################################################


def calc_pll_bw(params):
    params["bw"] = bw_pipll(params["k"])
    return params

##########################################################################
#   Estimate LF parameters
##########################################################################

def design_filters(fom, rms_jit, fref, fosc, alpha_max, p, kdco, beta=BETA, alpha_max_sc=0.1):
    """ calc sc filter and BBPD filter
    """
    s0 = s0_osc(fom, p, fosc)
    if rms_jit == 0.0: # causes divide by zero, avoid...
        rms_jit = 1e-18
    # Calculate for BBPD jitter first
    print("\nCalculating LF for BBPD jitter")
    k_jit = kopt_bbpd_jit(s0, rms_jit, fref, fosc)
    int_pn_jit = total_int_pn_bbpd_jit(s0, rms_jit, k_jit, fref, fosc)
    bw_jit = bw_pipll(k_jit)
    alpha = bw_jit/fref
    if alpha > alpha_max:
        print("\tBBPD jitter optimized alpha=%f > alpha_max=%f)"%(alpha, alpha_max))
        print("\tSetting alpha=alpha_max")
        alpha = alpha_max
        k_jit = k_fixed_bw(alpha_max, fref)
        int_pn_jit = total_int_pn_bbpd_jit(s0, rms_jit, k_jit, fref, fosc)
    print("\tComputed integrated PN = %f [rad^2]"%int_pn_jit)

    # Calculate LF for emergent BB behavior
    print("\nCalculating LF for Emergent BB behavior")
    k_emerg = kopt(beta, fref)
    int_pn_emerg = total_int_pn(s0, k_emerg, fref, beta)
    print("\tComputed integrated PN = %f [rad^2]"%int_pn_emerg)

    # take result with worse total integrated phase noise
    if int_pn_jit > int_pn_emerg:
        print("\nBBPD jitter selected as dominant")
        k = k_jit
        int_pn = int_pn_jit
    else:
        print("\nBB Emergent behavior determined as dominant")
        k = k_emerg
        int_pn = int_pn_emerg
    bw = bw_pipll(k)
    alpha = bw/fref
    print("\tBW = %E [Hz]"%bw)
    print("\tkbbpd/kbbpd0 = %f"%kbbpd_gain_reduction(s0, rms_jit, k, fref, fosc))
    print("\tMax BBPD jitter = %E [s]"%bbpd_jit_limit(s0, int_pn, alpha, fref, fosc))
    lf_bbpd = calc_lf_bbpd(kdco, k, int_pn)
    calc_discrete_lf(lf_bbpd, fref)
    lf_bbpd["bw"] = bw
    lf_bbpd["posc"] = p
    lf_bbpd["fosc"] = fosc
    lf_bbpd["fom"] = fom
    lf_bbpd["alpha"] = alpha
    lf_bbpd["rms_jit"] = rms_jit
    print("\nFinal BBPD LF Optimization result:")
    for k,v in lf_bbpd.items():
        print("\t%s\t->\t%r"%(k,v))

    # Opt synchronous counter
    print("\nSynchronous counter optimization")
    lf_sc = calc_lf_sc(alpha_max_sc, fref, fosc, kdco)
    calc_discrete_lf(lf_sc, fref)
    lf_sc["bw"] = fref*alpha_max
    lf_sc["posc"] = p
    lf_sc["fosc"] = fosc
    lf_sc["fom"] = fom
    lf_sc["alpha"] = alpha_max_sc
    print("\nFinal SC LF Optimization result:")
    for k,v in lf_sc.items():
        print("\t%s\t->\t%r"%(k,v))

    return {"sc"    : lf_sc,
            "bbpd"  : lf_bbpd,
           }



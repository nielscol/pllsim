""" Engine for running simulation
"""

import numpy as np
from typing import Callable
from copy import copy
from libpll.tools import timer
from libpll.pllcomp import ClockPhase, SCPDPhase, BBPD, DCOPhase, LoopFilterPIPhase
from libpll._signal import make_signal
import multiprocessing


#////////////////////////////////////////////////////////////////////////////////////////////
# BBPD PI-PLL simulator

def sim_bbpd_pipll(fclk, fs, sim_steps, div_n, use_bbpd, bbpd_rms_jit,
                 kdco1, kdco2, fine_bits, med_bits, fl_dco, krw_dco,
                 lf_i_bits, lf_f_bits, lf_params_sc, lf_params_bbpd,
                 tsettle_est, init_params, verbose=True, ignore_clk=False,
                 *args, **kwargs):
    # init pll component objects
    if verbose and args: print("Unused args = %r"%args)
    if verbose and kwargs: print("Unused kwargs = %r"%kwargs)
    clk = ClockPhase(f=fclk, dt=1/float(fs), init_phase=init_params["clk"])
    scpd = SCPDPhase(modulus=div_n, init_clk=init_params["clk"], init_out=init_params["scpd"],
                    ignore_clk=ignore_clk)
    bbpd = BBPD(rms_jit=bbpd_rms_jit, fclk=fclk, init_clk=init_params["clk"],
                init_out=init_params["bbpd"], ignore_clk=ignore_clk)
    dco = DCOPhase(kdco1=kdco1, kdco2=kdco2, f0=fl_dco, dt=1/float(fs), krw=krw_dco,
                   init_phase=init_params["osc"], quantize=True)
    lf = LoopFilterPIPhase(init_out=init_params["lf"], init_clk=init_params["clk"],
                            int_bits=lf_i_bits, frac_bits=lf_f_bits, ignore_clk=ignore_clk,
                            verbose=verbose, out_bits=fine_bits+med_bits, **lf_params_sc)

    bbpd_lf_switch = int(np.round(fclk*tsettle_est))
    if bbpd_lf_switch ==0:
        bbpd_lf_switch = 1
    print("LF gearswitch at %d samples"%bbpd_lf_switch)
    def eval_step(n, step):
        if n == bbpd_lf_switch and use_bbpd:
            if verbose: print("\n* Simulation step = %d, switching to BBPD optimized loop filter"%n)
            lf.change_filter(x1init=1, **lf_params_bbpd)
            # dco.clk_align()
        step["osc"]   = dco.update(step["fine"], step["med"])
        step["clk"]   = clk.update()
        step["scpd"]  = scpd.update(clk=step["clk"], xin=step["osc"])
        step["bbpd"]  = bbpd.update(clk=step["clk"], xin=step["osc"])
        if n < bbpd_lf_switch: step["error"] = step["scpd"]
        else: step["error"] = step["bbpd"]
        step["lf"]    = lf.update(xin=step["error"], clk=step["clk"])
        fine, med = fine_med(step["lf"], fine_bits, med_bits)
        step["fine"] = fine
        step["med"] = med
        return step

    data = run_sim(eval_step, sim_steps, init_params)

    for k,v in data.items():
        data[k] = make_signal(td=v, fs=fs)
    params = dict(fclk=fclk, fs=fs, sim_steps=sim_steps, div_n=div_n, use_bbpd=use_bbpd,
                  bbpd_rms_jit=bbpd_rms_jit, kdco1=kdco1, kdco2=kdco2, fl_dco=fl_dco,
                  krw_dco=krw_dco, lf_i_bits=lf_i_bits, lf_f_bits=lf_f_bits,
                  lf_params_sc=lf_params_sc, lf_params_bbpd=lf_params_bbpd,
                  fine_bits=fine_bits, med_bits=med_bits,
                  tsettle_est=tsettle_est, init_params=init_params, verbose=verbose)
    for k,v in kwargs.items():
        params[k] = v
    data["params"] = params
    return data



def sim_bbpd_pipll_mp(params, verbose=False):
    """ Method to launch sim with multiprocessing
    """
    # lf_init = int(round((params["fosc"]-params["fl_dco"]+params["init_f_err"])/params["kdco"]))
    # lf_ideal = int(round((params["fosc"]-params["fl_dco"])/params["kdco"]))
    lf_init = int(round((params["init_f"]-params["fl_dco"])/params["kdco"]))
    lf_ideal = int(round((params["fosc"]-params["fl_dco"])/params["kdco"]))
    params["init_params"]["lf"] = lf_init
    params["lf_init"] = lf_init
    params["lf_ideal"] = lf_ideal
    return sim_bbpd_pipll(verbose=verbose, **params)

#////////////////////////////////////////////////////////////////////////////////////////////
# Simulator sub-methods

def save_step(n, step_data, data):
    """ Saves simulation step result to dictionary with arrays
        containing full time evolution of simulation
        args:
            step_data: dict[
    """
    for k,v in step_data.items():
        data[k][n] = v


@timer
def run_sim(updater, steps, init_params):
    """ Runs/saves simulation data
    """
    step_data = copy(init_params)
    data = {}
    for k,v in init_params.items():
        data[k] = np.zeros(steps)
        data[k][0] = v
    for n in range(1, steps):
        step_data = updater(n, step_data)
        save_step(n, step_data, data)
    return data

#////////////////////////////////////////////////////////////////////////////////////////////
# Monte-Carlo and sweep engines


@timer
def sim_sweep(sim_engine, sim_params, sweep_param, sweep_vals, mp=True):
    """ Do a parametric sweep of PLL
        args:
            sim_engine - pll simulation method
            sim_params - parameters nominally passed to sim_engine
            sweep_param - parameter of sim_engine to sweep (str of argument keyword),
                or list of argument keywords to sweep
            sweep_vals - list of values that sweep_param should be simulated for
                or list of lists if multiple sweep_params is used.
    """
    if type(sweep_param) == list: hierarchy = [len(x) for x in sweep_vals]

    swept_params = make_sweep_sim_params(sim_params, sweep_param, sweep_vals)
    _swept_params = flatten(swept_params)

    if mp: # use multiprocessing
        cpus = multiprocessing.cpu_count()
        p = multiprocessing.Pool(cpus)
        _data = p.map(sim_engine, _swept_params)
    else:
        _data = [sim_engine(**params) for params in _swept_params]

    if type(sweep_param) == list: data = unflatten(_data, hierarchy)
    else: data = _data
    return data


def fine_med(lf_out, fine_bits, med_bits):
    if lf_out < 0:
        lf_out = 0
    elif lf_out > 2**(fine_bits+med_bits)-1:
        lf_out = 2**(fine_bits+med_bits)-1
    med = np.floor(lf_out/2**(fine_bits))
    fine = lf_out - med*2**(fine_bits)
    return fine, med


@timer
def sim_mc(sim_engine, sim_params, var_param, stdev, samples, mp=True):
    """ Do a variance analysis of PLL via monte-carlo sampling
        args:
            sim_engine - pll simulation method
            sim_params - parameters nominally passed to sim_engine
            var_param - parameter of sim_engine to be varied (str of argument keyword),
                or list of argument keywords to sweep
            stdev - standard deviation that var_param should be simulated for
                or list of lists if multiple var_params is used.
    """
    swept_params = []
    if type(var_param) == list:
        for n in range(samples):
            _sim_params = copy(sim_params)
            for _param, _stdev in zip(var_param, stdev):
                nom = sim_params[_param]
                _sim_params[_param] = np.random.normal(nom, nom*_stdev)
            swept_params.append(_sim_params)
    else:
        for n in range(samples):
            _sim_params = copy(sim_params)
            nom = sim_params[var_param]
            _sim_params[var_param] = np.random.normal(nom, nom*stdev)
            swept_params.append(_sim_params)
    if mp: # use multiprocessing
        cpus = multiprocessing.cpu_count()
        p = multiprocessing.Pool(cpus)
        data = p.map(sim_engine, swept_params)
    else:
        data = [sim_engine(**params) for params in swept_params]

    return data

def flatten(items):
    """ Use to flatten list for perfoming map operation with multiprocessing
    """
    flattened = []
    for item in items:
        if type(item) == list:
            flattened.extend(flatten(item))
        else:
            flattened.append(item)
    return flattened

def unflatten(items, hierarchy):
    """ Undo list flattening done for map operation
    """
    unflattened = [items]
    for sweep_len in hierarchy[-1:0:-1]:
        unflattened.append([])
        n = 0
        unflattened[-1].append([])
        for item in unflattened[-2]:
            if n!=0 and n%sweep_len == 0:
                unflattened[-1].append([])
            unflattened[-1][-1].append(item)
            n += 1
    return unflattened[-1]

def make_sweep_sim_params(sim_params, sweep_param, sweep_vals):
    swept_sim_params = []
    if type(sweep_param) == list and len(sweep_param) > 1:
        for val in sweep_vals[0]:
            _sim_params = copy(sim_params)
            _sim_params[sweep_param[0]] = val
            swept_params = make_sweep_sim_params(_sim_params, sweep_param[1:], sweep_vals[1:])
            swept_sim_params.append(swept_params)
    else:
        if type(sweep_param) == list:
            sweep_param = sweep_param[0]
            sweep_vals = sweep_vals[0]
        for val in sweep_vals:
            _sim_params = copy(sim_params)
            _sim_params[sweep_param] = val
            swept_sim_params.append(_sim_params)
    return swept_sim_params



#////////////////////////////////////////////////////////////////////////////////////////////
# Old TDC and divider based PLL model

# def pllsim_int_n(fclk, fs, sim_steps, div_n, tdc_steps, use_bbpd, kbbpd, bbpd_tsu, bbpd_th,
                 # kdco, fl_dco, krwro_dco, lf_i_bits, lf_f_bits, lf_params, lf_params_bbpd,
                 # tsettle_est, init_params, verbose=True, *args, **kwargs):
    # # init pll component objects
    # if verbose and args: print("Unused args = %r"%args)
    # if verbose and kwargs: print("Unused kwargs = %r"%kwargs)
    # clk = ClockPhase(f=fclk, dt=1/float(fs), init_phase=init_params["clk"])
    # tdc = TDCPhase(tdc_steps=tdc_steps, init_clk=init_params["clk"],
                   # init_out=init_params["tdc"])
    # bbpd = BBPD(tsu=bbpd_tsu, th=bbpd_th, fclk=fclk, init_clk=init_params["clk"],
                # init_out=init_params["bbpd"])
    # dco = DCOPhase(kdco=kdco, f0=fl_dco, dt=1/float(fs), krwro=krwro_dco,
                   # init_phase=init_params["osc"], quantize=True)
    # lf = LoopFilterIIRPhase(init_out=init_params["lf"], init_clk=init_params["clk"],
                            # int_bits=lf_i_bits, frac_bits=lf_f_bits,
                            # verbose=verbose, **lf_params)
# 
    # bbpd_lf_switch = int(np.ceil(fclk*tsettle_est))
# 
    # def eval_step(n, step):
        # if n == bbpd_lf_switch and use_bbpd:
            # if verbose: print("\n* Simulation step = %d, switching to BBPD optimized loop filter"%n)
            # lf.change_filter(**lf_params_bbpd)
        # if n >= bbpd_lf_switch: bbpd_weight = 1.0
        # else: bbpd_weight = 0.0
        # step["osc"]   = dco.update(step["lf"])
        # step["div"]   = step["osc"]/float(div_n)
        # step["clk"]   = clk.update()
        # step["tdc"]   = tdc.update(clk=step["clk"], xin=step["div"])
        # _bbpd = step["bbpd"]
        # step["bbpd"]  = bbpd.update(clk=step["clk"], xin=step["div"])
        # if _bbpd!=step["bbpd"] and use_bbpd:
            # step["kbbpd"] = 0.5*step["kbbpd"] if step["kbbpd"] > 2*kbbpd else kbbpd
        # step["error"] = step["tdc"] + step["kbbpd"]*step["bbpd"]*bbpd_weight
        # step["lf"]    = lf.update(xin=step["error"], clk=step["clk"])
        # return step
# 
    # if not use_bbpd: init_params["kbbpd"] = 0
    # data = run_sim(eval_step, sim_steps, init_params)
# 
    # for k,v in data.items():
        # data[k] = make_signal(td=v, fs=fs)
    # params = dict(fclk=fclk, fs=fs, sim_steps=sim_steps, div_n=div_n, tdc_steps=tdc_steps,
                  # kbbpd=kbbpd, bbpd_tsu=bbpd_tsu, bbpd_th=bbpd_th, kdco=kdco, fl_dco=fl_dco,
                  # krwro_dco=krwro_dco, lf_i_bits=lf_i_bits, lf_f_bits=lf_f_bits,
                  # lf_params=lf_params, init_params=init_params, verbose=verbose)
    # for k,v in kwargs.items():
        # params[k] = v
    # data["params"] = params
    # return data
# 

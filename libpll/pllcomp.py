""" PLL component model class implementations for discrete time simulation
    Cole Nielsen 2019
"""

import numpy as np
from libpll.tools import fixed_point, is_edge_phase

###############################################################################
# Oscillator
###############################################################################


class DCOPhase:
    """ PHASE DOMAIN digitally controlled oscillator with random-walk phase noise
        With two tuning ranges
    """
    def __init__(self, kdco1, kdco2, f0, dt, quantize=True, krw=0,
                init_phase=0.0, tau=None):
        """ args:
                kdco1 - gain per freq ctrl word LSB of input1
                kdco2 - gain per freq ctrl word LSB of input2
                f0 - frequency with 0 control word
                krw - random walk phase gain (for phase noise modeling)
                krorw - random walk gain if using ring oscillator model
                dt - time step per iteration (fixed)
        """
        self.kdco1 = kdco1
        self.kdco2 = kdco2
        self.f0 = f0
        self.krw = krw
        self.phase = init_phase
        self.dt = dt
        self.quantize = quantize
        if tau: self.wpt = 2*dt/tau
        else: self.wpt = np.inf

    def update(self, fctrl1, fctrl2):
        """ args:
                fctrl - frequency control word of DCO
            returns:
                oscillator output at updated time
        """
        if self.quantize: fctrl1 = round(fctrl1)
        if self.quantize: fctrl2 = round(fctrl2)
        self.phase += 2*np.pi*(self.f0 + fctrl1*self.kdco1 + fctrl2*self.kdco2)*self.dt + self.krw*np.random.choice([-1,1])
        return self.phase

    def reset(self, init_phase=0.0):
        self.phase = init_phase

    def clk_align(self):
        self.phase = np.ceil(self.phase/(2*np.pi))*(2*np.pi)


###############################################################################
# Clock Source
###############################################################################

class ClockPhase:
    """ PHASE DOMAIN ideal clock source
    """
    def __init__(self, f, dt, init_phase=0.0):
        """ args:
                f - clock frequency
                dt - simulation time step
        """
        self.f = f
        self.dt = dt
        self.init_phase = init_phase
        self.n = 0

    def update(self):
        """ returns:
                clock output at updated time
        """
        self.n += 1
        return 2*np.pi*self.f*self.n*self.dt + self.init_phase

    def reset(self, init_phase=0.0):
        self.n = 0
        self.init_phase = init_phase

###############################################################################
# Time to digital converter
###############################################################################

class TDCPhase:
    """ PHASE DOMAIN ideal time to digital converter, timed from rising edges
    """
    def __init__(self, tdc_steps, init_out=0.0, init_clk=0.0):
        self.tdc_steps = tdc_steps
        self.out = init_out
        self.clk_last = init_clk

    def update(self, clk, xin):
        """ Inputs in phase (radians)
        """
        if is_edge_phase(clk, self.clk_last):
            self.out = round(self.tdc_steps*(clk-xin)/(2*np.pi))
            self.out = (self.out+self.tdc_steps/2)%self.tdc_steps-self.tdc_steps/2
        self.clk_last = clk
        return self.out

    def reset(self,):
        self.out = 0
        self.clk_last = 0

class SCPDPhase:
    """ PHASE DOMAIN synchronous counter phase detector
    """
    def __init__(self, modulus, init_out=0.0, init_clk=0.0, ignore_clk=False):
        self.modulus = modulus
        self.out = init_out
        self.count = 0
        self.clk_last = init_clk
        self.ignore_clk = ignore_clk

    def update(self, clk, xin):
        """ Inputs in phase (radians)
        """
        if is_edge_phase(clk, self.clk_last) or self.ignore_clk:
            new_count = np.floor(xin/(2*np.pi))
            self.out = -new_count + self.count + self.modulus
            self.count = new_count
        self.clk_last = clk
        return self.out

    def reset(self,):
        self.out = 0
        self.clk_last = 0
        self.count = 0


class BBPD:
    """ PHASE DOMAIN ideal bang-bang phase detector
    """
    def __init__(self, rms_jit, fclk, init_clk=0, init_out=0, ignore_clk=False):
        self.cyc_jit = rms_jit*fclk
        self.clk_last = init_clk
        self.out = init_out
        self.ignore_clk = ignore_clk

    def update(self, clk, xin):
        """ Inputs in phase (radians)
        """
        if is_edge_phase(clk, self.clk_last): # or self.ignore_clk:
            xin /= 2*np.pi
            xin -= np.floor(xin)
            # error = (clk-xin)
            if xin > self.cyc_jit and xin <= 0.5:    self.out = -1.0
            elif xin < 1-self.cyc_jit and xin > 0.5: self.out = +1.0
            else:   self.out = np.random.choice((-1.0,1.0))
        self.clk_last = clk
        return self.out

    def reset(self,):
        self.clk_last = 0
        self.out = 0

###############################################################################
# Loop filter
###############################################################################

class LoopFilterIIRPhase:
    def __init__(self, a0, a1, b1, b2, a2=0.0, init_out=0, init_clk=0, int_bits=None,
                 frac_bits=None, verbose=True, ignore_clk=False, quant_filt=True,
                 out_bits=32, *args, **kwargs):
        """ Loop filter with tunable pole, zero
        """
        if int_bits and frac_bits: self.quantize = True
        else: self.quantize = False
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.verbose = verbose
        self.ignore_clk = ignore_clk
        self.max_int = 2**out_bits-1

        self.y1 = init_out
        self.y2 = init_out
        self.x1 = 0
        self.a0 = fixed_point(a0, int_bits, frac_bits) if self.quantize else a0
        self.a1 = fixed_point(a1, int_bits, frac_bits) if self.quantize else a1
        self.a2 = fixed_point(a2, int_bits, frac_bits) if self.quantize else a2
        self.b1 = fixed_point(b1, int_bits, frac_bits) if self.quantize else b1
        self.b2 = fixed_point(b2, int_bits, frac_bits) if self.quantize else b2
        self.clk_last = init_clk
        if self.quantize and verbose and quant_filt:
            print("\n* Finite word representations - int bits = %d, frac bits = %d"%(int_bits, frac_bits))
            print("\ta0 = %E, error = %E"%(self.a0, -a0+self.a0))
            print("\ta1 = %E, error = %E"%(self.a1, -a1+self.a1))
            if a2: print("\ta2 = %E, error = %E"%(self.a2, -a2+self.a2))
            print("\tb1 = %E, error = %E"%(self.b1, -b1+self.b1))
            print("\tb2 = %E, error = %E"%(self.b2, -b2+self.b2))
            print("\ta0+a1 = %E, error=%E"%(self.a0+self.a1, a0+a1-self.a0-self.a1))
            print("\t-b1-b2 = %E, error=%E"%(-self.b1-self.b2, -b1-b2+self.b1+self.b2))

    def change_filter(self, a0, a1, b1, b2, a2=0.0, *args, **kwargs):
        self.a0 = fixed_point(a0, self.int_bits, self.frac_bits) if self.quantize else a0
        self.a1 = fixed_point(a1, self.int_bits, self.frac_bits) if self.quantize else a1
        self.a2 = fixed_point(a2, self.int_bits, self.frac_bits) if self.quantize else a2
        self.b1 = fixed_point(b1, self.int_bits, self.frac_bits) if self.quantize else b1
        self.b2 = fixed_point(b2, self.int_bits, self.frac_bits) if self.quantize else b2
        if self.quantize and self.verbose:
            print("\n* Finite word representations - int bits = %d, frac bits = %d"%(self.int_bits, self.frac_bits))
            print("\ta0 = %E, error = %E"%(self.a0, -a0+self.a0))
            print("\ta1 = %E, error = %E"%(self.a1, -a1+self.a1))
            print("\tb1 = %E, error = %E"%(self.b1, -b1+self.b1))
            print("\tb2 = %E, error = %E"%(self.b2, -b2+self.b2))
            print("\ta0+a1 = %E, error=%E"%(self.a0+self.a1, a0+a1-self.a0-self.a1))
            print("\t-b1-b2 = %E, error=%E"%(-self.b1-self.b2, -b1-b2+self.b1+self.b2))

    def update(self, xin, clk):
        """ args:
                xin - filter input
                clk - clk of filter
            returns:
                filtered version of input
        """
        if is_edge_phase(clk, self.clk_last) or self.ignore_clk:
            ynew = self.a0*xin + self.a1*self.x1 - self.b1*self.y1 - self.b2*self.y2
            if self.quantize:
                ynew = fixed_point(ynew, self.int_bits, self.frac_bits)
            self.x1 = xin
            self.y2 = self.y1
            self.y1 = ynew
        self.clk_last = clk

        if self.y1 < 0: return 0
        elif self.y1 > self.max_int: return self.max_int
        else: return np.floor(self.y1)

    def reset(self,):
        pass


class LoopFilterPIPhase:
    """ Phase domain loop filter for PI-controller only operation.
    """
    def __init__(self, b0, b1, init_out=0, init_clk=0, int_bits=None,
                 frac_bits=None, verbose=True, ignore_clk=False, quant_filt=True,
                 out_bits=32, *args, **kwargs):
        """ b0 is assumed to be 0
        """
        if int_bits and frac_bits: self.quantize = True
        else: self.quantize = False
        self.quant_filt = quant_filt
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.verbose = verbose
        self.ignore_clk = ignore_clk
        self.max_int = 2**out_bits-1
        self.out_bits = out_bits

        self.y1 = init_out
        self.x1 = 0
        self.b0 = fixed_point(b0, int_bits, frac_bits) if self.quantize else b0
        self.b1 = fixed_point(b1, int_bits, frac_bits) if self.quantize else b1
        self.clk_last = init_clk
        if self.quantize and verbose and quant_filt:
            print("\n* Finite word representations - int bits = %d, frac bits = %d"%(int_bits, frac_bits))
            print("\tb0 = %E, error = %E"%(self.b0, -b0+self.b0))
            print("\tb1 = %E, error = %E"%(self.b1, -b1+self.b1))
            print("\tb0+b1 = %E, error=%E"%(self.b0+self.b1, b0+b1-self.b0-self.b1))

    def change_filter(self, b0, b1, x1init=0, *args, **kwargs):
        self.b0 = fixed_point(b0, self.int_bits, self.frac_bits) if self.quantize else b0
        self.b1 = fixed_point(b1, self.int_bits, self.frac_bits) if self.quantize else b1
        self.x1 = x1init
        print("!!!!!!!!!1 x1init=", x1init)
        if self.quantize and self.verbose and self.quant_filt:
            print("\n* Finite word representations - int bits = %d, frac bits = %d"%(self.int_bits, self.frac_bits))
            print("\tb0 = %E, error = %E"%(self.b0, -b0+self.b0))
            print("\tb1 = %E, error = %E"%(self.b1, -b1+self.b1))
            print("\tb0+b1 = %E, error=%E"%(self.b0+self.b1, b0+b1-self.b0-self.b1))

    def update(self, xin, clk):
        """ args:
                xin - filter input
                clk - clk of filter
            returns:
                filtered version of input
        """
        if is_edge_phase(clk, self.clk_last) or self.ignore_clk:
            ynew = self.b0*xin + self.b1*self.x1 + self.y1
            if self.quantize:
                ynew = fixed_point(ynew, self.out_bits, self.frac_bits)
            self.x1 = xin
            self.y1 = ynew
        self.clk_last = clk

        if self.y1 < 0: return 0
        elif self.y1 > self.max_int: return self.max_int
        else: return np.floor(self.y1)

    def reset(self,):
        pass





#!/usr/bin/env python

from loadh5 import *


## the antenna-delay functions ##
def lnlike(params, y, yerr, nChan=1024, oChan=0):
    '''
    params: a single parameter clock offset (d) in units of sampling cycle
    y, yerr: phase and error across the channels, in radian
    '''

    # y.shape = (nChan,)
    # params.shape = (nGrid,)   # num of gridded params to check
    nChan = y.shape[0]
    nGrid = params.shape[0]
    phimod = np.zeros((nGrid, nChan))
    for i in range(nGrid):
        d = params[i]
        phimod[i] = phimodel(d, nChan, oChan)
    

    #ymod = pwrap2(phimod - y.reshape((1,nChan)))
    ymod = phimod - y.reshape((1,nChan))
    #ymod -= np.median(ymod, axis=1, keepdims=True)
    tmpc = np.ma.exp(1.j*ymod)
    #tmpc = np.ma.exp(1.j*phimod) * np.ma.exp(-1.j*y.reshape(1,nChan))
    tmpx = np.ma.median(tmpc.real)
    tmpy = np.ma.median(tmpc.imag)
    ymod -= np.ma.angle((tmpx+1.j*tmpy))
    #tmpc -= (tmpx+1.j*tmpy)
    #ymod = np.ma.angle(tmpc)

    # log likelihood
    #like = -0.5*np.ma.sum(ymod**2/(yerr.reshape((1,nChan)))**2, axis=1)
    like = -0.5*np.ma.sum(pwrap2(ymod)**2/(yerr.reshape((1,nChan)))**2, axis=1)
    return like

def phimodel(d, nChan=1024, oChan=0):
    '''
    given a single delay (d) in clock cycles, return the phase as a function of channels
    central channel can be specified with oChan
    output shape: (nChan,)
    '''
    #y = np.linspace(-np.pi*d, np.pi*d, nChan, endpoint=False)
    s = np.pi*2./nChan*d
    y = s*(np.arange(nChan)-nChan/2+oChan)
    return y

def phimodel2(d, freq):
    '''
    given a 1D array of delays (d) and an array of frequencies 
    return the phase as a function of frequency
    output shape: (nDelay, nChan)

    note:
        if the unit of d is in clocks, then freq should be normalized channels
            with the appropriate start and end channels
            e.g. freq = np.arange(1024, 2048)/1024   for 400-800MHz
        if the unit of d is in seconds, then freq should be in Hz
            e.g. freq = np.linspace(400e6, 800e6, endpoint=False)
    '''
    if (not isinstance(d, np.ndarray)):
        d = np.array(d)
    if (not isinstance(freq, np.ndarray)):
        freq = np.array(freq)
    #s = (np.pi*2./nChan*d).reshape((-1,1))
    #y = (np.arange(nChan)-nChan/2+oChan).reshape((1,nChan))
    s = d.reshape((-1,1))
    y = (2.*np.pi*s)*freq.reshape((1,-1))
    return y

def phiCorr(d, freq):
    '''
    given a 1D array of delays (d), and an array of frequencies
    return an array of complex coefficients for phase correction
    shape: (nDelay, nChan)

    note:
        if the unit of d is in clocks, then freq should be in channels
            with the appropriate start and end channels
            e.g. freq = np.arange(1024, 2048)/1024   for 400-800MHz
        if the unit of d is in seconds, then freq should be in Hz
            e.g. freq = np.linspace(400e6, 800e6, endpoint=False)
    '''
    phi = phimodel2(d, freq)
    c_arr = np.exp(1.j*phi)
    return c_arr


def lnprob(params, ranages, freq, y, yerr, nChan=1024):
    return lnflat(params, ranges) + lnlike(params, freq, y, yerr, nChan)



## general  function ##
def lnflat(params, ranges):
    # flat prior
    # input:
    #   params (N)
    #   ranges (N, 2) --> the lower and upper limits for each param
    # return log probability
    prob = 0.
    for p, r in zip(params, ranges):
        if (p>r.max() or p<=r.min()):
            prob = -np.inf
    return prob


def pwrap2(phi, lim=np.pi):
    '''
    wrap input phase between +/- lim

    input:
        phi     array of phase, any shape
        lim     e.g. np.pi or 180 (deg)
    output:
        phi     same shape as the input
    '''
    phi2 = phi.copy()
    while(np.any(phi2 >= lim)):
        phi2[phi2 >= lim] -= 2.*lim
    while(np.any(phi2 < -lim)):
        phi2[phi2 < -lim] += 2.*lim

    return phi2


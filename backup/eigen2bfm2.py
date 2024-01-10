#!/usr/bin/env python

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from glob import glob
import time

from packet_func import *
from loadh5 import *

t0 = time.time()

nAnt  = 16
nChan = 1024
flim  = [400., 800.]


if (nAnt==4):
    aconf = 'FUSHAN6_test.config'
    fvis = 'result_short2.vish5'
    theta_deg = [-35, -25, -15, -5]
    theta_deg = np.array(theta_deg)
    theta_rad = theta_deg / 180. * np.pi
    sin_theta_m = np.sin(theta_rad)
    ant_flag = []
    asel = [0,3,6,9]

elif (nAnt==16):
    aconf = 'FUSHAN6_230418.config'
    #feig = 'fpga0.1031113710.bin.eigen.h5'
    #feig = 'fpga0.1106113710.bin.eigen.h5'
    #ant_flag = [3]
    #feig = 'fpga1.1031113710.bin.eigen.h5'
    feig = 'fpga2.1121114000.bin.eigen.h5'
    ant_flag = []
    sep = 1.    # meters
    lamb0 = 2.998e8/400e6   # longest wavelength
    #sin_theta_m = lamb0/sep/nAnt*(np.arange(nAnt)-nAnt//2)  # centered above
    sin_theta_m = lamb0/sep/nAnt*(np.arange(nAnt)-nAnt)    # offset toward East (HA<0)
    theta_deg = np.arcsin(sin_theta_m)/np.pi*180.
    asel = np.arange(nAnt)


# loading the phase correction
winSec = getData(feig, 'winSec')
# eigenvectors from the scaled mode
V2  = getData(feig, 'V2_scale') # shape: (nChan, nAnt, nMode)
# bandpass normalization for the scaled mode
N2  = getData(feig, 'N2_scale') # shape: (nAnt, nChan)
N2.mask[N2==0] = True
for ai in ant_flag:
    N2.mask[ai] = True
# take the leading eigenmode corresponding to the calibrator (at transit time)
refV2  = V2[:,:,-1]         # averaged window, last mode
refV2 /= np.abs(refV2)      # keep only the phase info
print('refV2.shape:', refV2.shape)


# define wavelengths
fMHz = np.linspace(flim[0], flim[1], nChan, endpoint=False)  # in MHz
lamb = 2.998e8 / (fMHz * 1e6)  # in meters
#print('wavelengths in m:', lamb)


# loading antenna positions
pos = np.loadtxt(aconf, usecols=(1,))
print('pos (m):', pos)


# define beamform matrix
## beamforming: raw, positive correction, negative correction
nBeams = len(theta_deg)
# raw beamform; 1st line below for swapped Re/Im; 2nd line below for older format
#BFM0 = np.exp(-2.j*np.pi*pos.reshape((1,nAnt,1))/lamb.reshape((1,1,nChan))*sin_theta_m.reshape((nBeams,1,1)))
BFM0 = np.exp(+2.j*np.pi*pos.reshape((1,nAnt,1))/lamb.reshape((1,1,nChan))*sin_theta_m.reshape((nBeams,1,1)))
# delay correction term
#TauCorr = np.exp(-2.j*np.pi*antTau.reshape((1,nAnt,1))*fMHz.reshape((1,1,nChan))*1e-3)

## new phase correction term
TauCorr = refV2.T.reshape((1,nAnt,nChan))
BFM1 = BFM0 * TauCorr
BFM2 = BFM0 * TauCorr.conjugate()
# BFM.shape = (nBeams, nAnt, nChan)

t1 = time.time()
print('BFMs done', ' elapsed:%.1fsec'%(t1-t0))


# bandpass equalization
print('debug: N2 min/max', N2.min(), N2.max(), np.ma.median(N2, axis=1))
N2 = np.ma.median(N2, axis=1, keepdims=True)
BEQ = 1./N2     # shape: (nAnt, nChan)
BEQ /= np.ma.abs(BEQ).max()
BEQ *= 4096
BEQ.fill_Value = 0.
BEQ = BEQ.filled()
print('debug: BEQ min/max', BEQ.min(), BEQ.max())
print('debug: norm BEQ min/max', BEQ.min()/4096*127, BEQ.max()/4096*127)

## override BEQ, no equalization
#BEQ = np.ones((nAnt, nChan), dtype=float)*127


allBFM = [BFM0, BFM1, BFM2]     # phase-only beamform matrix
allTYP = ['raw', 'pos', 'neg']
nType = len(allTYP)

for ii in range(nType):
    tp = allTYP[ii]
    yy = allBFM[ii]
    ofile = '%s.%s.bfm'%(feig, tp)
    print(ofile)

    BFM = np.zeros((nBeams, nAnt, nChan, 2), dtype=np.short)
    BFM[:,:,:,0] = BEQ.reshape((1, nAnt, -1)) * yy.real
    BFM[:,:,:,1] = BEQ.reshape((1, nAnt, -1)) * yy.imag
    np.save('BFM.%s.npy'%tp, BFM)

    matrix = BFM.transpose((2,1,0,3))  # output shape: (nChan, nAnt, nBeam, 2)
    matrix_size = nChan*nAnt*nBeams*2

    buf = struct.pack('<%dh'%matrix_size, *matrix.flatten())
    print('writing to:', ofile)
    fh = open(ofile, 'wb')
    fh.write(buf)
    fh.close()



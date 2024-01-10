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
    #feig = 'fpga2.1121114000.bin.eigen.h5'
    feig = 'fpga0.1111114010.bin.eigen.h5'
    ant_flag = [3]
    sep = 1.    # meters
    lamb0 = 2.998e8/400e6   # longest wavelength
    #sin_theta_m = lamb0/sep/nAnt*(np.arange(nAnt)-nAnt//2)  # centered above
    sin_theta_m = lamb0/sep/nAnt*(np.arange(nAnt)-nAnt)    # offset toward East (HA<0)
    theta_deg = np.arcsin(sin_theta_m)/np.pi*180.
    asel = [0, 2, 9, 15] # bl: 2, 6, 7, 9, 13, 15
    #asel = [0, 1, 4, 15] # bl: 1, 3, 4, 11, 14, 15


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
nBeam = len(theta_deg)
# raw beamform; 1st line below for swapped Re/Im; 2nd line below for older format
#BFM0 = np.exp(-2.j*np.pi*pos.reshape((1,nAnt,1))/lamb.reshape((1,1,nChan))*sin_theta_m.reshape((nBeam,1,1)))
BFM0 = np.exp(+2.j*np.pi*pos.reshape((1,nAnt,1))/lamb.reshape((1,1,nChan))*sin_theta_m.reshape((nBeam,1,1)))
# delay correction term
#TauCorr = np.exp(-2.j*np.pi*antTau.reshape((1,nAnt,1))*fMHz.reshape((1,1,nChan))*1e-3)

## new phase correction term
TauCorr = refV2.T.reshape((1,nAnt,nChan))
BFM1 = BFM0 * TauCorr
BFM2 = BFM0 * TauCorr.conjugate()
# BFM.shape = (nBeam, nAnt, nChan)


## identity matrix, for antenna voltages in 4bit with calibration and EQ
IDT = np.zeros((nBeam,nAnt,nChan), dtype=complex)
for ai in range(nAnt):
    IDT[ai,ai,:] = 1+0j
IDT *= TauCorr  # positive correction


## intensity for visibility
nsel = len(asel)
VIS = np.zeros((nBeam,nAnt,nChan), dtype=complex)
for ii in range(nsel):
    ai = asel[ii]
    VIS[ii,ai] = 1+0j

b = -1
for ii in range(nsel-1):
    ai = asel[ii]
    for jj in range(ii+1,nsel):
        aj = asel[jj]
        b += 1      # bl number
        r = 4+b*2   # row (beam) number
        VIS[r,ai] = 1+0j
        VIS[r,aj] = 1+0j
        VIS[r+1,ai] = 1+0j
        VIS[r+1,aj] = 0+1j


t1 = time.time()
print('BFMs done', ' elapsed:%.1fsec'%(t1-t0))


# bandpass equalization
print('debug: N2 min/max', N2.min(), N2.max(), np.ma.median(N2, axis=1))
#N2 = np.ma.median(N2, axis=1, keepdims=True)
BEQ = 1./N2     # shape: (nAnt, nChan)
#BEQ /= np.ma.abs(BEQ).max()
print('debug: BEQ.max', BEQ.max())
BEQ /= BEQ.max()
BEQ *= 8192 #4096
#BEQ.fill_Value = 0.
#BEQ = BEQ.filled()
BEQ[BEQ.mask] = 0.
BEQ[BEQ>4096] = 4096
print('debug: BEQ min/max', BEQ.min(), BEQ.max())
print('debug: norm BEQ min/max', BEQ.min()/4096*127, BEQ.max()/4096*127)
print('debug: norm BEQ median', np.median(BEQ)/4096*127)

## override BEQ, no equalization
#BEQ = np.ones((nAnt, nChan), dtype=float)*127


#allBFM = [BFM0, BFM1, BFM2]     # phase-only beamform matrix
#allTYP = ['raw', 'pos', 'neg']
allBFM = [BFM1, VIS, IDT]     # phase-only beamform matrix
allTYP = ['pos', 'vis', 'idt']
nType = len(allTYP)

for ii in range(nType):
    tp = allTYP[ii]
    yy = allBFM[ii]
    ofile = '%s.%s.npy'%(feig, tp)
    print(ofile)

    BFM = np.zeros((nBeam, nAnt, nChan, 2), dtype=np.short)
    BFM[:,:,:,0] = BEQ.reshape((1, nAnt, -1)) * yy.real
    BFM[:,:,:,1] = BEQ.reshape((1, nAnt, -1)) * yy.imag
    np.save(ofile, BFM)

    continue    # skip the bfm file
    matrix = BFM.transpose((2,1,0,3))  # output shape: (nChan, nAnt, nBeam, 2)
    matrix_size = nChan*nAnt*nBeam*2

    buf = struct.pack('<%dh'%matrix_size, *matrix.flatten())
    ofile = '%s.%s.bfm'%(feig, tp)
    print('writing to:', ofile)
    fh = open(ofile, 'wb')
    fh.write(buf)
    fh.close()



#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

BFM = np.load('fpga0.1111114010.bin.eigen.h5.pos.npy')
cBFM = BFM[:,:,:,0] + 1j*BFM[:,:,:,1]

nAnt = 16
nChan = 1024

y = np.arange(nAnt)
ch = np.arange(nChan)
CC, YY = np.meshgrid(ch, y)

for pt in range(3):
    if (pt==0):     # amp
        ptstr = 'amp'
        z = np.abs(cBFM)
    elif (pt==1):   # pha
        ptstr = 'pha'
        z = np.angle(cBFM)
    elif (pt==2):   # profile
        ptstr = 'prf'

    fig, s2d = plt.subplots(4,4,figsize=(12,9), sharex=True, sharey=True)
    sub = s2d.flatten()

    if (pt<=1):
        for bm in range(nAnt):
            ax = sub[bm]
            ax.pcolormesh(CC,YY,z[bm],shading='auto')
    elif (pt==2):
        for bm in range(nAnt):
            ax = sub[bm]
            for ai in range(nAnt):
                ax.plot(ch, np.abs(cBFM[bm,ai,:]))

    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig('test_%s.png'%ptstr)
    plt.close(fig)



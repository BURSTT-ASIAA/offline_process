#!/usr/bin/env python

import sys, os.path
from loadh5 import *
import matplotlib.pyplot as plt
from subprocess import call
from astropy.time import Time


nAnt    = 16        # antenna number
nChan   = 1024      # channel number
flim    = [400., 800.]  # freq limit in MHz
emode   = 1         # M-th strongest eigenmode to plot
Aref    = 8         # a reference antenna for phase plot
xtype   = 1         # 1: UT, 2: win_ID
vmode   = 3         # 2: scale, 3: coeff (in makeCov)


inp = sys.argv[0:]
pg  = inp.pop(0)

#fin = 'win_eigenmodes.h5'

usage = '''
plot the results from 'check_eigenmodes.py'
syntax:
    %s <h5_file> [options]

options are:
    -a Aref         # reference antenna for phase plot
                    # default: %d
    -m MODE         # plot the M-th strongest eigenmode
                    # default: %d
    --xt type       # x-axis type:: 1: UT; 2: win_ID


''' % (pg, Aref, emode)

if (len(inp)<1):
    sys.exit(usage)

while (inp):
    k = inp.pop(0)
    if (k == '-m'):
        emode = int(inp.pop(0))
    elif (k == '-a'):
        Aref = int(inp.pop(0))
    elif (k == '--xt'):
        xtype = int(inp.pop(0))
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        fin = k

if (xtype==1):
    xname = 'UT'
elif (xtype==2):
    xname = 'win'

odir = fin.replace('.h5', '.mode%d.Aref%d.%s.plots'%(emode,Aref,xname))
if (not os.path.isdir(odir)):
    call('mkdir -p %s'%odir, shell=True)
print('output saved in:', odir)


attrs = getAttrs(fin)
epoch0 = attrs['unix_utc_open']
tsec = getData(fin, 'winSec')

if (vmode == 2):
    savW2 = getData(fin, 'win_W_scale')
    savV2 = getData(fin, 'win_V_scale')
    nWin, nChan, nAnt, nMode = savV2.shape
if (vmode == 3):
    savW3 = getData(fin, 'win_W_coeff')
    savV3 = getData(fin, 'win_V_coeff')
    nWin, nChan, nAnt, nMode = savV3.shape

ut = Time(tsec+epoch0, format='unix').to_datetime()

freq = getData(fin, 'freq')
if (freq is None):
    freq = np.linspace(flim[0], flim[1], nChan, endpoint=False)

chan = np.arange(nChan)
if (xtype==1):
    tx   = ut
    xlab = 'time (UT)'
elif (xtype==2):
    tx   = np.arange(nWin)
    xlab = 'win_ID'
X, Y = np.meshgrid(tx, freq, indexing='xy')


if (vmode == 2):
    V2last  = savV2[:,:,:,-emode]
    V2ref  = V2last[:,:,Aref]          # arbitrarily choose ant8 as phase ref
    V2norm = V2ref / np.abs(V2ref)  # keep only the phase info
    V2rel  = V2last / V2norm.reshape((nWin,nChan,1))  # phase relative to Vref
if (vmode == 3):
    V3last  = savV3[:,:,:,-emode]
    V3ref  = V3last[:,:,Aref]          # arbitrarily choose ant8 as phase ref
    V3norm = V3ref / np.abs(V3ref)  # keep only the phase info
    V3rel  = V3last / V3norm.reshape((nWin,nChan,1))  # phase relative to Vref

## phase plots
for pt in range(4):
    if (pt == 0):
        if (vmode==2):
            continue
        z = np.angle(V3last)
        png = '%s/eigenvec.coeff.V.phase.png'%odir
        sptitle = 'file: %s, coeff eigenvector, phase'%fin
    elif (pt == 1):
        if (vmode==3):
            continue
        z = np.angle(V2last)
        png = '%s/eigenvec.scale.V.phase.png'%odir
        sptitle = 'file: %s, scale eigenvector, phase'%fin
    elif (pt == 2):
        if (vmode==2):
            continue
        z = np.angle(V3rel)
        png = '%s/eigenvec.coeff.Vrel.phase.png'%odir
        sptitle = 'file: %s, coeff eigenvector, phase, ref=Ant%d'%(fin,Aref)
    elif (pt == 3):
        if (vmode==3):
            continue
        z = np.angle(V2rel)
        png = '%s/eigenvec.scale.Vrel.phase.png'%odir
        sptitle = 'file: %s, scale eigenvector, phase, ref=Ant%d'%(fin,Aref)

    print('plotting:', png)

    fig, s2d = plt.subplots(4,4,figsize=(16,12), sharex=True, sharey=True)
    sub = s2d.flatten()

    #for ai in range(nAnt):
    ai = -1
    for ii in range(4):
        for jj in range(4):
            ai += 1
            ax = sub[ai]
            ax.pcolormesh(X,Y,z[:,:,ai].T, vmin=-3.2, vmax=3.2, shading='nearest')
            ax.set_title('ant%02d'%ai)
            if (ii==3):
                #ax.set_xlabel('window_id')
                ax.set_xlabel(xlab)
            if (jj==0):
                ax.set_ylabel('freq (MNhz)')
        
    sptitle += ' zlim:[%.1f, %.1f]'%(-3.2, 3.2)

    if (xtype==1):
        fig.autofmt_xdate()
    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.subplots_adjust(wspace=0., hspace=0.1)
    fig.suptitle(sptitle)
    fig.savefig(png)
    plt.close(fig)



## ampld plots
for pt in range(2):
    if (pt == 0):
        if (vmode==2):
            continue
        z = np.abs(V3last) * np.sqrt(savW3[:,:,-1]).reshape((nWin,nChan,1))
        png = '%s/eigenvec.coeff.V.ampld.png'%odir
        sptitle = 'file: %s, coeff eigenvector, ampld'%fin
        zmin = 0
        zmax = 0.6
    elif (pt == 1):
        if (vmode==3):
            continue
        z = np.abs(V2last) * np.sqrt(savW2[:,:,-1]).reshape((nWin,nChan,1))
        png = '%s/eigenvec.scale.V.ampld.png'%odir
        sptitle = 'file: %s, scale eigenvector, ampld'%fin
        zmin = z.min()
        zmax = z.max()/10

    print('plotting:', png)
    print('zmin, zmax:', zmin, zmax)
    fig, s2d = plt.subplots(4,4,figsize=(16,12), sharex=True, sharey=True)
    sub = s2d.flatten()

    #for ai in range(nAnt):
    ai = -1
    for ii in range(4):
        for jj in range(4):
            ai += 1
            ax = sub[ai]
            ax.pcolormesh(X,Y,z[:,:,ai].T, vmin=zmin, vmax=zmax, shading='nearest')
            ax.set_title('ant%02d'%ai)
            if (ii==3):
                #ax.set_xlabel('window_id')
                ax.set_xlabel(xlab)
            if (jj==0):
                ax.set_ylabel('freq (MNhz)')
        
    sptitle += ' zlim:[%.1f, %.1f]'%(zmin, zmax)

    if (xtype==1):
        fig.autofmt_xdate()
    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.subplots_adjust(wspace=0., hspace=0.1)
    fig.suptitle(sptitle)
    fig.savefig(png)
    plt.close(fig)

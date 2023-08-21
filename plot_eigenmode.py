#!/usr/bin/env python

import sys, os.path
from loadh5 import *
import matplotlib.pyplot as plt
from subprocess import call
from astropy.time import Time

inp = sys.argv[0:]
pg  = inp.pop(0)

#fin = 'win_eigenmodes.h5'

usage = '''
plot the results from 'check_eigenmodes.py'
syntax:
    %s <h5_file>

''' % (pg,)

if (len(inp)<1):
    sys.exit(usage)

while (inp):
    k = inp.pop(0)

    fin = k


odir = fin.replace('.h5', '.plots')
if (not os.path.isdir(odir)):
    call('mkdir -p %s'%odir, shell=True)
print('output saved in:', odir)


attrs = getAttrs(fin)
epoch0 = attrs['unix_utc_open']
tsec = getData(fin, 'winSec')

savW2 = getData(fin, 'win_W_scale')
savV2 = getData(fin, 'win_V_scale')
savW3 = getData(fin, 'win_W_coeff')
savV3 = getData(fin, 'win_V_coeff')
nWin, nChan, nAnt, nMode = savV3.shape

ut = Time(tsec+epoch0, format='unix').to_datetime()


chan = np.arange(nChan)
#tx   = np.arange(nWin)
tx   = ut
X, Y = np.meshgrid(tx, chan, indexing='xy')

V2last  = savV2[:,:,:,-1]
V3last  = savV3[:,:,:,-1]
#Vhlast = savV[:,:,-1,:].conjugate()

V2ref  = V2last[:,:,8]          # arbitrarily choose ant8 as phase ref
V2norm = V2ref / np.abs(V2ref)  # keep only the phase info
V2rel  = V2last / V2norm.reshape((nWin,nChan,1))  # phase relative to Vref
V3ref  = V3last[:,:,8]          # arbitrarily choose ant8 as phase ref
V3norm = V3ref / np.abs(V3ref)  # keep only the phase info
V3rel  = V3last / V3norm.reshape((nWin,nChan,1))  # phase relative to Vref

## phase plots
for pt in range(4):
    if (pt == 0):
        z = np.angle(V3last)
        png = '%s/eigenvec.coeff.V.phase.png'%odir
        sptitle = 'file: %s, coeff eigenvector, phase'%fin
    elif (pt == 1):
        z = np.angle(V2last)
        png = '%s/eigenvec.scale.V.phase.png'%odir
        sptitle = 'file: %s, scale eigenvector, phase'%fin
    elif (pt == 2):
        z = np.angle(V3rel)
        png = '%s/eigenvec.coeff.Vrel.phase.png'%odir
        sptitle = 'file: %s, coeff eigenvector, phase, ref=Ant8'%fin
    elif (pt == 3):
        z = np.angle(V2rel)
        png = '%s/eigenvec.scale.Vrel.phase.png'%odir
        sptitle = 'file: %s, scale eigenvector, phase, ref=Ant8'%fin

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
                ax.set_xlabel('time (UT)')
            if (jj==0):
                ax.set_ylabel('freq (MNhz)')
        
    sptitle += ' zlim:[%.1f, %.1f]'%(-3.2, 3.2)

    fig.autofmt_xdate()
    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.subplots_adjust(wspace=0., hspace=0.1)
    fig.suptitle(sptitle)
    fig.savefig(png)
    plt.close(fig)



## ampld plots
for pt in range(2):
    if (pt == 0):
        z = np.abs(V3last) * np.sqrt(savW3[:,:,-1]).reshape((nWin,nChan,1))
        png = '%s/eigenvec.coeff.V.ampld.png'%odir
        sptitle = 'file: %s, coeff eigenvector, ampld'%fin
        zmin = 0
        zmax = 0.6
    elif (pt == 1):
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
                ax.set_xlabel('time (UT)')
            if (jj==0):
                ax.set_ylabel('freq (MNhz)')
        
    sptitle += ' zlim:[%.1f, %.1f]'%(zmin, zmax)

    fig.autofmt_xdate()
    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.subplots_adjust(wspace=0., hspace=0.1)
    fig.suptitle(sptitle)
    fig.savefig(png)
    plt.close(fig)

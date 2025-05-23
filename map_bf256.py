#!/usr/bin/env python

from packet_func import *
from loadh5 import *
import sys, os.path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation as animation
from glob import glob
from subprocess import call
from datetime import datetime
from astropy.time import Time



inp = sys.argv[0:]
pg  = inp.pop(0)


nRow    = 16
nAnt    = 16
nBeam   = nRow*nAnt
nChan0  = 1024
nChan   = 128       # for 256-ant
flim    = [400., 800.]
freq0   = np.linspace(flim[0], flim[1], nChan0, endpoint=False)
sep     = [1.0, 0.5]    # amtemma spacing in X and Y (meters)
beam0   = [-7.5, -7.5]  # beam0 in X and Y
chlim   = [0, nChan0]
use_id  = False

blocklen = 51200    # number of frames per block
nSum    = 400       # integration number

verbose = 0

odir2   = 'intensity.plots'
read_raw = False
zlim    = None
pts     = [1]
do_trans = False

tDelay  = 100       # animation, frame delay in ms


usage   = '''
plot amplitude of the spectrum as a function of frequency and time

syntax:
    %s -r <ring_id> <dir> [options]

options are:
    --sum SUM       # the integration for the intensity data
                    # (default: %d)
    --frame FRAME   # number of frames per block
                    # (default: %d)
    -o <odir>       # specify an output dir
    --redo          # force reading raw data
    --zlim zmin zmax# set the min/max color scale
    -v              # verbose
    --raw           # plot the raw intensity
                    # (default is to plot the normalized intensity)
    --both          # plot both raw and normalized intensities
    --trans         # transpose intensity from baseband
    --beam0 X Y     # specify the beam0 in X and Y direction
                    # (default: -7.5, -7.5)
    --sep X Y       # specify the antenna spacing in X and Y (meters)
                    # (default: 1.0, 0.5)
    --chlim lo hi   # channel range used for spectral avg
                    # (default: 0, 1024)
    --id            # show xlabel and ylabel in beam_id

''' % (pg, nSum, blocklen)


if (len(inp)<1):
    sys.exit(usage)

dirs  = []
rings = []
while (inp):
    k = inp.pop(0)
    if (k=='-o'):
        odir2 = inp.pop(0)
    elif (k=='--redo'):
        read_raw = True
    elif (k=='--zlim'):
        zmin = float(inp.pop(0))
        zmax = float(inp.pop(0))
        zlim = [zmin, zmax]
    elif (k=='-r'):
        ring_id = int(inp.pop(0))
        idir = inp.pop(0)
        dirs.append(idir)
        rings.append(ring_id)
    elif (k == '--id'):
        use_id = True
    elif (k == '--sep'):
        sep[0] = float(inp.pop(0))
        sep[1] = float(inp.pop(0))
    elif (k == '--beam0'):
        beam0[0] = float(inp.pop(0))
        beam0[1] = float(inp.pop(0))
    elif (k == '--chlim'):
        chlim[0] = int(inp.pop(0))
        chlim[1] = int(inp.pop(0))
    elif (k=='-v'):
        verbose=1
    elif (k=='--raw'):
        pts = [0]
    elif (k=='--both'):
        pts = [0, 1]
    elif (k == '--trans'):
        do_trans = True
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        sys.exit('extra argument: %s'%k)


nTime   = blocklen//nSum
nElem   = nTime*nChan*nBeam
nByte   = nElem * 2
print(nTime, nElem, nByte)

nDir = len(dirs)


arrInts  = []
arrNInts = []
freqs    = []
tsecs    = []
for j in range(nDir):
    idir = dirs[j]
    ring_id = rings[j]
    ring_name = 'ring%d'%(ring_id%2)

    i0 = nChan*ring_id
    freq = freq0[i0:i0+nChan]
    freqs.append(freq)

    if (ring_name in idir):
        ofile = idir + '.inth5'
    else:
        ofile = '%s.%s.inth5'%(idir, ring_name)
    print('output to:', ofile)
    odir = '%s.plots'%ofile

    adoneh5(ofile, freq, 'freq')

    files   = glob('%s/intensity_%s_*'%(idir,ring_name))
    files.sort()
    nFile   = len(files)
    if (False):
        nFile = 10
        files = files[:nFile]
    print('nFile:', nFile)

    attrs   = {}
    attrs['idir'] = idir
    attrs['ring_id'] = ring_id
    attrs['nChan'] = nChan
    attrs['nRow'] = nRow

    # intensity array
    if (os.path.isfile(ofile)):
        tmp = getData(ofile, 'intensity')
        if (tmp is None):
            read_raw = True
        else:
            if (not read_raw):
                arrInt = tmp
                arrNInt = getData(ofile, 'norm.intensity')
                winSec = getData(ofile, 'winSec')
                attrs = getAttrs(ofile)
                epoch0 = attrs.get('unix_utc_open')
                if (do_trans):
                    arrInt = arrInt.transpose((0,3,1,2))
                    arrNInt = arrNInt.transpose((0,3,1,2))
    else:
        read_raw = True

    if (read_raw):
        #arrInt  = np.ma.array(np.zeros((nFile, nRow, nAnt, nChan)), mask=False)  # unmasked by default
        arrInt  = np.ma.array(np.zeros((nFile, nChan, nRow, nAnt)), mask=False)  # unmasked by default
        fepoch  = filesEpoch(files, hdver=2, meta=0)
        epoch0  = fepoch[0] # UTC
        attrs['unix_utc_open'] = epoch0
        winSec  = fepoch - epoch0
        for i in range(nFile):
            fbin = files[i]
            print('reading: %s (%d/%d)'%(fbin, i+1,nFile))

            fh = open(fbin, 'rb')
            # read 1 block, sum128
            fh.seek(64) # skip header
            buf = fh.read(nByte)
            data = np.frombuffer(buf, dtype=np.float16).reshape((nTime,nChan,nRow,nAnt))
            arrInt[i] = data.mean(axis=0)
            #data = np.frombuffer(buf, dtype=np.uint16).reshape((nTime,nChan,nRow,nAnt))
            #data = np.frombuffer(buf, dtype=np.float16).reshape((nTime,nChan,nAnt,nRow))
            #arrInt[i] = data.mean(axis=0).transpose((1,2,0))
            #arrInt[i] = data.mean(axis=0).transpose((2,1,0))
            fh.close()

        adoneh5(ofile, arrInt, 'intensity')
        adoneh5(ofile, winSec, 'winSec')

        putAttrs(ofile, attrs)

        ## bandpass normalization
        arrNInt = arrInt / np.median(arrInt,axis=0)
        adoneh5(ofile, arrNInt, 'norm.intensity')

    arrInts.append(arrInt)
    arrNInts.append(arrNInt)
    print(j, arrInt.shape, arrNInt.shape)
    tsecs.append(winSec)

#arrInt = np.concatenate(arrInts, axis=3)
#freq = np.concatenate(freqs, axis=0)
#print('combine', arrInt.shape, arrNInt.shape)

print('epoch0:', epoch0)
loc0 = epoch0 + 3600*8  # convert back to local time
#winDT = Time(loc0+winSec, format='unix').to_datetime()
#matDT = mdates.date2num(winDT)
#X, Y = np.meshgrid(winSec, freq, indexing='xy')
#X, Y = np.meshgrid(winDT, freq, indexing='xy')
#X = winDT
#Y = freq
#print('X:', X)

if (nDir==1):
    odir2 = odir

if (not os.path.isdir(odir2)):
    call('mkdir %s'%odir2, shell=True)


## warning if number tsecs not the same
nWins = []
for i in range(nDir):
    nWins.append(len(tsecs[i]))
if (len(np.unique(nWins))>1):
    print('warning: number of windows not equal')
    print(nWins)
else:
    nWin = np.unique(nWins)[0]

arrNInt = np.concatenate(arrNInts, axis=1)  # combine along freq
if (chlim[1] > arrNInt.shape[1]):
    chlim[1] = arrNInt.shape[1]
mapNInt = arrNInt[:,chlim[0]:chlim[1]].mean(axis=1)
if (zlim is None):
    zlim = [mapNInt.min(), mapNInt.max()]

## movie of freq-integrated maps
gif = '%s/map256_animate.gif' % (odir2,)
#fig, ax = plt.subplots(1,1,figsize=(10,7.5))
fig, ax = plt.subplots(1,1,figsize=(6,9))
ax.set_aspect('equal')

if (use_id):
    ax.set_xlabel('EW beams')
    ax.set_ylabel('NS beams')
    X = np.arange(16)
    Y = np.arange(16)
else:
    ax.set_xlabel('EW angle (deg)')
    ax.set_ylabel('NS angle (deg)')

    lamb0 = 2.998e8/400e6 # meter
    sin_theta_X = lamb0/sep[0]/nAnt*(np.arange(nAnt)+beam0[0])
    sin_theta_Y = lamb0/sep[1]/nRow*(np.arange(nRow)+beam0[1])
    X = np.arcsin(sin_theta_X)/np.pi*180.
    Y = np.arcsin(sin_theta_Y)/np.pi*180.

var = mapNInt[0]    # (nRow, nAnt)
winSec = tsecs[0][0]
winDT  = Time(loc0+winSec, format='unix').to_datetime()
title = 'win: %04d, time: %s'%(0, winDT.strftime('%y%m%d_%H%M%S'))
ax.set_title(title)
s = ax.pcolormesh(X,Y,var,vmin=zlim[0],vmax=zlim[1])
ax.set_xlim(-np.array(ax.get_xlim()))
#s = ax.pcolormesh(np.flip(X),Y,np.flip(var,axis=1),vmin=zlim[0],vmax=zlim[1])
cb = plt.colorbar(s,ax=ax)
fig.tight_layout()

def update(i):
    print('window:',i)

    var = mapNInt[i]    # (nRow, nAnt)
    winSec = tsecs[0][i]
    winDT  = Time(loc0+winSec, format='unix').to_datetime()
    title = 'win: %04d, time: %s'%(i, winDT.strftime('%y%m%d_%H%M%S'))
    ax.set_title(title)

    s = ax.pcolormesh(X,Y,var,vmin=zlim[0],vmax=zlim[1])
    #s = ax.pcolormesh(np.flip(X),Y,np.flip(var,axis=1),vmin=zlim[0],vmax=zlim[1])
    #cb = plt.colorbar(s,ax=ax)


ani = animation.FuncAnimation(fig=fig, func=update, frames=nWin, interval=tDelay)
ani.save(gif, writer='pillow')
plt.close(fig)

print('X angles (deg):', X)
print('Y angles (deg):', Y)
print('flipped')

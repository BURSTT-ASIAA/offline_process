#!/usr/bin/env python

from packet_func import *
from loadh5 import *
import sys, os.path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from glob import glob
from subprocess import call
from datetime import datetime
from astropy.time import Time

import matplotlib.dates as mdates


inp = sys.argv[0:]
pg  = inp.pop(0)


nRow    = 1
nAnt    = 16
nBeam   = nRow*nAnt
nChan0  = 1024
nChan   = 1024       # for 16-ant
flim    = [400., 800.]
freq0    = np.linspace(flim[0], flim[1], nChan0, endpoint=False)

blocklen = 51200    # number of frames per block
nSum    = 400       # integration number

verbose = 0

odir2   = 'intensity.plots'
read_raw = False
zlim    = None
pts     = [1]


usage   = '''
plot amplitude of the spectrum as a function of frequency and time

syntax:
    %s -f <fpga_id> <dir> [options]

    example: the data are named 'intensity_fpga2_sum400_yymmdd_HHMMSS'
    then <fpga_id> = 2, i.e. -f 2

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
    elif (k=='-f'):
        ring_id = int(inp.pop(0))
        idir = inp.pop(0)
        dirs.append(idir)
        rings.append(ring_id)
    elif (k=='-v'):
        verbose=1
    elif (k=='--raw'):
        pts = [0]
    elif (k=='--both'):
        pts = [0, 1]
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
    #ring_name = 'ring%d'%ring_id
    ring_name = 'fpga%d'%ring_id

    #i0 = nChan*ring_id
    i0 = 0
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
#arrNInt = np.concatenate(arrNInts, axis=3)
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

## 64 beams in one plot
png = '%s/norm.16beams.png' % (odir2)
#fig, sub = plt.subplots(4,16,figsize=(32,8), sharex=True, sharey=True)
#fig, tmp = plt.subplots(8,16,figsize=(32,12), sharex=True, height_ratios=[2,1,2,1,2,1,2,1])
#fig, tmp = plt.subplots(2,16,figsize=(32,3), sharex=True, height_ratios=[2,1])
fig, tmp = plt.subplots(4,8,figsize=(16,6), sharex=True, height_ratios=[2,1,2,1])
sub  = tmp[0::2].flatten()
sub2 = tmp[1::2].flatten()


for kk in range(nDir):
    arrNInt = arrNInts[kk]
    freq = freqs[kk]
    winSec = tsecs[kk]
    winDT = Time(loc0+winSec, format='unix').to_datetime()
    X = winDT
    Y = freq

    if (zlim is None):
        vmin = arrNInt.min()
        vmax = arrNInt.max()
    else:
        vmin = zlim[0]
        vmax = zlim[1]
    print('zmin,zmax:', vmin, vmax)

    b = -1
    for j in range(nRow):
        for ai in range(nAnt):
            b += 1
            ax = sub[b] #sub[nRow-1-j, ai]
            ax.pcolormesh(X,Y,arrNInt[:,:,j,ai].T, vmin=vmin, vmax=vmax, shading='auto')

            prof = np.ma.median(arrNInt[:,:,j,ai], axis=1) # avg in freq, each node separately
            ax2 = sub2[b] #sub2[nRow-1-j, ai]
            ax2.plot(winDT, prof)
            ax2.set_ylim(vmin, vmax)

fig.autofmt_xdate()
fig.tight_layout(rect=[0,0.03,1,0.95])
fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle(odir2)
fig.savefig(png)
plt.close(fig)


## another figure for spec.avg intensity vs time
fig,ax = plt.subplots(1,1,figsize=(12,4))
for j in range(nRow):
    for ai in range(nAnt):
        prof = np.ma.median(arrNInt[:,:,j,ai], axis=1) # avg in freq, each node separately
        ax.plot(winDT, prof, label='%d'%ai)
ax.set_xlabel('time')
ax.set_ylabel('norm.intensity')
ax.legend()
ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=np.arange(0,60,5)))
ax.grid(which='both')
ax.tick_params(which='major', length=8)

fig.autofmt_xdate()
fig.tight_layout(rect=[0,0.03,1,0.95])
fig.savefig('%s/prof_vs_time.png'%odir2)
plt.close(fig)

sys.exit('finished')


## plot each row separately
for j in range(nRow):
    fig, s2d = plt.subplots(8,4,figsize=(16,12), sharex=True, height_ratios=[2,1,2,1,2,1,2,1])
    sub = s2d[0::2,:].flatten()
    sub2 = s2d[1::2,:].flatten()
    for ii in range(4):
        s2d[ii*2,0].set_ylabel('freq (MHz)')
        s2d[ii*2+1,0].set_ylabel('amp')
        s2d[7,ii].set_xlabel('time (sec)')

    for kk in range(nDir):
        arrInt = arrInts[kk]
        arrNInt = arrNInts[kk]
        freq = freqs[kk]
        winSec = tsecs[kk]
        winDT = Time(loc0+winSec, format='unix').to_datetime()
        X = winDT
        Y = freq

        ## plotting
        #for pt in range(1,2):
        for pt in pts:
            if (pt==0):
                png = '%s/raw.row%d.png' % (odir2,j)
                arr = arrInt[:,:,j]
                #print(arrInt.shape, arr.shape)
            elif (pt==1):
                png = '%s/norm.row%d.png' % (odir2,j)
                arr = arrNInt[:,:,j]
            print('plotting:', png, '...')

            if (zlim is None):
                vmin = arr.min()
                vmax = arr.max()
            else:
                vmin = zlim[0]
                vmax = zlim[1]
            print('zmin,zmax:', vmin, vmax)


            for ai in range(nAnt):
                ax = sub[ai]
                #print(arr[:,:,ai].shape)
                #ax.imshow(arrInt[:,ai].T, origin='lower', aspect='auto')
                ax.pcolormesh(X,Y,arr[:,:,ai].T, vmin=vmin, vmax=vmax, shading='auto')

                ax2 = sub2[ai]
                y2d = arr[:,:,ai].T
                y1d = y2d.mean(axis=0)
                ax2.plot(winDT, y1d, marker='.')
                ax2.set_ylim(vmin, vmax)
                ax2.grid()


    fig.autofmt_xdate()
    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(png)
    plt.close(fig)




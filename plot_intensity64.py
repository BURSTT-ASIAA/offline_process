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


#idir    = 'data_230822'

inp = sys.argv[0:]
pg  = inp.pop(0)


nAnt    = 16
nChan0  = 1024
nRow    = 4         # for 64-ant
nChan   = 512       # for 64-ant
nBeam   = nAnt
flim    = [400., 800.]
freq0    = np.linspace(flim[0], flim[1], nChan0, endpoint=False)

p0      = 0
nPack   = 1000
nBlock  = 1
#blocklen= 800000
blocklen= 128000
bitwidth= 4
hdver   = 2
order_off= -2
meta    = 64
verbose = 0

odir2   = 'intensity.plots'
read_raw = False
zlim    = None
no_bitmap = False


usage   = '''
plot amplitude of the spectrum as a function of frequency and time

syntax:
    %s -r <ring_id> <dir> [options]

options are:
    -n nPack        # number of packets to load (%d)
    --p0 p0         # starting packet (%d)
    --16bit         # is 16bit data (default is 4bit data)
    --hd VER        # header version: 1 or 2 (%d)
    --meta bytes    # ring buffer or file metadata length in bytes (%d)
    --nBlock nBlock # number of blocks in the file (%d)
    --blocklen blocklen
                    # the number of packets in a block (%d)
    -o odir2        # the output dir for combined plots
    --redo          # force reading raw data
    --zlim zmin zmax# set the min/max color scale
    --no-bitmap     # ignore bitmap from the data
    -v              # verbose

''' % (pg, nPack, p0, hdver, meta, nBlock, blocklen)


if (len(inp)<1):
    sys.exit(usage)

dirs  = []
rings = []
while (inp):
    k = inp.pop(0)
    if (k=='-n'):
        nPack = int(inp.pop(0))
    elif (k=='--p0'):
        p0 = int(inp.pop(0))
    elif (k=='--16bit'):
        bitwidth = 16
    elif (k=='--hd'):
        hdver = int(inp.pop(0))
    elif (k == '--meta'):
        meta = int(inp.pop(0))
    elif (k=='--nBlock'):
        nBlock = int(inp.pop(0))
    elif (k=='--blocklen'):
        blocklen = int(inp.pop(0))
    elif (k=='-o'):
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
    elif (k=='--no-bitmap'):
        no_bitmap = True
    elif (k=='-v'):
        verbose=1
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        sys.exit('extra argument: %s'%k)



nDir = len(dirs)


arrInts  = []
arrNInts = []
freqs    = []
for j in range(nDir):
    idir = dirs[j]
    ring_id = rings[j]
    ring_name = 'ring%d'%ring_id

    i0 = nChan*ring_id
    freq = freq0[i0:i0+nChan]
    order_off = -2-ring_id      # for 64-ant
    freqs.append(freq)

    if (ring_name in idir):
        ofile = idir + '.inth5'
    else:
        ofile = '%s.%s.inth5'%(idir, ring_name)
    print('output to:', ofile)
    odir = '%s.plots'%ofile

    adoneh5(ofile, freq, 'freq')

    files   = glob('%s/%s.*.bin'%(idir,ring_name))
    files.sort()
    nFile   = len(files)
    if (False):
        nFile = 10
        files = files[:nFile]
    print('nFile:', nFile)

    attrs   = {}
    attrs['idir'] = idir
    attrs['p0'] = p0
    attrs['nPack'] = nPack
    attrs['nBlock'] = nBlock
    attrs['blocklen'] = blocklen
    attrs['bitwidth'] = bitwidth
    attrs['hdver'] = hdver
    attrs['order_off'] = order_off
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
        arrInt  = np.ma.array(np.zeros((nFile, nRow, nAnt, nChan)), mask=True)  # masked by default
        winSec  = np.zeros(nFile)
        dt0 = None
        for i in range(nFile):
            fbin = files[i]
            print('reading: %s (%d/%d)'%(fbin, i+1,nFile))
            ftime = os.path.basename(fbin).split('.')[1]
            if (len(ftime)==10):
                fstr = '23'+ftime
            elif (len(ftime)==14):
                fstr = ftime[2:]
            dt = datetime.strptime(fstr, '%y%m%d%H%M%S')
            if (dt0 is None):
                dt0 = dt
                epoch0 = Time(dt0, format='datetime').to_value('unix')
                epoch0 -= 3600*8    # convert to UTC
                attrs['unix_utc_open'] = epoch0
            winSec[i] = (dt-dt0).total_seconds()

            fh = open(fbin, 'rb')
            #tick, spec = loadSpec(fh, p0, nPack, bitmap=bitmap, bitwidth=bitwidth, order_off=order_off, hdver=hdver, verbose=1, meta=meta)
            # spec.shape = (nFrame, nAnt, nChan)
            ringSpec = loadNode(fh, p0, nPack, order_off=order_off, verbose=verbose, meta=meta, nFPGA=nRow, no_bitmap=no_bitmap)
            # ringSpec.shape = (nRow, nFrame, nAnt, nChan)
            fh.close()

            aspec = np.ma.abs(ringSpec).mean(axis=1)
            arrInt[i] = aspec   # shape: (nFile, nAnt, nChan)
            arrInt.mask[i] = aspec.mask


        adoneh5(ofile, arrInt, 'intensity')
        adoneh5(ofile, winSec, 'winSec')

        putAttrs(ofile, attrs)

        ## bandpass normalization
        arrNInt = arrInt / arrInt.mean(axis=0)
        adoneh5(ofile, arrNInt, 'norm.intensity')

    arrInts.append(arrInt)
    arrNInts.append(arrNInt)
    print(j, arrInt.shape, arrNInt.shape)

arrInt = np.concatenate(arrInts, axis=3)
arrNInt = np.concatenate(arrNInts, axis=3)
freq = np.concatenate(freqs, axis=0)
print('combine', arrInt.shape, arrNInt.shape)

print('epoch0:', epoch0)
loc0 = epoch0 + 3600*8  # convert back to local time
winDT = Time(loc0+winSec, format='unix').to_datetime()
#matDT = mdates.date2num(winDT)
#X, Y = np.meshgrid(winSec, freq, indexing='xy')
#X, Y = np.meshgrid(winDT, freq, indexing='xy')
X = winDT
Y = freq
#print('X:', X)

if (nDir==1):
    odir2 = odir

if (not os.path.isdir(odir2)):
    call('mkdir %s'%odir2, shell=True)

for j in range(nRow):
    ## plotting
    for pt in range(1,2):
        if (pt==0):
            png = '%s/raw.row%d.png' % (odir2,j)
            arr = arrInt[:,j]
        elif (pt==1):
            png = '%s/norm.row%d.png' % (odir2,j)
            arr = arrNInt[:,j]
        print('plotting:', png, '...')

        if (zlim is None):
            vmin = arr.min()
            vmax = arr.max()
        else:
            vmin = zlim[0]
            vmax = zlim[1]
        print('zmin,zmax:', vmin, vmax)


        fig, s2d = plt.subplots(8,4,figsize=(16,12), sharex=True, height_ratios=[2,1,2,1,2,1,2,1])
        sub = s2d[0::2,:].flatten()
        sub2 = s2d[1::2,:].flatten()
        for ii in range(4):
            s2d[ii*2,0].set_ylabel('freq (MHz)')
            s2d[ii*2+1,0].set_ylabel('amp')
            s2d[7,ii].set_xlabel('time (sec)')


        for ai in range(nAnt):
            ax = sub[ai]
            #ax.imshow(arrInt[:,ai].T, origin='lower', aspect='auto')
            ax.pcolormesh(X,Y,arr[:,ai].T, vmin=vmin, vmax=vmax, shading='auto')

            ax2 = sub2[ai]
            y2d = arr[:,ai].T
            y1d = y2d.mean(axis=0)
            ax2.plot(winDT, y1d)
            ax2.set_ylim(vmin, vmax)
            ax2.grid()


        fig.autofmt_xdate()
        fig.tight_layout(rect=[0,0.03,1,0.95])
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(png)
        plt.close(fig)



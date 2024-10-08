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
nChan   = 1024
nBeam   = nAnt
flim    = [400., 800.]
freq    = np.linspace(flim[0], flim[1], nChan, endpoint=False)

p0      = 0
nPack   = 1000
nBlock  = 1
#blocklen= 800000
blocklen= 128000
bitwidth= 4
hdver   = 2
order_off= -2
meta    = 64

ofile   = 'intensity.inth5'
read_raw = False
zlim    = None
no_bitmap = False


usage   = '''
plot amplitude of the spectrum as a function of frequency and time

syntax:
    %s <data_dir> [options]

options are:
    -n nPack        # number of packets to load (%d)
    --p0 p0         # starting packet (%d)
    --16bit         # is 16bit data (default is 4bit data)
    --hd VER        # header version: 1 or 2 (%d)
    --meta bytes    # ring buffer or file metadata length in bytes (%d)
    --nBlock nBlock # number of blocks in the file (%d)
    --blocklen blocklen
                    # the number of packets in a block (%d)
    --ooff order_off# add this offset to the packet_order (%d)
    -o ofile        # the output file name (<data_dir>.inth5)
    --redo          # force reading raw data
    --zlim zmin zmax# set the min/max color scale
    --no-bitmap     # ignore bitmap from the data

''' % (pg, nPack, p0, hdver, meta, nBlock, blocklen, order_off)


if (len(inp)<1):
    sys.exit(usage)

dirs = []
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
    elif (k=='--ooff'):
        order_off = int(inp.pop(0))
    elif (k=='-o'):
        ofile = inp.pop(0)
    elif (k=='--redo'):
        read_raw = True
    elif (k=='--zlim'):
        zmin = float(inp.pop(0))
        zmax = float(inp.pop(0))
        zlim = [zmin, zmax]
    elif (k=='--no-bitmap'):
        no_bitmap = True
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        #idir = k
        dirs.append(k)


for idir in dirs:
    ofile = idir + '.inth5'
    print('output to:', ofile)

    files   = glob('%s/*.bin'%idir)
    files.sort()
    nFile   = len(files)
    #nFile   = 10
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
        arrInt  = np.ma.array(np.zeros((nFile, nAnt, nChan)), mask=True)  # masked by default
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
            BM = loadFullbitmap(fh, nBlock, blocklen=blocklen, meta=meta)
            bitmap = BM[p0:p0+nPack]
            if (no_bitmap):
                bitmap = np.ones(nPack, dtype=bool) # override real bitmap, testing
            nValid = np.count_nonzero(bitmap)
            fValid = float(nValid/nPack)
            if (fValid < 0.1):
                print('invalid block, skip!')
                continue

            tick, spec = loadSpec(fh, p0, nPack, bitmap=bitmap, bitwidth=bitwidth, order_off=order_off, hdver=hdver, verbose=1, meta=meta)
            # spec.shape = (nFrame, nAnt, nChan)
            aspec = np.ma.abs(spec).mean(axis=0)
            arrInt[i] = aspec
            arrInt.mask[i] = aspec.mask


        adoneh5(ofile, arrInt, 'intensity')
        adoneh5(ofile, winSec, 'winSec')

        putAttrs(ofile, attrs)

        ## bandpass normalization
        arrNInt = arrInt / arrInt.mean(axis=0)
        adoneh5(ofile, arrNInt, 'norm.intensity')


    print('epoch0:', epoch0)
    loc0 = epoch0 + 3600*8  # convert back to local time
    winDT = Time(loc0+winSec, format='unix').to_datetime()
    #matDT = mdates.date2num(winDT)
    #X, Y = np.meshgrid(winSec, freq, indexing='xy')
    X, Y = np.meshgrid(winDT, freq, indexing='xy')
    #print('X:', X)
    ## plotting
    for pt in range(2):
        if (pt==0):
            png = '%s.raw.png' % ofile
            arr = arrInt
        elif (pt==1):
            png = '%s.norm.png' % ofile
            arr = arrNInt
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
            ax.pcolormesh(X,Y,arr[:,ai].T, vmin=vmin, vmax=vmax)

            ax2 = sub2[ai]
            y2d = arr[:,ai].T
            y1d = y2d[:768].mean(axis=0)
            ax2.plot(winDT, y1d)
            ax2.grid()


        fig.autofmt_xdate()
        fig.tight_layout(rect=[0,0.03,1,0.95])
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(png)
        plt.close(fig)



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
nRow    = 16        # for 256-ant
nChan   = 128       # for 256-ant
nBeam   = nAnt
flim    = [400., 800.]
freq0    = np.linspace(flim[0], flim[1], nChan0, endpoint=False)

p0      = 0
nPack   = 1024
nBlock  = 1
#blocklen= 800000
blocklen= 128000    # not used?
bitwidth= 4
hdver   = 2
order_off= 0
meta    = 64
verbose = 0

odir2   = None
read_raw = True
zlim    = None
no_bitmap = False
redo    = False


usage   = '''
plot amplitude of the spectrum as a function of frequency and time

syntax:
    %s <dirs> [options]

all .bin files are loaded from the provided dirs
the packet order is automatically assigned from packet header

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
        redo = True
    elif (k=='--zlim'):
        zmin = float(inp.pop(0))
        zmax = float(inp.pop(0))
        zlim = [zmin, zmax]
    elif (k=='--no-bitmap'):
        no_bitmap = True
    elif (k=='-v'):
        verbose=1
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        dirs.append(k)



nDir = len(dirs)
if (meta == 64):
    hdver = 2

# for bf256
total_order = 8
freqs = freq0.reshape(total_order, nChan)


if (nDir == 1):
    oname = dirs[0].rstrip('/')
    fh5  = '%s.inth5'%oname
    odir = '%s.plots'%oname
else:
    fh5  = 'intensity.inth5'
    odir = 'intensity.plots'


if (os.path.isfile(fh5)):
    tmp = getData(fh5, 'arrInt')
    if (tmp is not None):
        read_raw = False
        arrInt = tmp
        arrNInt = getData(fh5, 'arrNInt')
        freq = getData(fh5, 'freq')
        tsecs = getData(fh5, 'tsecs')
        attrs = getAttrs(fh5)
        loc0 = attrs['unix_utc_open'] + 3600*8
        winDT = Time(loc0+tsecs, format='unix').to_datetime()

if (redo):
    read_raw = True

if (read_raw):
    tsecs = []
    allInts  = []
    allNInts = []
    for o in range(total_order):
        allInts.append([])
        allNInts.append([])

    attrs   = {}
    attrs['dirs'] = dirs
    attrs['p0'] = p0
    attrs['nPack'] = nPack
    attrs['nBlock'] = nBlock
    attrs['blocklen'] = blocklen
    attrs['bitwidth'] = bitwidth
    attrs['hdver'] = hdver
    attrs['order_off'] = order_off
    attrs['nChan'] = nChan
    attrs['nChan0'] = nChan0
    attrs['nAnt'] = nAnt
    attrs['nRow'] = nRow

    epoch0 = None
    for j in range(nDir):
        idir = dirs[j]

        files   = glob('%s/*.bin'%idir)
        files.sort()
        nFile   = len(files)
        if (False):
            nFile = 16
            files = files[:nFile]
            print(files)
        print(idir, 'nFile:', nFile)

        fileOrder = np.zeros(nFile)
        fileInt  = np.ma.array(np.zeros((nFile, nRow, nAnt, nChan)), mask=True)  # masked by default
        ep0, fileSec  = filesEpoch(files, hdver=2, meta=64, split=True)
        if (epoch0 is None):
            epoch0 = ep0
        attrs['unix_utc_open'] = epoch0

        for i in range(nFile):
            fbin = files[i]
            print('reading: %s (%d/%d)'%(fbin, i+1,nFile))

            fh = open(fbin, 'rb')
            mt = fh.read(64)
            hd = fh.read(64)
            tmp = decHeader2(hd)
            od = tmp[4]
            fileOrder[i] = od

            # spec.shape = (nFrame, nAnt, nChan)
            ringSpec = loadNode(fh, p0, nPack, order_off=order_off, verbose=verbose, meta=meta, nFPGA=nRow, no_bitmap=no_bitmap)
            # ringSpec.shape = (nRow, nFrame, nAnt, nChan)
            fh.close()

            aspec = np.ma.abs(ringSpec).mean(axis=1)
            fileInt[i] = aspec   # shape: (nFile, nAnt, nChan)
            fileInt.mask[i] = aspec.mask

        for o in range(total_order):
            if (len(tsecs)==0):
                tsecs = fileSec[fileOrder==o]
                print(tsecs)
            if (len(allInts[o])==0):    # use array if none exists
                allInts[o] = fileInt[fileOrder==o]
            else:   # concatenate if array already exists
                allInts[o] = np.concatenate([allInts[o], fileInt(fileOrder==o)], axis=0)
                if (len(tsecs) < allInts[o].shape[0]):
                    tsecs = np.concatenate([tsecs, fileSec[fileOrder==o]], axis=0)

    putAttrs(fh5, attrs)

    ## bandpass normalization
    for o in range(total_order):
        len_order = len(allInts[o])
        print('order', o, 'len:', len_order)
        if (len_order > 0):
            allNInts[o] = allInts[o] / allInts[o].mean(axis=0)
            name = 'order%d'%o

    use_order = np.unique(fileOrder).astype(int)
    print(use_order)

    arrInt = np.concatenate([allInts[o] for o in use_order], axis=3)
    arrNInt = np.concatenate([allNInts[o] for o in use_order], axis=3)
    freq = np.concatenate(freqs[use_order], axis=0)
    print('combine', arrInt.shape, arrNInt.shape, freq.shape)
    adoneh5(fh5, arrInt, 'arrInt')
    adoneh5(fh5, arrNInt, 'arrNInt')

    print('epoch0:', epoch0)
    loc0 = epoch0 + 3600*8  # convert back to local time
    winDT = Time(loc0+tsecs, format='unix').to_datetime()
    adoneh5(fh5, freq, 'freq')
    adoneh5(fh5, tsecs, 'tsecs')

#X, Y = np.meshgrid(winSec, freq, indexing='xy')
#X, Y = np.meshgrid(winDT, freq, indexing='xy')
X = winDT
Y = freq
#print('X:', X)

if (odir2 is None):
    odir2 = odir

if (not os.path.isdir(odir2)):
    call('mkdir %s'%odir2, shell=True)

## 256 beams in one plot
png = '%s/norm.256beams.png' % (odir2)
#fig, sub = plt.subplots(4,16,figsize=(32,8), sharex=True, sharey=True)
hratio = np.zeros(32)
hratio[0::2] = 2
hratio[1::2] = 1
fig, tmp = plt.subplots(32,16,figsize=(32,48), sharex=True, height_ratios=hratio)
sub  = tmp[0::2]
sub2 = tmp[1::2]


if (zlim is None):
    vmin = arrNInt.min()
    vmax = arrNInt.max()
else:
    vmin = zlim[0]
    vmax = zlim[1]
print('zmin,zmax:', vmin, vmax)

for j in range(nRow):
    for ai in range(nAnt):
        ax = sub[nRow-1-j, ai]
        if (j==0 and ai==0):
            print(X.shape, Y.shape, arrNInt[:,j,ai].shape)
        ax.pcolormesh(X,Y,arrNInt[:,j,ai].T, vmin=vmin, vmax=vmax, shading='auto')

        prof = arrNInt[:,j,ai].mean(axis=1) # avg in freq, each node separately
        ax2 = sub2[nRow-1-j, ai]
        ax2.plot(winDT, prof)
        ax2.set_ylim(vmin, vmax)

fig.autofmt_xdate()
fig.tight_layout(rect=[0,0.03,1,0.95])
fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle('%s, %s'%(os.getcwd(), odir2))
fig.savefig(png)
plt.close(fig)







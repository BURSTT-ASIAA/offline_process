#!/usr/bin/env python

import sys, os.path
import time, gc
from glob import glob
import matplotlib.pyplot as plt
from astropy.time import Time

from packet_func import *
from calibrate_func import *
from loadh5 import *

inp = sys.argv[0:]
pg  = inp.pop(0)

bitwidth = 16
nAnt    = 16
nChan   = 1024
nBlock  = 0
autoblock = True
nPack   = 16384
p0      = 0
blocklen = 102400
bytePack = 8256
hdver   = 2
no_bitmap = False
meta    = 64
nPool   = 4

#idir = 'data_monitor'
fout = 'win_eigenmodes.h5'

usage = '''
given a folder calculate the eigenmodes of each binary file
the results are saved in a single file 'win_eigenmodes.py'
syntax:
    %s <data_folder> [options]

    options are:
    -n nPack        # number of packets to construct the covariance matrix
                    # which is then used to solve for eigenvalue/eigenvector
                    # (default: %d)
    --p0 P0         # reading starts from this packet
                    # (default: %d)
    --blocklen blocklen
                    # the number of packets in a block
                    # (default: %d)
    --hd ver        # the header version: 1=old, 2=new
                    # (default: %d)
    --nB nBlock     # specify the number of Block
                    # will disable autoblock
                    # (default: 0, autoblock=True)
    --meta META     # specify the length of meta data in bytes
                    # (0 for older data, 64 for newer data)
                    # (default: %d)
    --pool nPool    # number of threads to calculate covariance matrix
                    # (default: %d)
    --no-bitmap     # ignore the bitmap in the file
    -o out_h5       # output .h5 filename
                    # (default: %s)

Note: to plot the results, use 'plot_eigenmode.py'

''' % (pg, nPack, p0, blocklen, hdver, meta, nPool, fout)

if (len(inp)<1):
    sys.exit(usage)

while(inp):
    k = inp.pop(0)
    if (k == '-n'):
        nPack = int(inp.pop(0))
    elif (k == '--p0'):
        p0 = int(inp.pop(0))
    elif (k == '--blocklen'):
        blocklen = int(inp.pop(0))
    elif (k == '--hd'):
        hdver = int(inp.pop(0))
    elif (k == '--nB'):
        nBlock = int(inp.pop(0))
    elif (k == '--meta'):
        meta = int(inp.pop(0))
    elif (k == '--pool'):
        nPool = int(inp.pop(0))
    elif (k == '--no-bitmap'):
        no_bitmap = True
    elif (k == '-o'):
        fout = inp.pop(0)
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        idir = k

byteBlock = blocklen * bytePack

print('reading data from:', idir)
files = glob('%s/*.bin'%idir)
files.sort()
#files = files[-10:] # testing only
nFile = len(files)

#nFile2 = 10 # testing only
nFile2 = nFile
tx   = np.arange(nFile2)
chan = np.arange(nChan)
Vlast = np.zeros((nFile2, nChan, nAnt), dtype=complex)

attrs = {}
attrs['bitwidth'] = bitwidth
attrs['nPack'] = nPack
attrs['files'] = files

ftime0 = None
tsec = []
#savW2 = []  # 2 for scaled
#savV2 = []
savW3 = []  # 3 for coeff
savV3 = []

t0 = time.time()

ii = -1
for i in range(nFile):
#for i in range(36, nFile, 6):   # testing, starting from 7:00am, every 30min
    print('(%d/%d)...'%(i+1, nFile), files[i])
    ii += 1

    fbin = files[i]

    fbase = os.path.basename(fbin)   # use 1st dir as reference
    tmp = fbase.split('.')
    ftpart = tmp[1]
    if (len(ftpart)==10):
        ftstr = '23'+ftpart # hard-coded to 2023!!
    elif (len(ftpart)==14):
        ftstr = ftpart[2:]
    ftime = datetime.strptime(ftstr, '%y%m%d%H%M%S')
    if (ftime0 is None):
        ftime0 = ftime
        unix0 = Time(ftime0, format='datetime').to_value('unix')    # local time
        unix0 -= 3600.*8.                                           # convert to UTC
        attrs['unix_utc_open'] = unix0

    dt = (ftime - ftime0).total_seconds()
    print('(%d/%d)'%(ii+1,nFile), fbin, 'dt=%dsec'%dt)


    if (nBlock == 0 and autoblock):
        fsize = os.path.getsize(fbin)
        nBlock = np.rint(fsize/byteBlock).astype(int)
        print('autoblock:', nBlock)


    fh = open(fbin, 'rb')
    BM = loadFullbitmap(fh, nBlock, blocklen=blocklen, meta=meta)
    bitmap = BM[p0:p0+nPack]
    if (no_bitmap):
        bitmap = np.ones(nPack, dtype=bool)
    fvalid = float(np.count_nonzero(bitmap)/nPack)
    if (fvalid < 0.1): # an arbitrary threshold
        print('no valid block. skip.')
        fh.close()
        continue

    # start appending only when the block is valid
    tsec.append(dt)

    tick, spec0 = loadSpec(fh, p0, nPack, bitmap=bitmap, bitwidth=bitwidth, hdver=hdver, verbose=1, meta=meta)
    #tick, spec0 = loadSpec(fh, p0, nPack, nBlock=nBlock, bitwidth=bitwidth, verbose=1)
    fh.close()

    spec = spec0.transpose((1,0,2))
    del tick, spec0
    gc.collect()
    t1 = time.time()
    print('... data loaded. elapsed:', t1-t0)

    if (False):
        #Cov2, norm2 = makeCov(spec, scale=True, coeff=False)
        Cov2, norm2 = makeCov(spec, scale=True, bandpass=True, coeff=False, nPool=nPool)
        W2, V2 = Cov2Eig(Cov2)
        savW2.append(W2)
        savV2.append(V2)
    Cov3, norm3 = makeCov(spec, scale=False, coeff=True, nPool=nPool)
    W3, V3 = Cov2Eig(Cov3)
    savW3.append(W3)
    savV3.append(V3)

    #Vlast[ii] = V[:,:,nAnt-1]
    t2 = time.time()
    print('... eigenmode got. elapsed:', t2-t0)

print('files loaded')
#savW2 = np.array(savW2)
#savV2 = np.array(savV2)
savW3 = np.array(savW3)
savV3 = np.array(savV3)

#adoneh5(fout, savW2, 'win_W_scale')
#adoneh5(fout, savV2, 'win_V_scale')
adoneh5(fout, savW3, 'win_W_coeff')
adoneh5(fout, savV3, 'win_V_coeff')

adoneh5(fout, tsec, 'winSec')
putAttrs(fout, attrs)

print('done... results saved in:', fout)

sys.exit()

# skip plotting, use plot_eigenmode.py
X, Y = np.meshgrid(tx, chan, indexing='xy')
fig, s2d = plt.subplots(4,4,figsize=(16,12))
sub = s2d.flatten()

for ai in range(nAnt):
    ax = sub[ai]
    ax.pcolormesh(X,Y,np.angle(Vlast[:,:,ai]).T, shading='nearest')

fig.tight_layout()
fig.savefig('eigenvec.coeff.V.phsae.png')
plt.close(fig)



#!/usr/bin/env python

import sys, os.path
import time, gc
from glob import glob
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.stats import sigma_clip

from packet_func import *
from calibrate_func import *
from loadh5 import *

inp = sys.argv[0:]
pg  = inp.pop(0)

bitwidth = 16
nAnt    = 16
nChan   = 1024
flim    = [400., 800.]
blocklen = 128000
nBlock  = 1
autoblock = True
nPack   = 10000
p0      = 0
hdver   = 2
meta    = 64
order_off = 0
fout    = 'output.eigen.h5'
user_fout = False
no_bitmap = False
hdlen   = 64
paylen  = 8192
combine = False
redo    = False
nPool   = 4

ant_flag = []
nFlag   = 0

usage   = '''
compute covariance matrix and save the eigenmodes from a FPGA binary file
if multiple files are given, two modes are available:
    the default mode is to process each file (16-element) independently
    the combine mode will group the files into a larger array 
        (nFile*nAnt-element) for covariance computation

syntax:
    %s <bin file(s)> [options]

options are:
    -n nPack    # read nPack from the binary file (%d)

    --p0 p0     # starting packet offset (%d)

    --blocklen blocklen
                # change the packet number per block
                # (default: %d)
    --nB nB     # specify the block length of the binary data
                # (default: determined by autoblock %d)
    -o fout     # specify an output file name
                # (default: %s)
    --flag 'ant(s)'
                # specify the input number (0--15) to be flagged
    --hd VER    # header version (1, 2)
                # (default: %d)
    --meta bytes # number of bytes in the ring buffer or file metadata
                # ring buffer: 128 bytes
                # file: 64 bytes
                # (default: %d)
    --combine   # combine the bin files to for a larger covariance matrix
    --redo      # re-generate eigenmodes
                # (default is to plot existing eigenmodes)
    --pool nPool
                # number of threads used to parallel process makeCov
                # (default: %d)

    (special)
    --no-bitmap # ignore the bitmap
    --4bit      # read 4-bit data
    --ooff OFF  # offset added to the packet_order

''' % (pg, nPack, p0, blocklen, nBlock, fout, hdver, meta, nPool)

if (len(inp) < 1):
    sys.exit(usage)

files0 = []
while (inp):
    k = inp.pop(0)
    if (k == '-n'):
        nPack = int(inp.pop(0))
    elif (k == '--p0'):
        p0 = int(inp.pop(0))
    elif (k == '--blocklen'):
        blocklen = int(inp.pop(0))
    elif (k == '--nB'):
        nBlock = int(inp.pop(0))
    elif (k == '-o'):
        user_fout = True
        fout = inp.pop(0)
    elif (k == '--flag'):
        tmp = inp.pop(0).split()
        ant_flag = [int(x) for x in tmp]
        print('ant_flag:', ant_flag)
        nFlag = len(ant_flag)
    elif (k == '--no-bitmap'):
        no_bitmap = True
    elif (k == '--4bit'):
        bitwidth = 4
    elif (k == '--hd'):
        hdver = int(inp.pop(0))
    elif (k == '--meta'):
        meta = int(inp.pop(0))
    elif (k == '--ooff'):
        order_off = int(inp.pop(0))
    elif (k == '--combine'):
        combine = True
    elif (k == '--redo'):
        redo = True
    elif (k == '--pool'):
        nPool = int(inp.pop(0))
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        files0.append(k)

# for autoblock
byteBlockBM = blocklen//8
byteBlock = (hdlen + paylen)*blocklen + byteBlockBM
# frequency in MHz
freq = np.linspace(flim[0], flim[1], nChan, endpoint=False)



if (combine):
    nLoop = 1
    loop_files = [files0]
else:
    nLoop = len(files0)
    loop_files = [[x] for x in files0]

t00 = time.time()

for ll in range(nLoop):
    files = loop_files[ll]

    nFile = len(files)
    if (not user_fout):
        if (nFile == 1):   # override default fout if input is a single file
            fout = files[0] + '.eigen.h5'
    print('eigenmode is saved in:', fout, '...')

    if (os.path.isfile(fout) and not redo):
        attrs = getAttrs(fout)
        savN2 = getData(fout, 'N2_scale')
        savW2 = getData(fout, 'W2_scale')
        savV2 = getData(fout, 'V2_scale')
        savW3 = getData(fout, 'W3_coeff')
        savV3 = getData(fout, 'V3_coeff')
        tsec  = getData(fout, 'winSec')

    else:   # redo or file not exist
        attrs = {}
        attrs['bitwidth'] = bitwidth
        attrs['nPack'] = nPack
        attrs['p0'] = p0
        attrs['filename'] = files
        attrs['nAnt'] = nAnt
        attrs['nChan'] = nChan

        ftime0 = None
        tsec = []
        savW2 = []  # 2 for scaled
        savV2 = []
        savN2 = []  # normalization used to scale the data
        savN2mask = []  # normalization used to scale the data
        savW3 = []  # 3 for coeff
        savV3 = []

        if (bitwidth==4):
            nFrame = nPack // 2
        elif (bitwidth==16):
            nFrame = nPack // 8
        # note, the shape is after transpose
        spec = np.ma.array(np.zeros((nFile,nAnt,nFrame,nChan), dtype=complex), mask=True)

        t0 = time.time()

        ii = -1
        for i in range(nFile):
            print(i, files[i])
            ii += 1

            fbin = files[i]

            fbase = os.path.basename(fbin)   # use 1st dir as reference
            tmp = fbase.split('.')
            ftpart = tmp[1]
            if (len(ftpart)==10):
                ftstr = '23'+ftpart # hard-coded to 2023!!
            elif (len(ftpart)==14):
                ftstr = ftpart[2:]  # strip leading 20
            ftime = datetime.strptime(ftstr, '%y%m%d%H%M%S')
            if (ftime0 is None):
                ftime0 = ftime
                unix0 = Time(ftime0, format='datetime').to_value('unix')    # local time
                unix0 -= 3600.*8.                                           # convert to UTC
                attrs['unix_utc_open'] = unix0

            dt = (ftime - ftime0).total_seconds()
            print('(%d/%d)'%(ii+1,nFile), fbin, 'dt=%dsec'%dt)
            tsec.append(dt)

            if (autoblock):
                fsize = os.path.getsize(fbin)
                nBlock = np.rint((fsize-meta)/byteBlock).astype(int)
                print('auto block, nB:', nBlock)
                autoblock=False

            fh = open(fbin, 'rb')
            BM = loadFullbitmap(fh, nBlock, blocklen=blocklen, meta=meta)
            bitmap = BM[p0:p0+nPack]
            if (no_bitmap):
                bitmap = np.ones(nPack, dtype=bool)
            # spec0.shape = (nFrame, nAnt, nChan)
            tick, spec0 = loadSpec(fh, p0, nPack, bitmap=bitmap, bitwidth=bitwidth, hdver=hdver, order_off=order_off, verbose=1, meta=meta)
            #tick, spec0 = loadSpec(fh, p0, nPack, bitmap=bitmap, bitwidth=bitwidth, verbose=1)
            #tick, spec0 = loadSpec(fh, p0, nPack, nBlock=nBlock, bitwidth=bitwidth, verbose=1)
            fh.close()

            # trasposed shape = (nAnt, nFrame, nChan)
            spec[i] = spec0.transpose((1,0,2))
            del tick, spec0
            gc.collect()
            t1 = time.time()
            print('... data', i, 'loaded. elapsed:', t1-t0)

        spec = spec.reshape((-1,nFrame,nChan))
        ## new shape (nFile*nAnt, nFrame, nChan), only 3 axes

        Cov2, norm2 = makeCov(spec, scale=True, coeff=False, bandpass=True, ant_flag=ant_flag, nPool=nPool)
        #Cov2, norm2 = makeCov(spec, scale=True, coeff=False, bandpass=False, ant_flag=ant_flag)
        W2, V2 = Cov2Eig(Cov2)
        savW2.append(W2)
        savV2.append(V2)
        savN2.append(norm2)
        savN2mask.append(norm2.mask)
        Cov3, norm3 = makeCov(spec, scale=False, coeff=True, ant_flag=ant_flag, nPool=nPool)
        W3, V3 = Cov2Eig(Cov3)
        savW3.append(W3)
        savV3.append(V3)

        #Vlast[ii] = V[:,:,nAnt-1]
        t2 = time.time()
        print('... eigenmode got. elapsed:', t2-t0)

        print('files loaded')
        savW2 = np.array(savW2).mean(axis=0)
        savV2 = np.array(savV2).mean(axis=0)
        savN2 = np.ma.array(savN2, mask=savN2mask).mean(axis=0)
        savW3 = np.array(savW3).mean(axis=0)
        savV3 = np.array(savV3).mean(axis=0)
        tsec  = np.array(tsec)



        adoneh5(fout, savN2, 'N2_scale')    # shape (nAnt, nChan)
        adoneh5(fout, savW2, 'W2_scale')    # shape (nChan, nMode)
        adoneh5(fout, savV2, 'V2_scale')    # shape (nChan, nAnt, nMode)
        adoneh5(fout, savW3, 'W3_coeff')
        adoneh5(fout, savV3, 'V3_coeff')

        adoneh5(fout, tsec, 'winSec')
        putAttrs(fout, attrs)



    (nChan3, nAnt3, nMode3) = savV3.shape
    nCol = nAnt3//nAnt

    ## plot normalization and eigenvalues
    fig, sub = plt.subplots(3,1,figsize=(15,15),sharex=True)

    # normalization
    ax = sub[0]
    for ai in range(nAnt3):
        ax.plot(freq, savN2[ai], label='Ant%d'%ai)
    ax.set_yscale('log')
    ax.set_ylabel('voltage normalization')
    ax.legend(ncols=nCol)

    # eigenvalues
    ax = sub[1]
    for ai in range(nMode3):  # nMode = nAnt
        if (ai < nFlag):    # skip first n (weakest) modes correspondsng to the flagged antennas
            continue
        y = savW3[:,ai]
        y2 = sigma_clip(10.*np.log10(y), sigma=10)
        #ax.plot(freq, 10.*np.log10(y), label='Mode%d'%ai)
        ax.plot(freq, y2, label='Mode%d'%ai)
    ax.set_ylabel('power (dB)')
    ax.legend(ncols=nCol)

    # eigenvector of leading mode, phase
    ax = sub[2]
    for ai in range(nAnt3):
        if (ai < nFlag):
            continue
        y = savV3[:,ai,-1]
        ax.plot(freq, np.ma.angle(y), label='Ant%d'%ai)
    ax.set_ylabel('phase (rad)')
    ax.legend(ncols=nCol)
    ax.set_xlabel('freq (MHz)')
    ax.set_xlim(flim[0], flim[1]*1.25)

    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.suptitle(fout)
    fig.savefig('%s.png'%fout)
    plt.close(fig)


    t3 = time.time()
    print('... all done. elapsed:', t3-t00)

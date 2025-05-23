#!/usr/bin/env python

import sys, os.path
#from scapy.all import rdpcap
import struct
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
from astropy.stats import sigma_clip
import warnings
from packet_func import *


bitwidth = 4    # binary bit width
hdver    = 2    # header version
meta     = 64    # ring buffer or file metadata length in bytes

bpp = 8192  # bytes per packet (payload size)
#fpgaclock = 401.066667e6
fpgaclock = 400.e6

bytePayload = 8192      # packet payload size in bytes
byteHeader  = 64        # packet header size in bytes
#packBlock   = 1000000   # num of packets in a block
#packBlock   = 800000   # num of packets in a block
packBlock   = 102400   # num of packets in a block
autoblock = True
autop0    = True
order_off = 0
flim    = [400., 800.]  # freq limit in MHz


npack  = 1000
pack0 = 0
nBlock = 0
ylim = None
verbose = 0
no_bitmap = False

inp = sys.argv[0:]
pg  = inp.pop(0)

usage = '''
%s <rudp2huge_file(s)> [options]

Note: multiple files can be supplied. each one is plotted separately.

options are:
    -n packets      # specify how many packets to read and process
                    # (%d)
    -c clock_speed  # the estinmated FPGA clock speed in MHz (%.0f)
    -b byte_per_packet  # change the byte_per_packet variable (payload only)
                        # (%d)
    --16bit         # set this option if the data is from 16bit format
                    # (default is the 4bit format)
    --hd VER        # header version. 1 or 2
                    # (%d)
    --meta bytes    # ring buffer of file metadata length in bytes
                    # (%d)
    --p0 pack0      # the starting packet number
                    # disable autop0 (automatic switch block if p0=0 is bad)
    --blocklen blocklen
                    # number of packets in a block (%d)
    --nB nBlock     # the number of block (1M packets/block) in the binary file
                    # override auto-block
    --no-ab         # disable auto-determine of nBlock (auto-block)
    --ooff OFF      # offset the packet_order
    --no-bitmap     # ignore the bitmap
    --flim FMIN FMAX    # set the min/max frequency in MHz (%.1f, %.1f)
    --ylim YMIN YMAX    # set the plot range in y (default: auto)
    -v              # turn on some diagnostic info

''' % (pg, npack, fpgaclock/1e6, bpp, hdver, meta, packBlock, flim[0], flim[1])

if (len(inp) < 1):
    sys.exit(usage)

files = []
while inp:
    k = inp.pop(0)
    if (k == '-n'):
        npack = int(inp.pop(0))
    elif (k == '-c'):
        tmp = float(inp.pop(0))
        fpgaclock = tmp * 1e6
    elif (k == '-b'):
        bpp = int(inp.pop(0))
    elif (k == '--16bit'):
        bitwidth = 16
    elif (k == '--hd'):
        hdver = int(inp.pop(0))
    elif (k == '--meta'):
        meta = int(inp.pop(0))
    elif (k == '--p0'):
        pack0 = int(inp.pop(0))
        autop0 = False
    elif (k == '--blocklen'):
        packBlock = int(inp.pop(0))
    elif (k == '--nB'):
        nBlock = int(inp.pop(0))
        autoblock = False
    elif (k == '--no-ab'):
        autoblock = False
    elif (k == '--ooff'):
        order_off = int(inp.pop(0))
    elif (k == '--no-bitmap'):
        no_bitmap = True
    elif (k == '--flim'):
        fmin = float(inp.pop(0))
        fmax = float(inp.pop(0))
        flim = [fmin, fmax]
    elif (k == '--ylim'):
        ymin = float(inp.pop(0))
        ymax = float(inp.pop(0))
        ylim = [ymin, ymax]
    elif (k == '-v'):
        verbose = 1
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        files.append(k)

byteBMBlock = packBlock//8  # bitmap size of a block in bytes
byteBlock   = (bytePayload+byteHeader)*packBlock + byteBMBlock  # block size in bytes

if (bitwidth==4):
    ppf = 2
    pchan = 512
elif (bitwidth==16):
    ppf = 8     # packet-per-frame
    pchan = 128 # num of spec.channel in a packet
nchan = int(ppf * pchan)    # total num of spec.channel

nFile = len(files)
print(files)
for fii,fin in enumerate(files):
    if (not os.path.isfile(fin)):
        print('file not found: %s' % fin)
        continue
    else:
        print('plotting', fin, '...(%d/%d)'%(fii+1, nFile))

    t1 = time.time()

    nframe = npack // ppf     # the new spec2k_x4 bitcode outputs 4 packets per frame, each packet is 512ch, and a frame is 2048ch
    chan = np.arange(nchan)
    #samprate = fpgaclock * 2.
    #freq0 = np.fft.fftfreq(nchan*2, d=1/samprate) + fpgaclock
    #freq0 = np.fft.fftfreq(nchan, d=1/fpgaclock)
    #freq0 = np.fft.fftshift(freq0) + fpgaclock*1.5
    #samprate = fpgaclock * 4.
    #freq0 = np.fft.fftfreq(nchan*4, d=1/samprate)
    #freq = freq0[nchan:nchan*2] / 1e6
    #freq = np.linspace(0, fpgaclock, nchan, endpoint=False) + fpgaclock
    #freq /= 1e6
    #print(freq)
    #xlim = [freq[0], freq[-1]]
    freq = np.linspace(flim[0], flim[1], nchan, endpoint=False) # MHz
    print('flim:', flim)

    skipfile = False
    fh = open(fin, 'rb')

    if (nBlock == 0 and autoblock):
        fsize = os.path.getsize(fin)
        nBlock = np.rint(fsize/byteBlock).astype(int)
        print('autoblock:', nBlock)
    maxPack = nBlock * packBlock

    if (nBlock == 0):
        print('bitmap info not used.')

    fullBM = loadFullbitmap(fh, nBlock, blocklen=packBlock, meta=meta)
    p0 = pack0
    ap = autop0
    while (ap):
        bitmap = fullBM[p0:p0+npack]
        if (no_bitmap):
            bitmap = np.ones(npack, dtype=bool)
        fvalid = float(np.count_nonzero(bitmap)/npack)
        if (fvalid < 0.1):  # arbitrary threshold
            p0 += packBlock
            if (p0 >= maxPack):
                print('no valid block, skip')
                skipfile = True
                break
        else:
            ap = False

    if (verbose):
        print('pack0:', p0)

    bitmap = fullBM[p0:p0+npack]
    if (no_bitmap):
        bitmap = np.ones(npack, dtype=bool)
    fvalid = float(np.count_nonzero(bitmap)/npack)
    if (fvalid < 0.1):  # arbitrary threshold
        print('no valid block. skip.')
        skipfile = True

    if (skipfile):
        fh.close()
        continue

    #tick, spec = loadSpec(fh, p0, npack, bpp, ppf, nBlock=nBlock, verbose=verbose, bitwidth=bitwidth)
    tick, spec = loadSpec(fh, p0, npack, bpp, ppf, order_off=order_off, bitmap=bitmap, verbose=verbose, bitwidth=bitwidth, hdver=hdver, meta=meta)
    adata2d = np.abs(spec).mean(axis=0)

    fh.close()


    png1 = '%s.png' % fin
    f1, sub1 = plt.subplots(4,4,figsize=(15,9), sharex=True, sharey=True)
    s1 = sub1.flatten()

    warnings.filterwarnings('ignore')
    for ai in range(16):
        ax = s1[ai]
        title = 'Inp%d' % ai

        y = 20*np.log10(adata2d[ai])
        ax.plot(freq, y, marker='.')
        ax.grid()
        nnz = np.count_nonzero(np.abs(adata2d[ai]))
        ax.text(0.05, 0.90, 'nonzero:%d'%nnz, transform=ax.transAxes)
        pk = y.max()
        ipk = np.ma.argmax(y)
        fpk = freq[ipk]
        ax.text(0.05, 0.80, 'peak pow:%.2f\nch:%d\nfreq:%.2fMHz'%(pk,ipk,fpk), transform=ax.transAxes, va='top')
        ax.text(0.95, 0.90, title, transform=ax.transAxes, ha='right')
        if (ylim is not None):
            ax.set_ylim(ylim)
        if (ai >= 12):
            ax.set_xlabel('freq (MHz)')
        if (ai%4 == 0):
            ax.set_ylabel('power (dB)')
    warnings.filterwarnings('default')


    ax.set_xlim(flim)
    f1.tight_layout(rect=[0,0.03,1,0.95])
    f1.subplots_adjust(wspace=0, hspace=0)
    f1.suptitle('%s, npack=%d, nframe=%d, pack0=%d'%(fin, npack, nframe, p0))
    f1.savefig(png1)
    plt.close(f1)




    t3 = time.time()

    print('running time (sec):', t3-t1)

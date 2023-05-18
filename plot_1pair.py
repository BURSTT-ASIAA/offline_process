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
import gc
from subprocess import call


nInp  = 16
ppf   = 2       # packet-per-frame
bpp   = 8192    # bytes per packet (payload size)
pchan = 512     # num of spec.channel in a packet
nchan = int(ppf * pchan)    # total num of spec.channel
fpgaclock = 401.066667e6    # clock rate when locked with 10MHz ref (2022/Dec)

bytePayload = 8192      # packet payload size in bytes
byteHeader  = 64        # packet header size in bytes
packBlock   = 1000000   # num of packets in a block
byteBMBlock = packBlock//8  # bitmap size of a block in bytes
byteBlock   = (bytePayload+byteHeader)*packBlock + byteBMBlock  # block size in bytes
autoblock = True
autop0    = True


npack   = 1000
pack0   = 0
nBlock  = 0
ylim    = None
verbose = 0
idx1    = -1
idx2    = -1

inp = sys.argv[0:]
pg  = inp.pop(0)

usage = '''
%s <rudp_file_1> <rudp_file_2> --i1 <IDX1> --i2 <IDX2> [options]

specify two rudp files and the index of the RF path from each file
to form a correlation between these two paths

Note: if only one rudp file is specified, it implies intra-board correlation

options are:
    -n packets      # specify how many packets to read and process
                    # (%d)
    -c clock_speed  # the estinmated FPGA clock speed in MHz (%.0f)
    -b byte_per_packet  # change the byte_per_packet variable (payload only)
                        # (%d)
    --p0 pack0      # the starting packet number
                    # disable autop0 (automatic switch block if p0=0 is bad)
    --nB nBlock     # the number of block (1M packets/block) in the binary file
                    # override auto-block
    --no-ab         # disable auto-determine of nBlock (auto-block)
    --ylim YMIN YMAX    # set the plot range in y
    -v              # turn on some diagnostic info

''' % (pg, npack, fpgaclock/1e6, bpp)

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
    elif (k == '--p0'):
        pack0 = int(inp.pop(0))
        autop0 = False
    elif (k == '--nB'):
        nBlock = int(inp.pop(0))
        autoblock = False
    elif (k == '--no-ab'):
        autoblock = False
    elif (k == '--ylim'):
        ymin = float(inp.pop(0))
        ymax = float(inp.pop(0))
        ylim = [ymin, ymax]
    elif (k == '--i1'):
        idx1 = int(inp.pop(0))
    elif (k == '--i2'):
        idx2 = int(inp.pop(0))
    elif (k == '-v'):
        verbose = 1
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        files.append(k)


nFile = len(files)
if (nFile < 1):
    sys.exit('no rudp file provided.')
else:
    file1 = files[0]
    if (nFile == 1):
        file2 = file1
        mode = 'intra'
        files2 = [file1]
        odir = '%s.out' % os.path.basename(file1)
    else:
        file2 = files[1]
        mode = 'cross'
        files2 = [file1, file2]
        odir = '%s_%s.out' % (os.path.basename(file1), os.path.basename(file2))
print('correlating:')
print('  path1:', file1, ' idx:', idx1)
print('  path2:', file2, ' idx:', idx2)
if (not os.path.isdir(odir)):
    call('mkdir -p %s'%odir, shell=True)


pairSpec = []
for fid, fin in enumerate(files2):
    if (not os.path.isfile(fin)):
        print('file not found: %s' % fin)
        continue
    else:
        print('loading', fin)

    if (fid == 0):
        idx = idx1
    else:
        idx = idx2

    t1 = time.time()

    nframe = npack // ppf     # the new spec2k_x4 bitcode outputs 4 packets per frame, each packet is 512ch, and a frame is 2048ch
    chan = np.arange(nchan)
    samprate = fpgaclock * 2.
    freq0 = np.fft.fftfreq(nchan*2, d=1/samprate) + fpgaclock
    freq = freq0[:nchan] / 1e6
    #print(freq)
    xlim = [freq[0], freq[-1]]
    print('freq[0], freq[-1]:', freq[0], freq[-1])

    fh = open(fin, 'rb')

    if (nBlock == 0 and autoblock):
        fsize = os.path.getsize(fin)
        nBlock = np.rint(fsize/byteBlock).astype(int)
        print('autoblock:', nBlock)

    if (nBlock == 0):
        print('bitmap info not used.')

    fullBM = loadFullbitmap(fh, nBlock)
    p0 = pack0
    ap = autop0
    while (ap):
        bitmap = fullBM[p0:p0+npack]
        fvalid = float(np.count_nonzero(bitmap)/npack)
        if (fvalid < 0.1):  # arbitrary threshold
            p0 += packBlock
        else:
            ap = False
    if (verbose):
        print('pack0:', p0)

    tick, spec = loadSpec(fh, p0, npack, bpp, ppf, nBlock=nBlock, verbose=verbose)
    pairSpec.append(spec[:,idx])

    fh.close()

    t3 = time.time()
    print('loading time (sec):', t3-t1)


if (mode == 'intra'):
    idx = idx2
    pairSpec.append(spec[:,idx])


if (len(pairSpec) < 2):
    sys.exit('error loading spectra')
print('correlating ...')
cross = pairSpec[0] * pairSpec[1].conjugate()
mr = np.ma.median(cross.real, axis=0)
mi = np.ma.median(cross.imag, axis=0)
mcross = mr + 1.j*mi
#print(mcross)
across = cross.mean(axis=0)
cr = sigma_clip(cross.real, axis=0)
ci = sigma_clip(cross.real, axis=0)
cross2 = cr + 1.j*ci
across2 = cross2.mean(axis=0)

amp1  = np.ma.abs(pairSpec[0])
amp2  = np.ma.abs(pairSpec[1])
coeff = cross / (amp1*amp2)
acoeff = coeff.mean(axis=0)

print('plotting ...')
f1, sub1 = plt.subplots(3,1,figsize=(8,10),sharex=True)
png1 = '%s/pair_%02d_%02d.png' % (odir, idx1, idx2)
suptitle = 'idx1:%02d, idx2:%02d, mode:%s, npack:%d, pack0:%d' % (idx1, idx2, mode, npack, p0)
if (mode == 'intra'):
    suptitle += '\nfile1: %s' % file1
elif (mode == 'cross'):
    suptitle += '\nfile1: %s' % file1
    suptitle += '\nfile2: %s' % file2

# auto- and cross- power (in dB)
ax = sub1[0]
aamp1 = amp1.mean(axis=0)
#aamp1.mask[aamp1==0.] = True
aamp2 = amp2.mean(axis=0)
#aamp2.mask[aamp2==0.] = True
ax.plot(freq, 20.*np.ma.log10(aamp1), label='auto1')
ax.plot(freq, 20.*np.ma.log10(aamp2), label='auto2')
#ax.plot(freq, 10.*np.ma.log10(np.ma.abs(mcross)), label='med(corr)')
ax.plot(freq, 10.*np.ma.log10(np.ma.abs(across)), label='avg(corr)')
#ax.plot(freq, 10.*np.ma.log10(np.ma.abs(across2)), label='avg(clip(corr))')
ax.set_xlim(xlim)
ax.set_xlabel('freq (MHz)')
ax.set_ylabel('power (dB)')
ax.legend()

# correlation coefficient
ax = sub1[1]
ax.plot(freq, acoeff.real, label='real')
ax.plot(freq, acoeff.imag, label='imag')
ax.plot(freq, np.ma.abs(acoeff), label='abs')
ax.set_xlabel('freq (MHz)')
ax.set_ylabel('correlation')
ax.legend()

# correlation phase
ax = sub1[2]
ax.plot(freq, np.ma.angle(acoeff))
ax.set_xlabel('freq (MHz)')
ax.set_ylabel('phase (rad)')
ax.set_ylim([-3.3, 3.3])

f1.tight_layout(rect=[0,0.03,1,0.95])
f1.suptitle(suptitle)
f1.savefig(png1)
plt.close(f1)



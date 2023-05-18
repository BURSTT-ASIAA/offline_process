#!/usr/bin/env python

from packet_func import *
import gc, time
from scipy.signal import correlate
import matplotlib.pyplot as plt

## test_packets2.py
#   test synchronization of two fpga files
#   by cross-correlating 1000 packets from the middle of one file
#   to 1000 packets moving across the time index


f218 = 'fpga218_221121.bin'
f219 = 'fpga219_221121.bin'

nChan = 1024
npack = 1000
nloop = 100
#lstart= 100
nrepeat=1

if (sys.argv[0] == 'taskset'):
    inp = sys.argv[3:]
else:
    inp = sys.argv[0:]
pg  = inp.pop(0)

usage = '''
delay search by inverse-transform the spectra back to streams
and perform time-domain correlation
reference is 2 blocks (1000 packets/block)
or 1000 frames from fpga218

target is the same amount of frames taken from fpga219
changing 1 frame at a time
forming cross-correlation coefficients (normalized)

syntax:
    %s <starting block> [options]
    a block is 1000 packets
    (i.e. 1000 blocks for a file of 1M packets)

    options are:
    -l LOOP         # number of blocks to load into memory (%d)
    -r REPEAT       # repeat the operation in the next loop (%d)

''' % (pg, nloop, nrepeat)

if (len(inp) < 1):
    sys.exit(usage)

while (inp):
    k = inp.pop(0)
    if (k == '-l'):
        nloop = int(inp.pop(0))
    elif (k == '-r'):
        nrepeat = int(inp.pop(0))
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        lstart = int(k)


fh218 = open(f218, 'rb')
fh219 = open(f219, 'rb')


t0 = time.time()
print('reading fpga218:', 2000, 'packets')
pack00 = 499000
tick, fullspec0 = loadSpec(fh218, pack00, int(2*npack), 8192, 2)
spec0 = fullspec0[:,7]
stream0 = spec2Stream(spec0)
norm0 = stream0.var() * len(stream0)    # sum of variance
t1 = time.time()
print('... done:', t1-t0, 'sec')


print('reading fpga219:', npack*nloop*nrepeat, 'packets')
f1, ax1 = plt.subplots(1,1)

for jj in range(nrepeat):
    lstart1 = lstart + jj*nloop

    stream1 = np.zeros((nloop,nChan*npack//2), dtype=complex)
    spec1 = np.zeros((nloop,npack//2,nChan), dtype=complex)
    for ii in range(nloop):
        pack0 = int((ii+lstart1) * npack)
        print('pack0:', pack0)

        tick, spec = loadSpec(fh219, pack0, npack, 8192, 2)
        spec1[ii] = spec[:,7]
        stream1[ii] = spec2Stream(spec[:,7])

        if ((ii+1) % 10 == 0):
            print('#', ii+1, 'running time:', time.time()-t1)

    stream1 = stream1.flatten()
    print('stream1.shape:', stream1.shape)
    nlen = len(stream1)

    tcorr = correlate(stream1, stream0, mode='same')
    tcorr /= norm0
    atcorr = np.abs(tcorr)

    pk = atcorr.max()
    ipk = atcorr.argmax() - nlen//2

    ax1.plot(np.arange(nlen)-nlen//2, np.abs(tcorr))
    ax1.text(0.05, 0.95, 'max:%.3f @ %d'%(pk,ipk), transform=ax1.transAxes)

    spec1 = spec1.reshape((-1,nChan))

fh218.close()
fh219.close()



ax1.set_xlabel('sample offset')
ax1.set_ylabel('correlation coefficient')
ax1.set_title('pack0: %d, npack: %d'%(pack0, npack*nloop))

f1.savefig('tcorr_loop%d_start%d.png' % (nloop*nrepeat, lstart))
plt.close(f1)


f1, ax1 = plt.subplots(1,1)
ch = np.arange(nChan)
ax1.plot(ch, np.abs(spec0).mean(axis=0), label='fpga218')
ax1.plot(ch, np.abs(spec1).mean(axis=0), label='fpga219')
ax1.set_xlabel('chan')
ax1.set_ylabel('abs(spec)')

f1.savefig('checkspec_loop%d_start%d.png' %(nloop*nrepeat, lstart))
plt.close(f1)




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
delay search by sweeping through the spectra
reference is 2 blocks (1000 packets/block)
or 1000 frames from fpga218

target is the same amount of frames taken from fpga219
changing 1 frame at a time
forming cross-correlation coefficients (normalized)
cross-correlation of the 1000 frames is averaged
the median(abs(cross)) across spectral channel is recorded 

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
pack00 = 500000
tick, fullspec0 = loadSpec(fh218, pack00, int(2*npack), 8192, 2)
spec0 = fullspec0[:,7]
norm0 = np.abs(spec0)
t1 = time.time()
print('... done:', t1-t0, 'sec')


print('reading fpga219:', npack*nloop*nrepeat, 'packets')
f1, ax1 = plt.subplots(1,1)

for jj in range(nrepeat):
    lstart1 = lstart + jj*nloop
    fullspec1 = np.zeros((nloop+2, npack//2, 1024), dtype=complex)
    med_coeff = []
    for ii in range(nloop+2):
        pack0 = int((ii+lstart1) * npack)
        print('pack0:', pack0)

        tick, spec = loadSpec(fh219, pack0, npack, 8192, 2)

        fullspec1[ii] = spec[:,7]

        if ((ii+1) % 10 == 0):
            print('#', ii+1, 'running time:', time.time()-t1)

    fullspec1 = np.array(fullspec1)
    spec1 = fullspec1.reshape((-1,1024))
    print('spec1.shape:', spec1.shape)
    nlen = spec1.shape[0]

    tick00 = pack00//2
    tick10 = int(lstart1*npack/2)
    tick11 = int((lstart1+nloop)*npack/2)
    pack_off = np.arange(tick10-tick00, tick11-tick00)

    for ii in range(int(nloop*npack/2)):
        cross = spec0 * spec1[ii:ii+1000]
        norm1 = np.abs(spec1[ii:ii+1000])
        cross /= (norm0*norm1)
        coeff = cross.mean(axis=0)
        med_coeff.append(np.median(np.abs(coeff)))

    ax1.plot(pack_off, med_coeff)


fh218.close()
fh219.close()



ax1.set_xlabel('packet offset')
ax1.set_ylabel('correlation coefficient')
ax1.set_title('n_intg: %d'%(npack*nloop))

f1.savefig('scorr_loop%d_start%d.png' % (nloop*nrepeat, lstart))
plt.close(f1)


f1, ax1 = plt.subplots(1,1)
ch = np.arange(1024)
ax1.plot(ch, np.abs(spec0).mean(axis=0), label='fpga218')
ax1.plot(ch, np.abs(spec1).mean(axis=0), label='fpga219')
ax1.set_xlabel('chan')
ax1.set_ylabel('abs(spec)')

f1.savefig('checkspec2_loop%d_start%d.png' %(nloop*nrepeat, lstart))
plt.close(f1)




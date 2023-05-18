#!/usr/bin/env python

from packet_func import *
import gc, time
from scipy.signal import correlate
import matplotlib.pyplot as plt

## test_packets.py
#   test synchronization of two fpga files
#   by first forming time streams (inverse FFT of the spectra)
#   and doing time-domain cross-correlation

f218 = 'fpga218_221121.bin'
f219 = 'fpga219_221121.bin'

fh218 = open(f218, 'rb')
fh219 = open(f219, 'rb')

npack = 1000
nloop = 600
lstart= 200


spec0 = np.zeros((nloop, npack//2, 1024), dtype=complex)
spec1 = np.zeros((nloop, npack//2, 1024), dtype=complex)
t0 = time.time()
for ii in range(nloop):
    pack0 = int((ii+lstart) * npack)
    print('pack0:', pack0)

    for fi in range(2):
        if (fi==0):
            fh = fh218
            out2 = spec0
        else:
            fh = fh219
            out2 = spec1
        
        tick, spec = loadSpec(fh, pack0, npack, 8192, 2)
    
        out2[ii] = spec[:,7]

    if ((ii+1) % 10 == 0):
        print('#', ii+1, 'running time:', time.time()-t0)


spec0 = np.array(spec0)
spec1 = np.array(spec1)
stream0 = np.fft.ifft(np.fft.fftshift(spec0, axes=2), axis=2)
stream1 = np.fft.ifft(np.fft.fftshift(spec1, axes=2), axis=2)
print('stream0.shape, stream1.shape', stream0.shape, stream1.shape)


f1, ax1 = plt.subplots(1,1)
flat0 = stream0.flatten()
flat1 = stream1.flatten()
nlen = len(flat0)
tcorr = correlate(flat0, flat1, mode='same')
tcorr /= np.sqrt(flat0.var()*flat1.var())*nlen

ipk = np.abs(tcorr).argmax()

ax1.plot(np.arange(nlen), np.abs(tcorr))
ax1.text(0.05, 0.90, 'i_pk=%d'%(ipk-nlen//2), transform=ax1.transAxes)
ax1.set_xlabel('samples offset')
ax1.set_ylabel('correlation coefficient')
ax1.set_title('n_intg: %d'%(npack*nloop))

f1.savefig('tcorr_loop%d_start%d.png' % (nloop, lstart))
plt.close(f1)


f1, ax1 = plt.subplots(1,1)
ch = np.arange(1024)
ax1.plot(ch, np.abs(spec0).mean(axis=(0,1)), label='fpga218')
ax1.plot(ch, np.abs(spec1).mean(axis=(0,1)), label='fpga219')
ax1.set_xlabel('chan')
ax1.set_ylabel('abs(spec)')

f1.savefig('checkspec_loop%d_start%d.png' %(nloop, lstart))
plt.close(f1)



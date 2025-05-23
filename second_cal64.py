#!/usr/bin/env python

import sys, os.path, time
import matplotlib.pyplot as plt
from glob import glob
from subprocess import call

from packet_func import *
from calibrate_func import *
from pyplanet import *
from util_func import *


inp = sys.argv[0:]
pg  = inp.pop(0)

nChan0 = 1024
nAnt = 16

## bf64 ##
nFPGA = 4
nBeam = nFPGA*nAnt
nNode = 2
nChan = nChan0//nNode
start_off = -2          # for bf64, 400-600MHz is order=2
## bf64 ##

nFrame = 1000

no_bitmap = False
aref = 0
pad  = 32   # fft padding factor; for dividing the lag sample to finer resolution
flim = [400., 800.] # data freq range in MHz
chlim = None        # FT delay finding channel range

site = 'fushan6'
src  = 'sun'

sep  = 0.5          # row spacing in meters
theta_rot_deg = 0.0     # array misalignment angle in deg

## arbitrary number
f410 = 5.0e5 # Jy
f610 = 7.0e5 # Jy

usage = '''
derive delay between FPGAs from the baseband after 1st beamform (ring0, ring1 basebands)

syntax:
    %s <.bin files> [options]

currently, the .bin files are expected to be from the same snapshot
a .check folder will be created based on the files timestamp

options are:
    -n nFrame   # instead of nPack, specify number of frames to process
                # default: %d
    -s SEP      # row spacing in meters
                # default: %.1f
                # note: the default will be overridden by site settings
                #       for longtien (2.0)
    --rot ROT   # array misalignment angle in deg
                # default: %.1f
                # note: the default will be overriden by site settings
                #       for fushan6 (-3.0) and longtien (+0.5)
    --site SITE # site name for the calculation
                # default: %s
    --aref AREF # specify the reference FPGA
                # default: %d
    --flim min max  # data freq range in MHz
                # default: %.0f %.0f
    --chlim min max # FT delay fitting channel range
    --src SRC   # source to calculate the geometric delay
                # default: %s
    --interp f410 f610
                # specify the solar flux in Jy at 410MHz and 610MHz
                # default: f410=%.4e f610=%.4e
''' % (pg, nFrame, sep, theta_rot_deg, site, aref, flim[0], flim[1], src, f410, f610)

if (len(inp) < 1):
    sys.exit(usage)

files = []
while (inp):
    k = inp.pop(0)
    if (k == '-n'):
        nFrame = int(inp.pop(0))
    elif (k == '-s'):
        sep = float(inp.pop(0))
    elif (k == '--site'):
        site = inp.pop(0)
    elif (k == '--aref'):
        aref = int(inp.pop(0))
    elif (k == '--flim'):
        flim[0] = float(inp.pop(0))
        flim[1] = float(inp.pop(0))
    elif (k == '--chlim'):
        ch1 = int(inp.pop(0))
        ch2 = int(inp.pop(0))
        chlim = [ch1, ch2]
    elif (k == '--src'):
        src = inp.pop(0)
    elif (k == '--rot'):
        theta_rot_deg = float(inp.pop(0))
    elif (k.startswith('_')):
        sys.exit('unknown option: %s'%k)
    else:
        files.append(k)


nPack = nFPGA * nFrame

#files = glob('/burstt3/disk?/data/ring*.20240808115800.bin')
#files = glob('./burstt2/ring*.20240807115800.bin')
files.sort()
print(files)
epochs = filesEpoch(files, hdver=2, meta=64)

stamps = []
for f in files:
    fb = os.path.basename(f)
    i = fb.find('ring')
    #j = fb.find('.bin')
    tmp = fb[i+6:i+20]
    stamps.append(tmp)
nStamp = len(np.unique(stamps))
if (nStamp == 1):
    stamp_name = stamps[0]
    print(stamps, stamp_name)
    epoch0 = epochs[0]
    dt0 = Time(epoch0, format='unix').to_datetime()
else:
    print(stamps)
    sys.exit('more than one timestamp found. abort!')

cdir = 'cal_%s.check'%stamp_name
call('mkdir -p %s'%cdir, shell=True)


if (site == 'fushan6'):
    theta_rot_deg = -3.0
elif (site == 'longtien'):
    theta_rot_deg = 0.5
    sep = 2.0
elif (site == 'lyudao'):
    theta_rot_deg = 1.7
    sep = 1.0

print('site:', site, 'sep:', sep, 'rot:', theta_rot_deg)

# ## correlate 4 rows of FPGA (each server) separately

spec1 = np.ma.array(np.zeros((nFPGA, nFrame, nAnt, nChan0), dtype=complex), mask=False)
fMHz = np.linspace(flim[0], flim[1], nChan0, endpoint=False)

t0 = time.time()
for i in range(nNode):
    ooff = start_off - i
    start_chan = nChan * i
    wfreq = np.arange(start_chan, start_chan+nChan)
    fh0 = open(files[i], 'rb')
    mdict0 = metaRead(fh0)
    print(mdict0)
    data0 = loadNode(fh0, 0, nPack, order_off=ooff, verbose=1, no_bitmap=no_bitmap)
    print('ring%d'%i, data0.shape, 'time', time.time()-t0, 'sec')
    spec1[:,:,:,wfreq] = data0



inten1 = (np.abs(spec1)**2).mean(axis=3)    # shape(nFPGA, nFrame, nAnt)
xx = np.arange(nFrame)
## estimate the peak beam
inten2 = inten1.mean(axis=(0,1))    # shape(nAnt,) or nBeam
bb = np.argmax(inten2)

fig, sub = plt.subplots(8,8,figsize=(16,12),sharey=True, sharex=True)

for row in range(nFPGA):
    for ai in range(nAnt):
        if (ai<=7):
            ii = row*2
            jj = ai
        else:
            ii = row*2 + 1
            jj = ai-8
        ax = sub[ii,jj]
        if (ai == bb):
            color = 'r'
        else:
            color = 'b'
        ax.plot(xx, inten1[row,:,ai], color=color, label='row%d, beam-%d'%(row,ai))
        
        ax.legend()

fig.savefig('%s/intensity_vs_beam.png'%cdir)
plt.close(fig)



## choose the peak-beam for correlation
spec = spec1[:,:,bb]
auto = np.abs(spec).mean(axis=1)
nBl = nFPGA * (nFPGA-1) // 2
coeff1 = np.zeros((nBl,nChan0), dtype=complex)
b = -1
for ai in range(nFPGA-1):
    normi = auto[ai]
    for aj in range(ai+1, nFPGA):
        normj = auto[aj]
        b += 1
        normij = np.ma.abs(spec[ai])*np.ma.abs(spec[aj])
        cross = (spec[ai] * spec[aj].conjugate()/normij)
        cross[normij==0.] = 0j
        coeff1[b] = cross.mean(axis=0)
        



fig, sub = plt.subplots(3,3,figsize=(16,9), sharey=True, sharex=True)
for ii in range(1,3):
    for jj in range(ii):
        sub[ii,jj].remove()

b = -1
for ai in range(nFPGA-1):
    ii = ai
    for aj in range(ai+1, nFPGA):
        jj = aj-1
        b += 1
        ax = sub[ii,jj]
        #tau1 = taus1[ai]-taus1[aj]
        #tau2 = taus2[ai]-taus2[aj]
        ax.plot(fMHz, np.angle(coeff1[b]), 'b.')
        #ax.plot(fMHz, np.angle(coeff1[b]*np.exp(2j*np.pi*fMHz*1e-3*tau1)), 'c.', label='SEFD*lam^2')
        ax.set_ylim(-3.3, 3.3)
        ax.grid()
        
        if (ii==jj):
            ax.set_ylabel('phase (rad)')
            ax.set_xlabel('freq (MHz)')
        
fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('%s/coeff_phase_raw.png'%cdir)
plt.close(fig)




fig, sub = plt.subplots(3,3,figsize=(16,9), sharey=True, sharex=True)
for ii in range(1,3):
    for jj in range(ii):
        sub[ii,jj].remove()

b = -1
for ai in range(nFPGA-1):
    ii = ai
    for aj in range(ai+1, nFPGA):
        jj = aj-1
        b += 1
        ax = sub[ii,jj]
        ax.plot(fMHz, np.abs(coeff1[b]), 'b.')
        ax.grid()
        
        if (ii==jj):
            ax.set_ylabel('abs(coeff)')
            ax.set_xlabel('freq (MHz)')
        
fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('%s/coeff_ampld.png'%cdir)
plt.close(fig)



## calculate SEFD
flux = f410 + (fMHz-410)*(f610-f410)/400.

SEFD1 = flux.reshape((1,nChan0))/np.abs(coeff1) * (1.-np.abs(coeff1))
lam = 2.998e8/(fMHz*1e6)  # meter
mSEFD1 = SEFD1 / (fMHz/400.)**2


fig, sub = plt.subplots(3,3,figsize=(16,9), sharey=True, sharex=True)
for ii in range(1,3):
    for jj in range(ii):
        sub[ii,jj].remove()

b = -1
for ai in range(nFPGA-1):
    ii = ai
    for aj in range(ai+1, nFPGA):
        jj = aj-1
        b += 1
        ax = sub[ii,jj]
        ax.plot(fMHz, mSEFD1[b], 'b.', label='SEFD*lam^2')
        ax.plot(fMHz, SEFD1[b], 'c.', label='SEFD')
        ax.set_ylim(0, 5e5)
        ax.legend()
        ax.grid()
        
        if (ii==jj):
            ax.set_ylabel('SEFD (Jy)')
            ax.set_xlabel('freq (MHz)')
        
fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('%s/SEFD.png'%cdir)
plt.close(fig)


## convert to Tsys
G0 = 6.0 # dB
gain = 10.**(G0/10.)

Aeff = lam**2*gain/(4.*np.pi)  # meter^2
scale = Aeff / (2*1.38e-23) * 1e-26
Tsys1 = SEFD1 * scale


fig, sub = plt.subplots(3,3,figsize=(16,9), sharey=True, sharex=True)
for ii in range(1,3):
    for jj in range(ii):
        sub[ii,jj].remove()

b = -1
for ai in range(nFPGA-1):
    ii = ai
    for aj in range(ai+1, nFPGA):
        jj = aj-1
        b += 1
        ax = sub[ii,jj]
        ax.plot(fMHz, Tsys1[b], 'b.', label='Tsys')
        ax.set_ylim(0, 20)
        ax.grid()
        
        if (ii==jj):
            ax.set_ylabel('Tsys (K)')
            ax.set_xlabel('freq (MHz)')
        
fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('%s/Tsys.png'%cdir)
plt.close(fig)



## calculate covariance and eigenmodes

cov1 = np.zeros((4,4, 1024), dtype=complex)
for i in range(4):
    cov1[i,i] = 1.+0.j

b = -1
for i in range(3):
    for j in range(i+1,4):
        b += 1
        cov1[i,j] = coeff1[b]
        cov1[j,i] = coeff1[b].conjugate()

W1, V1 = Cov2Eig(cov1, ant_flag=[])


fig, ax = plt.subplots(1,1,figsize=(10,6), sharex=True, sharey=True)

for i in range(4):
    ax.plot(fMHz, 10*np.log10(W1[:,i]))

ax.set_xlabel('freq (MHz)')
ax.set_ylabel('W (dB)')
ax.set_ylim(-15, 10)

fig.savefig('%s/eigenvalue.png'%cdir)
plt.close(fig)




Vref1 = V1[:,:,-1]

fig, s2d = plt.subplots(2,2,figsize=(16,9), sharex=True, sharey=True)
sub = s2d.flatten()

Vref1p = Vref1 / (Vref1[:,aref]/np.abs(Vref1[:,aref])).reshape((-1,1))
Vref1p[np.isnan(Vref1p)] = 0.j

for i in range(4):
    ax = sub[i]
    ax.plot(fMHz, np.angle(Vref1p[:,i]), 'b.')

fig.savefig('%s/ant_phase_raw.png'%cdir)
plt.close(fig)



## define antenna positions
# start with no misalignment
print('array misalignment (deg):', theta_rot_deg)
theta_rot = theta_rot_deg / 180. * np.pi
pos0 = np.zeros((nFPGA,3))
pos0[:,1] = np.arange(nFPGA)*sep
pos = rot(pos0, theta_rot)
#print(pos0, pos)    # debug

dtarr = [dt0]
tauGeo = get_tauGeo(dtarr, pos, body=src, site=site, aref=aref)
ns_tau = tauGeo * 1e9
print(ns_tau.shape, ns_tau)


#ns_deg = -8 # Sun offset in NS direction, deg
#ns_rad = ns_deg/180*np.pi
#ns_tau = pos * ns_rad / 2.998e8 * 1e9 # delay in ns

# correct for geometric delay
VrefTau = Vref1p * np.exp(-2j*np.pi*ns_tau.reshape(1,-1)*fMHz.reshape(-1,1)*1e-3)
if (chlim is None):
    ch1 = 0
    ch2 = nChan0
else:
    ch1 = chlim[0]
    ch2 = chlim[1]
nChan1 = ch2-ch1
VrefTau = VrefTau[ch1:ch2]
print(Vref1p.shape, VrefTau.shape)
print(np.angle(Vref1p[:10,0]))
print(np.angle(VrefTau[:10,0]))

#VrefTau -= VrefTau.mean(axis=0, keepdims=True)
FTVref = np.fft.fft(VrefTau, n=int(nChan1*pad), axis=0)
FTVref = np.fft.fftshift(FTVref, axes=0)
print(np.angle(VrefTau[:10,0]))
peak_lag = np.abs(FTVref).argmax(axis=0) - int(pad*nChan1/2)
print(peak_lag)
peak_ns = peak_lag * 1e9/(400e6*nChan1/nChan0*pad) # convert to ns
print(peak_ns)
# coarse delay correction
VrefC = VrefTau*np.exp(-2j*np.pi*peak_ns.reshape((1,-1))*fMHz[ch1:ch2].reshape((-1,1))*1e-3)
# fine delay correction
medphi = np.median(np.angle(VrefC), axis=0)
dtau = medphi / (2.*np.pi) * lam[ch1:ch2].mean() / 2.998e8 * 1e9 # ns
#print(dtau)
VrefF = VrefC*np.exp(-2j*np.pi*dtau.reshape((1,-1))*fMHz[ch1:ch2].reshape((-1,1))*1e-3)
tauCorr = -(peak_ns + dtau)

ftau = '%s/ant_delay_correct.txt'%cdir
with open(ftau, 'w') as fh:
    print('# delay correction needed are (ns):', file=fh)
    line = ' '.join(['%.3f'%x for x in tauCorr])
    print("--ds '%s'"%line, file=fh)

print('\ndelay correction needed (ns):')
print("--ds '%s'"%line)
print('')


fig, s2d = plt.subplots(2,2,figsize=(12,8), sharex=True, sharey=True)
sub = s2d.flatten()

for i in range(4):
    ax = sub[i]
    ax.plot(fMHz[ch1:ch2], np.angle(Vref1p[ch1:ch2,i]), color='black', marker='.', ls='none', label='raw')
    ax.plot(fMHz[ch1:ch2], np.angle(VrefTau[:,i]), color='gray', marker='.', ls='none', label='rem_geo')
    ax.plot(fMHz[ch1:ch2], np.angle(VrefC[:,i]), color='orange', marker='.', ls='none', label='rem_coarse')
    ax.plot(fMHz[ch1:ch2], np.angle(VrefF[:,i]), color='b', marker='.', ls='none', label='rem_fine')
    ax.text(0.05, 0.90, 'tau: %.3f'%tauCorr[i], transform=ax.transAxes)
    ax.grid()
    if (i==0):
        ax.legend()

for i in range(2):
    s2d[i,0].set_ylabel('phase (rad)')
    s2d[1,i].set_xlabel('freq (MHz)')

fig.tight_layout(rect=[0,0.03,1,0.95])
png = '%s/ant_phase_correct.png'%cdir
fig.suptitle(png)
fig.savefig(png)
plt.close(fig)




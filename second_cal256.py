#!/usr/bin/env python

import sys, os.path, time
import matplotlib.pyplot as plt
from glob import glob
from subprocess import call, run

from packet_func import *
from calibrate_func import *
from pyplanet import *
from util_func import *


inp = sys.argv[0:]
pg  = inp.pop(0)

nChan0 = 1024
nAnt = 16

## bf256 ##
nFPGA = 16
nBeam = nFPGA*nAnt
nNode = 8
nChan = nChan0//nNode
start_off = 0          # for bf64, 400-600MHz is order=2
## bf256 ##

nFrame = 4096 #1000

no_bitmap = False
aref = 0
pad  = 32   # fft padding factor; for dividing the lag sample to finer resolution

site = 'fushan6'
src  = 'sun'
cdir_path = '/burstt14/disk12/2nd_cal'
if (not os.path.isdir(cdir_path)):
    cdir_path = '.'
make_copy = True

sep  = 0.5          # row spacing in meters
theta_rot_deg = 0.0     # array misalignment angle in deg
bmax = None         # auto-determine the max-intensity beam number
read_raw = True
#flim = [400, 800]
flim = [300, 700]

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
    --aref AREF # specify the reference antenna (phase ref)
                # default: 0
    --rot ROT   # array misalignment angle in deg
                # default: %.1f
                # note: the default will be overriden by site settings
                #       for fushan6 (-3.0) and longtien (+0.5)
    --site SITE # site name for the calculation
                # default: %s
    --src SRC   # source to calculate the geometric delay
                # default: %s
    --interp f410 f610
                # specify the solar flux in Jy at 410MHz and 610MHz
                # default: f410=%.4e f610=%.4e
    --flim FMIN FMAX
                # specify the frequency range in MHz (full 1024ch range)
    --bmax BB   # specify the beam number to analyze
                # default: auto-determine based on integrated intensity
    --copy DIR  # change the copying destination for 2nd_cal results
                # default: %s
''' % (pg, nFrame, sep, theta_rot_deg, site, src, f410, f610, cdir_path)

if (len(inp) < 1):
    sys.exit(usage)

files = []
while (inp):
    k = inp.pop(0)
    if (k == '-n'):
        nFrame = int(inp.pop(0))
    elif (k == '-s'):
        sep = float(inp.pop(0))
    elif (k == '--aref'):
        aref = int(inp.pop(0))
    elif (k == '--site'):
        site = inp.pop(0)
    elif (k == '--src'):
        src = inp.pop(0)
    elif (k == '--rot'):
        theta_rot_deg = float(inp.pop(0))
    elif (k == '--bmax'):
        bmax = int(inp.pop(0))
    elif (k == '--flim'):
        fmin = float(inp.pop(0))
        fmax = float(inp.pop(0))
        flim = [fmin, fmax]
    elif (k == '--copy'):
        cdir_path = inp.pop(0)
    elif (k.startswith('_')):
        sys.exit('unknown option: %s'%k)
    else:
        files.append(k)


nPack = nFPGA * nFrame // 4 ## bf256

#files = glob('/burstt3/disk?/data/ring*.20240808115800.bin')
#files = glob('./burstt2/ring*.20240807115800.bin')
files.sort()
print(files)
nFile = len(files)
epochs = filesEpoch(files, hdver=2, meta=64)

stamps = []
for f in files:
    fb = os.path.basename(f)
    i = fb.find('ring')
    j = fb.find('.bin')
    tmp = fb[i+6:j]
    tmp = tmp.split('.')[0]
    stamps.append(tmp)
nStamp = len(np.unique(stamps))
if (nStamp == 1):
    stamp_name = stamps[0]
    print(stamps, stamp_name)
    epoch0 = epochs[0]
    dt0 = Time(epoch0, format='unix').to_datetime()
else:
    sys.exit('more than one timestamp found. abort!')


cdir = 'cal_%s.check'%(stamp_name,)
call('mkdir -p %s'%cdir, shell=True)


if (site == 'fushan6'):
    theta_rot_deg = -3.0
elif (site == 'longtien'):
    theta_rot_deg = 0.5
    sep = 2.0


fMHz = np.linspace(flim[0], flim[1], nChan0, endpoint=False)

if (os.path.isfile('%s/ant_cov_coeff.npy'%cdir)):
    read_raw = False

if (read_raw):
    # ## correlate 4 rows of FPGA (each server) separately
    spec1 = np.ma.array(np.zeros((nFPGA, nFrame, nAnt, nChan0), dtype=complex), mask=True)

    t0 = time.time()
    for i in range(nFile):
        fh0 = open(files[i], 'rb')
        mdict0 = metaRead(fh0)
        #print(mdict0)
        data0, order0 = loadNode(fh0, 0, nPack, nFPGA=nFPGA, verbose=1, no_bitmap=no_bitmap, get_order=True)
        start_chan = nChan * order0
        wfreq = np.arange(start_chan, start_chan+nChan)
        print('order:', order0, data0.shape, 'time', time.time()-t0, 'sec')
        spec1[:,:,:,wfreq] = data0
        spec1[:,:,:,wfreq].mask = False



    inten1 = np.mean(np.ma.abs(spec1)**2, axis=3)    # shape(nFPGA, nFrame, nAnt)
    xx = np.arange(nFrame)
    ## estimate the peak beam
    inten2 = np.mean(inten1, axis=(0,1))    # shape(nAnt,) or nBeam
    if (bmax is None):
        bb = np.ma.argmax(inten2)
    else:
        bb = bmax

    fig, sub = plt.subplots(16,16,figsize=(32,24),sharey=True, sharex=True)
    for row in range(nFPGA):
        for ai in range(nAnt):
            ii = row
            jj = ai
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
    auto = np.ma.abs(spec).mean(axis=1) # shape=(nFPGA, nChan0)
    nBl = nFPGA * (nFPGA-1) // 2
    coeff1 = np.ma.zeros((nBl,nChan0), dtype=complex)
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

    ## calculate covarinace
    cov1 = np.zeros((nFPGA,nFPGA, 1024), dtype=complex)
    for i in range(nFPGA):
        cov1[i,i] = 1.+0.j

    b = -1
    for i in range(nFPGA-1):
        for j in range(i+1,nFPGA):
            b += 1
            cov1[i,j] = coeff1[b]
            cov1[j,i] = coeff1[b].conjugate()

    ## save the auto and cov
    ofile = '%s/ant_auto.npy'%cdir
    np.save(ofile, auto.data)

    ofile = '%s/ant_cov_coeff.npy'%cdir
    np.save(ofile, cov1)


else: # not read_raw
    auto = np.load('%s/ant_auto.npy'%cdir)
    cov1 = np.load('%s/ant_cov_coeff.npy'%cdir)
    nBl = int(nFPGA*(nFPGA-1)/2)
    coeff1 = np.ma.zeros((nBl,nChan0), dtype=complex)
    b = -1
    for ai in range(nFPGA-1):
        for aj in range(ai+1,nFPGA):
            b += 1
            #print(b, ai,aj)
            coeff1[b] = cov1[ai,aj]



fig, sub = plt.subplots(nFPGA-1,nFPGA-1,figsize=(32,18), sharey=True, sharex=True)
for ii in range(1,nFPGA-1):
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
        ax.plot(fMHz, np.ma.angle(coeff1[b]), 'b.')
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




fig, sub = plt.subplots(nFPGA-1,nFPGA-1,figsize=(32,18), sharey=True, sharex=True)
for ii in range(1,nFPGA-1):
    for jj in range(ii):
        sub[ii,jj].remove()

b = -1
for ai in range(nFPGA-1):
    ii = ai
    for aj in range(ai+1, nFPGA):
        jj = aj-1
        b += 1
        ax = sub[ii,jj]
        ax.plot(fMHz, np.ma.abs(coeff1[b]), 'b.')
        ax.grid()
        
        if (ii==jj):
            ax.set_ylabel('abs(coeff)')
            ax.set_xlabel('freq (MHz)')
        
fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('%s/coeff_ampld.png'%cdir)
plt.close(fig)



## calculate SEFD
flux = f410 + (fMHz-410)*(f610-f410)/200.

SEFD1 = flux.reshape((1,nChan0))/np.ma.abs(coeff1) * (1.-np.ma.abs(coeff1))
lam = 2.998e8/(fMHz*1e6)  # meter
mSEFD1 = SEFD1 / (fMHz/400.)**2


fig, sub = plt.subplots(nFPGA-1,nFPGA-1,figsize=(32,18), sharey=True, sharex=True)
for ii in range(1,nFPGA-1):
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


fig, sub = plt.subplots(nFPGA-1,nFPGA-1,figsize=(32,18), sharey=True, sharex=True)
for ii in range(1,nFPGA-1):
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


## solve eigenmodes
W1, V1 = Cov2Eig(cov1, ant_flag=[])

fig, ax = plt.subplots(1,1,figsize=(10,6), sharex=True, sharey=True)

for i in range(nFPGA):
    ax.plot(fMHz, 10*np.log10(W1[:,i]))

ax.set_xlabel('freq (MHz)')
ax.set_ylabel('W (dB)')

fig.savefig('%s/eigenvalue.png'%cdir)
plt.close(fig)




#Vref1 = V1[:,:,-1]
Vref1 = np.ma.array(V1[:,:,-1], mask=~np.isfinite(V1[:,:,-1]))

fig, s2d = plt.subplots(4,4,figsize=(32,18), sharex=True, sharey=True)
sub = s2d.flatten()

Vref1p = Vref1 / (Vref1[:,aref]/np.abs(Vref1[:,aref])).reshape((-1,1))

for i in range(nFPGA):
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
#print(ns_tau.shape)
print('tauGeo (ns):', ns_tau)


#ns_deg = -8 # Sun offset in NS direction, deg
#ns_rad = ns_deg/180*np.pi
#ns_tau = pos * ns_rad / 2.998e8 * 1e9 # delay in ns

# correct for geometric delay
VrefTau = Vref1p * np.exp(-2j*np.pi*ns_tau.reshape((1,-1))*fMHz.reshape((-1,1))*1e-3)

ofile = '%s/eigenvector_TauGeo_correct.npy'%cdir
np.save(ofile, VrefTau.data)

## plot the normalization and eigenvector for each row
fig, s2d = plt.subplots(4,4,figsize=(10,8), sharex=True, sharey=True)
sub = s2d.flatten()
med_auto = np.ma.median(auto, axis=0, keepdims=True)
auto2 = auto/med_auto
fcal2 = '%s/solution_2ndCal.npz'%(cdir,)
np.savez(fcal2, auto2=auto2, tau_i=VrefTau)
# auto2: relative ampld btw rows
# tau_i: eigenvector including instrument delay and weighting between rows

med_auto2 = np.median(auto2, axis=1)
med_Vamp = np.median(np.abs(VrefTau), axis=0)
print('norm:', med_auto2)
print('Vamp:', med_Vamp)
fnorm = '%s/ant_norm_correct.txt'%cdir
with open(fnorm, 'w') as fh:
    print('# normalization correction needed are (ns):', file=fh)
    line2 = ' '.join(['%.3f'%x for x in 1/med_auto2])
    print("--norm '%s'"%line2, file=fh)

for i in range(nFPGA):
    ax = sub[i]
    ax.plot(fMHz, 1/auto2[i], label='rel.gain')
    ax.plot(fMHz, np.abs(VrefTau[:,i])/0.25, label='abs(V)/0.25')
    if (i == 0):
        ax.legend()
    if (i>=12):
        ax.set_xlabel('freq (MHz)')
    if (i%4==0):
        ax.set_ylabel('rel.strength')
    ax.set_ylim(0, 2)

fig.tight_layout(rect=[0,0.03,1,0.95])
fig.subplots_adjust(wspace=0, hspace=0)
fig.savefig('%s/weighting.png'%cdir)
plt.close(fig)


## delay fitting
FTVref = np.fft.fft(VrefTau, n=int(nChan0*pad), axis=0)
FTVref = np.fft.fftshift(FTVref, axes=0)
peak_lag = np.abs(FTVref).argmax(axis=0) - int(pad*nChan0/2)
#print(peak_lag)
peak_ns = peak_lag * 1e9/400e6 / pad # convert to ns
#print(peak_ns)
# coarse delay correction
VrefC = VrefTau*np.exp(-2j*np.pi*peak_ns.reshape((1,-1))*fMHz.reshape((-1,1))*1e-3)
# fine delay correction
#medphi = np.median(np.angle(VrefC), axis=0)
med_r = np.median(VrefC.real, axis=0)
med_i = np.median(VrefC.imag, axis=0)
medphi = np.angle(med_r + 1.j*med_i)
dtau = medphi / (2.*np.pi) * lam.mean() / 2.998e8 * 1e9 # ns
#print(dtau)
VrefF = VrefC*np.exp(-2j*np.pi*dtau.reshape((1,-1))*fMHz.reshape((-1,1))*1e-3)
tauCorr = -(peak_ns + dtau)
#tauStr = ', '.join(['%.3f'%x for x in tauCorr])

ftau = '%s/ant_delay_correct.txt'%cdir
with open(ftau, 'w') as fh:
    print('# delay correction needed are (ns):', file=fh)
    line = ' '.join(['%.3f'%x for x in tauCorr])
    print("--ds '%s'"%line, file=fh)

print('\ndelay correction needed (ns):')
print("--ds '%s'"%line, "--norm '%s'"%line2)
print('')


fig, s2d = plt.subplots(4,4,figsize=(24,16), sharex=True, sharey=True)
sub = s2d.flatten()

for i in range(nFPGA):
    ax = sub[i]
    ax.plot(fMHz, np.angle(VrefTau[:,i]), color='gray', marker='.', ls='none', label='rem_geo')
    ax.plot(fMHz, np.angle(VrefC[:,i]), color='orange', marker='.', ls='none', label='rem_coarse')
    ax.plot(fMHz, np.angle(VrefF[:,i]), color='b', marker='.', ls='none', label='rem_fine')
    ax.text(0.05, 0.90, 'tau: %.3f'%tauCorr[i], transform=ax.transAxes)
    ax.grid()
    if (i==0):
        ax.legend()

for i in range(4):
    s2d[i,0].set_ylabel('phase (rad)')
    s2d[1,i].set_xlabel('freq (MHz)')

fig.tight_layout(rect=[0,0.03,1,0.95])
png = '%s/ant_phase_correct.png'%cdir
fig.suptitle(png)
fig.savefig(png)
plt.close(fig)


if (make_copy and os.path.isdir(cdir_path)):
    if (cdir_path == '.'):
        print('skip copying to self')
    else:
        #cdir2 = '%s/%s'%(cdir_path, cdir)
        #if (not os.path.isdir(cdir2)):
        cmd = 'rsync -avu %s %s/'%(cdir, cdir_path)
        print(cmd)
        res = run(cmd, shell=True, capture_output=True)
        print(res.stdout)


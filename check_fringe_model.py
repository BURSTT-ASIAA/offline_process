#!/usr/bin/env python
'''
check the fringe-stopping model used in show_fpga_window.py

syntax:
    check_fringe_model.py <.vish5_file> [CHMIN CHMAX]

    default channel range: 0 512
    outputs (plots and fs_check.npz) go to <.output>/fs_check/

    for each baseline, the fringe-stopped data (coeff - R)*conj(M)*conj(phase(A)) is examined:
    - residual delay and phase per window (should be flat if the fringe model is adequate)
    - coherence over time |<z>|/<|z|> (1 = no loss from the fringe model)
    - antenna-position corrections fitted to the residual delays, and the refit with them
    - source/static degeneracy kappa (short baselines with little fringe rotation)
    - source amplitude vs time compared with the beam attenuation model
'''
import sys, os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if (len(sys.argv) < 2):
    sys.exit(__doc__)
fvis = sys.argv[1]
chmin, chmax = 0, 512
if (len(sys.argv) >= 4):
    chmin, chmax = int(sys.argv[2]), int(sys.argv[3])
odir = os.path.join(fvis.replace('.vish5', '.output'), 'fs_check')
os.makedirs(odir, exist_ok=True)
with h5py.File(fvis, 'r') as f:
    d = {k.replace('.','_'):f[k][()] for k in ['winCoeff','winCoeff.mask','model_tau_m','atten','freq','winSec','el','az','ant_xyz']}
    if ('winCoeffClkCorr' in f):    # same choice of coeff as show_fpga_window.py
        d['winCoeff'] = f['winCoeffClkCorr'][()]
        d['winCoeff_mask'] = f['winCoeffClkCorr.mask'][()] if 'winCoeffClkCorr.mask' in f else np.zeros(d['winCoeff'].shape, bool)
ch = slice(chmin, chmax)
c0   = np.ma.array(d['winCoeff'], mask=d['winCoeff_mask'])[:,:,ch]
freq = d['freq'][ch]*1e6                     # Hz
f0   = freq.mean()
tau_m = d['model_tau_m']                     # (nWin, nBl) meters
att  = d['atten']
el, az = d['el'], d['az']
sec  = d['winSec']
xyz  = d['ant_xyz']
nWin, nBl, nChan = c0.shape
nAnt = xyz.shape[0]
pairs = [(i,j) for i in range(nAnt-1) for j in range(i+1,nAnt)]
blen = np.array([np.linalg.norm(xyz[j]-xyz[i]) for i,j in pairs])
cl = 2.998e8

# unit vector consistent with show_fpga_window.py (x=E, y=N, z=up; sign flipped)
s_hat = -np.array([np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el)]).T   # (nWin, 3)
def geo_tau(pos):
    return np.array([np.dot(s_hat, pos[j]-pos[i]) for i,j in pairs]).T   # (nWin, nBl) m
print('geometry check, max|tau_model - recomputed| = %.2e m' % np.abs(geo_tau(xyz)-tau_m).max())

w = (~np.ma.getmaskarray(c0)).astype(float)
c = np.ma.filled(c0, 0.)

def fringe_stop(tau):
    '''joint fit c = A*att*M + R per (bl,ch); return A, R, z (fringe-stopped, rotated by phase(A))'''
    M = np.exp(-2.j*np.pi*freq.reshape((1,1,-1))*(tau/cl)[:,:,np.newaxis])
    G = att.reshape((-1,1,1))*M
    n = w.sum(0); sG = (w*G).sum(0); sGG = (w*np.abs(G)**2).sum(0)
    sc = (w*c).sum(0); scG = (w*c*G.conj()).sum(0)
    det = sGG*n - np.abs(sG)**2
    A = (scG*n - sc*sG.conj())/det
    R = (sc - A*sG)/n
    z = (c - R)*M.conj()*(A.conj()/np.abs(A))*w
    kappa = np.abs(sG)**2/(n*sGG)       # template/constant degeneracy, 0=separable, 1=degenerate
    # matched-filter channel weights |A|/sigma^2, sigma from the quadrature scatter over time
    # (down-weights RFI channels whose discarded part fluctuates)
    sig = np.sqrt((z.imag**2).sum(0)/np.maximum(n,1))
    wch = np.abs(A)/sig**2
    wch /= np.median(wch, axis=1, keepdims=True)
    return A, R, z*wch[np.newaxis], kappa

def window_delay(z, pad=16):
    '''per (win,bl): residual delay (s) and phase at f0
    coarse: delay spectrum peak; fine: weighted LS of the phase slope across channels'''
    nf = nChan*pad
    df = freq[1]-freq[0]
    spec = np.fft.fft(z, n=nf, axis=2)
    k = np.abs(spec).argmax(axis=2)
    k = np.where(k > nf//2, k-nf, k)
    dtau = -k/(df*nf)
    fr = (freq-f0).reshape((1,1,-1))
    for it in range(2):
        zr = z*np.exp(2.j*np.pi*fr*dtau[:,:,np.newaxis])
        zc = zr.sum(2)
        ph = np.angle(zr*np.conj(zc)[:,:,np.newaxis])       # small residual phase per channel
        wt = np.abs(zr)**2
        sw = wt.sum(2); sf = (wt*fr).sum(2); sff = (wt*fr**2).sum(2); sp = (wt*ph).sum(2); sfp = (wt*fr*ph).sum(2)
        slope = (sw*sfp - sf*sp)/(sw*sff - sf**2)
        dtau = dtau - slope/(2*np.pi)
    zc = (z*np.exp(2.j*np.pi*fr*dtau[:,:,np.newaxis])).sum(2)
    return dtau, np.angle(zc)

def metrics(z):
    zt = z.sum(2)/np.maximum(w.sum(2),1)       # channel-avg per window (nWin, nBl)
    eta = np.abs(zt.sum(0))/np.abs(zt).sum(0)  # coherence over time
    prms = np.sqrt(np.mean(np.angle(zt)**2, axis=0))
    return zt, eta, prms

## 1. original model
A1, R1, z1, kap = fringe_stop(tau_m)
dt1, ph1 = window_delay(z1)
zt1, eta1, prms1 = metrics(z1)

## 2. fit antenna position corrections from the residual delays
# dtau*c = (dp_j - dp_i).(s(t) - <s>)  (time-mean absorbed by the per-channel A)
ds = s_hat - s_hat.mean(0)
y  = (dt1 - dt1.mean(0))*cl                # (nWin, nBl) m
X = np.zeros((nWin, nBl, 3*(nAnt-1)))      # ant0 fixed
for b,(i,j) in enumerate(pairs):
    if (j>0): X[:,b,3*(j-1):3*j] += ds
    if (i>0): X[:,b,3*(i-1):3*i] -= ds
X2 = X.reshape((-1, 3*(nAnt-1))); y2 = y.ravel()
# the solar track only spans part of the 3D direction space: truncated SVD
sv = np.linalg.svd(ds, compute_uv=False)
print('singular values of the s(t) variation (E,N,U mix):', np.round(sv/sv[0],3))
print('principal direction of s(t) variation (E,N,U):', np.round(np.linalg.svd(ds)[2][0],3))
rc = 0.2     # keep only the well-constrained direction(s)
sol, *_ = np.linalg.lstsq(X2, y2, rcond=rc)
for it in range(3):     # clip outliers
    res = y2 - X2.dot(sol)
    good = np.abs(res) < 4*1.4826*np.median(np.abs(res))
    sol, *_ = np.linalg.lstsq(X2[good], y2[good], rcond=rc)
dp = np.vstack([np.zeros(3), sol.reshape((nAnt-1,3))])
dp -= dp.mean(0)        # remove common offset (unconstrained)
xyz2 = xyz + dp
res = (y2 - X2.dot(sol)).reshape((nWin,nBl))
print('residual delay rms: before %.3f ns, after pos-fit %.3f ns' % (y.std()/cl*1e9, res.std()/cl*1e9))
print('antenna position corrections (cm):')
for a in range(nAnt):
    print('  ant%02d  dx=%6.2f dy=%6.2f dz=%6.2f' % (a, *(dp[a]*100)))

## 3. refit with the corrected positions
tau2 = geo_tau(xyz2)
A2, R2, z2, kap2 = fringe_stop(tau2)
dt2, ph2 = window_delay(z2)
zt2, eta2, prms2 = metrics(z2)

# antenna-based? solve per-window antenna delays from dt1 and check closure residual
Am = np.zeros((nBl, nAnt))
for b,(i,j) in enumerate(pairs):
    Am[b,i] = -1.; Am[b,j] = 1.
P = np.linalg.pinv(Am)
clo1 = dt1 - (Am.dot(P.dot(dt1.T))).T
clo2 = dt2 - (Am.dot(P.dot(dt2.T))).T

# expected phase rms from thermal noise: per-channel scatter of the quadrature part
# effective: from the weighted channel combination, sigma_zt = sqrt(sum wch^2 sig^2)/sum wch
zq = z1.imag
sig_t = np.sqrt((zq**2).sum(2))/np.abs(z1).sum(2).clip(1e-30)*np.abs(z1).sum(2)/np.maximum(w.sum(2),1)
exp_prms = np.median(sig_t/np.abs(zt1), axis=0)
exp_dtau = exp_prms/(2*np.pi)/(freq-f0).std()
print('expected (noise) phase rms med %.3f rad, delay rms med %.3f ns' % (np.median(exp_prms), np.median(exp_dtau)*1e9))
print('\n                         original      pos-corrected')
print('coherence eta  med      %.3f          %.3f' % (np.median(eta1), np.median(eta2)))
print('coherence eta  min      %.3f          %.3f' % (eta1.min(), eta2.min()))
print('phase rms (rad) med     %.3f          %.3f' % (np.median(prms1), np.median(prms2)))
print('delay rms (ns) med      %.3f          %.3f' % (np.median(dt1.std(0))*1e9, np.median(dt2.std(0))*1e9))
print('non-closing delay rms   %.3f          %.3f ns' % (clo1.std()*1e9, clo2.std()*1e9))
print('med|A|                  %.4f         %.4f' % (np.median(np.abs(A1)), np.median(np.abs(A2))))
print('med|R|                  %.4f         %.4f' % (np.median(np.abs(R1)), np.median(np.abs(R2))))
# phase-at-f0 vs geometric expectation (-2 pi f0 dtau), using time-mean-removed
pg = -2*np.pi*f0*(dt1-dt1.mean(0))
pm = np.angle(np.exp(1j*(ph1)))
cc = np.corrcoef(pg.ravel(), pm.ravel())[0,1]
print('corr(phase@f0, -2pi f0 dtau) = %.2f   (~1 if residual is geometric/true delay)' % cc)

np.savez(os.path.join(odir,'fs_check.npz'), dp=dp, xyz2=xyz2, eta1=eta1, eta2=eta2, dt1=dt1, dt2=dt2,
         kappa=kap, A1=A1, A2=A2, R1=R1, R2=R2, blen=blen)

## plots
blk = np.argsort(blen)
cmap = plt.get_cmap('viridis')
fig, sub = plt.subplots(2,3, figsize=(18,10))
for k, (dt, tag) in enumerate([(dt1,'original model'), (dt2,'pos-corrected model')]):
    ax = sub[k,0]
    for b in range(nBl):
        ax.plot(sec/3600, (dt[:,b]-dt[:,b].mean())*1e9, color=cmap(blen[b]/blen.max()), lw=0.6)
    ax.set_ylim(-2,2)
    ax.set_xlabel('Time (hr)'); ax.set_ylabel('residual delay - mean (ns)')
    ax.set_title('fringe-stopped residual delay, %s\n(color: baseline length)'%tag)
    ax = sub[k,1]
    for b in range(nBl):
        ax.plot(sec/3600, np.angle(zt1[:,b] if k==0 else zt2[:,b]), color=cmap(blen[b]/blen.max()), lw=0.6)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel('Time (hr)'); ax.set_ylabel('residual phase (rad), ch-avg')
    ax.set_title('fringe-stopped residual phase, %s'%tag)
ax = sub[0,2]
ax.scatter(blen, eta1, s=10, label='original')
ax.scatter(blen, eta2, s=10, marker='x', label='pos-corrected')
ax.set_xlabel('baseline length (m)'); ax.set_ylabel('coherence |<z>|/<|z|>')
ax.set_title('coherence over %.1f hr, ch%d-%d'%(sec[-1]/3600, chmin, chmax))
ax.legend()
ax = sub[1,2]
for k_, (lab, col) in enumerate([('300 MHz', 0), ('400 MHz', nChan//2), ('500 MHz', nChan-1)]):
    ax.scatter(blen, kap[:,col], s=10, label=lab)
ax.set_yscale('log')
ax.set_xlabel('baseline length (m)'); ax.set_ylabel(r'$\kappa=|\Sigma G|^2/(n\Sigma|G|^2)$')
ax.set_title('source/static degeneracy (1 = inseparable)')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(odir,'fringe_model_check.png'))
plt.close(fig)

fig, sub = plt.subplots(1,3, figsize=(16,5))
for k, lab in enumerate(['dx (E)', 'dy (N)', 'dz (up)']):
    ax = sub[k]
    ax.bar(np.arange(nAnt), dp[:,k]*100)
    ax.set_xlabel('Ant'); ax.set_ylabel('%s correction (cm)'%lab)
    ax.axhline(0, color='k', lw=0.5)
fig.suptitle('antenna position corrections fitted to the fringe-stopped residual delays')
fig.tight_layout()
fig.savefig(os.path.join(odir,'ant_pos_correction.png'))
plt.close(fig)

# amplitude vs time: beam (atten) model check
fig, sub = plt.subplots(1,2, figsize=(14,5))
ratio = np.abs(zt1)/np.median(np.abs(A1),axis=1).reshape((1,-1))
ax = sub[0]
ax.plot(sec/3600, np.median(ratio,axis=1), 'k-', lw=2, label='data: |z|/|A| (med over bl)')
ax.plot(sec/3600, att, 'r--', label='atten model')
ax.set_xlabel('Time (hr)'); ax.set_ylabel('relative source amplitude')
ax.legend(); ax.set_title('beam attenuation model check')
ax = sub[1]
ax.plot(freq/1e6, np.median(np.abs(A1),axis=0), label='|A| original')
ax.plot(freq/1e6, np.median(np.abs(A2),axis=0), label='|A| pos-corrected')
ax.plot(freq/1e6, np.median(np.abs(R1),axis=0), label='|R| original')
ax.plot(freq/1e6, np.median(np.abs(R2),axis=0), label='|R| pos-corrected')
ax.set_yscale('log'); ax.set_xlabel('Frequency (MHz)'); ax.set_ylabel('median over baselines')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(odir,'amp_check.png'))
plt.close(fig)

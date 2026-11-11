#!/usr/bin/env python
'''
re-form one BF512 beam (NS stage applied offline to the EW-beamformed trigger baseband)
and dedisperse a pulse, optionally after up-channelizing each 390-kHz channel with an N-point FFT
("quasi-coherent": intra-channel smearing reduced N times, time resolution N x 2.56 us)

syntax:
    crab_gp_bf512.py <YYYYMMDD> <HHMMSS.sss> <beam_id> [options]

    the time is the reported arrival time at 300 MHz (UTC); beam_id = ns*32 + ew_col
options:
    --dm DM         dispersion measure (default 56.72)
    --toff SEC      empirical offset added to the reported time (default -0.016)
    --pad SEC       half window around the expected arrival in each channel (default 0.05)
    --nfft N        up-channelization FFT length (default 16; 1 = none)
    --tbin SEC      output time bin (default 0.25e-3, rounded to a multiple of N x 2.56 us)
    --flip          reverse the sub-channel order within each channel (spectral-inversion test)
    --foff F        sub-channel frequency offset in units of the 390-kHz channel (default 0:
                    channel k centred at 300 + 0.390625 k MHz; 0.5 = k is the lower edge)
    --orders 'o..'  orders to process (default all 16)
    --dcal DIR      2nd_cal dir of the NS matrix (default: the latest cal_YYYYMMDD_HHMMSSZ.check
                    with a solution_2ndCal.npz in CALDIR that is earlier than the event)
    --caldir CALDIR folder searched for the default dcal (default /burstt12/disk12/2nd_cal)
    --odir DIR      output dir (default .); beam voltages are cached in DIR/vcache
    --glob PAT      baseband file pattern, {o} = order, {ts} = YYYYMMDD_HHMMSS
                    (default /burstt1*/disk*/baseband/baseband*_order{o}_{ts}Z_trig)
    --tag STR       extra suffix for the output names
'''
import os
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys, time
from glob import glob
from datetime import datetime, timezone
import numpy as np
from bf512_bfm2 import bfm2

inp = sys.argv[1:]
if len(inp) < 3:
    sys.exit(__doc__)
ymd, hms, beam = inp.pop(0), inp.pop(0), int(inp.pop(0))
dm, toff, pad, nfft, tbin, flip, foff = 56.72, -0.016, 0.05, 16, 0.25e-3, False, 0.
orders = list(range(16))
dcal = None
caldir = '/burstt12/disk12/2nd_cal'
odir = '.'
pat = '/burstt1*/disk*/baseband/baseband*_order{o}_{ts}Z_trig'
xtag = ''
while inp:
    k = inp.pop(0)
    if k == '--dm': dm = float(inp.pop(0))
    elif k == '--toff': toff = float(inp.pop(0))
    elif k == '--pad': pad = float(inp.pop(0))
    elif k == '--nfft': nfft = int(inp.pop(0))
    elif k == '--tbin': tbin = float(inp.pop(0))
    elif k == '--flip': flip = True
    elif k == '--foff': foff = float(inp.pop(0))
    elif k == '--orders': orders = [int(x) for x in inp.pop(0).split()]
    elif k == '--dcal': dcal = inp.pop(0)
    elif k == '--caldir': caldir = inp.pop(0)
    elif k == '--odir': odir = inp.pop(0)
    elif k == '--glob': pat = inp.pop(0)
    elif k == '--tag': xtag = inp.pop(0)
    else: sys.exit('unknown option %s' % k)
cdir = os.path.join(odir, 'vcache')
os.makedirs(cdir, exist_ok=True)

KDM = 4148.808
TF = 2.56e-6                 # s per frame
DF = 0.390625                # MHz per channel
PL, HD, NF, NCH, NB1 = 3904, 64, 16, 60, 32
PADX = 0.06                  # half window kept in the voltage cache (>= pad)
rows = np.arange(14)
ns, col = beam // NB1, beam % NB1
par = col % 2
t300 = datetime.strptime(ymd + hms, '%Y%m%d%H%M%S.%f').replace(tzinfo=timezone.utc).timestamp()
tref = t300 + toff           # expected 300-MHz arrival
ts = ymd + '_' + hms[:6]
tag0 = 'gp_%s_%s_b%d' % (ymd, hms.replace('.', ''), beam)
print('event %s %s  beam %d = NS %d, EW col %d (parity %d)  DM %.2f  t300+toff = %.3f' % (ymd, hms, beam, ns, col, par, dm, tref))
if dcal is None:
    cands = []
    for d in glob(os.path.join(caldir, 'cal_*Z.check')):
        try:
            tc = datetime.strptime(os.path.basename(d)[4:19], '%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
        if tc < t300 and os.path.isfile(os.path.join(d, 'solution_2ndCal.npz')):
            cands.append((tc, d))
    if not cands:
        sys.exit('no 2nd_cal before the event in %s; use --dcal' % caldir)
    dcal = max(cands)[1]
print('NS matrix from', dcal, '(note: assumes it was loaded before the event; check the bfm2 logs)')
if pad > PADX:
    sys.exit('pad must be <= %.2f (cache window)' % PADX)

# byte offsets of EW column `col` inside a packet payload: payload element e = ((s*30+g)*32+a)*2+c,
# channel = 2g+c, stored at byte e^1 (even/odd swap)
s_, g_, c_ = np.meshgrid(np.arange(2), np.arange(30), np.arange(2), indexing='ij')
e = ((s_*30 + g_)*NB1 + col)*2 + c_
bidx = HD + (e ^ 1).reshape(2, 60)                     # (stack, ch)
LUT = np.empty(256, np.complex64)
for b in range(256):
    i, q = b & 15, b >> 4
    LUT[b] = (i - 16*(i >= 8)) + 1j*(q - 16*(q >= 8)) + (0.5 + 0.5j)


def pkt_time(h):
    cnt = int(np.frombuffer(h[0:8].tobytes(), '<i8')[0])
    ep = int(np.frombuffer(h[16:20].tobytes(), '<u4')[0])
    return cnt, 2.0 + ep + cnt*TF/8.


def extract(o):
    '''beam voltages (frames, 60 ch) covering every channel's arrival +- PADX; cached'''
    fc = os.path.join(cdir, '%s_o%02d.npz' % (tag0, o))
    if os.path.isfile(fc):
        d = np.load(fc, allow_pickle=True)
        return d['y'], float(d['tfirst']), d['info'].item()
    fl = glob(pat.format(o='%02d' % o, ts=ts))
    if not fl:
        print('order %02d: no file' % o); return None
    fn = fl[0]
    m = np.memmap(fn, np.uint8, 'r')
    nCk = (m.size - HD)//(PL*NF)
    P = m[HD:HD + nCk*NF*PL].reshape(nCk, NF, PL)
    i0 = next(i for i in range(64) if bytes(P[i // 14, i % 14, 58:64]) in (b'BURSTT', b'RSTTTT'))
    c0, t0 = pkt_time(P[i0 // 14, i0 % 14]); k0 = i0 // 14
    t0 -= k0*2*TF
    ilast = next(i for i in range(nCk - 1, 0, -1) if bytes(P[i, 0, 58:64]) == b'BURSTT')
    c1, _ = pkt_time(P[ilast, 0])
    lin = (c1 - c0) - 16*(ilast - k0)
    f_o = 300. + DF*np.arange(o*NCH - 1, (o + 1)*NCH + 1)
    tch = tref - KDM*dm*(1/300.**2 - 1/f_o**2)
    ck0 = max(int(np.floor((tch.min() - PADX - t0)/TF))//2, 0)
    ck1 = min(int(np.ceil((tch.max() + PADX - t0)/TF))//2 + 2, nCk)
    if ck0 >= ck1:
        print('order %02d: window outside file' % o); return None
    M = bfm2(dcal, o)[par][:, ns, :][:, rows]           # (ch, row)
    y = np.zeros(((ck1 - ck0)*2, NCH), np.complex64)
    nbad = 0
    blk = 4096
    for a in range(ck0, ck1, blk):
        b = min(a + blk, ck1)
        Q = np.asarray(P[a:b, rows])                    # (n, 14, PL)
        ok = (Q[:, :, 58:64] == np.frombuffer(b'BURSTT', np.uint8)).all(-1)
        nbad += (~ok).sum()
        x = LUT[Q[:, :, bidx]]                          # (n, 14, 2, 60)
        x *= ok[:, :, None, None]
        x = x.transpose(0, 2, 1, 3).reshape(-1, len(rows), NCH)
        y[(a - ck0)*2:(b - ck0)*2] = np.einsum('frc,cr->fc', x, M)
    del m, P
    tfirst = t0 + ck0*2*TF
    info = dict(file=fn, t0=t0, nck=nCk, lin=lin, nbad=int(nbad), span=(ck1 - ck0)*2*TF)
    np.savez(fc, y=y, tfirst=tfirst, info=info)
    print('order %02d: %s  file start %s, %.3f s read, bad pkts %d, counter linearity %d' %
          (o, os.path.basename(fn), datetime.fromtimestamp(t0, timezone.utc).strftime('%H:%M:%S.%f')[:-3],
           info['span'], nbad, lin), flush=True)
    return y, tfirst, info


tsub = nfft*TF
nb = max(int(round(tbin/tsub)), 1)
nOut = int(round(2*pad/(nb*tsub)))
tgrid = (np.arange(nOut) + 0.5)*nb*tsub - pad          # relative to tref, dedispersed to 300 MHz
nsub = NCH*nfft
dyn = np.full((nOut, 16*nsub), np.nan, np.float32)
fsub = (300. + DF*(np.arange(16*NCH)[:, None] + foff + (np.arange(nfft)[None, :] - nfft//2)/nfft)).reshape(-1)
print('nfft %d: sub-channel %.1f kHz, sample %.2f us, output bin %d x = %.3f ms%s' %
      (nfft, DF/nfft*1e3, tsub*1e6, nb, nb*tsub*1e3, ', flipped' if flip else ''))
info = {}
T0 = time.time()
for o in orders:
    r = extract(o)
    if r is None:
        continue
    y, tfirst, info[o] = r
    nblk = y.shape[0]//nfft
    Y = y[:nblk*nfft].reshape(nblk, nfft, NCH)
    if nfft > 1:
        Y = np.fft.fftshift(np.fft.fft(Y, axis=1), axes=1)
        if flip:
            Y = Y[:, ::-1]
    pw = (Y.real**2 + Y.imag**2).transpose(0, 2, 1).reshape(nblk, nsub).astype(np.float32)   # (t, ch*nfft)
    fo = fsub[o*nsub:(o+1)*nsub]
    tarr = tref - KDM*dm*(1/300.**2 - 1/fo**2)
    st = np.floor((tarr - pad - tfirst)/tsub).astype(int)
    for j in range(nsub):
        s = st[j]
        if s < 0 or s + nOut*nb > nblk:
            continue
        dyn[:, o*nsub + j] = pw[s:s + nOut*nb, j].reshape(nOut, nb).mean(1)
    del y, Y, pw
print('dedispersion done [%.0f s]' % (time.time() - T0))

# normalize per sub-channel by off-pulse statistics (outer 40% of the window)
off = np.abs(tgrid) > 0.6*pad
# mean/std, not median/MAD: few-sample chi^2 bins are skewed, and a median offset of ~0.1 sigma per
# sub-channel adds up to a large false baseline over ~15k sub-channels
mu = np.nanmean(dyn[off], 0)
sd = np.nanstd(dyn[off], 0)
with np.errstate(invalid='ignore', divide='ignore'):
    dn = (dyn - mu)/sd
good = np.isfinite(dn).all(0) & (sd > 0)
rr = np.full(dn.shape[1], np.inf)
rr[good] = np.max(np.abs(dn[off][:, good]), 0)
good &= rr < 8.             # drop sub-channels with off-pulse outliers (RFI)
prof = np.nansum(dn[:, good], 1)/np.sqrt(good.sum())
prof = (prof - np.mean(prof[off]))/np.std(prof[off])     # S/N against the measured off-pulse scatter
pk = int(np.argmax(prof))
w = np.abs(tgrid - tgrid[pk]) < 1.5e-3
print('sub-channels used %d / %d' % (good.sum(), good.size))
print('peak S/N %.1f at %+.3f ms from t300+toff (t300 %+.3f ms); S/N within +-1.5 ms window sum %.1f' %
      (prof[pk], tgrid[pk]*1e3, (tgrid[pk] + toff)*1e3, prof[w].sum()/np.sqrt(w.sum())))
# per-channel S/N at the peak (sub-channels averaged back to the 390-kHz grid) for the plot
dch = np.nanmean(np.where(good, dn, np.nan).reshape(nOut, -1, nfft), 2)*np.sqrt(nfft)
tag = tag0 + ('_n%d' % nfft) + ('_flip' if flip else '') + xtag
np.savez(os.path.join(odir, tag + '.npz'), dch=dch, good=good, fsub=fsub, tgrid=tgrid, prof=prof,
         t300=t300, toff=toff, dm=dm, beam=beam, nb=nb, nfft=nfft, flip=flip, foff=foff, info=info)

import warnings; warnings.simplefilter('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(8, 9), sharex=True, gridspec_kw=dict(height_ratios=[1, 3]))
tms = tgrid*1e3
ax[0].plot(tms, prof, 'k-', lw=0.8)
ax[0].set_ylabel('S/N (band sum)')
ax[0].set_title('%s %s UTC  beam %d (NS %d, EW %d)  DM %.2f\nnfft %d (%.1f kHz), bin %.3f ms; t=0: reported t300 %+.0f ms; peak S/N %.1f at %+.2f ms' %
                (ymd, hms, beam, ns, col, dm, nfft, DF/nfft*1e3, nb*tsub*1e3, toff*1e3, prof[pk], tms[pk]), fontsize=10)
nf = 8
dd = np.nanmean(dch.reshape(nOut, -1, nf), 2)
ff = (300. + DF*np.arange(16*NCH)).reshape(-1, nf).mean(1)
v = np.nanpercentile(dd, [1, 99.5])
ax[1].imshow(dd.T, origin='lower', aspect='auto', extent=[tms[0], tms[-1], ff[0], ff[-1]], vmin=v[0], vmax=v[1], cmap='viridis')
ax[1].set_ylabel('freq (MHz)'); ax[1].set_xlabel('time (ms), dedispersed to 300 MHz')
fig.tight_layout()
fig.savefig(os.path.join(odir, tag + '.png'), dpi=110)
print('saved', os.path.join(odir, tag + '.{npz,png}'))

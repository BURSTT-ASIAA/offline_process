#!/usr/bin/env python
'''
re-form one BF512 or BF256 beam (NS stage applied offline to the EW-beamformed trigger baseband)
and dedisperse a pulse, optionally after up-channelizing each 390-kHz channel with an N-point FFT
("quasi-coherent": intra-channel smearing reduced N times, time resolution N x 2.56 us)

syntax:
    pulse_beamform.py <YYYYMMDD> <HHMMSS.sss> <beam_id> [options]
    pulse_beamform.py --bf256 <YYYYMMDD> <HHMMSS.sss> <beam_id> [options]

    the time is the reported arrival time at 300 MHz (UTC)
    BF512: beam_id = ns*32 + ew_col; 16 orders x 60 ch; files /burstt1*/disk*/baseband/baseband*_orderOO_<ts>Z_trig
    BF256: beam_id = ns*16 + ew_col;  8 orders x 128 ch; files /burstt1*/disk*/data/ring?.<ts>Z.trig.oO.bin
options:
    --bf256         BF256 mode (option may appear anywhere)
    --dm DM         dispersion measure (default 56.72)
    --toff SEC      empirical offset added to the reported time (default -0.016)
    --pad SEC       half window around the expected arrival in each channel (default 0.05)
    --nfft N        up-channelization FFT length (default 16; 1 = none)
    --tbin SEC      output time bin (default 0.25e-3, rounded to a multiple of N x 2.56 us)
    --flip/--noflip reverse the sub-channel order within each channel (default --flip: the FFT of a
                    390-kHz channel comes out spectrally inverted; verified on 3 Crab GPs, S/N 48 vs 12)
    --foff F        sub-channel frequency offset in units of the 390-kHz channel (default 0:
                    channel k centred at 300 + 0.390625 k MHz; 0.5 = k is the lower edge)
    --orders 'o..'  orders to process (default all: 16 for BF512, 8 for BF256)
    --dcal DIR      2nd_cal dir of the NS matrix. Default, BF512: the latest cal_YYYYMMDD_HHMMSSZ.check
                    with a solution_2ndCal.npz in CALDIR that is earlier than the event.
                    Default, BF256: the cal, beam angles, row mask and scale of the matrix that was
                    loaded at the event, from the latest bfm2 log before it (bf256_bfm2.py, LOGDIR)
    --caldir CALDIR folder searched for the default BF512 dcal (default /burstt12/disk12/2nd_cal)
    --logdir DIR    BF256 bfm2 log folder (default /home/ubuntu/burstt-log/bfm2)
    --odir DIR      output dir (default .); beam voltages are cached in DIR/vcache
    --glob PAT      baseband file pattern, {o} = order, {ts} = YYYYMMDD_HHMMSS
                    (default BF512 /burstt1*/disk*/baseband/baseband*_order{o}_{ts}Z_trig,
                     BF256 /burstt1*/disk*/data/ring?.{ts}Z.trig.o{o}.bin)
    --tag STR       extra suffix for the output names
    --wexp X        band-sum weight per sub-channel ~ f^-X (default 2: the general SEFD trend; 0 = uniform).
                    Each sub-channel is first normalized to unit off-pulse rms; the profile is
                    sum(w*dn)/sqrt(sum(w^2)), then re-normalized by its off-pulse scatter
    --wmax MS       widest boxcar for the width-adaptive integrated S/N (default 10 ms; also limited to
                    1/4 of the off-pulse span). Boxcar widths are ~1.4x steps from one output bin; each
                    boxcar S/N = sum/sqrt(n), divided by the measured off-pulse rms of that boxcar
                    relative to white noise when > 1 (guards against baseline drifts)
'''
import os
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys, time, json
from glob import glob
from datetime import datetime, timezone
import numpy as np
from bf512_bfm2 import bfm2

inp = sys.argv[1:]
BF256 = '--bf256' in inp
inp = [a for a in inp if a != '--bf256']
if len(inp) < 3:
    sys.exit(__doc__)
ymd, hms, beam = inp.pop(0), inp.pop(0), int(inp.pop(0))
dm, toff, pad, nfft, tbin, flip, foff = 56.72, -0.016, 0.05, 16, 0.25e-3, True, 0.
wexp, wmax = 2., 10e-3
dcal = None
caldir = '/burstt12/disk12/2nd_cal'
logdir = '/home/ubuntu/burstt-log/bfm2'
odir = '.'
if BF256:
    # 8 orders x 128 ch; packet = 64 B header + 8192 B = 4 frames x 64 ch-pairs x 16 EW beams x 2 ch;
    # 64 B file header; counter: 2 counts per frame; 4-bit samples symmetric (no +0.5 offset), saturating at +-7
    NORD, NCH, NB1, NSTK, PL, CPF, OFF = 8, 128, 16, 4, 64 + 8192, 2, 0.
    pat, ofmt, mode = '/burstt1*/disk*/data/ring?.{ts}Z.trig.o{o}.bin', '%d', 'bf256'
else:
    # 16 orders x 60 ch; packet = 64 B header + 3840 B = 2 frames x 30 ch-pairs x 32 EW beams x 2 ch;
    # 64 B ring header; counter: 8 counts per frame; 4-bit samples coded k+0.5
    NORD, NCH, NB1, NSTK, PL, CPF, OFF = 16, 60, 32, 2, 64 + 3840, 8, 0.5
    pat, ofmt, mode = '/burstt1*/disk*/baseband/baseband*_order{o}_{ts}Z_trig', '%02d', 'bf512'
orders = list(range(NORD))
xtag = ''
while inp:
    k = inp.pop(0)
    if k == '--dm': dm = float(inp.pop(0))
    elif k == '--toff': toff = float(inp.pop(0))
    elif k == '--pad': pad = float(inp.pop(0))
    elif k == '--nfft': nfft = int(inp.pop(0))
    elif k == '--tbin': tbin = float(inp.pop(0))
    elif k == '--flip': flip = True
    elif k == '--noflip': flip = False
    elif k == '--foff': foff = float(inp.pop(0))
    elif k == '--orders': orders = [int(x) for x in inp.pop(0).split()]
    elif k == '--dcal': dcal = inp.pop(0)
    elif k == '--caldir': caldir = inp.pop(0)
    elif k == '--logdir': logdir = inp.pop(0)
    elif k == '--odir': odir = inp.pop(0)
    elif k == '--glob': pat = inp.pop(0)
    elif k == '--tag': xtag = inp.pop(0)
    elif k == '--wexp': wexp = float(inp.pop(0))
    elif k == '--wmax': wmax = float(inp.pop(0))*1e-3
    else: sys.exit('unknown option %s' % k)
cdir = os.path.join(odir, 'vcache')
os.makedirs(cdir, exist_ok=True)

KDM = 4148.808
TF = 2.56e-6                 # s per frame
DF = 0.390625                # MHz per channel
HD, NF = 64, 16              # file header bytes, FPGAs (= physical rows) per chunk
PADX = 0.06                  # half window kept in the voltage cache (>= pad)
rows = np.arange(14)
ns, col = beam // NB1, beam % NB1
par = col % 2                # BF512 only: NS matrix parity of the EW column
t300 = datetime.strptime(ymd + hms, '%Y%m%d%H%M%S.%f').replace(tzinfo=timezone.utc).timestamp()
tref = t300 + toff           # expected 300-MHz arrival
ts = ymd + '_' + hms[:6]
tag0 = 'gp_%s_%s_b%d%s' % (ymd, hms.replace('.', ''), beam, '_bf256' if BF256 else '')
print('%s event %s %s  beam %d = NS %d, EW col %d%s  DM %.2f  t300+toff = %.3f' %
      (mode.upper(), ymd, hms, beam, ns, col, '' if BF256 else ' (parity %d)' % par, dm, tref))
if BF256:
    from bf256_bfm2 import matrix_for
    _, minfo = matrix_for(t300, 0, dcal, logdir)
    dcal = minfo['dcal']
    print('NS matrix: log %s\n    dcal %s, scale %g, rows masked %s, NS beam %d at %.2f deg' %
          (minfo['log'], dcal, minfo['scale'], np.where(np.asarray(minfo['mask']) == 0)[0].tolist() if minfo['mask'] is not None else '?',
           ns, minfo['theta_deg'][ns] if minfo['theta_deg'] is not None else np.nan))
elif dcal is None:
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
if not BF256:
    print('NS matrix from', dcal, '(note: assumes it was loaded before the event; check the bfm2 logs)')
if pad > PADX:
    sys.exit('pad must be <= %.2f (cache window)' % PADX)

# byte offsets of EW column `col` inside a packet payload: payload element e = ((s*NCH/2+g)*NB1+a)*2+c,
# channel = 2g+c, stored at byte e^1 (even/odd swap); same layout for BF512 and BF256 (loadNode/formSpec2)
s_, g_, c_ = np.meshgrid(np.arange(NSTK), np.arange(NCH//2), np.arange(2), indexing='ij')
e = ((s_*(NCH//2) + g_)*NB1 + col)*2 + c_
bidx = 64 + (e ^ 1).reshape(NSTK, NCH)                 # (stack, ch), after the 64 B packet header
LUT = np.empty(256, np.complex64)
for b in range(256):
    i, q = b & 15, b >> 4
    LUT[b] = (i - 16*(i >= 8)) + 1j*(q - 16*(q >= 8)) + OFF*(1 + 1j)


def pkt_time(h):
    cnt = int(np.frombuffer(h[0:8].tobytes(), '<i8')[0])
    ep = int(np.frombuffer(h[16:20].tobytes(), '<u4')[0])
    return cnt, 2.0 + ep + cnt*TF/CPF


def extract(o):
    '''beam voltages (frames, 60 ch) covering every channel's arrival +- PADX; cached'''
    fc = os.path.join(cdir, '%s_o%02d.npz' % (tag0, o))
    if os.path.isfile(fc):
        d = np.load(fc, allow_pickle=True)
        inf = d['info']
        inf = json.loads(str(inf)) if inf.dtype.kind == 'U' else inf.item()   # JSON (new) or pickled dict (old caches)
        return d['y'], float(d['tfirst']), inf
    fl = glob(pat.format(o=ofmt % o, ts=ts))
    if not fl:
        print('order %02d: no file' % o); return None
    fn = fl[0]
    m = np.memmap(fn, np.uint8, 'r')
    nCk = (m.size - HD)//(PL*NF)
    P = m[HD:HD + nCk*NF*PL].reshape(nCk, NF, PL)
    i0 = next(i for i in range(64) if bytes(P[i // 14, i % 14, 58:64]) in (b'BURSTT', b'RSTTTT'))
    c0, t0 = pkt_time(P[i0 // 14, i0 % 14]); k0 = i0 // 14
    t0 -= k0*NSTK*TF
    ilast = next(i for i in range(nCk - 1, 0, -1) if bytes(P[i, 0, 58:64]) == b'BURSTT')
    c1, _ = pkt_time(P[ilast, 0])
    nCk = ilast + 1          # BF256 files end with a bitmap after the last packet
    lin = (c1 - c0) - NSTK*CPF*(ilast - k0)
    f_o = 300. + DF*np.arange(o*NCH - 1, (o + 1)*NCH + 1)
    tch = tref - KDM*dm*(1/300.**2 - 1/f_o**2)
    ck0 = max(int(np.floor((tch.min() - PADX - t0)/TF))//NSTK, 0)
    ck1 = min(int(np.ceil((tch.max() + PADX - t0)/TF))//NSTK + 2, nCk)
    if ck0 >= ck1:
        print('order %02d: window outside file' % o); return None
    if BF256:
        M = matrix_for(t300, o, dcal, logdir)[0][:, ns, :][:, rows]   # (ch, row)
    else:
        M = bfm2(dcal, o)[par][:, ns, :][:, rows]       # (ch, row)
    y = np.zeros(((ck1 - ck0)*NSTK, NCH), np.complex64)
    nbad = 0
    blk = 4096
    for a in range(ck0, ck1, blk):
        b = min(a + blk, ck1)
        Q = np.asarray(P[a:b, rows])                    # (n, 14, PL)
        ok = (Q[:, :, 58:64] == np.frombuffer(b'BURSTT', np.uint8)).all(-1)
        nbad += (~ok).sum()
        x = LUT[Q[:, :, bidx]]                          # (n, 14, stack, ch)
        x *= ok[:, :, None, None]
        x = x.transpose(0, 2, 1, 3).reshape(-1, len(rows), NCH)
        y[(a - ck0)*NSTK:(b - ck0)*NSTK] = np.einsum('frc,cr->fc', x, M)
    del m, P
    tfirst = t0 + ck0*NSTK*TF
    info = dict(file=fn, t0=float(t0), nck=int(nCk), lin=int(lin), nbad=int(nbad), span=float((ck1 - ck0)*NSTK*TF), mode=mode, dcal=dcal)
    np.savez(fc, y=y, tfirst=tfirst, info=json.dumps(info))      # JSON: readable under any numpy version
    print('order %02d: %s  file start %s, %.3f s read, bad pkts %d, counter linearity %d' %
          (o, os.path.basename(fn), datetime.fromtimestamp(t0, timezone.utc).strftime('%H:%M:%S.%f')[:-3],
           info['span'], nbad, lin), flush=True)
    return y, tfirst, info


tsub = nfft*TF
nb = max(int(round(tbin/tsub)), 1)
nOut = int(round(2*pad/(nb*tsub)))
tgrid = (np.arange(nOut) + 0.5)*nb*tsub - pad          # relative to tref, dedispersed to 300 MHz
nsub = NCH*nfft
dyn = np.full((nOut, NORD*nsub), np.nan, np.float32)
fsub = (300. + DF*(np.arange(NORD*NCH)[:, None] + foff + (np.arange(nfft)[None, :] - nfft//2)/nfft)).reshape(-1)
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


def band_profile(x):
    '''weighted band sum, weight ~ f^-x, normalized to unit off-pulse rms'''
    wt = (fsub[good]/400.)**(-x)
    p = (dn[:, good]*wt).sum(1)/np.sqrt((wt**2).sum())
    return (p - np.mean(p[off]))/np.std(p[off])


prof = band_profile(wexp)
prof0 = band_profile(0.) if wexp != 0 else prof
pk = int(np.argmax(prof))
print('sub-channels used %d / %d' % (good.sum(), good.size))
print('peak S/N (one %.3f-ms bin): %.1f with f^-%g weighting, %.1f uniform; at %+.3f ms from t300+toff (t300 %+.3f ms)' %
      (nb*tsub*1e3, prof[pk], wexp, prof0[np.argmax(prof0)], tgrid[pk]*1e3, (tgrid[pk] + toff)*1e3))

# width-adaptive integrated S/N: boxcar search over widths and positions
dtb = nb*tsub
noff = int(off.sum())
nmax = max(1, min(int(round(wmax/dtb)), noff//4))
widths = sorted(set(int(round(1.4142**i)) for i in range(40) if round(1.4142**i) <= nmax))
offL = np.where(tgrid < -0.6*pad)[0]; offR = np.where(tgrid > 0.6*pad)[0]


def boxcar(p, n):
    c = np.convolve(p, np.ones(n), 'valid')               # c[i] = sum p[i:i+n]
    # off-pulse boxcars lie entirely within one off-pulse side
    io = np.r_[offL[:max(len(offL) - n + 1, 0)], offR[:max(len(offR) - n + 1, 0)]]
    io = io[io < c.size]
    infl = np.std(c[io])/np.sqrt(n) if io.size > 2 else 1.
    return (c - np.mean(c[io]))/np.sqrt(n)/max(infl, 1.), infl


bx = []
print('boxcar  width(ms)   S/N    centre(ms)   off-pulse rms / white')
for n in widths:
    sn, infl = boxcar(prof, n)
    cen = (tgrid[:sn.size] + tgrid[n - 1:])/2
    i = int(np.argmax(np.where(np.abs(cen) <= 0.6*pad, sn, -np.inf)))    # centre within the on-pulse window
    bx.append((sn[i], n, i, infl))
    print('  %4d   %7.3f   %6.1f   %+8.3f     %.2f' % (n, n*dtb*1e3, sn[i], (tgrid[i] + tgrid[i + n - 1])/2*1e3, infl))
snb, nbest, ib, _ = max(bx)
tc = (tgrid[ib] + tgrid[ib + nbest - 1])/2
print('best: integrated S/N %.1f over %d bins = %.3f ms, centred %+.3f ms from t300+toff (t300 %+.3f ms)' %
      (snb, nbest, nbest*dtb*1e3, tc*1e3, (tc + toff)*1e3))
if nbest == widths[-1]:
    print('  (best width is the widest tried; raise --wmax/--pad if the pulse is broader)')
# per-channel S/N at the peak (sub-channels averaged back to the 390-kHz grid) for the plot
import warnings; warnings.simplefilter("ignore", RuntimeWarning)
dch = np.nanmean(np.where(good, dn, np.nan).reshape(nOut, -1, nfft), 2)*np.sqrt(nfft)
tag = tag0 + ('_n%d' % nfft) + ('_flip' if flip else '') + xtag
np.savez(os.path.join(odir, tag + '.npz'), dch=dch, good=good, fsub=fsub, tgrid=tgrid, prof=prof, prof0=prof0, wexp=wexp,
         boxcar=np.array(bx), snr_int=snb, width=nbest*dtb, tcen=tc,
         t300=t300, toff=toff, dm=dm, beam=beam, nb=nb, nfft=nfft, flip=flip, foff=foff, info=info)

import warnings; warnings.simplefilter('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(8, 9.3), sharex=True, gridspec_kw=dict(height_ratios=[1, 3]))
tms = tgrid*1e3
ax[0].plot(tms, prof, 'k-', lw=0.8, label='f$^{-%g}$ weighted' % wexp)
if wexp != 0:
    ax[0].plot(tms, prof0, '-', color='0.6', lw=0.6, zorder=0, label='uniform')
ax[0].axvspan((tc - nbest*dtb/2)*1e3, (tc + nbest*dtb/2)*1e3, color='#C44E52', alpha=0.2,
              label='best boxcar %.2f ms, S/N %.1f' % (nbest*dtb*1e3, snb))
ax[0].legend(fontsize=7, loc='upper right')
ax[0].set_ylabel('S/N (band sum)')
ax[0].set_title('%s %s %s UTC  beam %d (NS %d, EW %d)  DM %.2f\nnfft %d (%.1f kHz), bin %.3f ms; t=0: reported t300 %+.0f ms; peak S/N %.1f at %+.2f ms\nf$^{-%g}$-weighted band sum; best boxcar %.2f ms: integrated S/N %.1f' %
                (mode.upper(), ymd, hms, beam, ns, col, dm, nfft, DF/nfft*1e3, nb*tsub*1e3, toff*1e3, prof[pk], tms[pk], wexp, nbest*dtb*1e3, snb), fontsize=9.5)
nf = 8
dd = np.nanmean(dch.reshape(nOut, -1, nf), 2)
ff = (300. + DF*np.arange(NORD*NCH)).reshape(-1, nf).mean(1)
v = np.nanpercentile(dd, [1, 99.5])
ax[1].imshow(dd.T, origin='lower', aspect='auto', extent=[tms[0], tms[-1], ff[0], ff[-1]], vmin=v[0], vmax=v[1], cmap='viridis')
ax[1].set_ylabel('freq (MHz)'); ax[1].set_xlabel('time (ms), dedispersed to 300 MHz')
fig.tight_layout()
fig.savefig(os.path.join(odir, tag + '.png'), dpi=110)
print('saved', os.path.join(odir, tag + '.{npz,png}'))

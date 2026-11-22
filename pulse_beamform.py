#!/usr/bin/env python
'''
re-form one BF512, BF256 or BF64 beam (NS stage applied offline to the EW-beamformed trigger baseband)
and dedisperse a pulse, optionally after up-channelizing each 390-kHz channel with an N-point FFT
("quasi-coherent": intra-channel smearing reduced N times, time resolution N x 2.56 us)

syntax:
    pulse_beamform.py <YYYYMMDD> <HHMMSS.sss> <beam_id> [options]
    pulse_beamform.py --bf256 <YYYYMMDD> <HHMMSS.sss> <beam_id> [options]
    pulse_beamform.py --bf64  <YYYYMMDD> <HHMMSS.sss> <beam_id> [options]

    the time is the reported arrival time at 300 MHz (UTC)
    BF512: beam_id = ns*32 + ew_col; 16 orders x 60 ch; files /burstt1*/disk*/baseband/baseband*_orderOO_<ts>Z_trig
    BF256: beam_id = ns*16 + ew_col;  8 orders x 128 ch; files /burstt1*/disk*/data/ring?.<ts>Z.trig.oO.bin
    BF64:  beam_id = the Fushan (BF256) trigger beam id, ns*16 + ew_col. The GRN (or --site) beam is picked from it:
           the FUS beam centre (FUS beam0 (EW, NS) = (-7.5, -7.5) unless --fus_beam0) is taken to J2000 at the event
           time and back to the BF64 station (latitude/longitude and array rotation from getStationConfig); the EW
           column is the nearest BF64 EW beam (EW stage: 16 beams, beam0 -7.5, lam 0.75 m, Spacing_EW_m) and the NS
           stage is steered offline to the source's NS direction (all frequencies). --ew / --ns_deg override,
           --radec / --src replace the FUS beam as the direction.
           2 ring files x 512 ch (one per 200 MHz), 4 FPGA rows,
           1 frame per packet (loadNode nFPGA=4, as in second_cal64.py); the file order is read from the packet
           header and mapped to the second_cal64 1024-ch grid: header orders 0/1 -> 300-700 MHz, 2/3 -> 400-800 MHz
           (--flim0 overrides). The counter rate (counts per frame) is measured from each file.
options:
    --bf256         BF256 mode (option may appear anywhere)
    --bf64          BF64 mode (option may appear anywhere)
    --dm DM         dispersion measure (default 56.72)
    --toff SEC      empirical offset added to the reported time (default -0.016)
    --pad SEC       half window around the expected arrival in each channel (default 0.05)
    --nfft N        up-channelization FFT length (default 16; 1 = none)
    --tbin SEC      output time bin (default 0.25e-3, rounded to a multiple of N x 2.56 us)
    --flip/--noflip reverse the sub-channel order within each channel (default --flip: the FFT of a
                    390-kHz channel comes out spectrally inverted; verified on 3 Crab GPs, S/N 48 vs 12)
    --foff F        sub-channel frequency offset in units of the 390-kHz channel (default 0:
                    channel k centred at 300 + 0.390625 k MHz; 0.5 = k is the lower edge)
    --orders 'o..'  orders to process (default all: 16 for BF512, 8 for BF256, 2 for BF64 = the two 512-ch halves)
    --dcal DIR      2nd_cal dir of the NS matrix. Default, BF512: the latest cal_YYYYMMDD_HHMMSSZ.check
                    with a solution_2ndCal.npz in CALDIR that is earlier than the event.
                    Default, BF256: the cal, beam angles, row mask and scale of the matrix that was
                    loaded at the event, from the latest bfm2 log before it (bf256_bfm2.py, LOGDIR)
                    BF64: the latest second_cal64.py solution before the event in CALDIR (the GRN writer applies
                    phiCorr x normCorr of a --dcal dir, like BF256, but does not log which one)
    --caldir CALDIR folder searched for the default BF512 / BF64 dcal (default /burstt12/disk12/2nd_cal for BF512,
                    /burstt6/disk8/2nd_cal for BF64)
    --logdir DIR    BF256 bfm2 log folder (default /home/ubuntu/burstt-log/bfm2)
    --odir DIR      output dir (default .); beam voltages are cached in DIR/vcache
    --glob PAT      baseband file pattern, {o} = order, {ts} = YYYYMMDD_HHMMSS
                    (default BF512 /burstt1*/disk*/baseband/baseband*_order{o}_{ts}Z_trig,
                     BF256 /burstt1*/disk*/data/ring?.{ts}Z.trig.o{o}.bin)
    --tag STR       extra suffix for the output names
    --chwin         keep only each channel's own window (arrival +- 0.06 s incl. intra-channel smear) in the cache;
                    automatic when the sweep within an order exceeds 1 s (high DM: bounded memory and cache size)
  BF64 NS matrix (bf64_bfm2.py): steering x phiCorr x normCorr x mask, the same form as BF256, with phiCorr and
  normCorr from --dcal DIR/solution_2ndCal.npz written by second_cal64.py (shape (1, 4, 1024)). Without --dcal
  the real-time matrix is rebuilt (steering x FPGA delays x mask, write_2nd_matrix_64_log.py: no calibration).
    --bfm2log FILE  real-time bfm2-<host>-ring<r>-4row-*.npz log: beam angles, row mask, NS spacing, FPGA delays
                    (default: per ring, the entry of LOGDIR/bfm2-<host>-ring<r>.log in effect at the event)
    --fus_beam0 'EW NS'  beam0 of the Fushan EW and NS stages for the trigger beam id (default '-7.5 -7.5')
    --radec 'RA Dec'  J2000 source position (e.g. '03:32:59.37 +54:34:43.6') used instead of the FUS beam centre
    --src NAME      known source used instead of the FUS beam centre: astropy SkyCoord.from_name (e.g. B0329+54,
                    'PSR J1935+2154', 'Cyg A'; network), Sun, Moon, or an offline table (B0329+54, Crab, B0950+08,
                    B1133+16, B1933+16, Cas A, Cyg A, Tau A, Vir A)
    --ew COL        BF64 EW column (0-15) to use instead of the automatic pick
    --ew_beam0 B    beam0 of the BF64 EW stage for the automatic pick (default -7.5, the zcu216 --beam0 in the bfm1
                    log; B0329 on 11/05 and 11/14 peaked ~0.5 column higher than this predicts, as if -8.0)
    --rot DEG       array rotation of the BF64 station (default Rotate_deg of getStationConfig)
    --ns_deg DEG    NS beam angle to use instead of the source's NS direction (asin m at the station)
    --site CODE     BF64 station code (getStationConfig; default GRN)
    --bitmap MODE   packet bitmap at the end of each dump file (1 = packet received): auto (default) applies it
                    only if it agrees with the packet counters on ~2000 sampled packets, on / off. Independently,
                    packets whose counter is off the file's contiguous sequence (not from the same ring pass) are
                    dropped and reported: a sign of a ring-buffer dump problem, not a normal condition.
    --sep M         NS row spacing in m (default: the log value, else 0.5)
    --mask 'rows'   rows to exclude (e.g. '3')
    --flim0 F       lower edge of the 1024-ch grid, 300 or 400 MHz (default from the header order)
    BF64 --glob default /burstt*/disk*/data/ring?.{ts}Z.trig.o{o}.bin (all matching files are opened and
    assigned by their header order); {ts14} = YYYYMMDDHHMMSS is also available
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
import bf64_bfm2

inp = sys.argv[1:]
BF256 = '--bf256' in inp
BF64 = '--bf64' in inp
if BF256 and BF64:
    sys.exit('choose one of --bf256, --bf64')
inp = [a for a in inp if a not in ('--bf256', '--bf64')]
if len(inp) < 3:
    sys.exit(__doc__)
ymd, hms, beam = inp.pop(0), inp.pop(0), int(inp.pop(0))
dm, toff, pad, nfft, tbin, flip, foff = 56.72, -0.016, 0.05, 16, 0.25e-3, True, 0.
wexp, wmax = 2., 10e-3
chwin = False
dcal = None
caldir = '/burstt12/disk12/2nd_cal'
logdir = '/home/ubuntu/burstt-log/bfm2'
odir = '.'
bfm2log, sep64, mask64, flim0 = None, None, None, None
radec, src, site, fus_beam0, ew_man, nsdeg_man, bitmap_mode = None, None, 'GRN', (-7.5, -7.5), None, None, 'auto'
EW0_64 = -7.5                # BF64 EW stage beam0 (zcu216 -t pos): column c points at l = lam400/(16 dEW) (c + EW0_64)
rot_man = None
NF, rows = 16, np.arange(14)   # packets (FPGAs = physical rows) per chunk, rows used
if BF64:
    # 2 rings x 512 ch; packet = 64 B header + 8192 B = 1 frame x 256 ch-pairs x 16 EW beams x 2 ch; 4 FPGAs per chunk;
    # 64 B file header; counter rate measured per file (CPF None); 4-bit samples symmetric
    NORD, NCH, NB1, NSTK, PL, CPF, OFF = 2, 512, 16, 1, 64 + 8192, None, 0.
    pat, ofmt, mode = '/burstt*/disk*/data/ring?.{ts}Z.trig.o{o}.bin', None, 'bf64'
    NF, rows = 4, np.arange(4)
elif BF256:
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
    elif k == '--chwin': chwin = True
    elif k == '--wmax': wmax = float(inp.pop(0))*1e-3
    elif k == '--bfm2log': bfm2log = inp.pop(0)
    elif k == '--fus_beam0': fus_beam0 = tuple(float(x) for x in inp.pop(0).split())
    elif k == '--src': src = inp.pop(0)
    elif k == '--ew': ew_man = int(inp.pop(0))
    elif k == '--ew_beam0': EW0_64 = float(inp.pop(0))
    elif k == '--rot': rot_man = float(inp.pop(0))
    elif k == '--ns_deg': nsdeg_man = float(inp.pop(0))
    elif k == '--bitmap': bitmap_mode = inp.pop(0)
    elif k == '--sep': sep64 = float(inp.pop(0))
    elif k == '--mask': mask64 = [int(x) for x in inp.pop(0).split()]
    elif k == '--flim0': flim0 = float(inp.pop(0))
    elif k == '--radec': radec = inp.pop(0).split()
    elif k == '--site': site = inp.pop(0)
    else: sys.exit('unknown option %s' % k)
cdir = os.path.join(odir, 'vcache')
os.makedirs(cdir, exist_ok=True)

KDM = 4148.808
FBASE = 300.                 # MHz at channel 0 of order 0 (BF64: set from the header order)
TF = 2.56e-6                 # s per frame
DF = 0.390625                # MHz per channel
HD = 64                      # file header bytes
PADX = 0.06                  # half window kept in the voltage cache (>= pad)
CHWIN_S = 1.0                # sweeps longer than this keep per-channel windows only (or --chwin)
ns, col = beam // NB1, beam % NB1
par = col % 2                # BF512 only: NS matrix parity of the EW column
t300 = datetime.strptime(ymd + hms, '%Y%m%d%H%M%S.%f').replace(tzinfo=timezone.utc).timestamp()
tref = t300 + toff           # expected 300-MHz arrival
ts = ymd + '_' + hms[:6]
tag0 = 'gp_%s_%s_b%d%s' % (ymd, hms.replace('.', ''), beam, '_bf256' if BF256 else '_bf64' if BF64 else '')
print('%s event %s %s  %sbeam %d = NS %d, EW col %d%s  DM %.2f  t300+toff = %.3f' %
      (mode.upper(), ymd, hms, 'FUS trigger ' if BF64 else '', beam, ns, col, '' if (BF256 or BF64) else ' (parity %d)' % par, dm, tref))
if BF256:
    from bf256_bfm2 import matrix_for
    _, minfo = matrix_for(t300, 0, dcal, logdir)
    dcal = minfo['dcal']
    print('NS matrix: log %s\n    dcal %s, scale %g, rows masked %s, NS beam %d at %.2f deg' %
          (minfo['log'], dcal, minfo['scale'], np.where(np.asarray(minfo['mask']) == 0)[0].tolist() if minfo['mask'] is not None else '?',
           ns, minfo['theta_deg'][ns] if minfo['theta_deg'] is not None else np.nan))
elif BF64:
    # files: open every match, read the order of the first valid packet (header byte 36)
    cand = sorted(glob(pat.format(ts=ts, ts14=ymd + hms[:6], o='*')))
    hord = {}
    for fn in cand:
        m = np.memmap(fn, np.uint8, 'r')
        for i in range(64):
            h = m[HD + i*PL:HD + i*PL + 64]
            if bytes(h[58:64]) in (b'BURSTT', b'RSTTTT'):
                hord[fn] = int(h[36]); break
        del m
    if not hord:
        sys.exit('BF64: no readable file matches %s' % pat.format(ts=ts, ts14=ymd + hms[:6], o='*'))
    if flim0 is None:
        flim0 = 400. if min(hord.values()) >= 2 else 300.
    if dcal is None:
        # the real-time writer (rudp64_wrbd_3/write_2nd_matrix_64_log.py) applies --dcal phiCorr x normCorr but does not
        # log the directory: take the latest second_cal64 solution before the event (names cal_YYYYMMDD_HHMMSSZ.check
        # or cal_YYYYMMDDHHMMSS.check)
        cd64 = caldir if caldir != '/burstt12/disk12/2nd_cal' else '/burstt6/disk8/2nd_cal'
        cands = []
        for d_ in glob(os.path.join(cd64, 'cal_*.check')):
            stamp = os.path.basename(d_)[4:-6].replace('_', '').rstrip('Z')
            try:
                tc = datetime.strptime(stamp, '%Y%m%d%H%M%S').replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue
            if tc < t300 and os.path.isfile(os.path.join(d_, 'solution_2ndCal.npz')):
                cands.append((tc, d_))
        if cands:
            dcal = max(cands)[1]
            print('BF64 cal: latest solution before the event: %s (not logged by the writer; --dcal to override)' % dcal)
        else:
            print('BF64 cal: none found in %s; NS beams without calibration' % cd64)
    start_off = -2 if flim0 == 400. else 0            # as second_cal64.py
    files64 = {}
    for fn, od in hord.items():
        files64.setdefault(od + start_off, fn)
    FBASE = flim0
    # ---- pointing: FUS trigger beam centre (or --radec / --src) -> J2000 -> this station's array frame ----
    from astropy.coordinates import Angle
    def _rd(ra, dec):
        return '%s %s' % (Angle(ra, 'deg').to_string('hourangle', sep=':', precision=1, pad=True),
                          Angle(dec, 'deg').to_string(sep=':', precision=0, alwayssign=True, pad=True))
    stc = bf64_bfm2.station(site)
    if rot_man is not None:
        stc['Rotate_deg'] = rot_man
        bf64_bfm2.ROT_OVERRIDE[site] = rot_man
    dl64 = bf64_bfm2.LAM400/(NB1*stc['Spacing_EW_m'])          # EW-beam step in l
    def _pick(ra, dec):
        az, el, l, m = bf64_bfm2.radec_to_lm(site, ra, dec, t300)
        return az, el, l, m, l/dl64 - EW0_64, np.degrees(np.arcsin(np.clip(m, -1, 1)))
    if radec is not None or src is not None:
        if radec is not None:
            ra_d, dec_d, how = Angle(radec[0], 'hourangle').deg, Angle(radec[1], 'deg').deg, "--radec"
        else:
            ra_d, dec_d, how = bf64_bfm2.resolve_source(src, t300)
            how = '--src %s (%s)' % (src, how)
        print('direction from %s: J2000 %s (the FUS trigger beam %d is not used for pointing)' % (how, _rd(ra_d, dec_d), beam))
    else:
        nsf, ewf = beam // 16, beam % 16
        lf, mf = bf64_bfm2.beam_lm(beam, 'FUS', fus_beam0)
        ra_d, dec_d = bf64_bfm2.lm_to_radec('FUS', lf, mf, t300)
        print('FUS trigger beam %d (NS %d, EW %d), FUS beam0 (EW, NS) = (%g, %g): l %+.3f m %+.3f -> J2000 %s' %
              (beam, nsf, ewf, fus_beam0[0], fus_beam0[1], lf, mf, _rd(ra_d, dec_d)))
        # extent of the FUS beam cell (half a beam step each way) as seen from this station
        for lab, dlf, dmf in (('EW -1/2', -0.5, 0), ('EW +1/2', 0.5, 0), ('NS -1/2', 0, -0.5), ('NS +1/2', 0, 0.5)):
            r_, d_ = bf64_bfm2.lm_to_radec('FUS', lf + dlf*bf64_bfm2.LAM400/16, mf + dmf*bf64_bfm2.LAM400/8, t300)
            p_ = _pick(r_, d_)
            print('    FUS cell edge %s: J2000 %s -> %s EW col %.1f, NS %.1f deg' % (lab, _rd(r_, d_), site, p_[4], p_[5]))
    az, el, lg, mg, xcol, nsd = _pick(ra_d, dec_d)
    col_auto = int(np.clip(np.rint(xcol), 0, NB1 - 1))
    col = ew_man if ew_man is not None else col_auto
    ns_deg = nsdeg_man if nsdeg_man is not None else nsd
    ns = 0                                                      # the steered NS beam (index in the offline matrix)
    print('%s (lat %s, rot %g deg): az %.1f el %.1f, l %+.3f m %+.3f -> EW col %.2f (EW beam0 %g; auto %d%s), NS %.2f deg%s' %
          (site, stc['Latitude'], stc.get('Rotate_deg', 0.), az, el, lg, mg, xcol, EW0_64, col_auto,
           ', using --ew %d' % col if ew_man is not None else '', ns_deg, ' (--ns_deg)' if nsdeg_man is not None else ''))
    if el < 0:
        print('WARNING: the direction is below the horizon at %s' % site)
    if ew_man is None and abs(xcol - np.rint(xcol)) > 0.35 and 0 <= xcol <= NB1 - 1:
        print('NOTE: the direction falls near the boundary of EW columns %d and %d; consider also --ew %d' %
              (np.floor(xcol), np.floor(xcol) + 1, np.floor(xcol) + (1 if col_auto == np.floor(xcol) else 0)))
    if not -0.5 <= xcol <= NB1 - 0.5:
        print('WARNING: the direction is outside the %s EW beam fan (col %.1f); the edge column is used' % (site, xcol))
    if ew_man is not None and abs(ew_man - xcol) > 1:
        print('WARNING: --ew %d is %.1f beams from the computed direction' % (ew_man, abs(ew_man - xcol)))
    # NS matrix parameters per slot (ring): explicit log, else the log entry in effect at the event (for row mask,
    # NS spacing and FPGA delays; its real-time beam angles are only reported); the beam is steered to ns_deg
    nsp = {}
    for sl in sorted(files64):
        lf_ = bfm2log or bf64_bfm2.find_log(t300, sl, logdir)
        li = bf64_bfm2.load_log(lf_, flim0) if lf_ else {}
        mk = np.asarray(li.get('mask', np.ones(4)), float).copy()
        if mask64:
            mk[mask64] = 0.
        nsp[sl] = dict(log=lf_, theta=np.full(4, ns_deg), theta_rt=li.get('theta_deg'),
                       sep=sep64 if sep64 is not None else li.get('sep', stc['Spacing_NS_m']), mask=mk, delay=li.get('delay_ns'))
    tag0 = 'gp_%s_%s_f%d_e%02d_n%+.1f_bf64' % (ymd, hms.replace('.', ''), beam, col, ns_deg)
    print('BF64 files (slot: header order, file): %s;  grid %.0f MHz + 0.390625 k' %
          ({k: (hord[v], os.path.basename(v)) for k, v in sorted(files64.items())}, flim0))
    for sl, q in sorted(nsp.items()):
        print('NS matrix slot %d: %s; log %s\n    steered to %.2f deg (real-time beams %s deg), sep %.2f m, rows masked %s%s' %
              (sl, 'cal %s (phiCorr x normCorr)' % dcal if dcal else 'NO calibration (real-time rebuild)',
               os.path.basename(q['log']) if q['log'] else 'none', ns_deg,
               np.round(q['theta_rt'], 2) if q['theta_rt'] is not None else '?', q['sep'],
               np.where(q['mask'] == 0)[0].tolist(), '' if dcal or q['delay'] is None else ', FPGA delays %s ns' % np.round(q['delay'], 2)))
    orders = [o for o in orders if o in files64] or orders
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
if not (BF256 or BF64):
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


def pkt_time(h, cpf=None):
    cnt = int(np.frombuffer(h[0:8].tobytes(), '<i8')[0])
    ep = int(np.frombuffer(h[16:20].tobytes(), '<u4')[0])
    return cnt, 2.0 + ep + cnt*TF/(cpf or CPF or 1)


def pbits(bmb, c0, c1):
    '''bitmap of chunks c0..c1-1 as (c1-c0, NF) bool; bit p (LSB first) = file packet p = chunk p//NF, slot p%NF'''
    p0, p1 = c0*NF, c1*NF
    b = np.unpackbits(np.asarray(bmb[p0//8:(p1 + 7)//8]), bitorder='little')
    return b[p0 % 8:p0 % 8 + p1 - p0].reshape(-1, NF).astype(bool)


def extract(o):
    '''beam voltages (frames, NCH ch) covering every channel's arrival +- PADX; cached'''
    fc = os.path.join(cdir, '%s%s_o%02d.npz' % (tag0, '_bm' + bitmap_mode if BF64 and bitmap_mode != 'auto' else '', o))
    if os.path.isfile(fc):
        d = np.load(fc, allow_pickle=True)
        inf = d['info']
        inf = json.loads(str(inf)) if inf.dtype.kind == 'U' else inf.item()   # JSON (new) or pickled dict (old caches)
        tf = d['tfirst']
        return d['y'], (float(tf) if tf.ndim == 0 else tf), inf
    fl = ([files64[o]] if o in files64 else []) if BF64 else glob(pat.format(o=ofmt % o, ts=ts))
    if not fl:
        print('order %02d: no file' % o); return None
    fn = fl[0]
    m = np.memmap(fn, np.uint8, 'r')
    nCk = (m.size - HD)//(PL*NF)
    P = m[HD:HD + nCk*NF*PL].reshape(nCk, NF, PL)
    i0 = next(i for i in range(64) if bytes(P[i // NF, i % NF, 58:64]) in (b'BURSTT', b'RSTTTT'))
    k0 = i0 // NF
    ilast = next(i for i in range(nCk - 1, 0, -1) if bytes(P[i, 0, 58:64]) == b'BURSTT')
    nCk = ilast + 1          # BF256 files end with a bitmap after the last packet
    # counter model from ~500 sampled chunks: chunk k holds counter cref + cpf*NSTK*k within one ring pass
    ks = np.unique(np.linspace(k0, ilast, 513).astype(int))
    cs = np.array([pkt_time(P[k, 0])[0] if bytes(P[k, 0, 58:64]) == b'BURSTT' else -1 for k in ks], dtype=np.int64)
    v = cs >= 0
    ks, cs = ks[v], cs[v]
    cpf = CPF
    if cpf is None:          # BF64: counts per frame = median rate between consecutive samples
        r = np.median(np.diff(cs)/(NSTK*np.diff(ks)))
        cpf = max(int(round(r)), 1)
        if abs(r - cpf) > 0.01:
            print('order %02d: WARNING median counter rate %.4f counts/frame is not an integer; using %d' % (o, r, cpf))
    ep = int(np.frombuffer(P[ks[0], 0, 16:20].tobytes(), '<u4')[0])
    res = cs - cpf*NSTK*ks
    cref = int(np.median(res))
    lin = int(np.sum(np.abs(res - cref) > cpf*NSTK))           # sampled chunks off the contiguous sequence
    if lin:
        nps = np.round((res - cref)*TF/cpf, 1)
        print('order %02d: WARNING %d of %d sampled chunks are off the contiguous counter sequence of the file (time offsets '
              's: count %s): the dump is corrupted (ring-buffer problem); off-sequence packets are dropped' %
              (o, lin, len(ks), dict(zip(*[x.tolist() for x in np.unique(nps, return_counts=True)]))))
    # packet bitmap (BF64 dump files: after the data, 1 bit per packet, LSB first; file header: block_number <u2 at 8,
    # packet_number <u4 at 12, bitmask_offset <u8 at 32)
    bmb, use_bm = None, False
    if BF64 and bitmap_mode != 'off':
        bn, pn = int(m[8:10].view('<u2')[0]), int(m[12:16].view('<u4')[0])
        bo = int(m[32:40].view('<u8')[0])
        if pn % NF == 0 and bo == HD + bn*pn*PL and bo + bn*pn//8 <= m.size:
            bmb = m[bo:bo + bn*pn//8]
            kk = ks[:256]
            Qs = np.asarray(P[kk])                                                   # (n, NF, PL) sampled chunks
            cnt_s = np.ascontiguousarray(Qs[:, :, 0:8]).view('<i8')[..., 0]
            g_s = (Qs[:, :, 58:64] == np.frombuffer(b'BURSTT', np.uint8)).all(-1) & \
                (np.abs(cnt_s - (cref + cpf*NSTK*kk)[:, None]) <= cpf*NSTK)
            b_s = np.concatenate([pbits(bmb, k, k + 1) for k in kk])
            f_miss = 1 - b_s[g_s].mean() if g_s.any() else np.nan               # good packets the bitmap calls missing
            f_fake = b_s[~g_s].mean() if (~g_s).any() else np.nan               # bad packets the bitmap calls received
            use_bm = bitmap_mode == 'on' or (g_s.sum() > 100 and f_miss < 0.02)
            print('order %02d: bitmap %.1f%% set; on %d sampled packets: %.1f%% of valid ones marked missing, %s of invalid '
                  'ones marked received -> %s' % (o, 100*np.unpackbits(np.asarray(bmb[::max(len(bmb)//65536, 1)])).mean(),
                  g_s.size, 100*f_miss, 'n/a' if np.isnan(f_fake) else '%.0f%%' % (100*f_fake),
                  'used for flagging' if use_bm else 'NOT used (inconsistent with the packet counters; --bitmap on forces)'))
        else:
            print('order %02d: no valid bitmap (header block_number %d, packet_number %d, bitmask_offset %d)' % (o, bn, pn, bo))
    t0 = 2.0 + ep + cref*TF/cpf                               # time of chunk 0
    f_o = FBASE + DF*np.arange(o*NCH - 1, (o + 1)*NCH + 1)
    tch = tref - KDM*dm*(1/300.**2 - 1/f_o**2)
    ck0 = max(int(np.floor((tch.min() - PADX - t0)/TF))//NSTK, 0)
    ck1 = min(int(np.ceil((tch.max() + PADX - t0)/TF))//NSTK + 2, nCk)
    if ck0 >= ck1:
        print('order %02d: window outside file' % o); return None
    if BF64:
        q = nsp[o]
        M = bf64_bfm2.bfm2_64(o, flim0, dcal, q['theta'], q['mask'], q['sep'], q['delay'])[:, ns, :][:, rows]   # (ch, row)
    elif BF256:
        M = matrix_for(t300, o, dcal, logdir)[0][:, ns, :][:, rows]   # (ch, row)
    else:
        M = bfm2(dcal, o)[par][:, ns, :][:, rows]       # (ch, row)
    # per-channel windows when the sweep is long (high DM): keep only each channel's own arrival +- PADX
    # (incl. its intra-channel smear) instead of every channel over the whole sweep
    fe = FBASE + DF*(o*NCH + np.arange(NCH))
    tlo = tref - KDM*dm*(1/300.**2 - 1/(fe + DF/2)**2) - PADX
    thi = tref - KDM*dm*(1/300.**2 - 1/(fe - DF/2)**2) + PADX
    use_win = chwin or (tch.max() - tch.min()) > CHWIN_S
    nbad = nmiss = nseq = nbm = 0
    blk = 4096
    if use_win:
        s0 = np.clip(np.floor((tlo - t0)/TF).astype(int), ck0*NSTK, None)       # first frame of each channel's window
        nwin = int(np.ceil((thi - tlo).max()/TF)) + NSTK
        s0 = np.minimum(s0, ck1*NSTK - nwin)
        ck0 = max(int(s0.min())//NSTK, 0); ck1 = min(int(s0.max() + nwin)//NSTK + 1, nCk)
        y = np.zeros((nwin, NCH), np.complex64)
    else:
        y = np.zeros(((ck1 - ck0)*NSTK, NCH), np.complex64)
    for a in range(ck0, ck1, blk):
        b = min(a + blk, ck1)
        aF, bF = a*NSTK, b*NSTK
        if use_win:
            act = np.flatnonzero((s0 < bF) & (s0 + nwin > aF))      # channels whose window overlaps this block
            if act.size == 0:
                continue
        else:
            act = slice(None)
        Q = np.asarray(P[a:b, rows])                    # (n, rows, PL)
        ok = (Q[:, :, 58:64] == np.frombuffer(b'BURSTT', np.uint8)).all(-1)
        nmiss += (~ok).sum()
        cnt = np.ascontiguousarray(Q[:, :, 0:8]).view('<i8')[..., 0]                    # (n, rows)
        cok = np.abs(cnt - (cref + cpf*NSTK*np.arange(a, b))[:, None]) <= cpf*NSTK     # counter matches the chunk
        nseq += (ok & ~cok).sum()
        ok &= cok
        if use_bm:
            bq = pbits(bmb, a, b)[:, rows]
            nbm += (ok & ~bq).sum()
            ok &= bq
        nbad += (~ok).sum()
        x = LUT[Q[:, :, bidx[:, act]]]                  # (n, rows, stack, ch)
        x *= ok[:, :, None, None]
        x = x.transpose(0, 2, 1, 3).reshape(-1, len(rows), x.shape[-1])
        yb = np.einsum('frc,cr->fc', x, M[act])
        if use_win:
            for k, j in enumerate(act):
                lo, hi = max(s0[j], aF), min(s0[j] + nwin, bF)
                y[lo - s0[j]:hi - s0[j], j] = yb[lo - aF:hi - aF, k]
        else:
            y[(a - ck0)*NSTK:(b - ck0)*NSTK] = yb
    del m, P
    tfirst = t0 + s0*TF if use_win else t0 + ck0*NSTK*TF
    info = dict(file=fn, t0=float(t0), nck=int(nCk), lin=int(lin), nbad=int(nbad), nmiss=int(nmiss), nseq=int(nseq),
                nbm=int(nbm), bitmap=bool(use_bm), span=float((ck1 - ck0)*NSTK*TF), mode=mode, dcal=dcal, cpf=int(cpf),
                chwin=bool(use_win))
    np.savez(fc, y=y, tfirst=tfirst, info=json.dumps(info))      # JSON: readable under any numpy version
    ntot = (ck1 - ck0)*len(rows)
    print('order %02d: %s  file start %s, %.3f s read; packets flagged %d of %d (%.2f%%): no marker %d, off-sequence %d, '
          'bitmap %s%s' % (o, os.path.basename(fn), datetime.fromtimestamp(t0, timezone.utc).strftime('%H:%M:%S.%f')[:-3],
           info['span'], nbad, ntot, 100.*nbad/max(ntot, 1), nmiss, nseq, nbm if use_bm else '-',
           ', per-channel windows of %.3f s kept' % (y.shape[0]*TF) if use_win else ''), flush=True)
    if nseq > 0.01*ntot:
        print('order %02d: WARNING %.1f%% of the packets in the read window are from another ring pass (corrupted dump)'
              % (o, 100.*nseq/ntot))
    return y, tfirst, info


tsub = nfft*TF
nb = max(int(round(tbin/tsub)), 1)
nOut = int(round(2*pad/(nb*tsub)))
tgrid = (np.arange(nOut) + 0.5)*nb*tsub - pad          # relative to tref, dedispersed to 300 MHz
nsub = NCH*nfft
dyn = np.full((nOut, NORD*nsub), np.nan, np.float32)
fsub = (FBASE + DF*(np.arange(NORD*NCH)[:, None] + foff + (np.arange(nfft)[None, :] - nfft//2)/nfft)).reshape(-1)
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
    tf_sub = np.repeat(tfirst, nfft) if np.ndim(tfirst) else tfirst      # per-channel windows: start time per channel
    st = np.floor((tarr - pad - tf_sub)/tsub).astype(int)
    for j in range(nsub):
        s = st[j]
        if s < 0 or s + nOut*nb > nblk:
            continue
        dyn[:, o*nsub + j] = pw[s:s + nOut*nb, j].reshape(nOut, nb).mean(1)
    del y, Y, pw
if not info:
    sys.exit('no data in the event window in any order (see the messages above)')
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
         t300=t300, toff=toff, dm=dm, beam=beam, ew=col, ns_deg=ns_deg if BF64 else np.nan, nb=nb, nfft=nfft, flip=flip, foff=foff, info=info)

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
bstr = ('FUS beam %d -> %s EW %d, NS %.1f deg' % (beam, site, col, ns_deg)) if BF64 else 'beam %d (NS %d, EW %d)' % (beam, ns, col)
ax[0].set_title('%s %s %s UTC  %s  DM %.2f\nnfft %d (%.1f kHz), bin %.3f ms; t=0: reported t300 %+.0f ms; peak S/N %.1f at %+.2f ms\nf$^{-%g}$-weighted band sum; best boxcar %.2f ms: integrated S/N %.1f' %
                (mode.upper(), ymd, hms, bstr, dm, nfft, DF/nfft*1e3, nb*tsub*1e3, toff*1e3, prof[pk], tms[pk], wexp, nbest*dtb*1e3, snb), fontsize=9.5)
nf = 8
dd = np.nanmean(dch.reshape(nOut, -1, nf), 2)
ff = (FBASE + DF*np.arange(NORD*NCH)).reshape(-1, nf).mean(1)
v = np.nanpercentile(dd, [1, 99.5])
ax[1].imshow(dd.T, origin='lower', aspect='auto', extent=[tms[0], tms[-1], ff[0], ff[-1]], vmin=v[0], vmax=v[1], cmap='viridis')
ax[1].set_ylabel('freq (MHz)'); ax[1].set_xlabel('time (ms), dedispersed to 300 MHz')
fig.tight_layout()
fig.savefig(os.path.join(odir, tag + '.png'), dpi=110)
print('saved', os.path.join(odir, tag + '.{npz,png}'))

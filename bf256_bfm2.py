#!/usr/bin/env python
'''rebuild the real-time BF256 2nd-stage (NS) matrix for any of the 8 orders
(write_2nd_matrix_256_amx_channel_log.py: BFM = exp(+2j pi pos/lam sin_theta) * phiCorr * normCorr * SCALE * mask)

bfm2_256(dcal, order, theta_deg, mask, scale) -> M, shape (128 ch, 16 NS beams, 16 rows), complex
    NS beam y[ns] = sum_r M[ch, ns, r] x[r]   (x = EW-beamformed baseband of row r)
find_log(t_unix) -> (path, dict) of the latest saved bfm2 log written before t (UTC names, burstt16 ring0)
    dict has dcal, theta_deg, mask, scale parsed from the log (raw_cmd, theta_deg, mask)
matrix_for(t_unix, order) -> (M, info): the matrix that was loaded at time t

syntax (check against saved logs):  bf256_bfm2.py 'glob of bfm2-*.npz' [order ...]
    order defaults to the ring number for burstt16 files (ring0 = o0, ring1 = o1)
'''
import os, re, glob
from datetime import datetime, timezone
import numpy as np

LOGDIR = '/home/ubuntu/burstt-log/bfm2'
nRow, nChan2, sep = 16, 128, 0.5
DF = 0.390625   # MHz; channel k (0..1023) at 300 + k*DF


def bfm2_256(dcal, order, theta_deg=None, mask=None, scale=0.5, beam0=-7.5):
    d = np.load('%s/solution_2ndCal.npz' % dcal)
    phiCorr, normCorr = d['phiCorr'], d['normCorr']                   # (1, 16, 1024)
    f = np.linspace(300., 700., 1024, endpoint=False)
    lamb = 2.998e8/(f*1e6)
    pos = np.arange(nRow)*sep
    pos -= pos.mean()
    if theta_deg is None:
        s = 2.998e8/400e6/sep/nRow*(np.arange(nRow) + beam0)
    else:
        s = np.sin(np.radians(np.asarray(theta_deg, float)))
    if mask is None:
        mask = np.ones(nRow); mask[[14, 15]] = 0.
    B = np.exp(2j*np.pi*pos[None, :, None]/lamb[None, None, :]*s[:, None, None])   # (ns, row, 1024)
    B = B*phiCorr*normCorr*scale*np.asarray(mask, float)[None, :, None]
    j0 = order*nChan2
    return B[:, :, j0:j0+nChan2].transpose(2, 0, 1)                                 # (ch, ns, row)


def log_time(fn):
    m = re.search(r'-(\d{6}_\d{6})-cal_', os.path.basename(fn))
    return datetime.strptime(m.group(1), '%y%m%d_%H%M%S').replace(tzinfo=timezone.utc).timestamp() if m else None


def find_log(t_unix, logdir=LOGDIR, ring=0):
    c = [(log_time(f), f) for f in glob.glob('%s/bfm2-burstt16-ring%d-*-cal_*.npz' % (logdir, ring))]
    c = [x for x in c if x[0] is not None and x[0] < t_unix]
    if not c:
        return None, None
    fn = max(c)[1]
    d = np.load(fn, allow_pickle=True)
    cmd = str(d['raw_cmd']) if 'raw_cmd' in d.files else ''
    m = re.search(r'--dcal\s+(\S+)', cmd)
    dcal = m.group(1) if m else '/burstt12/disk12/2nd_cal/' + re.search(r'(cal_\d{8}_\d{6}Z)', fn).group(1) + '.check'
    m = re.search(r'--scale\s+(\S+)', cmd)
    scale = float(m.group(1)) if m else 0.5
    info = dict(log=fn, dcal=dcal, scale=scale, mask=d['mask'], cmd=cmd,
                theta_deg=d['theta_deg'] if 'theta_deg' in d.files else None,
                beam0=float(d['beam0'][0]))
    return fn, info


def matrix_for(t_unix, order, dcal=None, logdir=LOGDIR):
    '''matrix loaded at t (from the latest log); dcal overrides the cal named in the log'''
    fn, info = find_log(t_unix, logdir)
    if info is None:
        if dcal is None:
            raise RuntimeError('no bfm2 log before the event in %s; give a dcal' % logdir)
        info = dict(log=None, dcal=dcal, scale=0.5, mask=None, theta_deg=None, beam0=-7.5, cmd='')
    if dcal is not None:
        info['dcal'] = dcal
    M = bfm2_256(info['dcal'], order, info['theta_deg'], info['mask'], info['scale'], info['beam0'])
    return M, info


def load_saved(fnpz):
    b = np.load(fnpz)['BFM2']
    return b[..., 0] + 1j*b[..., 1]                                                    # (ch, ns, row)


if __name__ == '__main__':
    import sys
    files = sorted(glob.glob(sys.argv[1]))
    for fs in files:
        d = np.load(fs, allow_pickle=True)
        cmd = str(d['raw_cmd']); dcal = re.search(r'--dcal\s+(\S+)', cmd).group(1)
        sc = float(re.search(r'--scale\s+(\S+)', cmd).group(1)) if '--scale' in cmd else 0.5
        th = d['theta_deg'] if 'theta_deg' in d.files else None
        S = load_saved(fs)
        orders = [int(x) for x in sys.argv[2:]] or range(8)
        best = min((np.abs(S - bfm2_256(dcal, o, th, d['mask'], sc)).max(), o) for o in orders)
        print('%s: best order %d, max |saved - rebuilt| %.1e (max |M| %.2f)' % (os.path.basename(fs), best[1], best[0], np.abs(S).max()))

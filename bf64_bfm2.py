#!/usr/bin/env python
'''offline BF64 2nd-stage (NS) matrix: 4 FPGA rows -> 4 NS beams, 1024 channels on the second_cal64 grid

layout follows write_2nd_matrix_64_log.py (real-time writer) and second_cal64.py (calibration):
    steering  W[ns, row, f] = exp(+2j pi pos_row/lam_f sin(theta_ns)),  pos = row*sep (centred here; a common phase)
    theta     sin(theta) = lam(F0)/sep0/4 * (arange(4) + beam0), sep0 = 0.5 m; or theta_deg from a bfm2 log / --theta
    real time W * exp(+2j pi delay_ns f) * mask * 4096       (no calibration term)
    offline   W * phiCorr * normCorr * mask                    (phiCorr, normCorr (1, 4, 1024) from <dcal>/solution_2ndCal.npz,
                                                                same form and use as BF256; replaces the FPGA delays)
    the real-time BFM2 in the log is stored as (512 ch, row, beam, re/im); this module returns (ch, ns, row) like bf256_bfm2
frequency grid: f = F0 + 0.390625 k, k = 0..1023; F0 = 300 MHz (header orders 0, 1) or 400 MHz (orders 2, 3),
    slot = order + start_off (start_off = 0 or -2) selects channels 512*slot .. 512*slot + 511

functions:
    theta_beam0(beam0, F0)                          default beam angles (deg)
    load_log(fn)                                    dict from a bfm2-<host>-ring<r>-4row-*.npz log
    bfm2_64(slot, F0, dcal, theta_deg, mask, sep, delay_ns, beam0) -> M (512, 4 ns, 4 rows), complex
    load_saved(fn)                                  real-time matrix of a log as (512, ns, row), unit scale
pointing (sky <-> array direction cosines; array frame x = along a row (EW), y = across rows (NS), rotated by the
station's Rotate_deg as in check_times_bf256_target.py: l = cos(r) uE + sin(r) uN, m = -sin(r) uE + cos(r) uN):
    station(site)                                   getStationConfig dict
    beam_lm(beamid, site, beam0=(-7.5, -7.5))       beam-centre (l, m) of a 16 x 16 BF256 beam id (ns*16 + ew):
                                                    l = lam400/(16 dEW) (ew + beam0_EW), m = lam400/(16 dNS) (ns + beam0_NS)
    lm_to_radec(site, l, m, t_unix)                 J2000 (RA, Dec) in deg of an array direction at time t
    radec_to_lm(site, ra, dec, t_unix)              (az, el, l, m) at the site (deg, deg, array-frame cosines)
    resolve_source(name, t_unix)                    J2000 (RA, Dec) in deg: Sun/Moon, astropy SkyCoord.from_name (CDS
                                                    Sesame, also tried as 'PSR <name>'), else a small offline table
syntax (check a log against the rebuild, up to a common phase per channel and beam):
    bf64_bfm2.py bfm2-*.npz [F0]
'''
import numpy as np

nRow, nChan2, DF, SEP0, SCALE = 4, 512, 0.390625, 0.5, 4096.


def theta_beam0(beam0=-1.5, F0=400.):
    lam0 = 2.998e8/(F0*1e6)
    return np.degrees(np.arcsin(lam0/SEP0/nRow*(np.arange(nRow) + beam0)))


def _arr(d, k, default=None):
    if k not in d.files:
        return default
    v = d[k]
    return np.asarray(v.item() if v.dtype == object and v.shape == () else v, dtype=float)


def theta_from_matrix(B, sep, F0, ring):
    '''beam angles (deg) fitted from a saved real-time matrix (ch, row, beam, 2): row-phase slope at each channel'''
    M = B[..., 0].astype(float) + 1j*B[..., 1]
    f = F0 + DF*(ring*nChan2 + np.arange(nChan2))
    lam = 2.998e8/(f*1e6)
    k = slice(nChan2//4, 3*nChan2//4)
    dph = np.angle(M[k, 1:, :]*np.conj(M[k, :-1, :]))                      # (ch, row-1, beam): 2 pi sep/lam sin
    s = np.median(dph/(2*np.pi*sep/lam[k, None, None]), axis=(0, 1))
    return np.degrees(np.arcsin(np.clip(s, -1, 1)))


def load_log(fn, F0=300.):
    '''real-time NS matrix parameters; when theta_deg is not stored (GRN logs keep None), the angles are read from the
    file name (-th<a>_<b>_<c>_<d>-) or, failing that, fitted from the saved matrix'''
    import re, os
    d = np.load(fn, allow_pickle=True)          # GRN logs store some fields as object arrays
    sep = float(np.atleast_1d(_arr(d, 'sep_m', [0.5]))[0])
    th = d['theta_deg'] if 'theta_deg' in d.files else None
    th = None if th is None or (th.dtype == object and th.shape == () and th.item() is None) else np.asarray(th, float).ravel()
    if th is None or th.size != nRow:
        m = re.search(r'-th(-?[\d.]+(?:_-?[\d.]+){%d})-' % (nRow - 1), os.path.basename(fn))
        mr = re.search(r'-ring(\d)', os.path.basename(fn))
        if m:
            th = np.array([float(x) for x in m.group(1).split('_')])
        elif 'BFM2' in d.files:
            th = theta_from_matrix(d['BFM2'], sep, F0, int(mr.group(1)) if mr else 0)
    return dict(log=fn, theta_deg=th, mask=_arr(d, 'mask', np.ones(nRow)), sep=sep, delay_ns=_arr(d, 'delay_fpga'),
                beam0=float(np.atleast_1d(_arr(d, 'beam0', [-1.5]))[0]),
                theta_fit=theta_from_matrix(d['BFM2'], sep, F0, int(re.search(r'-ring(\d)', os.path.basename(fn)).group(1)))
                if 'BFM2' in d.files and re.search(r'-ring(\d)', os.path.basename(fn)) else None)


def find_log(t_unix, ring, logdir='/home/ubuntu/burstt-log/bfm2'):
    '''the bfm2 log entry in effect at t (latest line of bfm2-<host>-ring<r>.log before t); None if not found'''
    import glob, os
    from datetime import datetime
    best = None
    for lf in glob.glob('%s/bfm2-*-ring%d.log' % (logdir, ring)):
        for line in open(lf):
            p = line.split()
            if len(p) < 2:
                continue
            try:
                t = datetime.fromisoformat(p[0]).timestamp()
            except ValueError:
                continue
            if t < t_unix and (best is None or t > best[0]):
                best = (t, p[1])
    if best is None:
        return None
    fn = best[1] if os.path.isfile(best[1]) else os.path.join(logdir, os.path.basename(best[1]))
    return fn if os.path.isfile(fn) else None


def bfm2_64(slot, F0, dcal=None, theta_deg=None, mask=None, sep=0.5, delay_ns=None, beam0=-1.5):
    f = F0 + DF*np.arange(2*nChan2)                                   # MHz
    lam = 2.998e8/(f*1e6)
    pos = np.arange(nRow)*sep
    pos -= pos.mean()
    if theta_deg is None or len(np.atleast_1d(theta_deg)) != nRow:
        theta_deg = theta_beam0(beam0, F0)
    s = np.sin(np.radians(np.asarray(theta_deg, float)))
    B = np.exp(2j*np.pi*pos[None, :, None]/lam[None, None, :]*s[:, None, None])        # (ns, row, 1024)
    if dcal is not None:
        d = np.load('%s/solution_2ndCal.npz' % dcal)
        B = B*d['phiCorr'].reshape(1, nRow, -1)*d['normCorr'].reshape(1, nRow, -1)
    elif delay_ns is not None:
        B = B*np.exp(2j*np.pi*np.asarray(delay_ns, float)[None, :, None]*1e-9*f[None, None, :]*1e6)
    if mask is not None:
        B = B*np.asarray(mask, float)[None, :, None]
    j0 = slot*nChan2
    return B[:, :, j0:j0 + nChan2].transpose(2, 0, 1)                                  # (ch, ns, row)


def load_saved(fn):
    b = np.load(fn, allow_pickle=True)['BFM2'].astype(float)                                             # (ch, row, beam, 2)
    return (b[..., 0] + 1j*b[..., 1]).transpose(0, 2, 1)/SCALE                        # (ch, ns, row)


C = 2.998e8
LAM400 = C/400e6
# offline fallback for --src (J2000)
KNOWN = {'B0329+54': ('03:32:59.37', '+54:34:43.6'), 'B0531+21': ('05:34:31.94', '+22:00:52.2'),
         'CRAB': ('05:34:31.94', '+22:00:52.2'), 'TAUA': ('05:34:31.94', '+22:00:52.2'),
         'B0950+08': ('09:53:09.31', '+07:55:35.8'), 'B1133+16': ('11:36:03.25', '+15:51:04.5'),
         'B1933+16': ('19:35:47.83', '+16:16:40.0'), 'CASA': ('23:23:24.0', '+58:48:54'),
         'CYGA': ('19:59:28.36', '+40:44:02.1'), 'VIRA': ('12:30:49.42', '+12:23:28.0')}


ROT_OVERRIDE = {}            # site -> rotation (deg) replacing the getStationConfig value


def station(site):
    from getStationConfig import GetStationConfig
    st = GetStationConfig(site)
    if not st:
        raise ValueError('unknown station %s' % site)
    st = dict(st)
    if site in ROT_OVERRIDE:
        st['Rotate_deg'] = ROT_OVERRIDE[site]
    return st


def _observer(st, t_unix):
    import ephem
    from datetime import datetime, timezone
    ob = ephem.Observer()
    ob.lon, ob.lat, ob.elevation = st['Longitude'], st['Latitude'], st['Elevation']
    ob.pressure = 0
    ob.date = ephem.Date(datetime.fromtimestamp(t_unix, timezone.utc).replace(tzinfo=None))
    return ob


def beam_lm(beamid, site='FUS', beam0=(-7.5, -7.5), nb=16):
    st = station(site)
    ns, ew = beamid//nb, beamid % nb
    return (LAM400/(nb*st['Spacing_EW_m'])*(ew + beam0[0]), LAM400/(nb*st['Spacing_NS_m'])*(ns + beam0[1]))


def lm_to_radec(site, l, m, t_unix):
    import ephem
    st = station(site)
    r = np.radians(st.get('Rotate_deg', 0.))
    uE, uN = np.cos(r)*l - np.sin(r)*m, np.sin(r)*l + np.cos(r)*m
    uz = np.sqrt(max(1 - uE**2 - uN**2, 0.))
    ob = _observer(st, t_unix)
    ra, dec = ob.radec_of(np.arctan2(uE, uN), np.arcsin(uz))
    eq = ephem.Equatorial(ephem.Equatorial(ra, dec, epoch=ob.date), epoch=ephem.J2000)
    return np.degrees(float(eq.ra)), np.degrees(float(eq.dec))


def radec_to_lm(site, ra_deg, dec_deg, t_unix):
    import ephem
    st = station(site)
    ob = _observer(st, t_unix)
    b = ephem.FixedBody()
    b._ra, b._dec, b._epoch = np.radians(ra_deg), np.radians(dec_deg), ephem.J2000
    b.compute(ob)
    az, el = float(b.az), float(b.alt)
    uE, uN = np.cos(el)*np.sin(az), np.cos(el)*np.cos(az)
    r = np.radians(st.get('Rotate_deg', 0.))
    return np.degrees(az), np.degrees(el), np.cos(r)*uE + np.sin(r)*uN, -np.sin(r)*uE + np.cos(r)*uN


def resolve_source(name, t_unix):
    key = name.upper().replace(' ', '').replace('PSR', '')
    if key in ('SUN', 'MOON'):
        import ephem
        from datetime import datetime, timezone
        b = ephem.Sun() if key == 'SUN' else ephem.Moon()
        b.compute(ephem.Date(datetime.fromtimestamp(t_unix, timezone.utc).replace(tzinfo=None)), epoch=ephem.J2000)
        return np.degrees(float(b.a_ra)), np.degrees(float(b.a_dec)), 'ephem (J2000 astrometric at the event)'
    try:
        from astropy.coordinates import SkyCoord
        for q in (name, 'PSR ' + name):
            try:
                c = SkyCoord.from_name(q)
                return float(c.icrs.ra.deg), float(c.icrs.dec.deg), 'SkyCoord.from_name(%r)' % q
            except Exception:
                continue
    except ImportError:
        pass
    k2 = key.replace('-', '').replace('_', '')
    for k, (ra, dec) in KNOWN.items():
        if k == key or k == k2:
            from astropy.coordinates import Angle
            return Angle(ra, 'hourangle').deg, Angle(dec, 'deg').deg, 'offline table'
    raise ValueError('cannot resolve source %r (use --radec)' % name)


if __name__ == '__main__':
    import sys, re, os
    F0 = float(sys.argv[2]) if len(sys.argv) > 2 else 400.
    fn = sys.argv[1]
    info = load_log(fn)
    ring = int(re.search(r'-ring(\d)', os.path.basename(fn)).group(1))
    S = load_saved(fn)
    M = bfm2_64(ring, F0, None, info['theta_deg'], info['mask'], info['sep'], info['delay_ns'], info['beam0'])
    # remove the per-(ch, beam) common phase (the writer does not centre the positions)
    ref = np.where(np.abs(S[:, :, :1]) > 0, S[:, :, :1]/np.maximum(np.abs(S[:, :, :1]), 1e-9), 1)
    refm = M[:, :, :1]/np.maximum(np.abs(M[:, :, :1]), 1e-9)
    print('%s: theta %s, mask %s, sep %.2f, delays %s ns; max |saved - rebuilt| = %.1e (int16 rounding ~2e-4)' %
          (os.path.basename(fn), np.round(info['theta_deg'], 2), info['mask'], info['sep'], info['delay_ns'],
           np.abs(S/ref - M/refm).max()))

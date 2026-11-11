#!/usr/bin/env python
'''rebuild the real-time BF512 2nd-stage (NS) matrix for all 16 orders
(write_2nd_matrix_512_log.py --auto --dcal <2nd_cal dir> -m '14 15', i.e. the 11/10 06:15Z setting)

bfm2(dcal, order) -> M, shape (2 parity, 60 ch, 16 NS beams, 16 rows), complex
    parity 0 = even EW columns (beam0 -7.75), 1 = odd EW columns (beam0 -7.25)
    NS beam y[ns] = sum_r M[parity, ch, ns, r] x[r]   (x = EW-beamformed baseband of row r)
'''
import numpy as np

nRow, nChan2, sep0, sep = 16, 60, 0.5, 0.5
DF = 0.390625   # MHz per channel; channel k (0..959) at 300 + k*DF


def bfm2(dcal, order, beam0=(-7.75, -7.25), mask_rows=(14, 15), scale=(0.25, 1.0)):
    '''scale reproduces the deployed writer: BFM01 (even EW columns) is multiplied by SCALE=0.5 twice,
    BFM02 (odd columns) not at all (verified to 2.5e-8 against the saved burstt16 ring0-3 = orders 0-3)'''
    d = np.load('%s/solution_2ndCal.npz' % dcal)
    phiCorr, normCorr = d['phiCorr'], d['normCorr']          # (1, 16, 960)
    f = 300. + DF*np.arange(960)
    lamb = 2.998e8/(f*1e6)
    pos = np.arange(nRow)*sep
    pos -= pos.mean()
    lamb0 = 2.998e8/400e6
    mask = np.ones(nRow)
    mask[list(mask_rows)] = 0.
    out = []
    for p in range(2):
        s = lamb0/sep0/nRow*(np.arange(nRow) + beam0[p])
        B = np.exp(2j*np.pi*pos[None, :, None]/lamb[None, None, :]*s[:, None, None])   # (ns, row, 960)
        B = B*phiCorr*normCorr*mask[None, :, None]
        j0 = order*nChan2
        out.append(B[:, :, j0:j0+nChan2].transpose(2, 0, 1)*scale[p])                 # (ch, ns, row)
    return np.array(out)


def load_saved(fnpz):
    d = np.load(fnpz)
    b = d['BFM2']
    return b[..., 0] + 1j*b[..., 1]                                                   # (2, ch, ns, row)


if __name__ == '__main__':
    import sys, glob
    dcal = sys.argv[1]
    for fs in sorted(glob.glob(sys.argv[2])):
        S = load_saved(fs)
        best = None
        for o in range(16):
            M = bfm2(dcal, o)
            for p in range(2):
                ok = np.abs(S[p]) > 0
                if ok.sum() == 0:
                    continue
                r = S[p][ok]/M[p][ok]
                dev = np.std(r/np.median(r.real))
                if best is None or dev < best[0]:
                    best = (dev, o)
        print(fs.split('/')[-1], 'best order', best[1], 'rel dev %.2e' % best[0])
        o = best[1]
        M = bfm2(dcal, o)
        for p in range(2):
            ok = np.abs(S[p]) > 0
            r = S[p][ok]/M[p][ok]
            print('   parity %d: saved/rebuilt median %.4f%+.4fj, rel scatter %.2e, zero pattern same: %s' %
                  (p, np.median(r.real), np.median(r.imag), np.std(r)/abs(np.median(r)), np.array_equal(ok, np.abs(M[p]) > 0)))

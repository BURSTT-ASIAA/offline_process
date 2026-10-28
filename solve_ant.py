#!/usr/bin/env python

from loadh5 import *
from util_func import *
from delay_func2 import *
from numpy.linalg import pinv
import matplotlib.colors as colors
import sys, os.path

inp = sys.argv[0:]
pg  = inp.pop(0)

ant_flag = []
zmin    = 0.
zmax    = 250.
z2min   = 0.
z2max   = 5.
tmin    = 0
tmax    = 0
nAnt    = 16
gdiff   = np.zeros(nAnt)
aref    = 0
chlim   = None


def solve_masked(B, D):
    '''
    weighted least-squares solution of B . X = D, ignoring masked or non-finite D
    (equivalent to pinv(B) . D when nothing is masked)
    B:  shape (nBl, nAnt)
    D:  masked array, shape (nBl, ...)
    return X, masked array of shape (nAnt, ...)
    an antenna is masked where fewer than 2 of its baselines are valid
    '''
    sh = D.shape
    D2 = D.reshape((sh[0], -1))
    d = np.ma.getdata(D2)
    w = (~np.ma.getmaskarray(D2) & np.isfinite(d)).astype(float)   # (nBl, K)
    w[np.all(B==0, axis=1)] = 0.        # flagged baselines
    d = np.where(w>0, d, 0.)
    nb, na = B.shape
    BB = (B[:,:,np.newaxis]*B[:,np.newaxis,:]).reshape((nb, -1))
    N = np.dot(w.T, BB).reshape((-1, na, na))   # (K, nAnt, nAnt)
    rhs = np.dot((w*d).T, B)                    # (K, nAnt)
    X = np.einsum('kij,kj->ik', np.linalg.pinv(N), rhs)
    nvalid = np.dot((B!=0).astype(float).T, w)  # (nAnt, K)
    X = np.ma.array(X, mask=nvalid<2)
    return X.reshape((na,)+sh[1:])


usage = '''
solve antenna-based delay and Tsys from the baseline-based measurements
need to have run the two scripts:
    multi_fpag_vis.py
    show_fpga_windows.py

syntax:
    %s <.vish5_file(s)> [options]

    multiple files provided will be solved separately

options are:
    --flag 'list of antennas'
                        # specify a list of antennas to exclude from the analysis
                        # string quotes are needed if there are more than one antenna
                        # e.g. --flat '4 7'
                        # the antenna numbers are between 0, 15
    --tlim tmin tmax    # limiting the plotting and median time range in seconds
    --zlim zmin zmax    # specify the plot range for Tsys2D
                        # (default: %.0f %.0f K)
    --z2lim z2min z2max # specify the plot range for SEFD2D
                        # (default: %.3f %.3f MJy)
    --gdiff ANT GD_dB   # if a certain antenna had a different gain from the common value
                        # the difference can be specified here in dB
                        # e.g. a common 6.5dB gain was used in Tsys derivation
                        # but an antenna is 9.0dB, then the difference is 2.5dB
                        # default is 0dB for all antennas
                        # if multiple antennas are different, use --gdiff multiple times
    --aref ANT          # select reference antenna for Tau and Phi solutions
    --chlim MIN MAX     # set channel range for SEFD and Tsys average

    if the fringe-stopped results (winTsys_fs, winTsys2D_fs, winSEFD2D_fs, winCoeffRFI)
    from show_fpga_window.py exist, they are also solved and saved with the '_fs' suffix

''' % (pg, zmin, zmax, z2min, z2max)

if (len(inp)<1):
    sys.exit(usage)

files = []
while (inp):
    k = inp.pop(0)
    if (k == '--flag'):
        tmp = inp.pop(0).split()
        ant_flag = [int(x) for x in tmp]
    elif (k == '--tlim'):
        tmin = float(inp.pop(0))
        tmax = float(inp.pop(0))
    elif (k == '--zlim'):
        #tmp = inp.pop(0).split()
        #zmin, zmax = [float(x) for x in tmp]
        zmin = float(inp.pop(0))
        zmax = float(inp.pop(0))
    elif (k == '--z2lim'):
        z2min = float(inp.pop(0))
        z2max = float(inp.pop(0))
    elif (k == '--gdiff'):
        ai = int(inp.pop(0))
        gdiff[ai] = float(inp.pop(0))
    elif (k == '--aref'):
        aref = int(inp.pop(0))
    elif (k == '--chlim'):
        chmin = int(inp.pop(0))
        chmax = int(inp.pop(0))
        chlim = [chmin,chmax]
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        #fvis = k
        files.append(k)


for fvis in files:
    print('processing', fvis)
    odir = fvis.replace('.vish5', '.output')
    print('Tsys2D zlim (K):', zmin, zmax)
    print('SEFD2D z2lim (MJy):', z2min, z2max)


    winSec = getData(fvis, 'winSec')
    winNFT = getData(fvis, 'winNFT')
    nWin, nAnt, nChan = winNFT.shape
    gdiff = gdiff[:nAnt]    # truncate the array if fewer antennas were selected
    print('gdiff:', gdiff)

    if (chlim is None):
        chlim = [0, nChan]

    nBl = int(nAnt*(nAnt-1)/2)

    winTAU = getData(fvis, 'winTauRes')    # shape (nWin, nBl)
    #winTAU = np.ma.array(winTAU, mask=False)
    winPHI = getData(fvis, 'winPHI')    # shape (nWin, nBl)
    #winPHI = np.ma.array(winPHI, mask=False)
    winTsys = getData(fvis, 'winTsys')
    #winTsys is already a masked array
    do_Tsys2D = False
    tmp = getData(fvis, 'winTsys2D')    # shape (nWin, nBl, nChan)
    if (tmp is not None):
        winTsys2D = tmp
        do_Tsys2D = True
    do_SEFD2D = False
    tmp = getData(fvis, 'winSEFD2D')    # shape (nWin, nBl, nChan)
    if (tmp is not None):
        winSEFD2D = tmp
        do_SEFD2D = True
    # fringe-stopped results (optional)
    winTsys_fs   = getData(fvis, 'winTsys_fs')      # shape (nWin, nBl)
    winTsys2D_fs = getData(fvis, 'winTsys2D_fs')    # shape (nWin, nBl, nChan)
    winSEFD2D_fs = getData(fvis, 'winSEFD2D_fs')    # shape (nWin, nBl, nChan)
    do_fs = winTsys_fs is not None

    #if (True):  # make Tsys correction in this script
    if (False):  # Tsys correction already made in show_fpag_vis.py
        winCoeff = getData(fvis, 'winCoeff')
        medCoeff = np.ma.median(np.ma.abs(winCoeff), axis=2)
        winTsys *= (1.-medCoeff)

    if (tmax==0):
        tmax = winSec.max()
    tsel = (winSec>=tmin) * (winSec<=tmax)
    winSec = winSec[tsel]
    winTsys2D = winTsys2D[tsel]
    winSEFD2D = winSEFD2D[tsel]
    winTAU = winTAU[tsel]
    winPHI = winPHI[tsel]
    winTsys = winTsys[tsel]
    if (do_fs):
        winTsys_fs = winTsys_fs[tsel]
        winTsys2D_fs = winTsys2D_fs[tsel]
        winSEFD2D_fs = winSEFD2D_fs[tsel]
    nWin, nBl, nChan = winTsys2D.shape


    # solve X for equation: A . X = D   (Tau and Phi)
    # solve X for equation: B . X = D   (log(Tsys) and log(SEFD))
    # construct matrix A: 
    A = np.zeros((nBl, nAnt))
    B = np.zeros((nBl, nAnt))
    b = -1
    for ai in range(nAnt-1):
        for aj in range(ai+1, nAnt):
            b += 1
            if (ai in ant_flag or aj in ant_flag):
                #winTAU.mask[:,b] = True
                #winPHI.mask[:,b] = True
                #winTsys.mask[:,b] = True
                winTAU[:,b] = 0.
                winPHI[:,b] = 0.
                winTsys[:,b] = 0.
                if (do_Tsys2D):
                    winTsys2D[:,b] *= 0.
                if (do_SEFD2D):
                    winSEFD2D[:,b] *= 0.
                if (do_fs):
                    winTsys_fs[:,b] = 0.
                    winTsys2D_fs[:,b] *= 0.
                    winSEFD2D_fs[:,b] *= 0.
            else:
                A[b,ai] = 1.
                A[b,aj] = -1.
                B[b,ai] = 0.5
                B[b,aj] = 0.5


    # pseudo-inverse
    Ainv = pinv(A)
    Binv = pinv(B)
    # shape (nAnt, nBl)

    # model M = Ainv . D
    # residual R = A . M - D
    # and the same for B



    w0 = 4.
    h0 = 3.
    ww = w0 * nAnt
    hh = h0 * nAnt

    nPP = 4 if do_fs else 3
    for pp in range(nPP):
        if (pp == 0):   # TAU
            D = winTAU.T  # shape (nBl, nWin)
            name = 'Tau'
            ylab = 'res.Tau (ns)'
        elif (pp == 1):
            D = winPHI.T  # shape (nBl, nWin)
            name = 'Phi'
            ylab = 'res.Phi (rad)'
        elif (pp == 2):
            D = np.ma.log10(winTsys).T
            name = 'Tsys'
            #ylab = 'log10(Tsys/K)'
            ylab = 'Tsys (K)'
        elif (pp == 3):
            D = np.ma.log10(winTsys_fs).T
            name = 'Tsys_fs'
            ylab = 'Tsys (K), fringe-stop'

        if (pp < 2):
            M = np.dot(Ainv, D)  # shape (nAnt, nWin)
            M -= M[aref,:]   # Ant0 as reference Tau
            R = np.dot(A, M) - D
        else:
            M = solve_masked(B, D)  # shape (nAnt, nWin)
            BM = np.ma.dot(B, M)
            R = BM - D
        if (pp == 1):
            R = pwrap2(R)
        std = np.sqrt(R.var(axis=0))

        fig, sub = blsubplots(na=nAnt+1, figsize=(ww,hh))
        for ai in range(nAnt):
            for aj in range(ai+1,nAnt+1):
                ax = sub[ai,aj-1]
                ax.axhline(0., color='k')
                ax.axvline(0., color='k')

        # antenna solution
        savM_med = []
        savM_wta = []
        savM_min = []
        for ai in range(nAnt):
            ax = sub[ai,ai]
            if (pp < 2):        # Tau and Phi
                ax.plot(winSec, M[ai], 'r-')
                ax.fill_between(winSec, M[ai]-std, M[ai]+std, color='r', alpha=0.3)
                medM = np.median(M[ai])
                wtaM = np.ma.sum(M[ai]/std**2)/np.ma.sum(1./std**2)
                minM = M[ai].min()
                savM_med.append(medM)
                savM_wta.append(wtaM)
                savM_min.append(minM)
                ax.text(0.05, 0.90, 'med: %.3f'%(medM), transform=ax.transAxes)
                ax.text(0.05, 0.85, 'wt-avg: %.3f'%(wtaM), transform=ax.transAxes)
                if (pp==0): # TAU
                    ax.set_ylim(-20,20)
            else:               # Tsys
                ax.plot(winSec, 10.**M[ai], 'r-')
                ax.fill_between(winSec, 10.**(M[ai]-std), 10.**(M[ai]+std), color='r', alpha=0.3)
                # nan if the antenna is fully masked (e.g. flagged)
                medM = float(np.ma.filled(np.ma.median(M[ai]), np.nan))
                #print('ai', ai, 'std', std, 'M', M[ai])
                wtaM = float(np.ma.filled(np.ma.sum(M[ai]/std**2)/np.ma.sum(1./std**2), np.nan))
                minM = float(np.ma.filled(M[ai].min(), np.nan))
                savM_med.append(10**medM)
                savM_wta.append(10**wtaM)
                savM_min.append(10**minM)
                ax.text(0.05, 0.90, 'med: %.0fK'%(10**medM), transform=ax.transAxes)
                ax.text(0.05, 0.85, 'wt-avg: %.0fK'%(10**wtaM), transform=ax.transAxes)
                ax.text(0.05, 0.80, 'min: %.0fK'%(10**minM), transform=ax.transAxes)
                #ax.set_ylim(50, 200)
                ax.set_ylim(zmin, zmax)
                ax.grid()
            ax.set_title('Ant%d'%ai)
            ax.set_xlabel('Time (sec)')
            ax.set_ylabel(ylab)

        # baseline data and residual
        b = -1
        for ai in range(nAnt-1):
            for aj in range(ai+1, nAnt):
                b += 1
                corr = 'bl%d-%d' % (ai,aj)
                ax = sub[ai,aj]
                if (pp < 2):    # Tau and Phi
                    ax.plot(winSec, D[b])
                    ax.plot(winSec, R[b])
                    medD = np.median(D[b])
                    ax.text(0.05, 0.90, 'med: %.3f'%(medD), transform=ax.transAxes)
                else:           # Tsys
                    ax.plot(winSec, 10.**(D[b]))
                    ax.plot(winSec, 10.**(BM[b]))
                    medD = np.ma.median(D[b])
                    minD = D[b].min()
                    ax.text(0.05, 0.90, 'med: %.0fK'%(10**medD), transform=ax.transAxes)
                    ax.text(0.05, 0.85, 'min: %.0fK'%(10**minD), transform=ax.transAxes)
                    ax.grid()
                ax.set_title(corr)

        fig.text(0.03, 0.45, 'bl.data', color='C0')
        fig.text(0.03, 0.35, 'ant.solution', color='r')
        if (pp < 2):
            fig.text(0.03, 0.40, 'bl.residual', color='C1')
        else:
            fig.text(0.03, 0.40, 'bl.model', color='C1')



        fig.tight_layout(rect=[0,0.03,1,0.95])
        fig.suptitle('%s, antenna %s solution'%(fvis, name))
        png = '%s/antenna_%s.png'%(odir, name)
        fig.savefig(png)

        dname = 'ant%s'%name
        adoneh5(fvis, M.T, dname)
        attrs = {'median':savM_med, 'wt-avg':savM_wta,  'min':savM_min}
        putAttrs(fvis, attrs, dest=dname)
        dname = 'ant%s_err'%name
        adoneh5(fvis, std, dname)

        ## additional figure of antTau
        if (pp==0):
            fig, s2d = plt.subplots(4,4,figsize=(16,12),sharex=True,sharey=False)
            sub = s2d.flatten()
            for ai in range(nAnt):
                ax = sub[ai]
                ax.plot(winSec, M[ai])
                #ax.set_ylim(-20,20)
                ax.text(0.03, 0.90, 'ant%02d'%ai, transform=ax.transAxes)
                #ax.axhline(0, color='gray')
                medM = np.median(M[ai])
                ax.text(0.03, 0.80, 'med: %.3f'%medM, transform=ax.transAxes)
                ax.axhline(medM, color='r', ls=':')
                ax.fill_between(winSec, medM-0.1, medM+0.1, color='r', alpha=0.3)
                if (ai%4==0):
                    ax.set_ylabel('tau (ns)')
                if (ai>=12):
                    ax.set_xlabel('Time (sec)')
            fig.tight_layout(rect=[0,0.03,1,0.95])
            fig.subplots_adjust(hspace=0)
            fig.suptitle('ant instrument Tau, %s'%odir)
            fig.savefig('%s/antenna_Tau2.png'%odir)
            plt.close(fig)


    Tsys2D_list = []
    if (do_Tsys2D):
        Tsys2D_list.append(('', winTsys2D))
    if (do_fs):
        Tsys2D_list.append(('_fs', winTsys2D_fs))
    for sfx, Y2D in Tsys2D_list:
        print('solve Tsys2D%s ...'%sfx)
        freq = getData(fvis, 'freq')    # MHz, nChan
        sec  = getData(fvis, 'winSec')  # sec, nWin
        sec = sec[tsel]
        T, F = np.meshgrid(sec, freq, indexing='xy')
        print(T.shape,F.shape)
        print(sec.shape, freq.shape)

        D = np.ma.log10(Y2D).transpose((1,2,0))     # new shape (nBl, nChan, nWin)
        M = solve_masked(B, D)                      # output shape (nAnt, nChan, nWin)
        print(gdiff.shape, gdiff)
        M += (gdiff/10.).reshape((-1,1,1))          # Tsys is proportional to gain (linear)
                                                    # M(Tsys) is in log10 scale,
                                                    # gdiff is in dB, which is 10.*log10(gain)
        vmin = zmin
        vmax = zmax

        M_freq  = np.ma.median(M, axis=2)
        M_freq2 = M.min(axis=2)
        print(M.shape)

        Mout = M.transpose((2,0,1))
        adoneh5(fvis, Mout, 'antTsys2D%s'%sfx)      # log10(Tsys2D), shape: (nWin, nAnt, nChan)

        #fig, sub = plt.subplots(4,4,figsize=(16,12),sharex=True,sharey=True)
        fig = plt.figure(figsize=(16,12))
        gs = fig.add_gridspec(4,8,height_ratios=[1,1,1,1], width_ratios=[3,1,3,1,3,1,3,1], wspace=0, hspace=0.15, left=0.07, right=0.96, bottom=0.08, top=0.93)

        sub = []
        sub2 = []
        for ii in range(4):
            for jj in range(4):
                ax = fig.add_subplot(gs[ii,jj*2])
                sub.append(ax)
                ax2 = fig.add_subplot(gs[ii,jj*2+1])
                sub2.append(ax2)

        for ai in range(nAnt):
            ax = sub[ai]
            ax2 = sub2[ai]

            ax.pcolormesh(T, F, 10**(M[ai]), vmin=vmin, vmax=vmax, shading='nearest')
            M_freq_med = 10**(np.ma.median(M_freq[ai,chlim[0]:chlim[1]]))
            ax.set_title('Ant%02d, %.0fK'%(ai,M_freq_med))
            if (ai%4==0):
                ax.set_ylabel('Frequency (MHz)')
            else:
                ax.set_yticks([])
            if (ai>=12):
                ax.set_xlabel('Time (sec)')
                ax2.set_xlabel('Tsys (K)')
            else:
                ax.set_xticks([])
                #ax2.set_xticks([])
                #ax2.set_xticklabels([])

            ax2.plot(10**(M_freq[ai]), freq, label='med')
            #ax2.plot(10**(M_freq2[ai]), freq, label='min')
            ax2.set_xlim(vmin, vmax)
            ax2.set_ylim(freq[0], freq[-1])
            #ax2.legend()
            #ax2.text(0.05, 1.02, 'T_sys=%.0fK'%10**(np.ma.median(M_freq2[ai])), transform=ax2.transAxes)
            #ax2.text(0.05, 1.02, 'T_sys=%.0fK'%10**(np.ma.median(M_freq[ai])), color='C0', transform=ax2.transAxes)
            ax2.axvline(M_freq_med, color='C1', linestyle=':')
            #print(ai, M_freq_med)
            #ax2.set_yticks([])
            ax2.set_yticklabels([])
            ax2.grid()

        fig.suptitle('%s, Tsys2D%s, antenna solution'%(fvis, sfx))
        fig.savefig('%s/antenna_Tsys2D%s.png'%(odir, sfx))
        plt.close(fig)


    SEFD2D_list = []
    if (do_SEFD2D):
        SEFD2D_list.append(('', winSEFD2D))
    if (do_fs):
        SEFD2D_list.append(('_fs', winSEFD2D_fs))
    for sfx, Y2D in SEFD2D_list:
        print('solve SEFD2D%s ...'%sfx)
        freq = getData(fvis, 'freq')    # MHz, nChan
        sec  = getData(fvis, 'winSec')  # sec, nWin
        sec = sec[tsel]
        T, F = np.meshgrid(sec, freq, indexing='xy')
        print(T.shape,F.shape)
        print(sec.shape, freq.shape)

        D = np.ma.log10(Y2D).transpose((1,2,0))     # new shape (nBl, nChan, nWin)
        lM = solve_masked(B, D)                      # output shape (nAnt, nChan, nWin)
        M = (10**lM)/1e6                             # convert back to linear scale in MJy

        vmin = z2min
        vmax = z2max

        M_freq  = np.ma.median(M, axis=2)
        M_freq2 = M.min(axis=2)
        print(M.shape)

        Mout = M.transpose((2,0,1))
        adoneh5(fvis, Mout, 'antSEFD2D%s'%sfx)      # SEFD2D, shape: (nWin, nAnt, nChan)

        #fig, sub = plt.subplots(4,4,figsize=(16,12),sharex=True,sharey=True)
        fig = plt.figure(figsize=(16,12))
        gs = fig.add_gridspec(4,8,height_ratios=[1,1,1,1], width_ratios=[3,1,3,1,3,1,3,1], wspace=0, hspace=0.15, left=0.07, right=0.96, bottom=0.08, top=0.93)

        sub = []
        sub2 = []
        for ii in range(4):
            for jj in range(4):
                ax = fig.add_subplot(gs[ii,jj*2])
                sub.append(ax)
                ax2 = fig.add_subplot(gs[ii,jj*2+1])
                sub2.append(ax2)

        for ai in range(nAnt):
            ax = sub[ai]
            ax2 = sub2[ai]

            ax.pcolormesh(T, F, M[ai], vmin=vmin, vmax=vmax, shading='nearest')
            M_freq_med = np.ma.median(M_freq[ai,chlim[0]:chlim[1]])
            ax.set_title('Ant%02d, %.3fMJy'%(ai,M_freq_med))
            if (ai%4==0):
                ax.set_ylabel('Frequency (MHz)')
            else:
                ax.set_yticks([])
            if (ai>=12):
                ax.set_xlabel('Time (sec)')
                ax2.set_xlabel('SEFD (MJy)')
            else:
                ax.set_xticks([])
                #ax2.set_xticks([])
                #ax2.set_xticklabels([])

            ax2.plot(M_freq[ai], freq, label='med')
            #ax2.plot(10**(M_freq2[ai]), freq, label='min')
            ax2.set_xlim(vmin, vmax)
            ax2.set_ylim(freq[0], freq[-1])
            #ax2.legend()
            #ax2.text(0.05, 1.02, 'T_sys=%.0fK'%10**(np.ma.median(M_freq2[ai])), transform=ax2.transAxes)
            #ax2.text(0.05, 1.02, 'T_sys=%.0fK'%10**(np.ma.median(M_freq[ai])), color='C0', transform=ax2.transAxes)
            ax2.axvline(M_freq_med, color='C1', linestyle=':')
            #print(ai, M_freq_med)
            #ax2.set_yticks([])
            ax2.set_yticklabels([])
            ax2.grid()

        fig.suptitle('%s, SEFD2D%s, antenna solution'%(fvis, sfx))
        fig.savefig('%s/antenna_SEFD2D%s.png'%(odir, sfx))
        plt.close(fig)



    ## RFI gauge from the fringe-stopping (the discarded part), per baseline
    winCoeffRFI = getData(fvis, 'winCoeffRFI')  # shape (nWin, nBl, nChan)
    winCoeffFS  = getData(fvis, 'winCoeffFS')   # shape (nWin, nBl, nChan)
    if (winCoeffRFI is not None and winCoeffFS is not None):
        print('plot RFI gauge ...')
        freq = getData(fvis, 'freq')    # MHz, nChan
        rfi = np.ma.abs(winCoeffRFI[tsel][:,:,chlim[0]:chlim[1]])
        src = winCoeffFS[tsel][:,:,chlim[0]:chlim[1]]
        med_rfi = np.ma.median(np.ma.median(rfi, axis=0), axis=1)                                        # (nBl,)
        med_src = np.ma.median(np.ma.median(src, axis=0), axis=1)                                         # (nBl,)
        mat_rfi = np.ma.masked_all((nAnt, nAnt))
        mat_src = np.ma.masked_all((nAnt, nAnt))
        b = -1
        for ai in range(nAnt-1):
            for aj in range(ai+1, nAnt):
                b += 1
                if (ai in ant_flag or aj in ant_flag):
                    continue
                mat_rfi[ai,aj] = mat_rfi[aj,ai] = med_rfi[b]
                mat_src[ai,aj] = mat_src[aj,ai] = med_src[b]

        fig, sub = plt.subplots(2,2,figsize=(14,12))
        for k, (mat, tit) in enumerate([(mat_src, 'fringe-stopped coeff (source)'), (mat_rfi, '|discarded| coeff (RFI gauge)')]):
            ax = sub[0,k]
            s = ax.imshow(mat, origin='lower', norm=colors.LogNorm())
            cb = plt.colorbar(s, ax=ax)
            cb.set_label('median coeff, ch%d-%d'%(chlim[0],chlim[1]))
            ax.set_xlabel('Ant')
            ax.set_ylabel('Ant')
            ax.set_title(tit)
        ax = sub[1,0]
        ax.plot(freq, np.ma.median(np.ma.median(winCoeffFS[tsel], axis=0), axis=0), label='fringe-stopped (source)')
        ax.plot(freq, np.ma.median(np.ma.median(np.ma.abs(winCoeffRFI[tsel]), axis=0), axis=0), label='|discarded| (RFI gauge)')
        ax.axvspan(freq[chlim[0]], freq[chlim[1]-1], color='gray', alpha=0.2)
        ax.set_yscale('log')
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('median coeff over time and baselines')
        ax.legend()
        ax = sub[1,1]
        ax.scatter(med_src, med_rfi, marker='.')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('fringe-stopped coeff (source)')
        ax.set_ylabel('|discarded| coeff (RFI gauge)')
        ax.set_title('per baseline, ch%d-%d'%(chlim[0],chlim[1]))
        fig.tight_layout(rect=[0,0.03,1,0.95])
        fig.suptitle('%s, fringe-stop RFI gauge'%fvis)
        fig.savefig('%s/baseline_RFI_fs.png'%odir)
        plt.close(fig)

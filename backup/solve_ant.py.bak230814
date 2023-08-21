#!/usr/bin/env python

from loadh5 import *
from util_func import *
from delay_func2 import *
from numpy.linalg import pinv
import sys, os.path

inp = sys.argv[0:]
pg  = inp.pop(0)

ant_flag = []
zmin    = 0.
zmax    = 300.
nAnt    = 16
gdiff   = np.zeros(nAnt)


usage = '''
solve antenna-based delay and Tsys from the baseline-based measurements
need to have run the two scripts:
    multi_fpag_vis.py
    show_fpga_windows.py

syntax:
    %s <.vish5_file> [options]

options are:
    --flag 'list of antennas'
                        # specify a list of antennas to exclude from the analysis
                        # string quotes are needed if there are more than one antenna
                        # e.g. --flat '4 7'
                        # the antenna numbers are between 0, 15
    --zlim zmin zmax    # specify the plot range for Tsys2D
                        # (default: %.0f %.0f K)
    --gdiff ANT GD_dB   # if a certain antenna had a different gain from the common value
                        # the difference can be specified here in dB
                        # e.g. a common 6.5dB gain was used in Tsys derivation
                        # but an antenna is 9.0dB, then the difference is 2.5dB
                        # default is 0dB for all antennas
                        # if multiple antennas are different, use --gdiff multiple times

''' % (pg, zmin, zmax)

if (len(inp)<1):
    sys.exit(usage)

while (inp):
    k = inp.pop(0)
    if (k == '--flag'):
        tmp = inp.pop(0).split()
        ant_flag = [int(x) for x in tmp]
    elif (k == '--zlim'):
        tmp = inp.pop(0).split()
        zmin, zmax = [float(x) for x in tmp]
    elif (k == '--gdiff'):
        ai = int(inp.pop(0))
        gdiff[ai] = float(inp.pop(0))
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        fvis = k

odir = fvis.replace('.vish5', '.output')
print('zlim:', zmin, zmax)


winSec = getData(fvis, 'winSec')
winNFT = getData(fvis, 'winNFT')
nWin, nAnt, nChan = winNFT.shape
gdiff = gdiff[:nAnt]    # truncate the array if fewer antennas were selected
print('gdiff:', gdiff)

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

#if (True):  # make Tsys correction in this script
if (False):  # Tsys correction already made in show_fpag_vis.py
    winCoeff = getData(fvis, 'winCoeff')
    medCoeff = np.ma.median(np.ma.abs(winCoeff), axis=2)
    winTsys *= (1.-medCoeff)


# solve X for equation: A . X = D   (Tau and Phi)
# solve X for equation: B . X = D   (log(Tsys))
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

for pp in range(3):
    if (pp == 0):   # TAU
        D = winTAU.T  # shape (nBl, nWin)
        name = 'Tau'
        ylab = 'res.Tau (ns)'
    elif (pp == 1):
        D = winPHI.T  # shape (nBl, nWin)
        name = 'Phi'
        ylab = 'res.Phi (rad)'
    elif (pp == 2):
        D = np.log10(winTsys).T
        name = 'Tsys'
        #ylab = 'log10(Tsys/K)'
        ylab = 'Tsys (K)'

    if (pp < 2):
        M = np.dot(Ainv, D)  # shape (nAnt, nWin)
        M -= M[0,:]   # Ant0 as reference Tau
        R = np.dot(A, M) - D
    else:
        M = np.dot(Binv, D)  # shape (nAnt, nWin)
        BM = np.dot(B, M)
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
            medM = np.median(M[ai])
            #print('ai', ai, 'std', std, 'M', M[ai])
            wtaM = np.ma.sum(M[ai]/std**2)/np.ma.sum(1./std**2)
            minM = M[ai].min()
            savM_med.append(10**medM)
            savM_wta.append(10**wtaM)
            savM_min.append(10**minM)
            ax.text(0.05, 0.90, 'med: %.0fK'%(10**medM), transform=ax.transAxes)
            ax.text(0.05, 0.85, 'wt-avg: %.0fK'%(10**wtaM), transform=ax.transAxes)
            ax.text(0.05, 0.80, 'min: %.0fK'%(10**minM), transform=ax.transAxes)
            ax.set_ylim(50, 200)
        ax.set_title('Ant%d'%ai)
        ax.set_xlabel('time (sec)')
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
                medD = np.median(D[b])
                minD = D[b].min()
                ax.text(0.05, 0.90, 'med: %.0fK'%(10**medD), transform=ax.transAxes)
                ax.text(0.05, 0.85, 'min: %.0fK'%(10**minD), transform=ax.transAxes)
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


if (do_Tsys2D):
    print('solve Tsys2D ...')
    freq = getData(fvis, 'freq')    # MHz, nChan
    sec  = getData(fvis, 'winSec')  # sec, nWin
    T, F = np.meshgrid(sec, freq, indexing='xy')
    print(T.shape,F.shape)
    print(sec.shape, freq.shape)

    D = np.log10(winTsys2D).transpose((1,2,0))  # new shape (nBl, nChan, nWin)
    M = np.tensordot(Binv, D, axes=(1,0))       # output shape (nAnt, nChan, nWin)
    M += (gdiff/10.).reshape((-1,1,1))          # Tsys is proportional to gain (linear)
                                                # M(Tsys) is in log10 scale,
                                                # gdiff is in dB, which is 10.*log10(gain)
    vmin = zmin
    vmax = zmax

    M_freq  = np.ma.median(M, axis=2)
    M_freq2 = M.min(axis=2)
    print(M.shape)


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
        ax.set_title('Ant%02d'%ai)
        if (ai%4==0):
            ax.set_ylabel('freq (MHz)')
        else:
            ax.set_yticks([])
        if (ai>=12):
            ax.set_xlabel('time (sec)')
            ax2.set_xlabel('Tsys (K)')
        else:
            ax.set_xticks([])
            ax2.set_xticks([])

        ax2.plot(10**(M_freq[ai]), freq, label='med')
        ax2.plot(10**(M_freq2[ai]), freq, label='min')
        ax2.set_xlim(vmin, vmax)
        #ax2.legend()
        ax2.text(0.05, 1.02, 'T_sys=%.0fK'%10**(np.ma.median(M_freq2[ai])), transform=ax2.transAxes)
        ax2.set_yticks([])

    fig.suptitle('%s, Tsys2D, antenna solution'%fvis)
    fig.savefig('%s/antenna_Tsys2D.png'%odir)
    plt.close(fig)


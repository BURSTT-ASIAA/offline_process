#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.time import Time
from astropy.stats import sigma_clip
import time

from loadh5 import *
from util_func import *
from pyplanet import *
from delay_func2 import *
from array_func import blsubplots

import plot_util

DB = loadDB()

inp = sys.argv[0:]
pg  = inp.pop(0)

#nChan = 1024       # get from .vish5 file
rfmin   = 0.
rfmax   = 0.
aconf   = None
ai      = 0
aj      = 1
srcflux = 0.
f410    = 0.
f610    = 0.
do_fitClkOff = False
maxClkOff = 20.     # clock offset search range (+/- maxClkOff)
dClkOff = 0.01      # step in clock offset search
noClkCorr = False
do_check = False
save_bl = False     # save the per-baseline plots (.output/bl*) and the .check plots
do_tausrc = False   # force redo of source delay fitting
maxTau1 = 100.       # samples @ 400MHz (i.e. 2.5ns)
dTau1   = 1.0
maxTau2 = 1.0
dTau2   = 0.01
blfit   = 0         # default disable fitting baslines
anglim  = 30.
angsep  = 3.
poslim  = 0.5
possep  = 0.1
gant    = 6.0       # antenna max gain in dB
Ehwhm   = 30.       # E-plane half-power half width in deg
Hhwhm   = 60.       # H-plane half-power half width in deg
dellim  = 150.
delsep  = 0.5
body    = 'sun'
revtau  = False
revpos  = False
swapEH  = False
zlim    = [0., 300.]
tlim    = None
freq_ref = 400e6    # reference frequency to scale the SEFD
flam_ref = 2.998e8/freq_ref # reference lambda
site    = None
theta_rot = None
clight  = 2.998e8


usage = '''
plot all windows of a given .vish5 file
multiple files can be given
each file has a corresponding plot

syntax:
    %s <.vish5_files> [options]

options are:
    -b BODY         # specify the target name for fringe estimate
                    # (default: %s)

    --flux JY       # specify the source flux in Jy for SEFD calculation
    --interp F410 F610
                    # instead of specifying a fixed flux
                    # specify the (Solar) flux at both 410MHz and 610MHz
                    # linear interpolation is used to estimate the flux
                    # (in Jy)   Note: overrides --flux

    --rf MIN MAX    # RF range to use for time plot and the per-window Tsys (winTsys, winTsys_fs)
                    # (both default to 0, for using the full range)

    --site SITE     # specify the site name, if not specified, will determine from ANT_CONF below
    --rot DEG       # specify the array rotation angle in degree
                    # FUS = -3.0
                    # LTN = +0.5
                    # GRN, KMN = 0.0
    --aconf ANT_CONT # specify an antenna configuration file
                    # default is none

    --gant GAIN     # antenna max gain in dB (%.1f dB)

    --hwhm EWdeg NSdeg
                    # change the E-W and N-S beam width in half-width at half maximum (in deg)
                    # default: %d %d

    --swapEH        # swap E and H plane attenuation (so dipoles point EW)
                    # (default: dipoles point NS)
    --zlim min max  # set the min/max of Tsys2D map in K
                    # the SEFD is 1/60 zlim in MJy
    --tlim min max  # set the min/max in time range to summarize Tsys2D and SEFD2D
                    # in seconds

    --save_bl       # save the per-baseline plots (<.output>/bl*.png)
                    # and the diagnostic plots in <.check>/
                    # (default: only the summary plot and the datasets are saved)

    ## when --aconf is given, a fringe-stopped estimate is also made:
    ##   coeff = A*atten(t)*exp(-2pi i f tau(t)) + R  is fitted per (bl,ch)
    ##   the per-window coeff along the source fringe --> winTsys*_fs, winSEFD2D_fs
    ##   the discarded part (not following the fringe) --> winCoeffRFI
    ##   (and the *Tsys2D_fs, *SEFD2D_fs, *rfi_fs.png plots with --save_bl)

    ## to select only a subset of antennas for processing
    --ai AI         # specify the antenna numbers (only needed if --aconf is used)
    --aj AJ         # (AI, AJ are default to 0, 1)

    ## for fitting clock offset
    --fit           # fit and remove the Clock Offset

    --clk 'LIMIT SEP'   # set the +/-LIMIT and SEP for clock offset fitting (in samples)
                    # default: (%.3f, %.3f)

    --no-clkcorr    # disable using ClkCorr Spec and Coeff for analysis

    --force-tausrc  # force redo of source delay fitting
                    # (FFT coarse delay + phase-slope fine delay, within +/-%.0f samples)
    --rev-tau       # reverse deflay direction, debugging
    --rev-pos       # reverse the antenna config file (for idt-beamformed data)

    --plot          # plot the Clock Offset phase correction and the waterfalls
                    # in <.check>/ (implies --save_bl)

    --del 'LIMIT SEP' # set the +/-LIMIT and SEP for delay (in samples)
                    # default: +/-150, 0.5

    ## for fitting antenna positions
    --init 'PARAMs' # assuming a linear array:
                    # param 1: direction of the array (in degree)
                    # param 2+: distance from Ant0 (in meter)
    --ang 'LIMIT SEP' # set the +/-LIMIT and SEP for angle (in degree)
                    # default: +/-30, 3
    --pos 'LIMIT SEP' # set the +/-LIMIT and SEP for antenna positions (in meter)
                    # default: +/-0.5, 0.1

''' % (pg, body, gant, Ehwhm, Hhwhm, maxClkOff, dClkOff, maxTau1)

if (len(inp) < 1):
    sys.exit(usage)

files = []
while (inp):
    k = inp.pop(0)
    if (k == '--rf'):
        rfmin = float(inp.pop(0))
        rfmax = float(inp.pop(0))
    elif (k == '-b'):
        body = inp.pop(0)
    elif (k == '--site'):
        site = inp.pop(0).lower()
    elif (k == '--rot'):
        theta_rot = float(inp.pop(0)) # degree
    elif (k == '--aconf'):
        aconf = inp.pop(0)
    elif (k == '--ai'):
        ai = int(inp.pop(0))
    elif (k == '--aj'):
        aj = int(inp.pop(0))
    elif (k == '--flux'):
        srcflux = float(inp.pop(0))
    elif (k == '--interp'):
        f410 = float(inp.pop(0))
        f610 = float(inp.pop(0))
    elif (k == '--gant'):
        gant = float(inp.pop(0))
    elif (k == '--hwhm'):
        Ehwhm = float(inp.pop(0))
        Hhwhm = float(inp.pop(0))
    elif (k == '--swapEH'):
        swapEH = True
    elif (k == '--zlim'):
        zlim[0] = float(inp.pop(0))
        zlim[1] = float(inp.pop(0))
    elif (k == '--tlim'):
        tmin = float(inp.pop(0))
        tmax = float(inp.pop(0))
        tlim = [tmin, tmax]
    elif (k == '--save_bl'):
        save_bl = True
    elif (k == '--fit'):
        do_fitClkOff = True
    elif (k == '--clk'):
        tmp = inp.pop(0).split()
        maxClkOff = float(tmp[0])
        dClkOff   = float(tmp[1])
    elif (k == '--no-clkcorr'):
        noClkCorr = True
    elif (k == '--force-tausrc'):
        do_tausrc = True
    elif (k == '--rev-tau'):
        revtau = True
    elif (k == '--rev-pos'):
        revpos = True
    elif (k == '--plot'):
        do_check = True
        save_bl = True
    elif (k == '--init'):
        tmp = inp.pop(0).split()
        ang0 = float(tmp.pop(0))
        pos0 = [float(x) for x in tmp]
        blfit = 1   # enable fitting basline
    elif (k == '--ang'):
        tmp = inp.pop(0).split()
        anglim = float(tmp[0])
        angsep = float(tmp[1])
    elif (k == '--pos'):
        tmp = inp.pop(0).split()
        poslim = float(tmp[0])
        possep = float(tmp[1])
    elif (k == '--del'):
        tmp = inp.pop(0).split()
        dellim = float(tmp[0])
        delsep = float(tmp[1])
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        files.append(k)
print('file:', files)


if (not site is None):
    if (site.lower() == 'fus'):
        theta_rot = -3.0
        if (aconf is None):
            aconf = '16x1.0y0.5'
    elif (site.lower() == 'ltn'):
        theta_rot = 0.5
        if (aconf is None):
            aconf = '16x1.0y1.0'
    elif (site.lower() == 'grn' or site.lower() == 'kmn'):
        theta_rot = 0.0
        if (aconf is None):
            aconf = '16x1.0y1.0'
    
if (theta_rot == None):
    theta_rot = 0.0


if (aconf is None):
    do_model = False
else:
    do_model = True
    tmp = os.path.basename(aconf)
    tmp2 = tmp.split('_')
    if (site is None):
        site = tmp2[0].lower()
    #antXYZ = np.loadtxt(aconf, usecols=(1,2,3), ndmin=2)
    if (os.path.isfile(aconf)):
        antXYZ = arrayConf(aconf, theta_rot=theta_rot)
    else:
        antXYZ = arrayConf(aconf, nRow=1, theta_rot=theta_rot) # only 1 row?

if (do_model and revpos):
    print('reversing the ant pos...')
    antXYZ = np.flip(antXYZ, axis=0)


G0 = 10.**(gant/10.)



##################################################
## analysis functions

def atten(x, hwhm):
    '''
    Gaussian attenuation (linear) given offset x and the half-power full width
    '''
    sig = hwhm/np.sqrt(2.*np.log(2.))
    return np.exp(-x**2/2./sig**2)


def circ_median(pha):
    '''
    median of phases (in rad), taken on the median cos and sin
    (takes care when the phase is close to +/-pi)
    '''
    cpx = np.ma.exp(1.j*pha)
    mr = np.ma.median(cpx.real)
    mi = np.ma.median(cpx.imag)
    return np.angle(mr+1.j*mi)


def cpx_median(spec, axis):
    '''median of the real and imag parts separately'''
    return np.ma.median(spec.real, axis=axis) + 1.j*np.ma.median(spec.imag, axis=axis)


def fit_clock_offset(spec1, coeff1, sec):
    '''
    find and remove the clock offset per window and baseline
    return: ClkOff (nWin, nBl), the corrected spec and coeff
    '''
    nWin, nBl, nChan = spec1.shape
    spec3 = spec1.copy()
    coeff3 = coeff1.copy()
    params = np.arange(-maxClkOff,maxClkOff,dClkOff)
    ClkOff = np.zeros((nWin, nBl))
    for i in range(nWin):
        print('...',i,'t%08.1f'%sec[i])
        for b in range(nBl):
            vispha = np.ma.angle(spec1[i,b])
            viserr = np.ones(nChan) * 0.3
            like = lnlike(params, vispha, viserr, nChan)
            maxpara = params[like.argmax()]
            phimod = phimodel(maxpara, nChan)
            spec3[i,b] = spec1[i,b] / np.exp(1.j*phimod)
            coeff3[i,b] = coeff1[i,b] / np.exp(1.j*phimod)
            ClkOff[i,b] = maxpara
    return ClkOff, spec3, coeff3


def srcdelay_phase(spec, freq):
    '''
    visibility phase with the spectral masking used in the source delay fitting
    spec: shape (..., nChan)
    '''
    vispha = np.ma.angle(spec)
    mm = np.ma.getmaskarray(vispha).copy()  # do not modify the mask of the input
    mm[...,freq<300.] = True    # low-freq RFI at Fushan
    mm[...,freq>750.] = True    # 4G signals
    vispha.mask = mm
    return vispha


def srcdelay_model(TAU, freq):
    '''
    phase model (rad) of the source delay TAU (in 400MHz samples) at RF freq (MHz)
    output shape: TAU.shape + freq.shape
    (identical to phimodel(TAU, 1024, oChan=1536) for the 400-800MHz band)
    '''
    return 2.*np.pi*np.multiply.outer(TAU, freq)/400.


def fit_source_delay(spec2, freq, sec, pad=32, niter=3):
    '''
    fit the source delay (not instrument delay) per window and baseline
    coarse: peak of the zero-padded FFT across channels (as in second_cal64.py)
    fine:   LS fit of the residual phase slope across channels, with outliers (RFI) clipped
    only the phase is used (unit-amplitude phasors), searched within +/-maxTau1
    return:
        TAU     source delay in 400MHz samples, shape (nWin, nBl)
        PHI     median residual phase in rad, shape (nWin, nBl)
                (phase at RF after removing the delay, i.e. the actual phase error)
    '''
    nWin, nBl, nChan = spec2.shape
    nfft = nChan*pad
    lag = np.fft.fftfreq(nfft, d=freq[1]-freq[0]) * 400.  # delay of each FFT bin, in 400MHz samples
    insearch = np.abs(lag) <= maxTau1
    fr = (freq - freq.mean()).reshape((1,-1))               # MHz
    TAU = np.zeros((nWin, nBl))
    PHI = np.zeros((nWin, nBl))
    for i in range(nWin):
        print('...',i,'t%08.1f'%sec[i])
        vispha = srcdelay_phase(spec2[i], freq)             # (nBl, nChan)
        w0 = (~np.ma.getmaskarray(vispha)).astype(float)
        v = np.ma.exp(1.j*vispha).filled(0.)
        # coarse delay
        ft = np.abs(np.fft.fft(v, n=nfft, axis=1))
        ft[:,~insearch] = 0.
        tau = lag[ft.argmax(axis=1)]
        # fine delay
        for it in range(niter):
            r = v*np.exp(-2.j*np.pi*fr*tau.reshape((-1,1))/400.)
            r0 = cpx_median(np.ma.array(r, mask=(w0==0)), axis=1).filled(1.)
            ph = np.angle(r*np.conj(r0).reshape((-1,1)))
            mad = np.ma.median(np.ma.array(np.abs(ph), mask=(w0==0)), axis=1).filled(np.pi)
            wt = w0 * (np.abs(ph) < 3.*1.4826*mad.reshape((-1,1)))
            sw = wt.sum(1); sf = (wt*fr).sum(1); sff = (wt*fr**2).sum(1)
            sp = (wt*ph).sum(1); sfp = (wt*fr*ph).sum(1)
            den = sw*sff - sf**2
            with np.errstate(divide='ignore', invalid='ignore'):
                slope = np.where(den>0, (sw*sfp - sf*sp)/den, 0.)  # rad per MHz
            tau += slope/(2.*np.pi)*400.
        TAU[i] = tau
        rescpx = np.ma.exp(1.j*(vispha - srcdelay_model(tau, freq)))
        PHI[i] = np.angle(cpx_median(rescpx, axis=1))
    return TAU, PHI


def source_geometry(tWin, nWin):
    '''
    az, el of the target and the beam attenuation
    return az, el, att0, NSoff (deg), EWoff (deg), unitVec (nWin, 3)
    '''
    b, obs = obsBody(body, time=tWin[0], site=site, retOBS=True, DB=DB)
    az = []
    el = []
    for ti in tWin:
        obs.date = ti
        b.compute(obs)
        az.append(b.az)
        el.append(b.alt)
    az = np.array(az)
    el = np.array(el)
    phi    = np.pi/2. - az
    theta  = np.pi/2. - el

    ## offsets from zenith in the EW and NS planes
    za = np.pi/2 - el
    pntr = np.sin(za)
    pntz = np.cos(za)
    pntx = pntr * np.sin(az)
    pnty = pntr * np.cos(az)
    EWoff = np.arctan2(pntx,pntz)
    NSoff = np.arctan2(pnty,pntz)

    ## apply calculate attenuation
    if (swapEH):
        Hatt = atten(NSoff/np.pi*180., Ehwhm)
        Eatt = atten(EWoff/np.pi*180., Hhwhm)
    else:
        Eatt = atten(EWoff/np.pi*180., Ehwhm)
        Hatt = atten(NSoff/np.pi*180., Hhwhm)
    att0 = Eatt*Hatt

    unitVec  = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], ndmin=2).T
    unitVec *= -1       # swapped definition
    return az, el, att0, NSoff/np.pi*180, EWoff/np.pi*180, unitVec


def fit_xyz(fa_spec2, unitVec, pairs, lam):
    '''
    choose the antenna positions that best match the spectral-avg phase vs time
    return the best xyz (nAnt, 3)
    '''
    if (blfit == 0):    # no fitting
        xyzparams = [antXYZ]
        fitparams = [[0,0,0,0]]
    elif (blfit == 1):  # 4 antennas in a straight line
        angrange = np.arange(-anglim, anglim, angsep) + ang0
        a1range = np.arange(-poslim, poslim, possep) + pos0[0]
        a2range = np.arange(-poslim, poslim, possep) + pos0[1]
        a3range = np.arange(-poslim, poslim, possep) + pos0[2]
        xyzparams = []
        fitparams = []
        for ang in angrange:
            th = ang/180.*np.pi
            for a1 in a1range:
                for a2 in a2range:
                    for a3 in a3range:
                        tmp = np.array([0., a1, a2, a3])
                        tmpx = tmp * np.cos(th)
                        tmpy = tmp * np.sin(th)
                        tmpz = np.zeros_like(tmp)
                        xyzparams.append(np.array([tmpx, tmpy, tmpz]).T)
                        fitparams.append([ang, a1, a2, a3])
        xyzparams = np.array(xyzparams)
    nParams = len(xyzparams)
    print('fitting xyz, nParams =', nParams)

    xyzlike = []
    for xyz in xyzparams:
        like = 0.
        for b, (ai, aj) in enumerate(pairs):
            Bproj = np.dot(unitVec, xyz[aj] - xyz[ai])  # in meters
            modComplex = np.exp(-2.j*np.pi*Bproj/lam)
            resPhase = np.angle(fa_spec2[:,b]/modComplex)
            resPhase -= np.median(resPhase)
            like += -0.5*(resPhase**2/0.3**2).sum()  # fixed phase error
        xyzlike.append(like)

    maxidx = np.argmax(xyzlike)
    maxxyz = xyzparams[maxidx]
    print('best xyz:', maxxyz)
    print('best fit param:', fitparams[maxidx])
    return maxxyz


def fringe_stop(coeff3, freq, Bproj, att0):
    '''
    fringe-stopping, alternative to removing the time-average
    source template per (bl,ch): G(t,f) = att0(t) * M(t,f), M = exp(-2pi i f_RF tau(t))
    joint LS fit over windows per (bl,ch): coeff3 = A*G + R
        A: source coeff at beam center; R: static coherent (RFI) term
    return:
        coeff_fs    per-window coeff projected along the source fringe (signed, no noise bias as in abs())
        coeff_rfi   the discarded part (not following the fringe), as a gauge of RFI
        fsA, fsR    shape (nBl, nChan)
    '''
    nWin, nBl, nChan = coeff3.shape
    fsM = np.exp(-2.j*np.pi*(freq*1e6).reshape((1,1,-1))*(Bproj/clight)[:,:,np.newaxis]) # (nWin,nBl,nChan)
    fsG = att0.reshape((-1,1,1)) * fsM
    fsw = (~np.ma.getmaskarray(coeff3)).astype(float)
    fsc = np.ma.filled(coeff3, 0.)
    fsn = fsw.sum(axis=0)
    sG  = (fsw*fsG).sum(axis=0)
    sGG = (fsw*np.abs(fsG)**2).sum(axis=0)
    sc  = (fsw*fsc).sum(axis=0)
    scG = (fsw*fsc*fsG.conj()).sum(axis=0)
    det = sGG*fsn - np.abs(sG)**2
    with np.errstate(divide='ignore', invalid='ignore'):
        fsA = (scG*fsn - sc*sG.conj())/det
        fsR = (sc - fsA*sG)/fsn
    fsbad = ~np.isfinite(fsA) | ~np.isfinite(fsR) | (fsA == 0)
    fsA[fsbad] = 0.
    fsR[fsbad] = 0.
    fsPha = np.ones_like(fsA)
    fsPha[~fsbad] = fsA[~fsbad]/np.abs(fsA[~fsbad])
    fsmask = (fsw == 0) | fsbad.reshape((1,nBl,nChan))
    coeff_fs = ((fsc - fsR) * fsM.conj() * fsPha.conj()).real
    coeff_fs = np.ma.array(coeff_fs, mask=fsmask)
    coeff_rfi = fsc - coeff_fs.filled(0.)*fsPha*fsM
    coeff_rfi = np.ma.array(coeff_rfi, mask=fsmask)
    fsA = np.ma.array(fsA, mask=fsbad)
    fsR = np.ma.array(fsR, mask=fsbad)
    return coeff_fs, coeff_rfi, fsA, fsR


def tsys_1d(med_c, flux, scale2, att=None):
    '''
    Tsys per window from the spectral-median coeff
    att: beam attenuation, same shape as med_c (or broadcastable), or None
    '''
    Tsys = flux / med_c * scale2
    if (att is not None):
        Tsys *= att
    # correct Tsys bias due to Solar flux
    Tsys *= (1.-med_c)
    return Tsys


def tsys_2d(c2d, flux, fscale2, att):
    '''
    Tsys per window and channel from the coeff, shape (nWin, ..., nChan)
    flux, fscale2: (nChan,) or (1,); att: (nWin,)
    '''
    ext = (1,)*(c2d.ndim-1)
    Tsys2D = flux.reshape(ext+(-1,)) / c2d * fscale2.reshape(ext+(-1,))
    Tsys2D *= att.reshape((-1,)+ext[1:]+(1,))
    # correct Tsys bias due to Solar flux
    Tsys2D *= (1.-c2d)
    return Tsys2D


def clip_columns(Y):
    '''sigma_clip each baseline (column) separately'''
    return np.ma.stack([sigma_clip(Y[:,b]) for b in range(Y.shape[1])], axis=1)


##################################################
## plotting functions

def plot_clock_phase(png, freq, spec1, spec3, i, nInp):
    fig, sub = blsubplots(na=nInp, figsize=(12,9), squeeze=False)
    b = -1
    for ai in range(nInp-1):
        for aj in range(ai+1,nInp):
            b += 1
            ax = sub[ai,aj-1]
            ax.plot(freq, np.ma.angle(spec1[i,b]), label='original')
            ax.plot(freq, np.ma.angle(spec3[i,b]), linestyle='none', marker='.', label='unwrap')
            ax.text(0.05, 0.90, 'med: %.2f'%np.ma.median(np.ma.angle(spec3[i,b])), transform=ax.transAxes)
            ax.set_xlabel('Frequency (MHz)')
            ax.set_ylabel('Phase (rad)')
            ax.legend(loc='upper right')
    fig.savefig(png)
    plt.close(fig)


def plot_waterfall(png, X, Y, pp_spec, linear):
    '''real/imag/power/phase waterfall of one baseline, pp_spec shape (nWin, nChan)'''
    fig, sub = plt.subplots(2,2,figsize=(15,10))
    for ax, part, tit in [(sub[0,0], pp_spec.real, 'Real'), (sub[0,1], pp_spec.imag, 'Imag')]:
        if (linear):
            s=ax.pcolormesh(X,Y,part.T,shading='auto')
        else:
            s=ax.pcolormesh(X,Y,part.T,shading='auto', norm=colors.SymLogNorm(linthresh=1e3))
        cb=plt.colorbar(s, ax=ax)
        cb.set_label('linear')
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Frequency (MHz)')
        ax.set_title(tit)
    # amp
    ax = sub[1,0]
    s=ax.pcolormesh(X,Y,10*np.ma.log10(np.ma.abs(pp_spec)).T,shading='auto')
    cb=plt.colorbar(s, ax=ax)
    cb.set_label('10*log10(abs())')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('Power')
    # pha
    ax = sub[1,1]
    s=ax.pcolormesh(X,Y,np.ma.angle(pp_spec).T,shading='auto')
    cb=plt.colorbar(s, ax=ax)
    cb.set_label('Phase (rad)')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('Phase')

    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.suptitle('%s'%png)
    fig.savefig(png)
    plt.close(fig)


def plot_clock_off(png, fvis, tWin, ClkOff, nInp, bsel):
    fig, sub = blsubplots(na=nInp, figsize=(12,9), squeeze=False)
    b = -1
    for ai in range(nInp-1):
        for aj in range(ai+1, nInp):
            b += 1
            ax = sub[ai,aj-1]
            ax.plot(tWin, ClkOff[:,b], marker='.', linestyle=':')
            ax.text(0.05, 0.90, 'min:%.2f max:%.2f'%(ClkOff[:,b].min(), ClkOff[:,b].max()), transform=ax.transAxes)
            ax.set_xlabel('Time (sec)')
            ax.set_ylabel('clock off (sample)')

    if (nInp >= 3):
        # add an axes for closure phase, if nInp>=3 and only for the first 3 inputs
        ClkClosure = ClkOff[:,bsel[0]]+ClkOff[:,bsel[2]]-ClkOff[:,bsel[1]]
        axclo = fig.add_axes(rect=[0.08,0.06,0.33,0.33])
        axclo.plot(tWin, ClkClosure, marker='.', linestyle=':')
        axclo.set_xlabel('Time')
        axclo.set_ylabel('closure off (sample)')
        axclo.set_title('closure: 01+12-02')

    fig.suptitle('%s'%(fvis,))
    fig.savefig(png)
    plt.close(fig)


def plot_source_delay(png, title, freq, spec2, TAU, PHI, i, nInp):
    if (nInp==1):
        ww=6.0
        hh=4.5
    elif(nInp<=4):
        ww=4.0
        hh=3.0
    else:
        ww=1.6
        hh=1.2
    figw=ww*(nInp-1)
    figh=hh*(nInp-1)

    fig, s2d = blsubplots(na=nInp, figsize=(figw,figh), squeeze=False, sharey=True, sharex=True)
    b = -1
    for ai in range(nInp-1):
        for aj in range(ai+1, nInp):
            b += 1
            ax = s2d[ai,aj-1]
            if (ai == aj-1):
                ax.set_xlabel('Frequency (MHz)')
                ax.set_ylabel('Phase (rad)')
                ax.set_ylim(-3.5, 5.0)
            vispha = srcdelay_phase(spec2[i,b], freq)
            mm = vispha.mask
            modpha = srcdelay_model(TAU[i,b], freq)
            respha = pwrap2(vispha - modpha)
            ax.plot(freq, vispha, color='C0', label='data')
            ax.scatter(freq[mm], vispha.data[mm], color='r', label='mask')
            ax.plot(freq, pwrap2(modpha), color='C2', label='model')
            ax.plot(freq, pwrap2(respha), color='C1', marker=',', linestyle='none', label='residual')
            ax.text(0.05, 0.92, 'tau: %.3f samp'%TAU[i,b], transform=ax.transAxes)
            ax.text(0.05, 0.82, 'phi: %.3f rad'%PHI[i,b], transform=ax.transAxes)

    lax = fig.add_axes((0.03,0.03,0.3,0.3))
    lax.set_xlim(0,1)
    lax.set_ylim(0,1)
    lax.set_frame_on(False)
    lax.set_axis_off()
    lax.plot(-1, 0, color='C0', label='data')
    lax.scatter(-1, 0, color='r', label='masked')
    lax.plot(-1, 0, color='C2', label='model')
    lax.plot(-1, 0, color='C1', label='residual')
    lax.legend(loc='upper left')

    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.suptitle(title)
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(png)
    plt.close(fig)


def plot_all_phase(png, sec, tWin, spec1, fa_spec3, nInp, bsel):
    '''spectral-avg phase vs time (all baselines)'''
    fig, sub = blsubplots(na=nInp, figsize=(12,9), squeeze=False)
    b = -1
    for ai in range(nInp-1):
        for aj in range(ai+1,nInp):
            b += 1
            ax = sub[ai,aj-1]
            ax.plot(tWin, np.ma.angle(fa_spec3[:,b]), marker='.', color='C0')
            ax.set_xlabel('Time')
            ax.set_ylabel('Phase (rad)')
    fig.autofmt_xdate(rotation=45)
    if (nInp >= 3):
        # add an axes for closure phase, if nInp>=3 and only for the first 3 inputs
        clospec1 = spec1[:,bsel[0]]*spec1[:,bsel[2]]/spec1[:,bsel[1]]
        closurep1 = np.ma.angle(clospec1).mean(axis=1)
        closeperr = np.ma.angle(clospec1).std(axis=1)/np.sqrt(np.count_nonzero(~clospec1.mask, axis=1))
        axclo = fig.add_axes(rect=[0.08,0.06,0.4,0.33])
        axclo.errorbar(sec, closurep1, yerr=closeperr, marker='.', linestyle=':', label='no-fit/closure/f_avg')
        axclo.text(0.05, 0.05, 'RMS:%.3frad'%closurep1.std(),transform=axclo.transAxes)
        axclo.set_xlabel('Time (sec)')
        axclo.set_ylabel('Closure phase (rad)')
        axclo.set_title('closure: 01+12-02')
    fig.tight_layout()
    fig.savefig(png)
    plt.close(fig)


def plot_bl_rf(png, suptitle, freq, rflim, sec, nft, spec, coeff, ai, aj, med_coeff, cfs_coeff=None, modPhase2=None):
    '''vs. RF plot of all windows for one baseline'''
    nWin = len(sec)
    f1, s1 = plt.subplots(3,2,figsize=(12,12))

    ax = s1[1,0]
    ax.set_title('cross01')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Xcorr coeff')
    ax.set_yscale('log')
    ax.set_ylim([2e-3, 2])
    ax = s1[1,1]
    ax.set_title('cross01')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Vis. phase (rad)')
    ax.set_ylim([-3.2,3.2])
    for j,jj in enumerate([ai,aj]):
        ax = s1[0,j]
        ax.set_title('Ant%d'%jj)
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Power (dBm)')

    for i in range(nWin):
        cc = 'C%d'%(i%10)
        s1[1,0].plot(freq, np.ma.abs(coeff[i]), color=cc)
        s1[1,1].plot(freq, np.ma.angle(spec[i]), color=cc, linestyle='none', marker='.')
        for j,jj in enumerate([ai,aj]):
            lab = '%04.0fsec'%sec[i] if (j==1) else ''
            s1[0,j].plot(freq, toPower(nft[i,jj]), color=cc, label=lab)

    par2 = np.percentile(med_coeff, [15,50,85])
    med_med_coeff = par2[1]
    std_med_coeff = (par2[2]-par2[0])/2.

    ax = s1[2,0]
    if (cfs_coeff is not None):
        ax.plot(sec, sigma_clip(cfs_coeff.real), 'b-', label='real')
    ax.plot(sec, sigma_clip(med_coeff), 'b:', label='abs')
    ax.legend()
    ax.set_xlabel('Window time (sec)')
    ax.set_ylabel('med_coefficient')
    ax.text(0.05, 0.90, 'med:%.3g+/-%.3g'%(med_med_coeff, std_med_coeff), transform=ax.transAxes)

    med_phase = np.ma.median(np.ma.angle(spec), axis=1)
    ax = s1[2,1]
    ax.plot(sec, med_phase)
    if (modPhase2 is not None):
        ax.plot(sec, modPhase2)
    ax.set_ylim([-3.2,3.2])
    ax.set_xlabel('Window time (sec)')
    ax.set_ylabel('med_phase')

    s1[0,0].get_shared_y_axes().join(s1[0,0], s1[0,1])
    s1[0,0].get_shared_x_axes().join(s1[0,0], s1[0,1], s1[1,0], s1[1,1])
    s1[2,0].get_shared_x_axes().join(s1[2,0], s1[2,1])

    s1[0,0].set_xlim(rflim[0], rflim[1])

    f1.tight_layout(rect=[0,0.03,1,0.95])
    f1.suptitle(suptitle)
    f1.savefig(png)
    plt.close(f1)


def plot_bl_vs_time(png, suptitle, tt, PHI, ns_data, cTsys, ns_model=None, cTsys_fs=None, el=None):
    '''vs. time plot for one baseline'''
    f2, s2 = plt.subplots(2,2, figsize=(15,10), sharex=True)

    # phase plot
    ax = s2[0,0]
    ax.plot(tt, PHI, ls='none', marker='*', label='data')
    ax.text(0.05, 0.90, 'med_dphi: %.3f rad'%circ_median(PHI), transform=ax.transAxes)
    ax.legend()
    ax.set_ylim([-3.5, 4.5])
    ax.set_xlabel('Time')
    ax.set_ylabel('Resid. phase (rad)')

    # delay plot (in ns)
    ax = s2[1,0]
    ax.plot(tt, ns_data, marker='.', label='data')
    if (ns_model is not None):
        ax.plot(tt, ns_model, label='model')
        ns_diff = ns_data - ns_model
        medns = np.median(ns_diff)
        ns_cdiff = sigma_clip(ns_diff)
        stdns = ns_cdiff.std()
        ax.scatter(tt, ns_diff, color='g', marker='*', label='data-model')
        ax.scatter(tt[ns_cdiff.mask], ns_diff[ns_cdiff.mask], marker='*', color='r')
        ax.fill_between(tt, np.ones(len(tt))*ns_cdiff.min(), np.ones(len(tt))*ns_cdiff.max(), color='b', alpha=0.2)
        ax.text(0.05, 0.90, 'median residual delay: %.3f ns (std dev: %.3f ns)'%(medns,stdns), transform=ax.transAxes)
        ax.legend()
        ax.set_ylim(-50, 50)
        ax.grid()
    ax.set_xlabel('Time')
    ax.set_ylabel('Delay (ns)') # src_delay

    # Tsys plot
    ax = s2[0,1]
    ax.set_xlabel('Time')
    ax.plot(tt, cTsys, 'b:', label='from abs')
    ax.text(0.05, 0.90, 'Tsys_min=%.0fK'%cTsys.min(), transform=ax.transAxes)
    if (cTsys_fs is not None):
        ax.plot(tt, cTsys_fs, 'r-', label='from fringe-stop')
        ax.text(0.05, 0.82, 'Tsys_fs_min=%.0fK'%cTsys_fs.min(), transform=ax.transAxes, color='r')
    ax.legend()
    ax.set_ylabel('Tsys (K)')
    ax.grid()

    # elevation plot
    ax = s2[1,1]
    if (el is not None):
        ax.plot(tt, el/np.pi*180, marker='.')
    ax.set_xlabel('Time')
    ax.set_ylabel('Elevation (deg)')

    f2.tight_layout(rect=[0,0.03,1,0.95])
    f2.suptitle(suptitle)
    f2.savefig(png)
    plt.close(f2)


def plot_bl_coeff2d(png, suptitle, T, Y, coeff):
    fig, sub = plt.subplots(2,2,figsize=(10,8),sharex=True,sharey=True)
    ax = sub[0,0]
    ax.pcolormesh(T,Y,coeff.real.T)
    ax.set_title('Real')
    ax.set_ylabel('Frequency (MHz)')
    ax = sub[0,1]
    ax.pcolormesh(T,Y,coeff.imag.T)
    ax.set_title('Imag')
    ax = sub[1,0]
    ax.pcolormesh(T,Y,np.ma.abs(coeff).T)
    ax.set_title('Abs')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_xlabel('Time')
    ax = sub[1,1]
    ax.pcolormesh(T,Y,np.ma.angle(coeff).T)
    ax.set_title('Phase')
    ax.set_xlabel('Time')

    plot_util.set_style(fig)

    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.subplots_adjust(wspace=0)
    fig.suptitle(suptitle)
    fig.savefig(png)
    plt.close(fig)


def plot_bl_map2d(png, suptitle, T, Y, freq, tWin, tsel, tt1, tt2, Y2D, vmin, vmax, cblabel, yname, unit):
    '''2D Tsys or SEFD map of one baseline, with time and freq medians'''
    nChan = len(freq)
    y_freq  = np.ma.median(Y2D[tsel], axis=0)   # med
    y_freq2 = Y2D[tsel].min(axis=0)             # min
    y_time  = np.ma.median(Y2D, axis=1)         # med
    y_time2 = Y2D.min(axis=1)                   # min

    fig = plt.figure(figsize=(12,9))
    gs = fig.add_gridspec(2,2,width_ratios=(3,1),height_ratios=(1,3), wspace=0, hspace=0, left=0.08, right=0.95, bottom=0.08, top=0.92)
    ax2D = fig.add_subplot(gs[1,0])
    axtp = fig.add_subplot(gs[0,0])
    axrt = fig.add_subplot(gs[1,1])

    ax2D.pcolormesh(T,Y,Y2D.T, vmin=vmin, vmax=vmax)
    ax2D.set_xlabel('Time (UT)')
    ax2D.set_ylabel('Frequency (MHz)')
    ax2D.axvline(tt1, linestyle='--', color='r')
    ax2D.axvline(tt2, linestyle='--', color='r')

    axtp.plot(tWin, y_time, label='med')
    axtp.plot(tWin, y_time2, linestyle=':', label='min')
    axtp.set_xlim(tWin[0], tWin[-1])
    axtp.legend()
    axtp.set_ylabel(cblabel)
    axtp.set_ylim(vmin, vmax)
    axtp.grid(axis='both')
    axtp.set_xticklabels([])

    if (np.count_nonzero(np.ma.getmaskarray(y_freq)) < nChan):
        axrt.plot(y_freq, freq, label='med')
    if (np.count_nonzero(np.ma.getmaskarray(y_freq2)) < nChan):
        axrt.plot(y_freq2, freq, linestyle=':', label='min')
    axrt.set_ylim(freq[0], freq[-1])
    axrt.legend()
    axrt.set_xlabel(cblabel)
    axrt.set_xlim(vmin, vmax)
    axrt.grid(axis='both')
    axrt.set_yticklabels([])
    axrt.axvline(np.ma.median(y_freq), color='k', linestyle=':')

    if (unit == 'K'):
        fig.text(0.76, 0.80, 'med(%s_freq.med) = %.0f K'%(yname, np.ma.median(y_freq)), color='C0')
        fig.text(0.76, 0.75, 'med(%s_freq.min) = %.0f K'%(yname, np.ma.median(y_freq2)), color='C1')
    else:
        fig.text(0.76, 0.80, 'med(%s_freq.med) = %.3f MJy'%(yname, np.ma.median(y_freq)), color='C0')
        fig.text(0.76, 0.75, 'med(%s_freq.min) = %.3f MJy'%(yname, np.ma.median(y_freq2)), color='C1')
        fig.text(0.76, 0.85, r'$\mathrm{SEFD} \cdot (400 MHz/f_{MHz})^2')

    plot_util.set_style(fig)

    fig.suptitle(suptitle)
    fig.savefig(png)
    plt.close(fig)


def plot_bl_rfi(png, suptitle, T, Y, freq, tWin, fsel, coeff_fs, coeff_rfi, fsA, fsR, coeff_sub):
    '''RFI gauge: the part discarded by fringe-stopping, for one baseline'''
    fig, sub = plt.subplots(2,2,figsize=(15,10))
    ax = sub[0,0]
    s = ax.pcolormesh(T,Y,10*np.ma.log10(np.ma.abs(coeff_rfi)).T, shading='auto')
    cb = plt.colorbar(s, ax=ax)
    cb.set_label('10*log10(abs(coeff))')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('discarded (not following fringe): Power')
    ax = sub[0,1]
    s = ax.pcolormesh(T,Y,np.ma.angle(coeff_rfi).T, shading='auto')
    cb = plt.colorbar(s, ax=ax)
    cb.set_label('Phase (rad)')
    ax.set_title('discarded (not following fringe): Phase')
    ax = sub[1,0]
    ax.plot(freq, np.ma.abs(fsA), label='|A| source (beam center)')
    ax.plot(freq, np.ma.abs(fsR), label='|R| static')
    ax.plot(freq, np.ma.median(np.ma.abs(coeff_rfi), axis=0), label='med_t |discarded|')
    ax.set_yscale('log')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Xcorr coeff')
    ax.legend()
    ax = sub[1,1]
    ax.plot(tWin, np.ma.median(coeff_fs[:,fsel], axis=1), marker='.', label='med_f fringe-stop')
    ax.plot(tWin, np.ma.median(np.ma.abs(coeff_rfi[:,fsel]), axis=1), marker='.', label='med_f |discarded|')
    ax.plot(tWin, np.ma.median(np.ma.abs(coeff_sub[:,fsel]), axis=1), marker='.', ls=':', label='med_f |sub_t-mean|')
    ax.set_xlabel('Time')
    ax.set_ylabel('Xcorr coeff')
    ax.legend()
    for ax in sub[0]:
        ax.set_xlabel('Time')
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.suptitle(suptitle)
    fig.savefig(png)
    plt.close(fig)



##################################################
## main

for fvis in files:
    if (not os.path.isfile(fvis)):
        print('file not found: %s --> skip' % fvis)
        continue
    print('processing:', fvis)

    cdir = fvis.replace('.vish5', '.check')
    if (save_bl and not os.path.isdir(cdir)):
        call('mkdir -p %s'%cdir, shell=True)
    odir = fvis.replace('.vish5', '.output')
    if (not os.path.isdir(odir)):
        call('mkdir -p %s'%odir, shell=True)


    attrs = getAttrs(fvis)
    attrs['EW_hwhm'] = Ehwhm
    attrs['NS_hwhm'] = Hhwhm
    attrs['sun_f410'] = f410
    attrs['sun_f610'] = f610
    if (not aconf is None):
        attrs['aconf'] = aconf
    attrs['Gant_dB'] = gant
    if (not site is None):
        attrs['site'] = site
    putAttrs(fvis, attrs)

    if (do_model):
        adoneh5(fvis, antXYZ, 'antXYZ')

    # file-open time
    unix_utc_open = attrs.get('unix_utc_open')
    if (unix_utc_open is None):
        fbase = os.path.basename(fvis)
        ii = fbase.find('_dev')
        dtstr = fbase[ii-13:ii]
        topen = datetime.strptime(dtstr, '%y%m%d_%H%M%S')   # local time
        atopen = Time(topen, format='datetime')
        atopen -= 8/24  # convert to UTC
    else:
        atopen = Time(unix_utc_open, format='unix')

    freq = getData(fvis, 'freq')   # RF in MHz
    nChan = len(freq)
    fullBW = freq.max() - freq.min()    # full bandwidth in MHz

    nft  = getData(fvis, 'winNFT')  # shape: nWin, nInp, nChan
    spec1 = getData(fvis, 'winSpec') # shape: nWin, nBl, nChan
    if (not isinstance(spec1, np.ma.MaskedArray)):
        spec1 = np.ma.array(spec1, mask=False)
    sec  = getData(fvis, 'winSec')  # shape: nWin
    coeff1 = getData(fvis, 'winCoeff') # shape: nWin, nBl, nChan
    if (not isinstance(coeff1, np.ma.MaskedArray)):
        coeff1 = np.ma.array(coeff1, mask=False)
    nWin = len(sec)

    # window time
    tmp = atopen + sec/86400        # convert to days
    tWin = tmp.to_datetime()

    # per-file time and RF ranges
    ftlim = [sec[0], sec[-1]] if (tlim is None) else tlim
    tsel = (sec>=ftlim[0]) * (sec<=ftlim[1])
    tt1 = tWin[np.argmin(np.abs(sec-ftlim[0]))]
    tt2 = tWin[np.argmin(np.abs(sec-ftlim[1]))]
    rf1 = freq.min() if (rfmin == 0) else rfmin
    rf2 = freq.max() if (rfmax == 0) else rfmax
    fsel = np.logical_and(freq>=rf1, freq<=rf2)


    if (len(spec1.shape) == 2): # backward-compatible with single_dev_vis2.py
        spec1  = spec1.reshape((nWin, 1, nChan))
        coeff1 = coeff1.reshape((nWin, 1, nChan))

    nInp = nft.shape[1]
    print('nInp =', nInp)
    nBl = int(nInp*(nInp-1)/2)
    pairs = [(ai,aj) for ai in range(nInp-1) for aj in range(ai+1,nInp)]

    # define pair id for closure(0,1,2)
    trio = [0, 1, 2]
    bsel = [b for b,(ai,aj) in enumerate(pairs) if (ai in trio and aj in trio)]

    if (f410>0 and f610>0): # override srcflux setting
        xs = (freq-410.)/(610.-410.)
        srcflux2 = f410 + (f610-f410)*xs # shape (nChan,)
    else:
        srcflux2 = np.array([srcflux])
    adoneh5(fvis, srcflux2, 'src_flux_Jy')


    ## clock offset; spec3/coeff3: clock-corrected, spec2/coeff2: time-mean subtracted
    if (do_fitClkOff):
        print('removing clock offset ...')
        savClkOff, spec3, coeff3 = fit_clock_offset(spec1, coeff1, sec)
        if (do_check):
            for i in range(nWin):
                plot_clock_phase('%s/correlate_pha2_t%08.1f.png'%(cdir, sec[i]), freq, spec1, spec3, i, nInp)
        adoneh5(fvis, savClkOff, 'winClkOff')
        adoneh5(fvis, spec3, 'winSpecClkCorr')
        adoneh5(fvis, coeff3, 'winCoeffClkCorr')
        print('... clock offset done.')
    else:
        savClkOff = getData(fvis, 'winClkOff')
        spec3 = getData(fvis, 'winSpecClkCorr')
        if (spec3 is None or noClkCorr):
            spec3 = spec1
        coeff3 = getData(fvis, 'winCoeffClkCorr')
        if (coeff3 is None or noClkCorr):
            coeff3 = coeff1
    spec2 = spec3 - spec3.mean(axis=0)
    coeff2 = coeff3 - coeff3.mean(axis=0)
    adoneh5(fvis, spec2, 'winSpecSub')
    adoneh5(fvis, coeff2, 'winCoeffSub')

    X,Y = np.meshgrid(sec, freq, indexing='xy')
    T,Y = np.meshgrid(tWin, freq, indexing='xy')

    if (do_check):
        ## waterfall of real/imag/amp/pha for spec1, spec2 and coeff1, coeff2
        for b, (ai,aj) in enumerate(pairs):
            bname = '%d-%d'%(ai,aj)
            for name, pp_spec, linear in [('spec_raw', spec1, False), ('spec_sub', spec2, False),
                                          ('coeff_raw', coeff1, True), ('coeff_sub', coeff2, True)]:
                png = '%s/waterfall_bl%s_%s.png' % (cdir, bname, name)
                plot_waterfall(png, X, Y, pp_spec[:,b,:], linear)

    if (save_bl and savClkOff is not None):
        plot_clock_off('%s/clock_off.png'%cdir, fvis, tWin, savClkOff, nInp, bsel)


    ## source delay (not instrument delay)
    savTAU = getData(fvis, 'winTAU')
    savPHI = getData(fvis, 'winPHI')
    if (savTAU is None or do_tausrc):
        print('fitting source delay ...')
        savTAU, savPHI = fit_source_delay(spec2, freq, sec)
        adoneh5(fvis, savTAU, 'winTAU')
        adoneh5(fvis, savPHI, 'winPHI')
        if (save_bl):
            for i in range(nWin):
                tag = 't%08.1f' % sec[i]
                plot_source_delay('%s/source_delay_%s.png'%(cdir, tag), '%s, source delay, %s'%(fvis, tag),
                                  freq, spec2, savTAU, savPHI, i, nInp)


    # spectral avg
    fa_spec2 = cpx_median(spec2, axis=2)    # shape (nWin, nBl)
    fa_spec3 = cpx_median(spec3, axis=2)    # shape (nWin, nBl)

    flam = clight / (freq * 1e6)
    lam = flam_ref

    chBW = fullBW*1e6 / nChan       # channel bandwidth in Hz
    tInt = 1./(fullBW*1e6) * nChan  # integration time per spectrum in sec
    scale = np.sqrt(1.*chBW*tInt)   # conversion between variance and SEFD

    # at reference freq --> 400MHz
    Tsys0 = 100.
    Aeff = lam**2 * G0 / (4.*np.pi)
    scale2 = Aeff * scale / (2*1.38e-23) * 1e-26 # Tsys = (S/Jy / SNR) * scale2
    SEFD2 = Tsys0 / scale2
    print('fiducial SEFD (%dK, %.2fm):'%(Tsys0,lam), '%.3f'%(SEFD2/1e6), 'MJy')
    fscale2 = flam**2 * G0 / (4.*np.pi) * scale / (2*1.38e-23) * 1e-26
    fscale3 = (flam/flam_ref)**2


    ## spectral-avg phase vs time (all baselines)
    plot_all_phase('%s/all_phase_vs_time.png'%odir, sec, tWin, spec1, fa_spec3, nInp, bsel)


    ## geometric model, fringe-stopping
    if (do_model):
        az, el, att0, NSoff_deg, EWoff_deg, unitVec = source_geometry(tWin, nWin)
        adoneh5(fvis, az, 'az')
        adoneh5(fvis, el, 'el')
        print('max atten:', att0.max())
        adoneh5(fvis, att0, 'atten')
        adoneh5(fvis, NSoff_deg, 'NSoff_deg')
        adoneh5(fvis, EWoff_deg, 'EWoff_deg')

        maxxyz = fit_xyz(fa_spec2, unitVec, pairs, lam)
        adoneh5(fvis, maxxyz, 'ant_xyz')
        # projected baseline (geometric delay in m), shape (nWin, nBl)
        savBproj = np.array([np.dot(unitVec, maxxyz[aj]-maxxyz[ai]) for ai,aj in pairs]).T
        blmodComplex = np.exp(-2.j*np.pi*savBproj/lam)  # model fringe at the ref freq, (nWin, nBl)
        if (revtau):
            savBproj *= -1
        adoneh5(fvis, savBproj, 'model_tau_m')

        print('fringe-stopping ...')
        coeff_fs, coeff_rfi, fsA, fsR = fringe_stop(coeff3, freq, savBproj, att0)
        adoneh5(fvis, coeff_fs, 'winCoeffFS')       # shape (nWin, nBl, nChan), real
        adoneh5(fvis, coeff_rfi, 'winCoeffRFI')     # shape (nWin, nBl, nChan), complex
        adoneh5(fvis, fsA, 'fsCoeffSrc')            # shape (nBl, nChan)
        adoneh5(fvis, fsR, 'fsCoeffStatic')         # shape (nBl, nChan)
        print('... fringe-stopping done. med|A|=%.3g, med|R|=%.3g' % (np.ma.median(np.ma.abs(fsA)), np.ma.median(np.ma.abs(fsR))))
    else:
        att0 = None
        el = None


    ## Tsys and delay residuals, all baselines at once
    # per-window Tsys from the spectral median (in --rf range) of the time-mean subtracted coeff
    med_coeff2 = np.ma.median(np.ma.abs(coeff2[:,:,fsel]), axis=2)     # (nWin, nBl)
    att1d = None if (att0 is None) else att0.reshape((-1,1))
    cTsys = clip_columns(tsys_1d(med_coeff2, srcflux2.mean(), scale2, att1d))
    ns_data = -savTAU/400e6*1e9         # source delay in ns
    if (do_model):
        # same from the fringe-stopped coeff
        med_cfs = np.ma.median(coeff_fs[:,:,fsel], axis=2)
        med_cfs = np.ma.masked_where(med_cfs<=0, med_cfs)
        cTsys_fs = clip_columns(tsys_1d(med_cfs, srcflux2.mean(), scale2, att1d))

        ns_model = savBproj/clight*1e9
        ns_diff = ns_data - ns_model

        # Tsys2D, SEFD2D, shape (nWin, nBl, nChan)
        absc2 = np.ma.abs(coeff2)
        Tsys2D = tsys_2d(absc2, srcflux2, fscale2, att0)
        SEFD2D = Tsys2D / fscale2.reshape((1,1,-1)) * fscale3.reshape((1,1,-1))
        cfs2D = np.ma.masked_where(coeff_fs<=0, coeff_fs)  # non-positive coeff masked
        Tsys2D_fs = tsys_2d(cfs2D, srcflux2, fscale2, att0)
        SEFD2D_fs = Tsys2D_fs / fscale2.reshape((1,1,-1)) * fscale3.reshape((1,1,-1))

        adoneh5(fvis, cTsys, 'winTsys')             # shape (nWin, nBl)
        adoneh5(fvis, Tsys2D, 'winTsys2D')          # shape (nWin, nBl, nChan)
        adoneh5(fvis, SEFD2D, 'winSEFD2D')          # shape (nWin, nBl, nChan)
        adoneh5(fvis, cTsys_fs, 'winTsys_fs')       # shape (nWin, nBl)
        adoneh5(fvis, Tsys2D_fs, 'winTsys2D_fs')    # shape (nWin, nBl, nChan)
        adoneh5(fvis, SEFD2D_fs, 'winSEFD2D_fs')    # shape (nWin, nBl, nChan)
        adoneh5(fvis, ns_model, 'winTauMod')
        adoneh5(fvis, ns_diff, 'winTauRes')
        adoneh5(fvis, np.median(ns_diff, axis=0), 'medTauRes')


    if (not save_bl):
        continue

    ## per-baseline plots
    print('plotting baselines ...')
    for b, (ai,aj) in enumerate(pairs):
        corr = '%d-%d'%(ai,aj)
        print('... bl%s'%corr)

        for pp in range(2):
            if (pp == 0):
                spec = spec3[:,b]
                coeff = coeff3[:,b]
                sfx = ''
                stitle = ''
            else:
                spec = spec2[:,b]
                coeff = coeff2[:,b]
                sfx = '.sub'
                stitle = ', sub_t-mean'
            med_coeff = np.ma.median(np.ma.abs(coeff[:,fsel]), axis=1)

            # fringe-stopped spectral-avg coeff (only for plotting)
            cfs_coeff = None
            modPhase2 = None
            if (do_model):
                fs_coeff = cpx_median(coeff, axis=1) / blmodComplex[:,b]
                ap_cal = np.ma.median(fs_coeff.real) + 1.j*np.ma.median(fs_coeff.imag)  # cal both amp and phase
                p_cal = ap_cal / np.abs(ap_cal) # cal only phase
                cfs_coeff = fs_coeff / p_cal
                modPhase2 = np.angle(blmodComplex[:,b] * p_cal)

            ## vs. RF plot
            png = '%s/bl%s.rf%d_%d.all_win%s.png' % (odir, corr, rf1, rf2, sfx)
            plot_bl_rf(png, 'file: %s, bl:%s%s' % (fvis, corr, stitle), freq, [rf1, rf2], sec, nft, spec, coeff, ai, aj,
                       med_coeff, cfs_coeff, modPhase2)

            ## vs. time plot
            if (pp == 0):
                cTsys_pp = sigma_clip(tsys_1d(med_coeff, srcflux2.mean(), scale2, att0))
            else:
                cTsys_pp = cTsys[:,b]
            png = '%s/bl%s.rf%d_%d.vs_time%s.png' % (odir, corr, rf1, rf2, sfx)
            suptitle = 'file: %s, bl:%s%s\nRF=[%.1f,%.1f]'%(fvis, corr, stitle, rf1, rf2)
            if (do_model):
                plot_bl_vs_time(png, suptitle, tWin, savPHI[:,b], ns_data[:,b], cTsys_pp,
                                ns_model[:,b], cTsys_fs[:,b], el)
            else:
                plot_bl_vs_time(png, suptitle, tWin, savPHI[:,b], ns_data[:,b], cTsys_pp)

            ## coeff plot
            png = '%s/bl%s.coeff2d%s.png' % (odir, corr, sfx)
            stitle2 = ', sub-t_mean' if (pp == 1) else ''
            plot_bl_coeff2d(png, 'file: %s, bl:%s%s' % (fvis, corr, stitle2), T, Y, coeff)

        if (do_model):
            ## 2D Tsys, SEFD plots
            suptitle = 'file: %s, bl:%s, sub-t_mean' % (fvis, corr)
            for name, Y2D, scl, cblabel, yname, unit in [
                    ('Tsys2D',    Tsys2D[:,b],        1, 'Tsys (K)', 'Tsys', 'K'),
                    ('SEFD2D',    SEFD2D[:,b]/1e6,   60, 'SEFD*(L/L0)^2 (MJy)', 'SEFD', 'MJy'),
                    ('Tsys2D_fs', Tsys2D_fs[:,b],     1, 'Tsys (K), fringe-stop', 'Tsys', 'K'),
                    ('SEFD2D_fs', SEFD2D_fs[:,b]/1e6, 60, 'SEFD*(L/L0)^2 (MJy), fringe-stop', 'SEFD', 'MJy')]:
                png = '%s/bl%s.rf%d_%d.%s.png' % (odir, corr, rf1, rf2, name)
                # SEFD range: 1/60 of zlim, i.e. 10MJy is roughly 300K
                plot_bl_map2d(png, suptitle, T, Y, freq, tWin, tsel, tt1, tt2, Y2D,
                              zlim[0]/scl, zlim[1]/scl, cblabel, yname, unit)

            ## RFI gauge: the part discarded by fringe-stopping
            png = '%s/bl%s.rfi_fs.png' % (odir, corr)
            plot_bl_rfi(png, 'file: %s, bl:%s, fringe-stop RFI gauge' % (fvis, corr), T, Y, freq, tWin, fsel,
                        coeff_fs[:,b], coeff_rfi[:,b], fsA[b], fsR[b], coeff2[:,b])


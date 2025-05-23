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

def atten(x, hwhm):
    '''
    Gaussian attenuation (linear) given offset x and the half-power full width
    '''
    sig = hwhm/np.sqrt(2.*np.log(2.))
    return np.exp(-x**2/2./sig**2)


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

    --rf MIN MAX    # RF range to use for time plot
                    # (both default to 0, for using the full range)

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

    ## to select only a subset of antennas for processing
    --ai AI         # specify the antenna numbers (only needed if --aconf is used)
    --aj AJ         # (AI, AJ are default to 0, 1)

    ## for fitting clock offset
    --fit           # fit and remove the Clock Offset

    --clk 'LIMIT SEP'   # set the +/-LIMIT and SEP for clock offset fitting (in samples)
                    # default: (%.3f, %.3f)

    --no-clkcorr    # disable using ClkCorr Spec and Coeff for analysis

    --force-tausrc  # force redo of source delay fitting
    --rev-tau       # reverse deflay direction, debugging
    --rev-pos       # reverse the antenna config file (for idt-beamformed data)

    --plot          # plot the Clock Offset phase correction

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

''' % (pg, body, gant, Ehwhm, Hhwhm, maxClkOff, dClkOff)

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



if (aconf is None):
    do_model = False
else:
    do_model = True
    tmp = os.path.basename(aconf)
    tmp2 = tmp.split('_')
    site = tmp2[0].lower()
    antXYZ = np.loadtxt(aconf, usecols=(1,2,3), ndmin=2)
if (do_model and revpos):
    print('reversing the ant pos...')
    antXYZ = np.flip(antXYZ, axis=0)


G0 = 10.**(gant/10.)




for fvis in files:
    if (not os.path.isfile(fvis)):
        print('file not found: %s --> skip' % fvis)
        continue
    print('processing:', fvis)

    cdir = fvis.replace('.vish5', '.check')
    if (not os.path.isdir(cdir)):
        call('mkdir -p %s'%cdir, shell=True)
    odir = fvis.replace('.vish5', '.output')
    if (not os.path.isdir(odir)):
        call('mkdir -p %s'%odir, shell=True)


    attrs = getAttrs(fvis)

    # file-open time
    unix_utc_open = attrs.get('unix_utc_open')
    if (unix_utc_open is None):
        fbase = os.path.basename(fvis)
        ii = fbase.find('_dev')
        dtstr = fbase[ii-13:ii]
        #print(fvis, ii, dtstr)
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
    #var  = getData(fvis, 'winVar')  # shape: nWin, nBl, nChan
    coeff1 = getData(fvis, 'winCoeff') # shape: nWin, nBl, nChan
    if (not isinstance(coeff1, np.ma.MaskedArray)):
        coeff1 = np.ma.array(coeff1, mask=False)
    nWin = len(sec)

    # window time
    tmp = atopen + sec/86400        # convert to days
    tWin = tmp.to_datetime()

    if (tlim is None):
        tlim = [sec[0], sec[-1]]
    tsel = (sec>=tlim[0]) * (sec<=tlim[1])
    tt1 = tWin[np.argmin(np.abs(sec-tlim[0]))]
    tt2 = tWin[np.argmin(np.abs(sec-tlim[1]))]


    if (len(spec1.shape) == 2): # backward-compatible with single_dev_vis2.py
        spec1  = spec1.reshape((nWin, 1, nChan))
        coeff1 = coeff1.reshape((nWin, 1, nChan))
        #var    = var.reshape((nWin, 1, nChan))

    nInp = nft.shape[1]
    print('nInp =', nInp)
    nBl = int(nInp*(nInp-1)/2)
    pairs = []
    for ai in range(nInp-1):
        for aj in range(ai+1,nInp):
            pairs.append([ai,aj])

    if (rfmin == 0):
        rfmin = freq.min()
    if (rfmax == 0):
        rfmax = freq.max()

    # define pair id for closure(0,1,2)
    trio = [0, 1, 2]
    bsel = []
    b = -1
    for ai in range(nInp-1):
        for aj in range(ai+1,nInp):
            b += 1
            if (ai in trio and aj in trio):
                bsel.append(b)

    if (f410>0 and f610>0): # override srcflux setting
        xs = (freq-410.)/(610.-410.)
        srcflux2 = f410 + (f610-f410)*xs # shape (nChan,)
    else:
        srcflux2 = np.array([srcflux])


    if (do_fitClkOff):
        ## find and remove clock offset
        print('removing clock offset ...')
        savClkOff = []
        spec3 = spec1.copy()
        coeff3 = coeff1.copy()
        params = np.arange(-maxClkOff,maxClkOff,dClkOff)
        #for i in range(1,2):   # testing
        for i in range(nWin):
            tag = 't%08.1f' % sec[i]
            print('...',i,tag)
            ClkOff = []
            for b in range(nBl):
                vispha = np.ma.angle(spec1[i,b])
                viserr = np.ones(nChan) * 0.3
                like = lnlike(params, vispha, viserr, nChan)
                maxlike = like.max()
                maxidx  = like.argmax()
                maxpara = params[maxidx]
                #print(b, maxpara)
                phimod = phimodel(maxpara, nChan)
                spec3[i,b] = spec1[i,b] / np.exp(1.j*phimod)
                coeff3[i,b] = coeff1[i,b] / np.exp(1.j*phimod)
                ClkOff.append(maxpara)
            savClkOff.append(ClkOff)

            if (do_check):
                png = '%s/correlate_pha2_%s.png' % (cdir, tag)
                fig, sub = blsubplots(na=nInp, figsize=(12,9), squeeze=False)
                b = -1
                for ai in range(nInp-1):
                    for aj in range(ai+1,nInp):
                        b += 1
                        ax = sub[ai,aj-1]
                        ax.plot(freq, np.ma.angle(spec1[i,b]), label='original')
                        ax.plot(freq, np.ma.angle(spec3[i,b]), linestyle='none', marker='.', label='unwrap')
                        ax.text(0.05, 0.90, 'med: %.2f'%np.ma.median(np.ma.angle(spec3[i,b])), transform=ax.transAxes)
                        ax.set_xlabel('freq (MHz)')
                        ax.set_ylabel('phase (rad)')
                        ax.legend(loc='upper right')
                fig.savefig(png)
                plt.close(fig)

        savClkOff = np.array(savClkOff)
        adoneh5(fvis, savClkOff, 'winClkOff')
        adoneh5(fvis, spec3, 'winSpecClkCorr')
        adoneh5(fvis, coeff3, 'winCoeffClkCorr')

        spec2 = spec3 - spec3.mean(axis=0)
        coeff2 = coeff3 - coeff3.mean(axis=0)
        print('... clock offset done.')

    else:
        savClkOff = getData(fvis, 'winClkOff')

        spec3 = getData(fvis, 'winSpecClkCorr')
        if (spec3 is None or noClkCorr):
            spec2 = spec1 - spec1.mean(axis=0)
            spec3 = spec1
        else:
            spec2 = spec3 - spec3.mean(axis=0)
        coeff3 = getData(fvis, 'winCoeffClkCorr')
        if (coeff3 is None or noClkCorr):
            coeff2 = coeff1 - coeff1.mean(axis=0)
            coeff3 = coeff1
        else:
            coeff2 = coeff3 - coeff3.mean(axis=0)
    adoneh5(fvis, spec2, 'winSpecSub')
    adoneh5(fvis, coeff2, 'winCoeffSub')

    xx = sec
    yy = freq
    X,Y = np.meshgrid(xx, yy, indexing='xy')
    tt = tWin
    T,Y = np.meshgrid(tt, yy, indexing='xy')
    ## plot waterfall of real/imag/amp/pha for spec1, spec2 and coeff1, coeff2
    if (do_check):

        for b, p in enumerate(pairs):
            ai, aj = p
            bname = '%d-%d'%(ai,aj)

            for pp in range(4):
                if (pp == 0):
                    png = '%s/waterfall_bl%s_spec_raw.png' % (cdir,bname)
                    pp_spec = spec1
                elif (pp == 1):
                    png = '%s/waterfall_bl%s_spec_sub.png' % (cdir,bname)
                    pp_spec = spec2
                elif (pp == 2):
                    png = '%s/waterfall_bl%s_coeff_raw.png' % (cdir,bname)
                    pp_spec = coeff1
                elif (pp == 3):
                    png = '%s/waterfall_bl%s_coeff_sub.png' % (cdir,bname)
                    pp_spec = coeff2

                fig, sub = plt.subplots(2,2,figsize=(15,10))

                # real
                ax = sub[0,0]
                if (pp > 1):
                    s=ax.pcolormesh(X,Y,pp_spec.real[:,b,:].T,shading='auto')
                else:
                    s=ax.pcolormesh(X,Y,pp_spec.real[:,b,:].T,shading='auto', norm=colors.SymLogNorm(linthresh=1e3))
                cb=plt.colorbar(s, ax=ax)
                cb.set_label('linear')
                ax.set_xlabel('time (sec)')
                ax.set_ylabel('freq (MHz)')
                ax.set_title('Real')
                # imag
                ax = sub[0,1]
                if (pp > 1):
                    s=ax.pcolormesh(X,Y,pp_spec.imag[:,b,:].T,shading='auto')
                else:
                    s=ax.pcolormesh(X,Y,pp_spec.imag[:,b,:].T,shading='auto', norm=colors.SymLogNorm(linthresh=1e3))
                cb=plt.colorbar(s, ax=ax)
                cb.set_label('linear')
                ax.set_xlabel('time (sec)')
                ax.set_ylabel('freq (MHz)')
                ax.set_title('Imag')
                # amp
                ax = sub[1,0]
                s=ax.pcolormesh(X,Y,10*np.ma.log10(np.ma.abs(pp_spec[:,b,:])).T,shading='auto')
                cb=plt.colorbar(s, ax=ax)
                cb.set_label('10*log10(abs())')
                ax.set_xlabel('time (sec)')
                ax.set_ylabel('freq (MHz)')
                ax.set_title('Power')
                # pha
                ax = sub[1,1]
                s=ax.pcolormesh(X,Y,np.ma.angle(pp_spec[:,b,:]).T,shading='auto')
                cb=plt.colorbar(s, ax=ax)
                cb.set_label('phase (rad)')
                ax.set_xlabel('time (sec)')
                ax.set_ylabel('freq (MHz)')
                ax.set_title('Phase')

                fig.tight_layout(rect=[0,0.03,1,0.95])
                fig.suptitle('%s'%png)
                fig.savefig(png)
                plt.close(fig)
        

    ## plot clock offset if possible
    if (savClkOff is not None):
        fig, sub = blsubplots(na=nInp, figsize=(12,9), squeeze=False)
        b = -1
        for ai in range(nInp-1):
            for aj in range(ai+1, nInp):
                b += 1
                ax = sub[ai,aj-1]
                ax.plot(tWin, savClkOff[:,b], marker='.', linestyle=':')
                ax.text(0.05, 0.90, 'min:%.2f max:%.2f'%(savClkOff[:,b].min(), savClkOff[:,b].max()), transform=ax.transAxes)
                ax.set_xlabel('time (sec)')
                ax.set_ylabel('clock off (sample)')

        if (nInp >= 3):
            # add an axes for closure phase, if nInp>=3 and only for the first 3 inputs
            ClkClosure = savClkOff[:,bsel[0]]+savClkOff[:,bsel[2]]-savClkOff[:,bsel[1]]
            axclo = fig.add_axes(rect=[0.08,0.06,0.33,0.33])
            axclo.plot(tWin, ClkClosure, marker='.', linestyle=':')
            axclo.set_xlabel('time')
            axclo.set_ylabel('closure off (sample)')
            axclo.set_title('closure: 01+12-02')

        fig.suptitle('%s'%(fvis,))
        fig.savefig('%s/clock_off.png' % cdir)
        plt.close(fig)


    # fit source delay (not instrument delay)
    tmp = getData(fvis, 'winTAU')
    if (not tmp is None):
        savTAU = tmp
        savPHI = getData(fvis, 'winPHI')

    if (tmp is None or do_tausrc):
        print('fitting source delay ...')
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

        savTAU = []
        savPHI = []
        #for i in range(3):
        for i in range(nWin):
            tag = 't%08.1f' % sec[i]
            print('...',i,tag)

            fig, s2d = blsubplots(na=nInp, figsize=(figw,figh), squeeze=False, sharey=True, sharex=True)
            #fig, s2d = blsubplots(na=nInp, figsize=(12,9), squeeze=False, sharey=True, sharex=True)
            sub = []
            b = -1
            for ai in range(nInp-1):
                for aj in range(ai+1, nInp):
                    ax = s2d[ai,aj-1]
                    sub.append(ax)
                    if (ai == aj-1):
                        ax.set_xlabel('freq (MHz)')
                        ax.set_ylabel('phase (rad)')
                        ax.set_ylim(-3.5, 5.0)

            # results per window
            TAU = []    # source delay in 400MHz samples 
            PHI = []    # measured phase at 600MHz
            for b in range(nBl):
                vispha = np.ma.angle(spec2[i,b])
                viserr = np.ones(nChan) * 0.3
                visamp = np.ma.abs(coeff2[i,b])
                # spectral masking
                mm = vispha.mask
                mm[freq<300.] = True        # low-freq RFI at Fushan
                mm[freq>750.] = True        # 4G signals
                #mm[visamp > 0.25] = True    # strong RFIs?
                vispha.mask = mm
                for ii in range(2): # 2 iterrations
                    if (ii == 0):
                        params = np.arange(-maxTau1,maxTau1,dTau1)
                    else:
                        params = np.arange(Tau1-maxTau2,Tau1+maxTau2,dTau2)
                    #like = lnlike(params, vispha, viserr, nChan, oChan=1536)
                    like = lnlike(params, vispha, viserr, nChan, oChan=0)
                    maxlike = like.max()
                    maxidx  = like.argmax()
                    maxpara = params[maxidx]
                    #print(b, maxpara)
                    if (ii == 0):
                        Tau1 = maxpara
                    else:
                        Tau2 = maxpara
                modpha = phimodel(Tau2, nChan, oChan=1536)
                #modpha = phimodel(Tau2, nChan, oChan=0)
                #spec4[i,b] = spec2[i,b] / np.exp(1.j*phimod)
                #coeff4[i,b] = coeff2[i,b] / np.exp(1.j*phimod)
                TAU.append(Tau2)

                # the residual phase
                #respha = vispha - pwrap2(modpha)
                #respha[respha> np.pi] -= np.pi*2
                #respha[respha<-np.pi] += np.pi*2
                respha = pwrap2(vispha - modpha)
                # calculate median of the residual phase (take care when respha is close to +/-pi)
                rescpx = np.ma.exp(1.j*respha)
                mr = np.ma.median(rescpx.real)
                mi = np.ma.median(rescpx.imag)
                medres = np.angle(mr+1.j*mi)
                PHI.append(medres)      
                # note that since the phimodel is centered at center of the spectrum (i.e. 600MHz)
                # the residual thus represents the measured phase at 600MHz
                ## new phimodel(oChan=1536) should have removed the 600MHz offset
                ## the residual should be actual phase error now

                ## plotting
                ax = sub[b]
                ax.plot(freq, vispha, color='C0', label='data')
                ax.scatter(freq[mm], vispha.data[mm], color='r', label='mask')
                ax.plot(freq, pwrap2(modpha), color='C2', label='model')
                ax.plot(freq, pwrap2(respha), color='C1', marker=',', linestyle='none', label='residual')
                ax.text(0.05, 0.92, 'tau: %.3f samp'%Tau2, transform=ax.transAxes)
                ax.text(0.05, 0.82, 'phi: %.3f rad'%medres, transform=ax.transAxes)

                #if (b == 0):
                #    ax.legend(loc='upper left', bbox_to_anchor=(0,-1.0,0.5,0.8))


            savTAU.append(TAU)
            savPHI.append(PHI)

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
            fig.suptitle('%s, source delay, %s'%(fvis, tag))
            fig.subplots_adjust(wspace=0, hspace=0)
            fig.savefig('%s/source_delay_%s.png'%(cdir, tag))
            plt.close(fig)


        savTAU = np.array(savTAU)
        savPHI = np.array(savPHI)

        adoneh5(fvis, savTAU, 'winTAU')
        adoneh5(fvis, savPHI, 'winPHI')


    # spectral avg
    fa_r = np.ma.median(spec2.real, axis=2)
    fa_i = np.ma.median(spec2.imag, axis=2)
    fa_spec2 = fa_r + 1.j*fa_i  # shape (nWin, nBl)
    fa_r = np.ma.median(spec3.real, axis=2)
    fa_i = np.ma.median(spec3.imag, axis=2)
    fa_spec3 = fa_r + 1.j*fa_i  # shape (nWin, nBl)


    fsel = np.logical_and(freq>=rfmin, freq<=rfmax)
    avgRF = freq[fsel].mean()
    lam = 2.998e8 / (avgRF * 1e6)   # wavelength in meters
    flam = 2.998e8 / (freq * 1e6)

    chBW = fullBW*1e6 / nChan       # channel bandwidth in Hz
    tInt = 1./(fullBW*1e6) * nChan  # integration time per spectrum in sec
    scale = np.sqrt(1.*chBW*tInt)   # conversion between variance and SEFD

    #G0 = 10.                       # on-axis gain (10dB --> 10)
    # fiducial num
    Tsys0 = 100.
    lam0  = 0.5
    Aeff0 = lam0**2 * G0 / (4*np.pi)
    SEFD0 = 2.*1.38e-23*1e26*Tsys0/Aeff0/scale  # converted to Jy
    #print('fiducial SEFD (150K, 0.5m):', SEFD0/1e6, 'MJy')

    Aeff = lam**2 * G0 / (4.*np.pi)
    scale2 = Aeff * scale / (2*1.38e-23) * 1e-26 # Tsys = (S/Jy / SNR) * scale2
    SEFD2 = Tsys0 / scale2
    print('fiducial SEFD (%dK, %.2fm):'%(Tsys0,lam), SEFD2/1e6, 'MJy')
    fscale2 = flam**2 * G0 / (4.*np.pi) * scale / (2*1.38e-23) * 1e-26 
    

    ## plot spectral-avg phase vs time (all baselines)
    fig, sub = blsubplots(na=nInp, figsize=(12,9), squeeze=False)
    # calculate phase error
    #tmp = spec3.var(axis=2) # along spec axis
    #std = np.sqrt(tmp/2)    # approx. scatter perpendicular to the ampld
    #amp = np.ma.abs(spec3).mean(axis=2)
    #perr = np.arctan2(std, amp) # phase scatter in rad
    ## calulate phase error, new
    #std = np.ma.sqrt(var/2) # approx scatter perpendicular to the ampld
    amp = np.ma.abs(spec3)
    #perr = np.arctan2(std, amp).mean(axis=2)
    #perr /= np.sqrt(np.count_nonzero(~amp.mask, axis=2))

    b = -1
    for ai in range(nInp-1):
        for aj in range(ai+1,nInp):
            b += 1
            ax = sub[ai,aj-1]
            #faphase = np.ma.median(np.ma.angle(spec3), axis=1)
            ax.plot(tWin, np.ma.angle(fa_spec3[:,b]), marker='.', color='C0')
            #ax.errorbar(tWin, np.ma.angle(fa_spec3[:,b]), perr[:,b], marker='.', linestyle=':', color='C0')
            ax.set_xlabel('time')
            ax.set_ylabel('phase (rad)')
            #mean_err = perr[:,b].mean()
            #rms_err = np.ma.angle(fa_spec3[:,b]).std()
            #print('%d%d:'%(ai,aj), 'RMS:%.3frad'%rms_err, '<err>:%.3frad'%mean_err)
    fig.autofmt_xdate(rotation=45)
    if (nInp >= 3):
        # add an axes for closure phase, if nInp>=3 and only for the first 3 inputs
        closurep = np.ma.angle(fa_spec3[:,bsel[0]]*fa_spec3[:,bsel[2]]/fa_spec3[:,bsel[1]])
        clospec1 = spec1[:,bsel[0]]*spec1[:,bsel[2]]/spec1[:,bsel[1]]
        closurep1 = np.ma.angle(clospec1).mean(axis=1)
        clospec3 = spec3[:,bsel[0]]*spec3[:,bsel[2]]/spec3[:,bsel[1]]
        closurep3 = np.ma.angle(clospec3).mean(axis=1)
        closeperr = np.ma.angle(clospec3).std(axis=1)/np.sqrt(np.count_nonzero(~clospec3.mask, axis=1))
        axclo = fig.add_axes(rect=[0.08,0.06,0.4,0.33])
        #axclo.plot(sec, closurep, marker='.', linestyle=':', label='fit/f_avg/closure')
        #axclo.plot(sec, closurep3, marker='o', linestyle=':', label='fit/closure/f_avg')
        #axclo.plot(sec, closurep1, marker='.', linestyle=':', label='no-fit/closure/f_avg')
        axclo.errorbar(sec, closurep1, yerr=closeperr, marker='.', linestyle=':', label='no-fit/closure/f_avg')
        axclo.text(0.05, 0.05, 'RMS:%.3frad'%closurep1.std(),transform=axclo.transAxes)
        #axclo.legend()
        axclo.set_xlabel('time (sec)')
        axclo.set_ylabel('closure phase (rad)')
        axclo.set_title('closure: 01+12-02')
        #print(axclo.get_xticks(), axclo.get_xticklabels())
        #axclo.set_xticks(axclo.get_xticks(), rotation=30, ha='right')
        #axclo.set_xticklabels(axclo.get_xticklabels(), rotation=30, ha='right')
    fig.tight_layout()
    fig.savefig('%s/all_phase_vs_time.png'%odir)
    plt.close(fig)


    ## SEFD plot
    if (False):
        if (srcflux > 0):
            medAmp = np.ma.median(np.ma.abs(spec2)) # median level
            calFac = medAmp / srcflux
            sefd = np.ma.sqrt(var) / calFac * scale # SEFD in Jy
            sefd.fill_value = 0.
            par = np.percentile(sefd.filled(), [0, 15, 85, 100], axis=0)
        else:
            par = np.zeros((4, nChan))

        f3, s3 = plt.subplots(1,1)
        png3 = '%s.SEFD.png' % fvis
        suptitle = 'file: %s, %.0fMHz, nWin=%d' % (fvis, fcen/1e6, nWin)

        ax = s3
        ax.fill_between(freq, par[0], par[3], alpha=0.2, color='b', label='min/max')
        ax.fill_between(freq, par[1], par[2], alpha=0.6, color='b', label='15%/85% pct')
        ax.grid(True, which='both', axis='y')
        ax.grid(True, which='major', axis='x')
        ax.legend()
        ax.set_yscale('log')
        ax.set_xlabel('freq (MHz)')
        ax.set_ylabel('SEFD (Jy)')
        ax.set_title(suptitle)
        f3.tight_layout()
        f3.savefig(png3)
        plt.close(f3)



    if (do_model):
        ut0 = tWin[0]
        b, obs = obsBody(body, time=ut0, site=site, retOBS=True, DB=DB)

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
        adoneh5(fvis, az, 'az')
        adoneh5(fvis, el, 'el')

        ## method 1
        ## convert the az,el to HA,DEC but assuming latitude=0
        ## if the dipole is in N-S direction, then
        ## DEC is the E-plane angle
        ## HA is the H-plane angle
        phi1 = 0
        DEC = np.arcsin(np.sin(el)*np.sin(phi1)+np.cos(el)*np.cos(phi1)*np.cos(az))
        HA  = np.arcsin((-1)*np.sin(az)*np.cos(el)/np.cos(DEC))

        ## method 2
        za = np.pi/2 - el
        pntr = np.sin(za)
        pntz = np.cos(za)
        pntx = pntr * np.sin(az)
        pnty = pntr * np.cos(az)
        EWoff = np.arctan2(pntx,pntz)
        NSoff = np.arctan2(pnty,pntz)
        # overrides method 1 now
        HA = -EWoff
        DEC = NSoff

        ## apply calculate attenuation
        if (swapEH):
            Hatt = atten(NSoff/np.pi*180., Ehwhm)
            Eatt = atten(EWoff/np.pi*180., Hhwhm)
        else:
            Eatt = atten(EWoff/np.pi*180., Ehwhm)
            Hatt = atten(NSoff/np.pi*180., Hhwhm)
        att0 = Eatt*Hatt
        print('max atten:', att0.max())

        unitVec  = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], ndmin=2).T
        unitVec *= -1       # swapped definition

        if (blfit == 0):    # no fitting
            xyzparams = [antXYZ] 
            fitparams = [[0,0,0,0]]
        elif (blfit == 1):  # 4 antennas in a straight line
            angrange = np.arange(-anglim, anglim, angsep) + ang0
            a1range = np.arange(-lenlim, lenlim, lensep) + pos0[0]
            a2range = np.arange(-lenlim, lenlim, lensep) + pos0[1]
            a3range = np.arange(-lenlim, lenlim, lensep) + pos0[2]
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
        xyzmodComplex = []
        for fiti in range(nParams):
            xyz = xyzparams[fiti]
            tmpmodComplex = []
            b = -1
            like = 0.
            for ai in range(nInp-1):
                for aj in range(ai+1, nInp):
                    b += 1
                    BVec = xyz[aj] - xyz[ai]
                    
                    Bproj  = np.zeros_like(theta)
                    for j in range(len(unitVec)):
                        Bproj[j]  = np.dot(BVec, unitVec[j])  # in meters
                    modComplex = np.exp(-2.j*np.pi*Bproj/lam)
                    tmpmodComplex.append(modComplex)
                    resPhase = np.angle(fa_spec2[:,b]/modComplex)
                    resPhase -= np.median(resPhase)
                    like += -0.5*(resPhase**2/0.3**2).sum()  # fixed phase error
            xyzlike.append(like)
            xyzmodComplex.append(tmpmodComplex)

        xyzlike = np.array(xyzlike)
        maxidx = xyzlike.argmax()
        maxxyz = xyzparams[maxidx]
        print('best xyz:', maxxyz)
        print('best fit param:', fitparams[maxidx])
        blmodComplex = xyzmodComplex[maxidx]
        blmodPhase = np.angle(blmodComplex)
        adoneh5(fvis, maxxyz, 'ant_xyz')

        # save the Bproj
        savBproj = []
        for i in range(nWin):
            Bproj = []
            for ai in range(nInp-1):
                for aj in range(ai+1,nInp):
                    BVec = maxxyz[aj] - maxxyz[ai]
                    Bproj.append(np.dot(BVec, unitVec[i]))
            savBproj.append(Bproj)
        savBproj = np.array(savBproj)
        if (revtau):
            savBproj *= -1


    ## plotting for each baseline
    savTsys = []
    savTsysmask = []
    savTsys2D = []
    savTsys2Dmask = []
    savSEFD2D = []
    savSEFD2Dmask = []
    savTauMod = []  # model delay
    savPhiMod = []  # model phase (600MHz)
    savTauRes = []  # residual delay
    savTauMed = []
    savPhiRes = []  # residual phase (600MHz)
    savPhiMed = []
    for b, pa in enumerate(pairs):
        ai, aj = pa
        corr = '%d-%d'%(ai,aj)

        if (do_model):
            modComplex = blmodComplex[b]
            modPhase = blmodPhase[b]

        ## vs. RF plot
        pwd  = os.getcwd()
        pwd1 = os.path.basename(pwd)
        for pp in range(2):
            if (pp == 0):
                spec = spec3[:,b]
                coeff = coeff3[:,b]
                png1 = '%s/bl%s.rf%d_%d.all_win.png' % (odir, corr, rfmin, rfmax)
                suptitle = 'file: %s, bl:%s' % (fvis, corr)
            else:
                spec = spec2[:,b]
                coeff = coeff2[:,b]
                png1 = '%s/bl%s.rf%d_%d.all_win.sub.png' % (odir, corr, rfmin, rfmax)
                suptitle = 'file: %s, bl:%s, sub_t-mean' % (fvis, corr)

            # spectral avg
            fr = np.ma.median(coeff.real, axis=1)
            fi = np.ma.median(coeff.imag, axis=1)
            fa_coeff = fr + 1.j*fi
            if (do_model):
                # fring-stopping
                fs_coeff = fa_coeff / modComplex
                cr = np.ma.median(fs_coeff.real)
                ci = np.ma.median(fs_coeff.imag)
                ap_cal = cr + 1.j*ci    # cal both amp and phase
                p_cal = ap_cal / np.abs(ap_cal) # cal only phase
                calPhase = np.ma.angle(p_cal)
                # median phase calibrate
                cfs_coeff = fs_coeff / p_cal
                modPhase2 = np.angle(modComplex * p_cal)


            #f1, s1 = plt.subplots(2,2,figsize=(12,9),sharex=True)
            f1, s1 = plt.subplots(3,2,figsize=(12,12))

            ax = s1[1,0]
            ax.set_title('cross01')
            ax.set_xlabel('freq (MHz)')
            #ax.set_ylabel('vis.power (dBm)')
            ax.set_ylabel('xcorr coeff')
            ax.set_yscale('log')
            ax.set_ylim([2e-3, 2])
            ax = s1[1,1]
            ax.set_title('cross01')
            ax.set_xlabel('freq (MHz)')
            ax.set_ylabel('vis.phase (rad)')
            ax.set_ylim([-3.2,3.2])
            #for j in range(2):
            for j,jj in enumerate([ai,aj]):
                ax = s1[0,j]
                ax.set_title('Ant%d'%jj)
                ax.set_xlabel('freq (MHz)')
                ax.set_ylabel('power (dBm)')


            for i in range(nWin):
                #print('   window: (%d/%d)'%(i+1,nWin))
                cc = 'C%d'%(i%10)

                ax = s1[1,0]
                #ax.plot(freq, toPower(np.ma.abs(spec[i]), mode='vis'), color=cc)
                ax.plot(freq, np.ma.abs(coeff[i]), color=cc)

                ax = s1[1,1]
                ax.plot(freq, np.ma.angle(spec[i]), color=cc, linestyle='none', marker='.')

                #for j in range(2):
                for j,jj in enumerate([ai,aj]):
                    ax = s1[0,j]
                    if (j==1):
                        lab = '%04.0fsec'%sec[i]
                    else:
                        lab = ''
                    ax.plot(freq, toPower(nft[i,jj]), color=cc, label=lab)

            med_coeff = np.ma.median(np.ma.abs(coeff), axis=1)
            par2 = np.percentile(med_coeff, [15,50,85])
            med_med_coeff = par2[1]
            std_med_coeff = (par2[2]-par2[0])/2.
            ax = s1[1,0]

            ax = s1[2,0]
            if (do_model):
                ax.plot(sec, sigma_clip(cfs_coeff.real), 'b-', label='real')
            ax.plot(sec, sigma_clip(med_coeff), 'b:', label='abs')
            ax.legend()
            ax.set_xlabel('window time (sec)')
            ax.set_ylabel('med_coefficient')
            ax.text(0.05, 0.90, 'med:%.3g+/-%.3g'%(med_med_coeff, std_med_coeff), transform=ax.transAxes)

            med_phase = np.ma.median(np.ma.angle(spec), axis=1)
            ax = s1[2,1]
            ax.plot(sec, med_phase)
            if (do_model):
                ax.plot(sec, modPhase2)
            ax.set_ylim([-3.2,3.2])
            ax.set_xlabel('window time (sec)')
            ax.set_ylabel('med_phase')


            #s1[0,1].legend()
            #s1[0,0].get_shared_y_axes().join(s1[0,0], s1[0,1], s1[1,0])
            s1[0,0].get_shared_y_axes().join(s1[0,0], s1[0,1])
            s1[0,0].get_shared_x_axes().join(s1[0,0], s1[0,1], s1[1,0], s1[1,1])
            s1[2,0].get_shared_x_axes().join(s1[2,0], s1[2,1])

            s1[0,0].set_xlim(rfmin, rfmax)

            f1.tight_layout(rect=[0,0.03,1,0.95])
            f1.suptitle(suptitle)
            f1.savefig(png1)
            plt.close(f1)


            ## vs. time plot
            if (pp == 0):
                png2 = '%s/bl%s.rf%d_%d.vs_time.png' % (odir, corr, rfmin, rfmax)
                suptitle = 'file: %s, bl:%s\nRF=[%.1f,%.1f]'%(fvis, corr, rfmin, rfmax)
            else:
                png2 = '%s/bl%s.rf%d_%d.vs_time.sub.png' % (odir, corr, rfmin, rfmax)
                suptitle = 'file: %s, bl:%s, sub_t-mean\nRF=[%.1f,%.1f]'%(fvis, corr, rfmin, rfmax)

            #f2, s2 = plt.subplots(3,1, figsize=(8,9), sharex=True)
            f2, s2 = plt.subplots(2,2, figsize=(15,10), sharex=True)

            # freq avg
            faspec = spec[:,fsel].mean(axis=1)
            faphase = np.ma.median(np.ma.angle(spec), axis=1)

            # vs sec or vs date
            #tt = sec
            tt = tWin
            
            # phase plot
            ax = s2[0,0]
            #ax.plot(tt, np.ma.angle(faspec), marker='.')
            #ax.plot(tt, faphase, marker='.')
            ax.plot(tt, savPHI[:,b], ls='none', marker='*', label='data')
            rescpx = np.exp(1.j*savPHI[:,b])
            mr = np.median(rescpx.real)
            mi = np.median(rescpx.imag)
            medres = np.angle(mr+1.j*mi)
            ax.text(0.05, 0.90, 'med_dphi: %.3f rad'%medres, transform=ax.transAxes)
            if (False): # new phase residual does not need to be subtracted
            #if (do_model):
                ax.plot(tt, modPhase, label='model')
                savPhiMod.append(modPhase)
                respha = savPHI[:,b] - modPhase
                savPhiRes.append(pwrap2(respha))
                ax.scatter(tt, pwrap2(respha), marker='*', label='data-model')
                rescpx = np.exp(1.j*respha)
                mr = np.median(rescpx.real)
                mi = np.median(rescpx.imag)
                medres = np.angle(mr+1.j*mi)
                ax.text(0.05, 0.90, 'med_dphi: %.3f rad'%medres, transform=ax.transAxes)
                savPhiMed.append(medres)
            ax.legend()
            ax.set_ylim([-3.5, 4.5])
            ax.set_xlabel('time')
            ax.set_ylabel('resid.phase (rad)')

            # delay plot (convert to ns)
            ax = s2[1,0]
            ns_data = -savTAU[:,b]/400e6*1e9
            ax.plot(tt, ns_data, marker='.', label='data')
            if (do_model):
                ns_model = savBproj[:,b]/2.998e8*1e9
                ax.plot(tt, ns_model, label='model')
                ns_diff = ns_data - ns_model
                medns = np.median(ns_diff)
                ns_cdiff = sigma_clip(ns_diff)
                stdns = ns_cdiff.std()
                ax.scatter(tt, ns_diff, marker='*', label='data-model')
                ax.scatter(tt[ns_cdiff.mask], ns_diff[ns_cdiff.mask], marker='*', color='r')
                ax.fill_between(tt, np.ones(len(tt))*ns_cdiff.min(), np.ones(len(tt))*ns_cdiff.max(), color='b', alpha=0.2)
                ax.text(0.05, 0.90, 'med_dtau: %.3f, std: %.3f ns'%(medns,stdns), transform=ax.transAxes)
                ax.set_ylim(-100, 100)
                ax.grid()
                if (pp==1):
                    savTauMod.append(ns_model)
                    savTauRes.append(ns_diff)
                    savTauMed.append(medns)
            ax.set_xlabel('time')
            ax.set_ylabel('src_delay (ns)')
                

            # power plot
            ax = s2[0,1]
            ax.set_xlabel('time')
            sel = 1
            if (sel==0):
                if (do_model):
                    ax.plot(tt, sigma_clip(cfs_coeff.real), 'b-', label='real')
                ax.plot(tt, sigma_clip(med_coeff), 'b:', label='abs')
                ax.legend()
                ax.set_ylabel('corr strength')
            elif (sel==1):
                #if (do_model):
                #    Tsys1 = srcflux / cfs_coeff.real * scale2
                #    #ax.plot(tt, sigma_clip(Tsys1), 'b-', label='from real')
                Tsys2 = srcflux2.mean() / med_coeff * scale2
                if (do_model):
                    Tsys2 *= att0
                # correct Tsys bias due to Solar flux
                Tsys2 *= (1.-med_coeff)
                cTsys2 = sigma_clip(Tsys2)
                if (pp==1):
                    savTsys.append(cTsys2)
                    savTsysmask.append(cTsys2.mask)
                ax.plot(tt, cTsys2, 'b:', label='from abs')
                ax.text(0.05, 0.90, 'Tsys_min=%.0fK'%cTsys2.min(), transform=ax.transAxes)
                ax.legend()
                ax.set_ylabel('Tsys (K)')
                ax.grid()


            # elevation plot
            ax = s2[1,1]
            if (do_model):
                ax.plot(tt, el/np.pi*180, marker='.')
            ax.set_xlabel('time')
            ax.set_ylabel('elevation (deg)')

            f2.tight_layout(rect=[0,0.03,1,0.95])
            f2.suptitle(suptitle)
            f2.savefig(png2)
            plt.close(f2)


            ## Coeff plot
            if (pp == 0):
                png = '%s/bl%s.coeff2d.png' % (odir, corr)
                suptitle = 'file: %s, bl:%s' % (fvis, corr)
            elif (pp == 1):
                png = '%s/bl%s.coeff2d.sub.png' % (odir, corr)
                suptitle = 'file: %s, bl:%s, sub-t_mean' % (fvis, corr)

            fig, sub = plt.subplots(2,2,figsize=(10,8),sharex=True,sharey=True)
            # real
            ax = sub[0,0]
            ax.pcolormesh(T,Y,coeff.real.T)
            ax.set_title('real')
            ax.set_ylabel('freq (MHz)')
            # real
            ax = sub[0,1]
            ax.pcolormesh(T,Y,coeff.imag.T)
            ax.set_title('imag')
            # ampld
            ax = sub[1,0]
            ax.pcolormesh(T,Y,np.ma.abs(coeff).T)
            ax.set_title('abs')
            ax.set_ylabel('freq (MHz)')
            ax.set_xlabel('time')
            # phase
            ax = sub[1,1]
            ax.pcolormesh(T,Y,np.ma.angle(coeff).T)
            ax.set_title('phase')
            ax.set_xlabel('time')

            fig.tight_layout(rect=[0,0.03,1,0.95])
            fig.subplots_adjust(wspace=0)
            fig.suptitle(suptitle)
            fig.savefig(png)
            plt.close(fig)



        ## 2D Tsys, SEFD plot
        if (do_model):
            Tsys2D = srcflux2.reshape((1,-1)) / np.ma.abs(coeff) * fscale2.reshape((1,-1))
            Tsys2D *= att0.reshape((-1,1))
            # correct Tsys bias due to Solar flux
            Tsys2D *= (1.-np.ma.abs(coeff))
            # Tsys.shape = (nWin, nChan)
            savTsys2D.append(Tsys2D)
            savTsys2Dmask.append(Tsys2D.mask)
            Tsys_time = np.ma.median(Tsys2D, axis=1)
            Tsys_freq = np.ma.median(Tsys2D[tsel], axis=0)
            Tsys_time2 = Tsys2D.min(axis=1)
            Tsys_freq2 = Tsys2D[tsel].min(axis=0)
            #vmin = Tsys2D.min()
            #vmax = Tsys2D.max()

            SEFD2D = Tsys2D / fscale2.reshape((1,-1))
            fscale3 = ((flam/flam[0])**2).reshape((1,-1))
            #print('debug', fscale3)
            SEFD2D *= fscale3
            savSEFD2D.append(SEFD2D)
            savSEFD2Dmask.append(SEFD2D.mask)
            SEFD2D_time = np.ma.median(SEFD2D, axis=1)
            SEFD2D_freq = np.ma.median(SEFD2D[tsel], axis=0)
            SEFD2D_time2 = SEFD2D.min(axis=1)
            SEFD2D_freq2 = SEFD2D[tsel].min(axis=0)


            for tp in range(2):
                if (tp==0):
                    Y2D = Tsys2D
                    png4 = '%s/bl%s.rf%d_%d.Tsys2D.png' % (odir, corr, rfmin, rfmax)
                    vmin = zlim[0]
                    vmax = zlim[1]
                    cblabel = 'Tsys (K)'
                    yname = 'Tsys'
                    y_freq  = Tsys_freq     # med
                    y_freq2 = Tsys_freq2    # min
                    y_time  = Tsys_time     # med
                    y_time2 = Tsys_time2    # min 
                elif (tp==1):
                    Y2D = SEFD2D/1e6    # convert to MJy
                    png4 = '%s/bl%s.rf%d_%d.SEFD2D.png' % (odir, corr, rfmin, rfmax)
                    vmin = zlim[0]/60
                    vmax = zlim[1]/60       # 10MJy, roughly 300K
                    cblabel = 'SEFD*(L/L0)^2 (MJy)'
                    yname = 'SEFD'
                    y_freq  = SEFD2D_freq/1e6     # med
                    y_freq2 = SEFD2D_freq2/1e6    # min
                    y_time  = SEFD2D_time/1e6     # med
                    y_time2 = SEFD2D_time2/1e6    # min 

                fig = plt.figure(figsize=(12,9))
                gs = fig.add_gridspec(2,2,width_ratios=(3,1),height_ratios=(1,3), wspace=0, hspace=0, left=0.08, right=0.95, bottom=0.08, top=0.92)
                ax2D = fig.add_subplot(gs[1,0])
                axtp = fig.add_subplot(gs[0,0])
                axrt = fig.add_subplot(gs[1,1])

                s = ax2D.pcolormesh(T,Y,Y2D.T, vmin=vmin, vmax=vmax)
                ax2D.set_xlabel('time (UT)')
                ax2D.set_ylabel('freq (MHz)')
                #cb = plt.colorbar(s, ax=ax2D)
                #cb.set_label('Tsys (K)')
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

                if (np.count_nonzero(y_freq.mask) < nChan):
                    axrt.plot(y_freq, freq, label='med')
                if (np.count_nonzero(y_freq2.mask) < nChan):
                    axrt.plot(y_freq2, freq, linestyle=':', label='min')
                axrt.set_ylim(freq[0], freq[-1])
                axrt.legend()
                axrt.set_xlabel(cblabel)
                axrt.set_xlim(vmin, vmax)
                axrt.grid(axis='both')
                axrt.set_yticklabels([])
                axrt.axvline(np.ma.median(y_freq), color='k', linestyle=':')

                if (tp==0):
                    fig.text(0.76, 0.80, 'med(%s_freq.med) = %.0f K'%(yname, np.ma.median(y_freq)), color='C0')
                    fig.text(0.76, 0.75, 'med(%s_freq.min) = %.0f K'%(yname, np.ma.median(y_freq2)), color='C1')
                elif (tp==1):
                    fig.text(0.76, 0.80, 'med(%s_freq.med) = %.3f MJy'%(yname, np.ma.median(y_freq)), color='C0')
                    fig.text(0.76, 0.75, 'med(%s_freq.min) = %.3f MJy'%(yname, np.ma.median(y_freq2)), color='C1')
                    fig.text(0.76, 0.85, 'SEFD*(400/fMHz)^2')
                fig.suptitle(suptitle)
                fig.savefig(png4)
                plt.close(fig)
                #print('debug', png4) 
            #sys.exit('debug done.')


    if (do_model):
        savTsys = np.ma.array(savTsys, mask=savTsysmask)
        adoneh5(fvis, savTsys.T, 'winTsys')   # shape (nWin, nBl), after the transpose
        savTsys2D = np.ma.array(savTsys2D, mask=savTsys2Dmask)
        adoneh5(fvis, savTsys2D.transpose((1,0,2)), 'winTsys2D') # shape (nWin, nBl, nChan)
        savSEFD2D = np.ma.array(savSEFD2D, mask=savSEFD2Dmask)
        adoneh5(fvis, savSEFD2D.transpose((1,0,2)), 'winSEFD2D') # shape (nWin, nBl, nChan)

        savTauMod = np.array(savTauMod).T
        savPhiMod = np.array(savPhiMod).T
        savTauRes = np.array(savTauRes).T
        savPhiRes = np.array(savPhiRes).T
        adoneh5(fvis, savTauMod, 'winTauMod')
        adoneh5(fvis, savPhiMod, 'winPhiMod')
        adoneh5(fvis, savTauRes, 'winTauRes')
        adoneh5(fvis, savPhiRes, 'winPhiRes')
        adoneh5(fvis, savTauMed, 'medTauRes')
        adoneh5(fvis, savPhiMed, 'medPhiRes')



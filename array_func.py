#!/usr/bin/env python
from loadh5 import *
from scipy.optimize import curve_fit
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as pat


def line(x, a, b):
    return a * x + b

def gauss(x, a, b, d):  # with fixed width (c=sig)
    y = a * np.exp(-(x-b)**2 / sig**2 / 2.)
    return y

def gauss2(x, a, b, c, d):
    y = a * np.exp(-(x-b)**2 / c**2 / 2.) + d
    return y

def pwrap(x, s, o): # wrap a phase slope
    # x is offset in arcmin
    # y is phase in rad
    y = s * x + o
    w = np.where(y>np.pi)
    while (w[0].size > 0):
        y[w] -= 2. * np.pi
        w =np.where(y>np.pi)
    w = np.where(y<-np.pi)
    while (w[0].size > 0):
        y[w] += 2. * np.pi
        w =np.where(y<-np.pi)
    return y



def pwrap2(p0, bounds=[-np.pi, np.pi]):
    # p0 is array-like. 
    # input phase in radians
    # output phase is kept within bounds

    p = p0.copy()   # try not to change the input

    while((p<bounds[0]).any()):
        p[p<bounds[0]] += 2. * np.pi

    while((p>=bounds[1]).any()):
        p[p>=bounds[1]] -= 2. * np.pi

    return p


def antpos2(skypol=180.):
    '''
    return the antenna position and baseline vectors in meters
    the X and Y coordicates corresponds to the u and v if the skypol is provided.
    optional input:
        skypol      scalar, the skypol value reported by ACU (degree)

    output:
        antxy       shape (na, 2), antenna (X, Y) in meters
        blxy        shape (nb, 2), baseline vector in meters
    '''
    na = 7
    nb = int(na * (na-1) / 2)
    b0   = 1.400        # shortest baseline length in m
    phi0 = 60.          # location of Ant1 (deg North from East)
    spcorr = -8.        # skypol correction

    dphi = (skypol + spcorr) - 180.
    phi0 += dphi

    antxy = np.zeros((na, 2))
    for ai in range(1, na):
        phi = (phi0 - 60. * (ai-1)) * np.pi / 180.
        antxy[ai, 0] = b0 * np.cos(phi)
        antxy[ai, 1] = b0 * np.sin(phi)

    blxy = np.zeros((nb, 2))
    bi = -1
    for ai in range(na-1):
        for aj in range(ai+1, na):
            bi += 1
            blxy[bi] = antxy[aj] - antxy[ai]

    return antxy, blxy

def solveAntPha(blresidual, loghz, bad=999.):

    if (not isinstance(blresidual, np.ma.MaskedArray)):
        blresidual = np.ma.array(blresidual)
        blresidual.mask = (blresidual == bad)

    X = np.ma.empty((2, na))
    P = np.ma.empty((2, na))
    M = np.ma.empty((2, nb))
    R = np.ma.empty((2, nb))

    M.mask = np.zeros_like(M, dtype=bool)
    R.mask = np.zeros_like(R, dtype=bool)
    
    for sb in range(2):
        if (sb == 0):       # lsb
            rf  = 84.0 + loghz - (462.5/1024.) * 2.24
        elif (sb == 1):     # usb
            rf  = 84.0 + loghz + (462.5/1024.) * 2.24
        lam     = 2.998e11 / rf / 1.e9  # lambda in mm
        scale   = lam / (2. * np.pi)

        D = blresidual[sb]

        A = np.zeros((nb, na))
        b = -1
        for ai in range(na-1):
            for aj in range(ai+1, na):
                b += 1
                if (not D.mask[b]):
                    A[b, ai] =  1.
                    A[b, aj] = -1.
                #else:
                #    D[b] = 0.

        Ainv = np.linalg.pinv(A, rcond=1.e-6)
        X[sb] = np.dot(Ainv, D)
        X[sb] -= X[sb, 0]           # reference to Ant0
        M[sb] = np.ma.dot(A, X[sb])
        R[sb] = D - M[sb]
        M[sb].mask  = D.mask
        R[sb].mask  = D.mask

        P[sb] = -X[sb] * scale  # deformation in mm

    return X, P, M, R


def pmodel(offdir, offlen, blxy, loghz):        # phase of a point source model
    rf0 = 84.0 + loghz          # central freq
    lam0    = 2.998e8 / rf0 / 1.e9      # lambda in m
    scale   = 2. * np.pi * (np.pi / 180. / 60.) / lam0 # scaling between projected baseline length and phase change per arcmin

    # blphase is in rad
    srad = offdir / 180. * np.pi
    sx = offlen * np.cos(srad)
    sy = offlen * np.sin(srad)
    svec = [sx, sy]

    varphase = np.empty((2, nb))
    for sb in range(2):
        for bi in range(nb):
            phi = np.dot(svec, blxy[bi]) * scale
            if (sb == 0):   # lsb
                rf1 = rf0 - (462.5/1024.)*2.24
            else:               # usb
                rf1 = rf0 + (462.5/1024.)*2.24
            varphase[sb, bi] = phi / rf0 * rf1

    while (np.any(varphase >  np.pi)):
        varphase[varphase  >  np.pi] -=  2. * np.pi
    while (np.any(varphase < -np.pi)):
        varphase[varphase  < -np.pi] +=  2. * np.pi

    return varphase


def phasediff(blphase, blperr, varphase, flag=999.):

    if (not isinstance(blphase, np.ma.MaskedArray)):    # probably both blphase and blperr are not np.ma.ndarray
        print('not masked')
        blphase = np.ma.array(blphase)
        blperr  = np.ma.array(blperr)
        mask = (blperr == flag) + (blperr == 0.)
        blphase.mask = mask
        blperr.mask  = mask

    presid = np.ma.zeros((2, nb))
    chi2 = 0.
    wt   = 0.
    for sb in range(2):
        for bi in range(nb):
            if (not blperr.mask[sb, bi]):
                # make difference = data - model
                dp  = -varphase[sb, bi] + blphase[sb, bi]
                while (np.abs(dp) > np.pi):
                    if (dp > np.pi):
                        dp -= 2. * np.pi
                    elif (dp < -np.pi):
                        dp += 2. * np.pi
                chi2 += dp**2 / blperr[sb, bi]**2
                wt   += 1.
            else:
                dp = flag

            presid[sb, bi] = dp
    presid.mask = blphase.mask

    if (wt > 0.):
        #print y, x, offlen, offdir, chi2
        chi2 /= wt

    return presid, chi2


def solvePointing(blphase, blperr, loghz, cmdoff=[0.,0.], ws=2., ds=0.1, skypol=180.):
    '''
    ** only for 7-element, in 1.4m pattern **
    find the best-fit point source location from the array phase information

    input: blphase, blperr has shape (nsb, nb), loghz is the LO in GHz

    optional:
        cmdoff is single corrdinate (dRA, dDEC)
        ws is the search window size in arcmin; searching in (-ws/2, ws/2)
        skypol is the ACU reported skypol of the data in deg

    output: bestx, besty, bestphase, bestresid, Xoff, Yoff, chis
    '''

    antxy, blxy = antpos2(skypol)

    #Xoff = np.arange(-10., 10., 0.2)
    #Yoff = np.arange(-10., 10., 0.2)
    Xoff = np.arange(cmdoff[0]-ws/2., cmdoff[0]+ws/2., ds)
    Yoff = np.arange(cmdoff[1]-ws/2., cmdoff[1]+ws/2., ds)
    chis = np.ones((len(Yoff), len(Xoff))) * 1.e10
    wt   = np.ones_like(chis)   # a weighting map to downweight peaks that are far away
    for yi in range(len(Yoff)):
        y = Yoff[yi]
        for xi in range(len(Xoff)):
            x = Xoff[xi]
            offdir = np.arctan2(y, x) / np.pi * 180.
            offlen = np.sqrt(x**2 + y**2)
            # given offset direction (offdir, in deg)
            # and offset length (offlen, in arcmin)
            # calculate chi-square of blphase

            varphase = pmodel(offdir, offlen, blxy, loghz)
            blpres = np.zeros_like(blphase)

            blpres, chi2 = phasediff(blphase, blperr, varphase)
            chis[yi, xi] = chi2

            r = np.sqrt((x-cmdoff[0])**2 + (y-cmdoff[1])**2)
            s0 = 0.1    # the fractional reduction at r == ws/2
            wt[yi, xi] = 1. - s0*(r/(ws/2.))


    #mindir = dirs[chis.argmin()]
    #min_idx = np.unravel_index(chis.argmin(), chis.shape)
    min_idx = np.unravel_index((chis*wt).argmin(), chis.shape)
    besty = Yoff[min_idx[0]]
    bestx = Xoff[min_idx[1]]
    bestdir = np.arctan2(besty, bestx) / np.pi * 180.
    bestlen = np.sqrt(bestx**2 + besty**2)
    bestphase = pmodel(bestdir, bestlen, blxy, loghz)
    bestresid, chi2 = phasediff(blphase, blperr, bestphase)

    return bestx, besty, bestphase, bestresid, Xoff, Yoff, chis


def imgDFT(vis, weight, loghz, center=[0., 0.], spol=180., ws=10., ds=0.1):
    '''
    invert the visibility to image with direct Fourier transform 
    input:
        vis: complex, shape = (nsb, nb)
        weight: real, shape = (nsb, nb)
        loghz: LO in GHz
        center: phase center in image (dRA, dDEC) in arcmin
        spol: ACU skypol in deg
        ws: map size in arcmin
        ds: map grid size in arcmin

    output:
        sky: real, shape = (nmap, nmap); nmap = int(ws / ds)    
    '''

    antxy, blxy = antpos2(spol)
    rf0     = 84.0 + loghz              # central freq
    lam0    = 2.998e8 / rf0 / 1.e9      # lambda in m
    scale   = 2. * np.pi / lam0         # blxy * scale = 2pi u

    nmap    = int(ws / ds)
    sky     = np.ma.zeros((nmap, nmap))

    if (np.isfinite(center).all()):    # phase center in image (arcmin)
        center = np.array(center)
        center *= np.pi / 180. / 60.    # radians

        # change in vis.phase, shape: (nb)
        dphi = np.dot(blxy, center) * scale * 2. * np.pi
        dphi = dphi.reshape((1, nb))
        vis *= np.exp(-1.j * dphi)

    x = np.arange(-ws/2., ws/2., ds) / 60. / 180. * np.pi
    y = np.arange(-ws/2., ws/2., ds) / 60. / 180. * np.pi
    X, Y = np.meshgrid(x, y, indexing='xy')

    sumwt = 0.
    for s in range(nsb):
        for b in range(nb):
            #sky += (np.exp(-1.j * scale * (blxy[b][0]*X + blxy[b][1]*Y)) * vis[s,b] / np.abs(vis[s,b])).real   # use phase info only, no weighting
            if (not vis.mask[s,b]):
                sky += (np.exp(-1.j * scale * (blxy[b][0]*X + blxy[b][1]*Y)) * vis[s,b] * weight[s,b]).real     # use phase info only
                sumwt += weight[s,b]

    sky /= sumwt

    return sky


def imgFFT(vis, weight, loghz, center=[0., 0.], spol=180., ws=10., ds=0.1, do_beam=False, do_aimg=False):
    '''
    invert the visibility to image with direct Fourier transform 
    input:
        vis: complex, shape = (nsb, nb)
        weight: real, shape = (nsb, nb)
        loghz: LO in GHz
        center: phase center in image (dRA, dDEC) in arcmin
        spol: ACU skypol in deg
        ws: map size in arcmin
        ds: map grid size in arcmin
        do_beam: whether to output the dirty beam (2x in width and height)
        do_aimg: whether to output the amplitude image (vis.phase set to 0)

    output:
        sky: real, shape = (nmap, nmap); nmap = int(ws / ds)
        if (beam=True)
            bm: real, shape = (nmap*2, nmap*2); nmap = int(ws / ds)
    '''

    antxy, blxy = antpos2(spol)
    rf0     = 84.0 + loghz              # central freq
    lam0    = 2.998e8 / rf0 / 1.e9      # lambda in m
    scale   = 1. / lam0                 # blxy * scale / du = uvpix

    nmap    = int(ws / ds)
    extra   = 4                         # enlarge factor of FFT; must be greater than 2
    nfft    = nmap * extra              # make larger map and cutout smaller part
                                        # to make uv pixel more accurate

    wfft    = ds * nfft / 60. / 180. * np.pi    # FFT map size in radian
    du      = 1. / wfft                 # uv pixel size


    if (np.isfinite(center).all()):    # phase center in image (arcmin)
        center = np.array(center)
        center *= np.pi / 180. / 60.    # radians

        # change in vis.phase, shape: (nb)
        dphi = np.dot(blxy, center) * scale * 2. * np.pi
        dphi = dphi.reshape((1, nb))
        vis *= np.exp(-1.j * dphi)


    uvpix   = np.zeros_like(blxy, dtype=int)
    for b in range(nb):
        for i in range(2):
            uvpix[b,i] = int(blxy[b,i] * scale / du)

    uv      = np.ma.zeros((nfft, nfft), dtype=complex)  # for dirty image
    uv2     = np.ma.zeros((nfft, nfft), dtype=float)    # for dirty beam
    uv3     = np.ma.zeros((nfft, nfft), dtype=complex)  # for amplitude image

    sumwt = 0.
    for s in range(nsb):
        for b in range(nb):
            #print 's,b,uvpix', s, b, uvpix[b]
            #print 'vis, weight', vis[s,b], vis[s,b].conjugate(), weight[s,b]
            if (not vis.mask[s,b]):
                uv[tuple( uvpix[b])] += (vis[s,b]             * weight[s,b])
                uv[tuple(-uvpix[b])] += (vis[s,b].conjugate() * weight[s,b])
                uv2[tuple( uvpix[b])] += weight[s,b]
                uv2[tuple(-uvpix[b])] += weight[s,b]
                uv3[tuple( uvpix[b])] += (np.abs(vis[s,b])    * weight[s,b])
                uv3[tuple(-uvpix[b])] += (np.abs(vis[s,b])    * weight[s,b])
                sumwt += weight[s,b]

    uv /= sumwt
    uv2 /= sumwt
    uv3 /= sumwt

    sky  = np.fft.ifft2(uv)
    sky2 = np.zeros((nmap, nmap), dtype=complex)
    asky  = np.fft.ifft2(uv3)
    asky2 = np.zeros((nmap, nmap), dtype=complex)
    #sky2[0:nmap/2,    0:nmap/2]    = sky[nfft-nmap/2:nfft, nfft-nmap/2:nfft]
    #sky2[0:nmap/2,    nmap/2:nmap] = sky[nfft-nmap/2:nfft, 0:nmap/2]
    #sky2[nmap/2:nmap, 0:nmap/2]    = sky[0:nmap/2,         nfft-nmap/2:nfft]
    #sky2[nmap/2:nmap, nmap/2:nmap] = sky[0:nmap/2,         0:nmap:2]
    for i2 in range(nmap):
        i = int(i2 + nfft - nmap/2)
        if (i >= nfft):
            i -= nfft
        for j2 in range(nmap):
            j = int(j2 + nfft - nmap/2)
            if (j >= nfft):
                j -= nfft
            sky2[i2, j2]  = sky[i, j]
            asky2[i2, j2] = asky[i, j]

    bm0   = np.fft.ifft2(uv2)
    sky2  /= bm0.max()
    asky2 /= bm0.max()
    bm    = bm0 / bm0.max()

    if (do_beam):
        bm2  = np.zeros((nmap*2, nmap*2), dtype=complex)
        for i2 in range(nmap*2):
            i = i2 + nfft - nmap
            if (i >= nfft):
                i -= nfft
            for j2 in range(nmap*2):
                j = j2 + nfft - nmap
                if (j >= nfft):
                    j -= nfft
                bm2[i2, j2] = bm[i, j]

    if (do_beam and do_aimg):
        return sky2.T.real, bm2.T.real, asky2.T.real
    elif (do_beam):
        return sky2.T.real, bm2.T.real
    elif (do_aimg):
        return sky2.T.real, asky2.T.real
    else:
        return sky2.T.real


def map2ax(img, extent=None, xlabel='X (pix)', ylabel='Y (pix)', cblabel='', **kwarg):
    '''
    return the fig, ax of an image
    assumes the img array starts from the lower-left corner of the image
    '''

    fig, ax = plt.subplots(1, 1, figsize=(10,8))

    if (extent != None):
        (xmin, xmax, ymin, ymax) = extent
        im = ax.imshow(img, origin='lower', extent=extent, **kwarg)
    else:
        im = ax.imshow(img, origin='lower', **kwarg)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cb = plt.colorbar(im, ax=ax)
    cb.set_label(cblabel)

    return fig, ax


def blsubplots(na=7, **kwargs):
    '''
    return a set of figure and subplots for baselines
    the subplots are organized as 2D array, indexed as [ai, aj-1]

    **kwargs will be passed to plt.subplots

    the default of kwargs if not overriden is:
        figsize = (10,8)
        sharex = True
        sharey = True

    '''

    if (not kwargs.get('figsize')):
        kwargs['figsize'] = (10,8)
    if (not kwargs.get('sharex')):
        kwargs['sharex'] = True
    if (not kwargs.get('sharey')):
        kwargs['sharey'] = True


    fig, sub = plt.subplots(na-1, na-1, **kwargs)
    for ii in range(na-1):
        for jj in range(na-1):
            if (ii > jj):
                sub[ii,jj].remove()
            else:
                corr = '%d%d' % (ii, jj+1)
                sub[ii,jj].set_title(corr)

    fig.tight_layout(rect=[0,0.03,1,0.95])

    return fig, sub


def azelpolar(**kwargs):
    '''
    return a polar plot (fig, ax) intended for (az,el)

    to plot, for example, use:
        ax.plot(az_rad, za_deg)
    az_rad is azimuth in radians
    za_deg is zenith angle in degrees (i.e. za_deg = 90.-el_deg)

    **kwargs will be passed to plt.figure

    the default of kwargs if not overriden is:
        figsize = (8,8)
    '''
    if (not kwargs.get('figsize')):
        kwargs['figsize'] = (8,8)

    fig = plt.figure(**kwargs)
    ax  = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2.)
    ax.set_theta_direction(-1)

    # elevation grids
    rlim = [0, 50.]
    egrids = np.arange(10., 51., 10.)
    elabel = ['80', '70', '60', '50', '40'] # corresponding labels
    # azimuth grids
    agrids = np.arange(0., 360., 30.)
    alabel = agrids.astype(int).astype(str)
    ax.set_rlim(rlim)
    #ax.set_rgrids([10, 20, 30, 40, 50], ['80', '70', '60', '50', '40'])
    ax.set_rgrids(egrids, elabel)
    ax.set_rlabel_position(0.)
    #ax.set_thetagrids(np.arange(0, 360,30))
    ax.set_thetagrids(agrids, alabel)
    ax.text(np.pi/4., rlim[1]*1.1, 'Azimuth', rotation=-45, horizontalalignment='right', fontsize=15)
    ax.text(0., rlim[1]/2., 'Elevation', rotation=90, horizontalalignment='right', fontsize=15)

    return fig, ax


def peakXY(imgs, ds=1., dmax=0.):
    '''
    given an image, find the maximum and position
    if dmax is given (>0), search the maximum around the image center,
    with a radius equal to dmax (units defined by ds)

    ds defines the pixel scale, defaults to 1., which means dmax is in pixels
    if ds will convert the pixels to arcmin, then dmax should be in arcmin
    '''
    if (imgs.ndim==2):
        (ny, nx) = imgs.shape
        nimg = 1
        imgs = imgs.reshape((nimg,ny,nx))
    elif (imgs.ndim==3):
        (nimg, ny, nx) = imgs.shape
        
    cx = int((nx - (nx%2)) / 2)
    cy = int((ny - (ny%2)) / 2)
    # grid coordinates in arcmin
    gx = ds * (np.arange(nx) - cx)
    gy = ds * (np.arange(nx) - cy)
    gX, gY = np.meshgrid(gx,gy)     # default indexing='xy'; put faster dimension in the first array
    darr = np.sqrt(gX**2 + gY**2)   # 2D array of pixel relative distance to central pixel

    peaks = np.zeros((nimg, 3))
    for i in range(nimg):
        if (not isinstance(imgs[i], np.ma.MaskedArray)):
            img = np.ma.array(imgs[i], mask=np.zeros((ny,nx), dtype=bool))      # a masked version
        else:
            img = imgs[i][:,:]                                                  # keep the mask if available

        if (dmax > 0.):
            dmask = darr > dmax
            img.mask += dmask

        pk = img.max()
        py, px = np.unravel_index(img.argmax(), img.shape)

        peaks[i,0] = pk
        peaks[i,1] = ds * (px - cx)
        peaks[i,2] = ds * (py - cy)

    return peaks


def loadLog(fconf):
    '''
    load the log file created by other script
    format in the log is:
        param1: string1
        param2: string2
        ...

    values are read as ASCII strings

    return a dict of the log file    
    '''
    if (not os.path.isfile(fconf)):
        print('log file not found: %s' % fconf)
        return None

    CONF = open(fconf, 'r')
    lines = CONF.readlines()
    CONF.close()
    attrs = {}
    for l2 in lines:
        line = l2.strip()
        if (line == ''):
            continue
        j = line.find(':')  # first occurence
        key = line[:j]
        val = line[j+2:]
        attrs[key] = val

    return attrs




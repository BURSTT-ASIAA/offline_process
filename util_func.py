#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


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


def toPower(x, mode='vol'):
    '''
    convert linear voltage to power in dB
    if mode=='vis', convert visibility to power in dB
    '''
    if (mode=='vol'):
        p = 20.*np.log10(x)
    elif (mode=='vis'):
        p = 10.*np.log10(x)
    return p



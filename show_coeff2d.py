#!/usr/bin/env python

from loadh5 import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

inp = sys.argv[0:]
pg  = inp.pop(0)

zlim = None
tlim = None

usage = '''
plot the correlation coefficients as a 2D map against freq and time
input is a .vish5 file from either single_dev_vis.py or two_dev_vis.py

note: the corr-coefficients are shown as log10(abs(x)), with max(x)=1

syntax:
    %s <.vish5(s)> [options]

    options are:
    --zlim ZMIN ZMAX    # the color scale range in log10(abs(x))
                        # e.g. ZMIN=-2.5, ZMAX=0.

    --tlim TMIN TMAX    # set the range in time (sec)

multiple file names can be supplied. each file will be plotted separately.

''' % (pg,)

if (len(inp) < 1):
    sys.exit(usage)

files = []
while (inp):
    k = inp.pop(0)
    if (k == '--zlim'):
        zmin = float(inp.pop(0))
        zmax = float(inp.pop(0))
        zlim = [zmin, zmax]
    elif (k == '--tlim'):
        tmin = float(inp.pop(0))
        tmax = float(inp.pop(0))
        tlim = [tmin, tmax]
    elif (k.startswith('-')):
        sys.exit('unknown option: %s' % k)
    else:
        files.append(k)

files.sort()
print(files)

for fvis in files:
    print('plotting ', fvis, ' ...')

    coeff  = getData(fvis, 'winCoeff')
    if (len(coeff.shape) == 2):     # backward compatibility for single_dev_vis.py
        nTime, nChan = coeff.shape
        coeff = coeff.reshape((nTime, 1, nChan))
    coeff2 = getData(fvis, 'winCoeffSub')
    freq = getData(fvis, 'freq')
    tsec = getData(fvis, 'winSec')

    if (tlim is None):
        tmin = tsec.min()
        tmax = tsec.max()
        tlim = [tmin, tmax]
    else:
        w = np.logical_and(tsec>=tlim[0], tsec<=tlim[1])
        tsec = tsec[w]
        coeff = coeff[w]
        coeff2 = coeff2[w]

    nTime, nBl, nChan = coeff2.shape

    X, Y = np.meshgrid(tsec, freq, indexing='ij')

    for b in range(nBl):
        png = '%s.2DCoeff.t%06d-%06d.bl%d.png' % (fvis, tlim[0], tlim[1], b)
        sptitle = '%s, tlim:[%d, %d]sec, baseline-%d' % (fvis, tlim[0], tlim[1], b)

        #fig, sub = plt.subplots(2,1,figsize=(10,9), sharex=True, sharey=True)
        fig = plt.figure(figsize=(16,9))
        gs = GridSpec(2,6, figure=fig)

        for pt in range(2):
            ## 2D map
            if (pt == 0):   ## raw coefficients
                z = coeff[:,b]
                title = 'raw x-coeff'
            elif (pt == 1): ## static-subtracted coefficients
                z = coeff2[:,b]
                title = 'static-subtracted x-coeff'
            Z = np.ma.log10(np.ma.abs(z))
            ax = fig.add_subplot(gs[pt,1:4])
            ax.set_title(title)
            if (zlim is None):
                zlim = [Z.min(), Z.max()]
            s = ax.pcolormesh(X, Y, Z, vmin=zlim[0], vmax=zlim[1])
            cb = plt.colorbar(s, ax=ax)
            cb.set_label('log10(abs(coeff))')
            ax.set_ylabel('freq (MHz)')
            if (pt==1):
                ax.set_xlabel('time (sec)')

            ## vs. freq
            ax = fig.add_subplot(gs[pt,0])
            #tmpr = np.ma.median(z.real, axis=0)
            #tmpi = np.ma.median(z.imag, axis=0)
            #med_z = tmpr + 1.j*tmpi
            #ax.plot(np.ma.log10(np.ma.abs(med_z)), freq)
            med_z = np.ma.median(Z, axis=0)
            ax.plot(med_z, freq)
            ax.set_xlim(zlim)
            ax.set_ylabel('freq (MHz)')
            if (pt==1):
                ax.set_xlabel('log10(abs(coeff))')
            if (pt==0):
                ax.set_title('median in time')

            ## vs. time
            ax = fig.add_subplot(gs[pt,4:6])
            #tmpr = np.ma.median(z.real, axis=1)
            #tmpi = np.ma.median(z.imag, axis=1)
            #med_z = tmpr + 1.j*tmpi
            #tmpr = np.ma.median(z.real)
            #tmpi = np.ma.median(z.imag)
            #med_z2 = tmpr + 1.j*tmpi
            #ax.plot(tsec, np.ma.log10(np.ma.abs(med_z)))
            med_z = np.ma.median(Z, axis=1)
            ax.plot(tsec, med_z)
            tmp = np.ma.median(Z)
            med_z2 = 10.**tmp
            ax.text(0.05, 0.90, 'med_coeff:%.3f (%.1fdB)'%(med_z2, tmp*10.), transform=ax.transAxes)
            ax.set_ylim(zlim)
            ax.set_ylabel('log10(abs(coeff))')
            if (pt==1):
                ax.set_xlabel('time (sec)')
            if (pt==0):
                ax.set_title('median in freq')


        fig.tight_layout(rect=[0,0.03,1,0.95])
        fig.suptitle(sptitle)
        fig.savefig(png)
        plt.close(fig)





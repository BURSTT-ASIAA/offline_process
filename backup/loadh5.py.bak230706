#!/usr/bin/env python
from analysisconf import *

import h5py
#import astropy.stats.biweight as bwt
from scipy import signal



def selectVis(mask, select, count=False):
    '''
    given a list of channel selections
    mask (True) the un-selected channels

    input mask is the visibility mask
    channel is axis=2

    select is a list of strings
    e.g. ['-510:-10', '377:379', 128]
    will select channels 10 to 510 in LSB
    channels 377 to 379 in USB
    and also channel 128 in USB
    '''
    if (count): # ignore input mask (which is required, so set to None)
        sh = (nsb,1,nch,1)
    else:
        sh = mask.shape

    in_mask = np.ones(sh, dtype=bool)   # init = all masked

    if (isinstance(select, (int,str))):
        if (isinstance(select, str)):
            sel = int(select)
        else:
            sel = select
        if (sel > 0):
            in_mask[1,:,sel,:] = False
        elif (sel < 0):
            in_mask[0,:,-sel,:] = False

    elif (isinstance(select, list)):
        for item in select:
            part = item.split(':')
            if (len(part) == 1):
                sel = int(part[0])
                if (sel > 0):
                    in_mask[1,:,sel,:] = False
                elif (sel < 0):
                    in_mask[0,:,-sel,:] = False
            elif (len(part) == 2):
                sel = [int(part[0]), int(part[1])]
                sel.sort()
                if (sel[0]>0 and sel[1]>0):
                    in_mask[1,:,sel[0]:sel[1]+1,:] = False
                elif (sel[0]<0 and sel[1]<0):
                    in_mask[0,:,-sel[1]:-sel[0]+1,:] = False
                else:   # sel[0]<=0 and sel[1]>=0
                    in_mask[1,:,:sel[1]+1,:] = False
                    in_mask[0,:,:-sel[0]+1,:] = False

    if (count):
        n_sel = np.count_nonzero(~in_mask)
        return n_sel
    else:
        return in_mask


def bldict():
    '''
    return a dictionary
    bl['ij'] = b
    '''
    bl = {}
    b = -1
    for ai in range(na-1):
        for aj in range(ai+1, na):
            b += 1
            corr = '%d%d' % (ai, aj)
            bl[corr] = b
    return bl


def getAttrs(fname, dest='/'):
    '''
    retrieve the top-level attributes from the correlator data
    (representative file: .lsb.auto.h5)
    input:
        specific filename of the h5 file

        dest is a string for the destination group or dataset
        default is to get the root level attributes

    return: dict
    '''

    if (not os.path.isfile(fname)):
        print('error finding the source h5 file:', fname)
        sys.exit()

    f = h5py.File(fname, 'r')
    try:
        g = f.get(dest)
    except:
        #sys.exit('error accessing file: %s' % fname)
        print('destination not found: %s' % dest)
        return None

    attrs = {}
    for k in list(g.attrs.keys()):
        attrs[k] = g.attrs.get(k)
    f.close()

    return attrs


def putAttrs(fname, attrs, dest='/'):
    '''
    put the attrs (dict) as attributes in the oneh5 file
    dest is a string for the destination group or dataset
    default is to put in root level attributes
    '''
    try:
        f = h5py.File(fname, 'a')
        g = f.get(dest)

        for k in list(attrs.keys()):
            g.attrs[k] = attrs[k]
        f.close()
    except:
        print('error accessing oneh5 file:', fname)
        sys.exit()

    return 1


def getData(fname, dname):
    '''
    return the <dname> dataset array
    if <dname>.mask exist, the data is load as masked array
    '''
    #print('debug: in getData')
    try:
        f = h5py.File(fname, 'r')
    except:
        sys.exit('error opening file: %s' % fname)

    if (f.get(dname)):
        d = f[dname][()]
    else:
        #sys.exit('%s not found in %s' % (dname, fname))
        print('%s not found in %s' % (dname, fname))
        print('... return None')
        return None

    mname = '%s.mask' % dname
    if (f.get(mname)):
        m = f[mname][()]
        arr = np.ma.array(d, mask=m)
    else:
        arr = d

    f.close()

    return arr    


def adoneh5(h5name, darray, dname, over=True):
    '''
    set over = False if you do not want to overwrite existing dataset
    '''
    #print datetime.now().isoformat()

    with h5py.File(h5name, 'a') as f:
        print('... ', dname)
        if (f.get(dname)):
            if (over):
                del f[dname]
            else:
                return      # do nothing if over==False
        f.create_dataset(dname, data = darray)

        if (isinstance(darray, np.ma.MaskedArray)):
            mname = '%s.mask' % dname
            if (f.get(mname)):
                del f[mname]
            f.create_dataset(mname, data = darray.mask)

    #print datetime.now().isoformat()



def vecSmooth(cspec, kern='g', gsigma=6., twidth=32, axis=2, mode=1):
    '''
    input:
        cspec       ndarray or a masked array
                    nominally, this is the complex visibility

        kern        'g' for Gaussian
                    't' for tophat

        gsigma      sigma of the Gaussian smoothing window

        twidth      width of the tophat

        axis        apply smoothing to which axis

        mode        0 for signal.convolve --> fast with n-D array, but fail for maskedArray
                    1 for np.convolve --> works with maskedArray, but 1-D at a time
    '''
    if (kern == 'g'):
        glen    = int(gsigma * 6.)
        win     = signal.gaussian(glen, gsigma) # 1-D array, length = glen
    elif (kern == 't'):
        glen    = int(twidth)
        win     = np.ones(glen)

    if (mode == 0):
        sh0         = np.ones_like(cspec.shape)
        try:
            sh0[axis] = glen
        except:
            print('error smoothing axis', axis)
            sys.exit()
        
        win     = win.reshape(tuple(sh0))
        #win    = win.reshape((1,1,glen))

        # spectrally smoothed quantities
        sspec   = signal.convolve(cspec, win/win.sum(), 'same')
        if (isinstance(cspec, np.ma.MaskedArray)):
            sspec   = np.ma.array(sspec, mask=cspec.mask)

    elif (mode == 1):
        if (axis !=2):
            print('vecSmooth mode=1 can only process array in the format spec[nsb, nb, nch]')
            sys.exit()
        sspec = cspec.copy()
        for sb in range(nsb):
            for b in range(nb):
                sspec[sb,b] = np.ma.convolve(cspec[sb,b], win/win.sum(), 'same')

    return sspec, win



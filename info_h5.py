#!/usr/bin/env python
#import numpy as np
#import h5py
#import sys, os.path
from loadh5 import *


def showAttrs(g, lv0):
    pre = '\t'*lv0
    akeys = list(g.attrs.keys())
    if (len(akeys) > 0):
        akeys.sort()
        print(pre, '<< Attrs >>')
        for k in akeys:
            print(pre, str(k), '=', g.attrs[k])
        print('')

def showGroup(h5o, name, lv0):
    # h5o = H5 obj (e.g. o1 = h5py.File(), o2 = o1.get(name))


    pre0 = '\t'*lv0
    print(pre0, name)
    print(pre0, '-----')

    lv = lv0 + 1
    pre = '\t'*lv

    g = h5o.get(name)
    #-- attrs
    showAttrs(g, lv)

    #-- all items
    items = list(g.items())
    items.sort()
    datas  = {}
    groups = {}
    for it in items:
        #print 'debug:'
        #print it
        if isinstance(it[1], h5py.Dataset):
            datas[it[0]] = it[1]
        if isinstance(it[1], h5py.Group):
            groups[it[0]] = it[1]

    #-- datasets
    if (len(datas) > 0):
        print(pre, '<< Datasets >>')
        dkeys = list(datas.keys())
        dkeys.sort()
        for k in dkeys:
            print(pre, str(k), ':', datas[k])
            d = g.get(k)
            showAttrs(d, lv+1)
        print('')

    #-- groups
    if (len(groups) > 0):
        print(pre, '<< Groups >>')
        gkeys = list(groups.keys())
        gkeys.sort()
        for k in gkeys:
            #print pre, str(k), ':', groups[k]
            showGroup(g, k, lv)
        print('')
        
    return None



inp     = sys.argv[0:]
pg      = inp.pop(0)
showtgt = False
files   = []


usage   = '''
usage %s <oneh5> [options]

    options are

    --target        display the target name of each observing unit
                    and their unit length (in sec)

''' % pg

if (len(inp) < 1):
    print(usage)
    sys.exit()

while (inp):
    k = inp.pop(0)
    if (k == '--target'):
        showtgt = True
    elif (k.startswith('-')):
        print('error recognizing argument:', k)
        sys.exit()
    else:
        #foneh5 = k
        files.append(k)


for foneh5 in files:
    try:
        f = h5py.File(foneh5, 'r')
    except IOError:
        print('error opening ', foneh5)
        sys.exit()


    print('=====')
    print('INFO for ', foneh5)
    print('=====')
    print('')

    if (not showtgt):
        showGroup(f, '/', 0)

    if (showtgt):
        showAttrs(f, 0)

        if (f.get('target')):
            target = getTarget(foneh5)

            pnt, pnthdr = getPointing(foneh5)
            length = pnt[3] - pnt[2]
            npat   = len(length)
            epoch  = (pnt[2] + pnt[3]) / 2.
            target = target.reshape(-1)

            if (foneh5.endswith('.mrgh5')):
                obsdate = getData(foneh5, 'obsdate')
            elif (foneh5.endswith('.avgh5')):
                fname = os.path.basename(foneh5)
                part = fname.split('.')
                ymd = datetime.strptime(part[0], '%Y_%b_%d_%H_%M_%S').strftime('%y%m%d')
                obsdate = np.empty(npat, dtype='S6')
                obsdate[:] = ymd


            print('<<Targets>>')
            if (len(pnthdr) > 4):
                print('#Unit  Target  T_int(sec)  epoch              Date     UT(sec) RA(hr)    DEC(deg)  Az(deg)  El(deg)  HP(deg) SP(deg) HA(deg)')
                for i in range(npat):
                    print('%03d    %-10s  % 5.0f   %.3f' % (i, target[i], length[i], epoch[i]), '    %6s' % obsdate[i], '   %.0f  %.5f  %.4f' % tuple(pnt[4:7,i]), '    %5.1f'*3 % tuple(pnt[7:10,i]), '  %.1f  % 6.1f' % (pnt[11,i], pnt[13,i]))
            else:
                print('#Unit  Target  T_int(sec)  epoch              Date')
                for i in range(npat):
                    print('%03d    %-10s  % 5.0f   %.3f' % (i, target[i], length[i], epoch[i]), '    %6s' % obsdate[i])
            print('')

        else:
            print('no target info in this file.')

    f.close()


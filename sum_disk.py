#!/usr/bin/env python

import sys, os.path
import numpy as np
from glob import glob
from subprocess import call

#idir = '/burstt1/disk3/data'

inp = sys.argv[0:]
pg  = inp.pop(0)
typ = 'bin' # bin or inten
patt = '*'

usage = '''
list data size for each FPGA/date
syntax:
    %s <dir> [options]

options are
    --inten     summarize intensity instead of baseband data
    --fpga      specifically finding calibration only
    --patt PATT additional pattern matching (e.g. for a specific month)

''' % (pg,)

if (len(inp)<1):
    sys.exit(usage)

inp_dirs = []
while(inp):
    k = inp.pop(0)
    if (k == '--inten'):
        typ = 'inten'
    elif (k == '--fpga'):
        typ = 'fpga'
    elif (k == '--patt'):
        patt = inp.pop(0)
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        idir = k
        inp_dirs.append(idir)


for idir in inp_dirs:
    print('=========')
    print(idir, patt)
    print('=========')

    if (typ == 'bin'):
        ifiles = glob('%s/%s.bin'%(idir,patt))
    elif (typ == 'fpga'):
        ifiles = glob('%s/fpga?.%s.bin'%(idir,patt))
    elif (typ == 'inten'):
        ifiles = glob('%s/intensity_%s'%(idir,patt))
    nTot = len(ifiles)

    dates = []
    names = []
    for i in range(nTot):
        f = ifiles[i]
        if (typ == 'bin' or typ == 'fpga'):
            tmp = f.split('.')
            if (len(tmp[1])==10):
                tmp2 = '.'.join([tmp[0], '2023'+tmp[1][:4]])
                tmp3 = '.'.join([tmp[0], tmp[1][:4]])
            elif (len(tmp[1])==14):
                tmp2 = '.'.join([tmp[0], tmp[1][:8]])
                tmp3 = '.'.join([tmp[0], tmp[1][:8]])
            elif ('_' in tmp[1]):
                ttmp = tmp[1].split('_')
                tmp2 = '.'.join([tmp[0], ttmp[0]])
                tmp3 = '.'.join([tmp[0], ttmp[0]])
            #print(i, f, tmp2)
            #sys.exit()
        elif (typ == 'inten'):
            dir0 = os.path.dirname(f)
            tmp = f.split('_')  # e.g.: intensity_ring1_sum400_260204_212722_f0
            tmp2 = '%s/'%dir0 + '_'.join(tmp[1:4])
            tmp3 = '_'.join(tmp[1:4])

        dates.append(tmp2)
        names.append(tmp3)

    uniq = np.unique(dates)
    #print(uniq)
    nUni = len(uniq)
    uniq2 = np.unique(names)

    for j in range(nUni):
        d = uniq[j]
        nm = uniq2[j]
        if (typ == 'bin' or typ == 'fpga'):
            jfiles = glob('%s*.bin'%nm)
        elif (typ == 'inten'):
            jfiles = glob('%s/*%s*'%(idir,nm))
        size  = 0
        sizet = 0
        for f in jfiles:
            if ('trig' in f):
                sizet += os.path.getsize(f)
            else:
                size += os.path.getsize(f)

        if (sizet > 0):
            print(d, '% 9.2f GiB / triggered'%(sizet/1e9))
        if (size > 0):
            print(d, '% 9.2f GiB / scheduled'%(size/1e9))


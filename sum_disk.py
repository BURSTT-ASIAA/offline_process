#!/usr/bin/env python

import sys, os.path
import numpy as np
from glob import glob
from subprocess import call

#idir = '/burstt1/disk3/data'

inp = sys.argv[0:]
pg  = inp.pop(0)
usage = '''
list data size for each FPGA/date
syntax:
    %s <dir>

''' % (pg,)

if (len(inp)<1):
    sys.exit(usage)

while(inp):
    k = inp.pop(0)
    idir = k


ifiles = glob('%s/*.bin'%idir)
nTot = len(ifiles)

dates = []
names = []
for i in range(nTot):
    f = ifiles[i]
    tmp = f.split('.')
    if (len(tmp[1])==10):
        tmp2 = '.'.join([tmp[0], '2023'+tmp[1][:4]])
        tmp3 = '.'.join([tmp[0], tmp[1][:4]])
    elif (len(tmp[1])==14):
        tmp2 = '.'.join([tmp[0], tmp[1][:8]])
        tmp3 = '.'.join([tmp[0], tmp[1][:8]])
    #print(i, f, tmp2)
    #sys.exit()

    dates.append(tmp2)
    names.append(tmp3)

uniq = np.unique(dates)
#print(uniq)
nUni = len(uniq)
uniq2 = np.unique(names)

for j in range(nUni):
    d = uniq[j]
    nm = uniq2[j]
    jfiles = glob('%s*.bin'%nm)
    size = 0
    for f in jfiles:
        size += os.path.getsize(f)
    print(d, '% 9.2f GiB'%(size/1e9))

#!/usr/bin/env python

import sys, os.path
import numpy as np

dest = '/mnt/bonsai/matrix'

inp = sys.argv[0:]
pg  = inp.pop(0)

usage = '''
copy the beamform matrix binary file to the system location
/mnt/bonsai/matrix

syntax:
    %s <bin_file>
''' % (pg,)

if (len(inp) < 1):
    sys.exit(usage)

while (inp):
    k = inp.pop(0)
    src = k

print('copying', src, 'to', dest)

buf = np.memmap(dest, dtype='i2', shape=(2**19))
buf[:] = np.fromfile(src, dtype='i2')

print('done')


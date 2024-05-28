#!/usr/bin/env python

import sys, os.path, time
from glob import glob
from subprocess import call
from packet_func import *

inp = sys.argv[0:]
pg  = inp.pop(0)

rmeta = 0

usage = '''
display the file meta data of saved files
needs to skip 64 bytes if used on ring buffers

syntax:
    %s <files> [options]

options are:
    --meta META     # ring buffer meta bytes to skip
                    # default: %d bytes

''' % (pg, rmeta)

if (len(inp) < 1):
    sys.exit(usage)

files = []
while (inp):
    k = inp.pop(0)
    if (k == '--meta'):
        rmeta = int(inp.pop(0))
    else:
        files.append(k)



for f in files:
    print('reading:', f)
    fh = open(f, 'rb')

    mdict = metaRead(fh)
    #print(mdict)
    for k in mdict.keys():
        print('  ', k, ':', mdict[k])



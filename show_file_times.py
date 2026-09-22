#!/usr/bin/env python

import sys, os.path, time
from packet_func import *
from glob import glob
from datetime import datetime, timedelta
from astropy.time import Time


inp = sys.argv[0:]
pg = inp.pop(0)

meta  = 64
hdlen = 64
pack_size = 8256
pack_num  = None
pack_off  = 1
bf    = 256
prate = 400e6/1024*2

usage = '''
show begin and end times of a file
syntax:
    %s <bin_files> [options]

options are
--ps SIZE       packet size in bytes (header+payload)
--pn NUM        number of packets per block
                (used to estimate last packet location)
                typical: 819200 for bf256
                         204800 for bf64
                         102400 for bf16
--po OFF        number of packet to offset from the end
--bf MODE       <256|64|16>
                (sets the default packet number and packet_offset)
--meta META     length of meta header in bytes

'''%(pg,)

if (len(inp)<1):
    sys.exit(usage)

files = []
while (inp):
    k = inp.pop(0)
    if (k == '--ps'):
        pack_size = int(inp.pop(0))
    elif (k == '--pn'):
        pack_num = int(inp.pop(0))
    elif (k == '--po'):
        pack_off = int(inp.pop(0))
    elif (k == '--bf'):
        bf = int(inp.pop(0))
    elif (k == '--meta'):
        meta = int(inp.pop(0))
    elif (k.startswith('-')):
        sys.exit(f'unknown option: {k}')
    else:
        files.append(k)

if (pack_num is None):
    if (bf == 256):
        pack_num = 819200
    elif (bf == 64):
        pack_num = 204800
    elif (bf == 16):
        pack_num = 102400

for fname in files:
    print(fname, 'times:')
    sz = os.path.getsize(fname)
    npack = (sz - meta)//8256
    #print(npack)
    nblock = npack//819200
    npack = nblock * 819200
    #print(nblock, npack)

    fh = open(fname, 'rb')
    buf = fh.read(meta) # meta
    # pack0
    hd = fh.read(hdlen)
    tmp = decHeader2(hd, ip=True)
    #print(tmp)
    ep0 = tmp[2]
    pcnt0 = tmp[0]
    ord0 = tmp[4]
    # last pack
    fh.seek(meta + 8256*(npack-pack_off))
    hd = fh.read(hdlen)
    tmp = decHeader2(hd, ip=True)
    #print(tmp)
    pcnt1 = tmp[0]
    ord1 = tmp[4]

    #print(prate)
    ep_begin = ep0 + 2 + (pcnt0-ord0)/prate
    ep_end = ep0 + 2 + (pcnt1-ord1)/prate
    print(Time([ep_begin, ep_end], format='unix').iso)



#!/usr/bin/env python

import sys, os.path, time
from glob import glob
from subprocess import call
from astropy.time import Time

from loadh5 import *
from packet_func import *


hdlen = 64
nChan = 1024
blocklen = 128000       # packets per block
ppf = 2                 # packets per frame
fpb = blocklen//ppf     # frames per block
bpb = 1. * fpb * nChan  # bytes per block
bphb = bpb + hdlen      # bytes per (block + header)

idir = 'data_240319_sun'
files = glob('%s/fpga*'%idir)
files.sort()
ofile = '%s.inth5'%idir

## testing 
#files = files[:4]

nFile = len(files)
print(nFile, files)

intr = []
pcnt = []
epoc = []
for i in range(nFile):
    fname = files[i]
    print('file:', fname)
    t1 = time.time()
    sp, pc, ep = loadBeam(fname, nBlock=1)    # load all blocks
    t2 = time.time()
    print('... loaded in', t2-t1, 'sec')
    raw_int = (np.abs(sp)**2).mean(axis=1)
    intr.append(raw_int)
    pcnt.append(pc)
    epoc.append(ep)

if (nFile>1):
    intr = np.concatenate(intr, axis=0)
    pcnt = np.concatenate(pcnt)
    epoc = np.concatenate(epoc)
else:
    intr = intr[0]
    pcnt = pcnt[0]
    epoc = epoc[0]

nWin, nChan = intr.shape
intr = intr.reshape((nWin,1,1,nChan))

intn = intr / intr.mean(axis=0)
tsec = epoc - epoc[0]

attrs = {}
attrs['unix_utc_open'] = epoc[0]
putAttrs(ofile, attrs)

adoneh5(ofile, intr, 'intensity')
adoneh5(ofile, intn, 'norm.intensity')
adoneh5(ofile, tsec, 'winSec')

freq = np.linspace(400., 800., nChan, endpoint=False)
adoneh5(ofile, freq, 'freq')


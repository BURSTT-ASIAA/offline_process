#!/usr/bin/env python

import sys, os.path
import numpy as np
from subprocess import call, run
from glob import glob
from datetime import datetime

from fpga_alias import *    # site_ips, fpga_alias, fpga_id

#site = 'LAN'
inp = sys.argv[0:]
pg  = inp.pop(0)
site = None

cals = ['antCal.npy', 'antCal2.npy', 'antCals.npz']

usage = '''
copy 1st cals into a folder for rfsoc
syntax:
    %s <site>
''' % (pg,)

if (len(inp)<1):
    sys.exit(usage)

while (inp):
    k = inp.pop(0)
    site = k.upper()


ips = site_ips[site]
mrow = len(ips) # max rows defined in fpga_alias

if (site == 'FUS'):
    dirs = ['b16', 'b15', 'b12', 'b11']
elif (site == 'LTN'):
    dirs = ['b5']
elif (site == 'GRN'):
    dirs = ['b6']
elif (site == 'KMN'):
    dirs = ['b1']
elif (site == 'LAN'):
    dirs = ['b2']

ndirs = len(dirs)
for di, d in enumerate(dirs):
    if (not os.path.isdir(d)):
        d = '.' # override for site with a single server
    print('### dir:', d)
    files = glob(f'{d}/fpga*.eigen.h5')
    files.sort()
    nfile = len(files)
    if (nfile == 0):
        sys.exit('no files found.')

    for f in files:
        print('## file:', f)
        fb = os.path.basename(f)
        tmp = fb.split('.')
        #print(tmp)
        rr = int(tmp[0][-1:]) + 4*di
        fname = f'{site}_row{rr+1:02d}'
        fid = fpga_id[fname]
        #print(fname, fid)

        ymd = datetime.strptime(tmp[1], '%Y%m%d_%H%M%SZ').strftime('%y%m%d')
        #print(ymd)

        odir = f'{site}_eigen_{ymd}'
        if not os.path.isdir(odir):
            call(f'mkdir {odir}', shell=True)

        idir = f'{f}.check'
        for c in cals:
            fc = f'{fb}.{c}'
            ifile = f'{idir}/{fc}'
            ofile = f'{odir}/{fname}.fpga{fid}.{fc}'
            #print(ifile, '->', ofile)
            cmd = f'cp {ifile} {ofile}'
            print(cmd)
            call(cmd, shell=True)


#!/usr/bin/env python

import sys, os.path
import re
import shutil
import numpy as np
from subprocess import call, run
from glob import glob
from datetime import datetime

from fpga_alias import *    # site_ips, fpga_alias, fpga_id

#site = 'LAN'
inp = sys.argv[0:]
pg  = inp.pop(0)
site = None
mode = 'eigen'
recal_source = None

cals = ['antCal.npy', 'antCal2.npy', 'antCals.npz']

usage = '''
copy 1st cals into a folder for rfsoc
syntax:
    %s <site> [--mode eigen|recal|1stcal] [--source check-folder]

    --mode eigen   copy per-FPGA eigenmode calibration files (default)
    --mode recal   copy 16-row recal256 calibration files
    --mode 1stcal  copy row calibration files from combined first calibration
    --source DIR   source .check directory (default: newest matching folder)
''' % (pg,)

if (len(inp)<1):
    sys.exit(usage)

while (inp):
    k = inp.pop(0)
    if (k == '--mode'):
        mode = inp.pop(0).lower()
    elif (k == '--source'):
        recal_source = inp.pop(0)
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    elif (site is None):
        site = k.upper()
    else:
        sys.exit('unexpected argument: %s'%k)


if (site is None):
    sys.exit(usage)
if (mode not in ('eigen', 'recal', '1stcal')):
    sys.exit('unknown mode: %s (expected eigen, recal, or 1stcal)'%mode)
if (site not in site_ips):
    sys.exit('unknown site: %s'%site)

ips = site_ips[site]
mrow = len(ips) # max rows defined in fpga_alias

if (mode in ('recal', '1stcal')):
    source_prefix = 'recal256' if mode == 'recal' else '1stcal'
    source_pattern = r'%s_(\d{8})_(\d{6})Z\.eigen\.h5\.check'%source_prefix
    if (recal_source is None):
        current_dir = os.path.abspath('.')
        current_name = os.path.basename(current_dir)
        if (re.fullmatch(source_pattern, current_name)):
            recal_source = current_dir
        else:
            candidates = glob('%s_*.eigen.h5.check'%source_prefix)
            candidates = [d for d in candidates if os.path.isdir(d)]
            if not candidates:
                sys.exit('no %s_*.eigen.h5.check directory found'%source_prefix)
            recal_source = max(candidates)
    recal_source = os.path.abspath(recal_source)
    source_name = os.path.basename(recal_source.rstrip(os.sep))
    match = re.fullmatch(source_pattern, source_name)
    if (not os.path.isdir(recal_source) or match is None):
        sys.exit('invalid %s source directory: %s'%(mode,recal_source))

    if (mode == 'recal'):
        if (mrow < 16):
            sys.exit('site %s has %d FPGA rows; recal mode requires 16'%(site,mrow))
        copy_rows = list(range(16))
    else:
        copy_rows = sorted(int(os.path.basename(rowdir)[3:]) for rowdir in glob(os.path.join(recal_source,'row[0-9][0-9]')) if os.path.isdir(rowdir))
        if not copy_rows:
            sys.exit('no rowNN directories found in %s'%recal_source)
        invalid_rows = [row for row in copy_rows if row >= mrow]
        if invalid_rows:
            sys.exit('site %s has no FPGA mapping for physical rows: %s'%(site,invalid_rows))

    ymd = datetime.strptime(match.group(1), '%Y%m%d').strftime('%y%m%d')
    odir = f'{site}_{mode}_{ymd}'
    os.makedirs(odir, exist_ok=True)
    eigen_basename = source_name[:-6]
    copies = []
    for rr in copy_rows:
        rowdir = os.path.join(recal_source, 'row%02d'%rr)
        fname = fpga_alias[ips[rr]]
        fid = fpga_id[fname]
        for c in cals:
            ifile = os.path.join(rowdir, '%s.%s'%(eigen_basename, c))
            ofile = os.path.join(odir, '%s.fpga%s.%s'%(fname, fid, os.path.basename(ifile)))
            copies.append((ifile, ofile))

    missing = [ifile for ifile, _ in copies if not os.path.isfile(ifile)]
    if missing:
        sys.exit('missing recal calibration file(s):\n%s'%'\n'.join(missing))
    for ifile, ofile in copies:
        shutil.copy2(ifile, ofile)
        print('cp %s %s'%(ifile, ofile))
    sys.exit(0)

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


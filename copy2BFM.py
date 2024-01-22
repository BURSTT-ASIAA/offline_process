#!/usr/bin/env python

import sys, os.path, time
from subprocess import call

inp = sys.argv[0:]
pg  = inp.pop(0)

home = os.getenv('HOME')
odir = '%s/rfsoc/python_zcu216/BFM'%home
idir = ''

usage = '''
copy the .npy files from selected directory to the BFM folder
(%s)
also copies the info.txt file for reference

syntax:
    %s <input_dir>

''' % (odir, pg)

if (len(inp)<1):
    sys.exit(usage)


idir = inp.pop(0)
if (not os.path.isdir(idir)):
    sys.exit('no such dir: %s'%idir)
if (not os.path.isdir(odir)):
    sys.exit('error finding dest folder: %s'%odir)

cmd = 'cp %s/*.npy %s'%(idir, odir)
call(cmd, shell=True)

cmd = 'cp %s/*.info.txt %s'%(idir, odir)
call(cmd, shell=True)


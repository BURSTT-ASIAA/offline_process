#!/usr/bin/env python3
import numpy as np
from ctypes import *
import time
import sys

N_BLK = 60
INT_SIZE = 100
N_PACKET = 128000
ARCH_BLKS = 400
FPGA_DIRS = ['/mnt/ens2f0np0', '/mnt/ens2f1np1', '/mnt/ens4f0np0', '/mnt/ens4f1np1']
OUT_PREFIX = '/burstt1/disk5/intensity'
FPGA_BLK_SZ = (8192+64)*N_PACKET
FPGA_DATA_OFFSET = 128

inp = sys.argv[0:]
pg  = inp.pop(0)
verbose = 0

usage = '''
this script saves the intensity ring buffer beam.bin into the disk

syntax:
    %s FPGA_ID DUR_sec [options]

    [mandatory arguments]
    FPGA_ID::   0,1,2,3 select the FPGA to use
    DUR_sec::   the duration to save

    [options]
    -v              output diagnostic info
    --sum INT_SIZE  change the integration number (100 or 400)

''' % (pg,)

if (len(inp)<2):
    sys.exit(usage)

fpga_id = int(inp.pop(0))
dur_sec = float(inp.pop(0))
while (inp):
    k = inp.pop(0)
    if (k == '-v'):
        verbose = 1
    elif (k == '--sum'):
        INT_SIZE = int(inp.pop(0))
        if (INT_SIZE==100 or INT_SIZE==400):
            if (verbose):
                print('integration:', INT_SIZE)
        else:
            sys.exit('invalid integration number:', INT_SIZE)

BLK_SIZE = 512*16*(N_PACKET//INT_SIZE)
IN_PREFIX = FPGA_DIRS[fpga_id]
FPGA_FILE = f'{IN_PREFIX}/fpga.bin'
INT_FILE = f'{IN_PREFIX}/intensity.bin'

fpga = np.memmap(FPGA_FILE, dtype='byte')
blk_ptr = cast(c_void_p(fpga.ctypes.data + 16), POINTER(c_ushort))
intensity = np.memmap(INT_FILE, dtype='u2')

last_blk = blk_ptr[0]
fn = None
nblk = 0

t0 = time.time()
t1 = time.time()
while (t1-t0 < dur_sec):
    tstr = time.strftime('%y%m%d_%H%M%S')
    if fn is None:
        #fn = f'{OUT_PREFIX}/intensity_{int(time.time())}'
        fn = f'{OUT_PREFIX}/intensity_fpga{fpga_id}_{tstr}'
        fd = open(fn, 'wb')

    while last_blk == blk_ptr[0]:
        time.sleep(0.01)
    last_blk = (last_blk + 1) % N_BLK

    p = FPGA_BLK_SZ * last_blk + FPGA_DATA_OFFSET
    header = fpga[p:p+64]
    header.tofile(fd)

    data = intensity[BLK_SIZE*last_blk:BLK_SIZE*(last_blk+1)]
    data.tofile(fd)

    nblk += 1
    if (verbose>0):
        print(f'Block #{last_blk} written to disk {disk}, count: {nblk}')
    if nblk >= ARCH_BLKS:
        nblk = 0
        fd.close()
        fn = None

    t1 = time.time()

print('fpga%d intensity_archive done'%fpga_id, time.asctime())


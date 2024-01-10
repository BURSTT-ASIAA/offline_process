#!/usr/bin/env python
import numpy as np
from ctypes import *
import time
import sys

N_BLK = 60
N_PACKET = 128000
ARCH_BLKS = 400
BLK_SIZE = 512*N_PACKET
FPGA_DIRS = ['/mnt/ens2f0np0', '/mnt/ens2f1np1', '/mnt/ens4f0np0', '/mnt/ens4f1np1']
OUT_PREFIX = ['/burstt1/disk1/beam', '/burstt1/disk2/beam', '/burstt1/disk3/beam', '/burstt1/disk4/beam']
OUT_PREFIX1 = '/sdisk1/beam'
FPGA_BLK_SZ = (8192+64)*N_PACKET
FPGA_DATA_OFFSET = 128

inp = sys.argv[0:]
pg  = inp.pop(0)
verbose = 0

usage = '''
new bursttd (rudpd) can extract a beam of data from selected FPGA and
put it in the ring buffer /mnt/(NIC)/beam.bin

this script saves the beam.bin into the disk

syntax:
    %s FPGA_ID N_disk DUR_sec [options]

    [mandatory arguments]
    FPGA_ID::   0,1,2,3 select the FPGA to use
    N_disk::    1 or 4; use 1 SSD or 4 HDD to save the data (needs 400MB/s)
    DUR_sec::   the duration to save

    [options]
    -v          output diagnostic info

''' % (pg,)


if (len(inp)<3):
    sys.exit(usage)

fpga_id = int(inp.pop(0))
ndisk = int(inp.pop(0))
dur_sec = float(inp.pop(0))
while (inp):
    k = inp.pop(0)
    if (k == '-v'):
        verbose = 1


IN_PREFIX = FPGA_DIRS[fpga_id]
FPGA_FILE = f'{IN_PREFIX}/fpga.bin'
BEAM_FILE = f'{IN_PREFIX}/beam.bin'

fpga = np.memmap(FPGA_FILE, dtype='byte')
blk_ptr = cast(c_void_p(fpga.ctypes.data + 16), POINTER(c_ushort))
beam = np.memmap(BEAM_FILE, dtype='byte')

last_blk = blk_ptr[0]
fds = []
nblk = 0
disk = 0

t0 = time.time()
t1 = time.time()
while (t1-t0 < dur_sec):
    if len(fds) == 0:
        tstr = time.strftime('%y%m%d_%H%M%S')
        if (ndisk == 1):
            n = 0
            fn = f'{OUT_PREFIX1}/fpga{fpga_id}_{tstr}_f{n}'
            fd = open(fn, 'wb')
            fds.append(fd)
        else:
            for n in range(ndisk):
                fn = f'{OUT_PREFIX[n]}/fpga{fpga_id}_{tstr}_f{n}'
                fd = open(fn, 'wb')
                fds.append(fd)

    while last_blk == blk_ptr[0]:
        time.sleep(0.01)
    last_blk = (last_blk + 1) % N_BLK

    p = FPGA_BLK_SZ * last_blk + FPGA_DATA_OFFSET
    header = fpga[p:p+64]
    header.tofile(fds[disk])

    data = beam[BLK_SIZE*last_blk:BLK_SIZE*(last_blk+1)]
    data.tofile(fds[disk])
    disk = (disk + 1) % ndisk

    nblk += 1
    if (verbose>0):
        print(f'Block #{last_blk} written to disk {disk}, count: {nblk}')
    if nblk >= ARCH_BLKS * ndisk:
        nblk = 0
        disk = 0
        for fd in fds:
            fd.close()
        fds = []

    t1 = time.time()

print('fpga%d beam_archive done'%fpga_id, time.asctime())


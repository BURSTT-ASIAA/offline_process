#!/usr/bin/env python

from packet_func import *


inp = sys.argv[0:]
pg  = inp.pop(0)

nPack   = 4
packSize = 8256
headSize = 64
meta    = 0
ip      = False

usage = '''
display header of the first few packets from the binary file
syntax:
    %s <file(s)> [options]

options are:
    -n nPack        # how many packets to display (%d)
    --meta bytes    # ring buffer or file metadata length in bytes
    --ip            # to also display the local and dest IPs

''' % (pg, nPack)

if (len(inp)<1):
    sys.exit(usage)

files = []
while(inp):
    k = inp.pop(0)
    if (k == '-n'):
        nPack = int(inp.pop(0))
    elif (k == '--meta'):
        meta = int(inp.pop(0))
    elif (k == '--ip'):
        ip = True
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        files.append(k)

nFile = len(files)

#bfile = '/burstt1/disk1/data/fpga0.0913235115.bin'
for j in range(nFile):
    bfile = files[j]
    print('file:', bfile)
    if (bfile.startswith('/mnt')):  # ring buffer
        off0 = 128
    else:                           # saved file
        off0 = 64

    fh = open(bfile, 'rb')
    for i in range(nPack):
        off = packSize * i + meta
        fh.seek(off)
        hd = fh.read(headSize)
        tmp = decHeader2(hd, ip=ip, verbose=False)
        print(tmp)
    fh.close()



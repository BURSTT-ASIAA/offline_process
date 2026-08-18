#!/usr/bin/env python

from packet_func import *
from astropy.time import Time


inp = sys.argv[0:]
pg  = inp.pop(0)

nPack   = 4
packSize = 8256
headSize = 64
meta    = 0
ip      = False
show_ts = False
prate   = 400e6/1024*2  # for 16bit, bf16, bf64, bf256
is_bf512= False         # need to change prate if is_bf512

usage = '''
display header of the first few packets from the binary file
syntax:
    %s <file(s)> [options]

options are:
    -n nPack        # how many packets to display (%d)
    --meta bytes    # ring buffer or file metadata length in bytes
    --ip            # to also display the local and dest IPs
    --ps packSize   # override packet size (%d)
    --time          # convert the packet counter to timestamp
    --prate RATE    # specify a different packet rate (pack/sec)
                    # (default: %d)
    --bf512         # for bf512, a different packet rate will be adopted
                    # packet size will also be different

''' % (pg, nPack, packSize, prate)

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
    elif (k == '--ps'):
        packSize = int(inp.pop(0))
    elif (k == '--time'):
        show_ts = True
    elif (k == '--prate'):
        prate = int(inp.pop(0))
    elif (k == '--bf512'):
        is_bf512 = True
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        files.append(k)

nFile = len(files)
if (is_bf512):
    prate   = 400e6/1024*8  # for bf512
    packSize = 3904

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
        if (show_ts):
            if (tmp is None):
                print(tmp)
            else:
                ep = tmp[2] + 2 + (tmp[0]-tmp[4])/prate
                ts = Time(ep, format='unix').to_datetime()
                print(tmp, ts)
        else:
            print(tmp)
    fh.close()



#!/usr/bin/env python

import sys, os.path
#from scapy.all import rdpcap
import struct
import numpy as np
import matplotlib.pyplot as plt
import time
from astropy.stats import sigma_clip
import warnings
from packet_func import *
from subprocess import call



ppf = 2     # packet-per-frame
nAnt = 16   # number of input channels
bpp = 8192  # bytes per packet (payload size)
pchan = 512 # num of spec.channel in a packet
nchan = int(ppf * pchan)    # total num of spec.channel
grp = 2            # number of bytes from the same antenna grouped together
blk = (nAnt * grp)  # number of bytes after all antennas cycled
nblk = int(bpp / blk)

fpgaclock = 400e6
order_off = 2


npack  = 1000   # default number of packets to read
mframe = 1000   # number of packets to time-avg

inp = sys.argv[0:]
pg  = inp.pop(0)

usage = '''
%s <RUDP_file1> <RUDP_file2> [options]

options are:
    -n packets      # specify how many packets to read and process
                    # (%d)
    -c clock_speed  # the estinmated FPGA clock speed in MHz (%.0f)
    -b byte_per_packet  # change the byte_per_packet variable (payload only)
                        (%d)
    --order OFFSET  # offset the packet order
                    # (to account for a bug where packet order always starts from 0)
                    # (default: %d)

''' % (pg, npack, fpgaclock/1e6, bpp, order_off)

if (len(inp) < 1):
    sys.exit(usage)

files = []
while inp:
    k = inp.pop(0)
    if (k == '-n'):
        npack = int(inp.pop(0))
    elif (k == '-c'):
        tmp = float(inp.pop(0))
        fpgaclock = tmp * 1e6
    elif (k == '-b'):
        bpp = int(inp.pop(0))
    elif (k == '--order'):
        order_off = int(inp.pop(0))
    else:
        files.append(k)

for fin in files:
    if (not os.path.isfile(fin)):
        sys.exit('file not found: %s' % fin)
nFile = len(files)

nframe = npack // ppf       # the new spec2k_x4 bitcode outputs 4 packets per frame
                            # each packet is 512ch, and a frame is 2048ch
pack_len = bpp + 64         # paylod length + header(64)

chan = np.arange(nchan)
samprate = fpgaclock * ppf
freq0 = np.fft.fftfreq(nchan*2, d=1/samprate)
freq = freq0[:nchan] / 1e6
## assuming only the upper half of packets have been transferred (in ppf packets)
freq += fpgaclock/1e6/ppf*order_off
#print(freq)
xlim = [freq[0], freq[-1]]


if (nFile != 2):
    print('invalid number of input files.')
    sys.exit()


### start reading data ###
t1 = time.time()

all_adata2d  = []
all_adata2d_mask  = []
all_across2d = []
all_across2d_mask = []

scan_across2d = []
scan_across2d_mask = []

## initialize the range ##
clk_skip = None

i1 = 0
do_next = True
while (do_next):
    ## initializing batch ##
    data0 = []  # the complex data of each packet
    clock = []  # the 5-bytes packet counter
    order = []  # the order in a 4-packet frame
    ftick = []  # frame counter
    rdata = []  # raw data
    for i in range(nFile):
        data0.append([])
        clock.append([])
        order.append([])


    i2 = i1 + mframe
    if (i2 >= npack):
        i2 = npack
        do_next = False

    ### find initial alignment ###
    if (clk_skip is None):
        clk_skip = [0, 0]
    else:
        clk_skip = [clk_skip[0]+mframe,clk_skip[1]+mframe]
    fh0 = open(files[0], 'rb')
    fh1 = open(files[1], 'rb')
    clk_tune = True
    ii = -1
    while (clk_tune):
        ii += 1
        skip_bytes = clk_skip[0] * pack_len
        fh0.seek(skip_bytes)
        hd0 = fh0.read(64)
        clk0, pko0 = headerUnpack(hd0, order_off=0)

        skip_bytes = clk_skip[1] * pack_len
        fh1.seek(skip_bytes)
        hd1 = fh1.read(64)
        clk1, pko1 = headerUnpack(hd1, order_off=0)
        clk_off = clk1 - clk0
        if (np.abs(clk_off) > 0):
            clk_tune = True
        else:
            clk_tune = False
        if (clk_off > 0):
            clk_skip[0] += clk_off
        else:
            clk_skip[1] += -clk_off
        print('iter:', ii, 'clk_off:', clk_off, 'clk0, clk1:', clk0, clk1)
    fh0.close()
    fh1.close()

    for fi, fin in enumerate(files):
        #print(fin)
        fh = open(fin, 'rb')
        skip_bytes = clk_skip[fi] * pack_len
        fh.seek(skip_bytes)
        #print('skip:', clk_skip[fi], 'packets, or:', skip_bytes, 'bytes')

        for i in range(i1, i2):
            buf = fh.read(pack_len)

            tmp = packetUnpack(buf, bpp, order_off=0)
            if (tmp is None):
                print('error in packet: %d'%i, fin)
                sys.exit()
            else:
                clk, pko, spec = tmp

            clock[fi].append(clk)
            order[fi].append(pko)
            data0[fi].append(spec)

            if (i%1000 ==0):
                print(fin, i)

        fh.close()

    print('batch reading done.')
    batch_adata2d = []
    batch_adata2d_mask = []
    for fi,fin in enumerate(files):
        ## rearranging packets into frames, taking into account lost packets
        clock[fi]  = np.array(clock[fi])

        # raw tick
        tick = clock[fi] - order[fi]        # packet of the same frame should have the same tick
                                            # i.e. tick is the frame id, delta-tick = ppf

        # legacy tick
        tick2 = np.unique(tick)             # keep only the unique ticks
        tick2 -= tick2[0] % ppf             # make sure the tick2 are multiples of ppf
        tick2 = tick2 // ppf                # normalize so that tick2 has an unit increment

        # new tick
        tick3 = tick - tick[0]              # tick3 starts from 0, increase by ppf (if no packet lost)
        tick3 = tick3 // ppf                # tick3 increment normalized to 1
        ntick = tick3.max() + 1
        print('ntick, ppf, bpp:', ntick, ppf, bpp)

        data  = np.ma.array(np.zeros((ntick, ppf, bpp), dtype=np.complex64), mask=True)
        data_tick = np.arange(ntick) + tick[0]
        ftick.append(data_tick)
        for i in range((i2-i1)):
            #it = int(tick[i]//4)
            it = tick3[i]
            io = int(order[fi][i])
            #print('debug:', i, it, io)
            data[it, io] = data0[fi][i]
            data[it, io].mask = False
        rdata.append(data)
        pack_flag = data.mask.all(axis=2)
        empty_frame = data.mask.all(axis=(1,2))
        fault_frame = data.mask.any(axis=(1,2))
        empty_pack  = data.mask.all(axis=(0,2))
        print('empty_pack:', empty_pack)
        #print('pack_flag:', pack_flag)

        print('nframes:', nframe, 'ntick:', ntick)
        print('nempty:', np.count_nonzero(empty_frame)) # frames that are totally empty
        #print('nfault:', np.count_nonzero(fault_frame)) # frames that are partially empty

        adata = np.abs(data).mean(axis=0)   # shape: (4, 8192)
        print('   data.shape', data.shape)
        print('   adata.shape', adata.shape)

        sel = 1
        if (sel == 0):
            adata2d = np.ma.array(np.zeros((nAnt, nchan), dtype=float), mask=False) # adata 1D is now abs
            for k in range(ppf):
                for i in range(nblk):
                    ch0 = i*grp + k*pchan
                    for j in range(nAnt):
                        serial = i*blk + j*grp
                        adata2d[j,ch0:ch0+grp] = adata[k,serial:serial+grp]
        elif (sel == 1):
            adata2d = adata.reshape((-1,nAnt,grp)).transpose((1,0,2)).reshape((nAnt,-1))

        batch_adata2d.append(adata2d)
        batch_adata2d_mask.append(adata2d.mask)
    all_adata2d.append(batch_adata2d)
    all_adata2d_mask.append(batch_adata2d_mask)


    ## cross-correlation ##
    print(len(ftick[0]), len(ftick[1]))
    mtick = np.min([len(ftick[0]), len(ftick[1])])
    print('mtick:', mtick)
    tick_diff = ftick[1][:mtick] - ftick[0][:mtick]
    #print(tick_diff)
    #print(ftick[0], ftick[1])
    print('min, max:', tick_diff.min(), tick_diff.max())

    cross = rdata[0][:mtick] * rdata[1][:mtick].conjugate()
    #cross /= (np.ma.abs(rdata[0][:mtick]) * np.ma.abs(rdata[1][:mtick]))  # correlation coefficient
    across = cross.mean(axis=0) # shape (ppf, 8192)

    sel = 1
    if (sel == 0):
        across2d = np.ma.array(np.zeros((nAnt, nchan), dtype=complex), mask=False)
        for k in range(ppf):
            for i in range(nblk):
                ch0 = i*grp + k*pchan
                for j in range(nAnt):
                    serial = i*blk + j*grp
                    across2d[j,ch0:ch0+grp] = across[k,serial:serial+grp]
    elif (sel == 1):
        across2d = across.reshape((-1,nAnt,grp)).transpose((1,0,2)).reshape((nAnt,-1))

    all_across2d.append(across2d)
    all_across2d_mask.append(across2d.mask)


    ostep = 50
    offs = []
    for off in range(-ntick//2, ntick//2-1, ostep):
        offs.append(off)
        if (off < 0):
            cross = rdata[0][-off:mtick] * rdata[1][:mtick+off].conjugate()
        else:
            cross = rdata[0][:mtick-off] * rdata[1][off:mtick].conjugate()
        across = cross.mean(axis=0)
        across2d = across.reshape((-1,nAnt,grp)).transpose((1,0,2)).reshape((nAnt,-1))
        scan_across2d.append(across2d)
        scan_across2d_mask.append(across2d.mask)

    scan_across2d = np.ma.array(scan_across2d, mask=scan_across2d_mask)


    ## next batch
    i1 = i2 + 1


t2 = time.time()
print('reading packet done: %d sec' % (t2-t1))
print('npack:',npack,'len(data0):',[len(data0[i]) for i in range(nFile)])

### data loading finished ###
all_adata2d = np.ma.array(all_adata2d, mask=all_adata2d_mask)
all_across2d = np.ma.array(all_across2d, mask=all_across2d_mask)

avg_adata2d  = all_adata2d.mean(axis=0)  # shape [nFile, nAnt, nChan]
avg_across2d = all_across2d.mean(axis=0) # shape [nAnt, nChan]

##  plotting individual FPGAs  ##
for fi, fin in enumerate(files):
    png1 = '%s.png' % fin
    f1, sub1 = plt.subplots(4,4,figsize=(12,9), sharex=True, sharey=True)
    s1 = sub1.flatten()

    adata2d = avg_adata2d[fi]

    warnings.filterwarnings('ignore')
    for ai in range(nAnt):
        ax = s1[ai]
        title = 'Inp%d' % ai

        #ax.scatter(freq, 20*np.log10(np.abs(adata2d[ai])))
        y = 20*np.log10(adata2d[ai])
        ax.plot(freq, y, marker='.')
        ax.grid()
        nnz = np.count_nonzero(np.abs(adata2d[ai]))
        ax.text(0.05, 0.90, 'nonzero:%d'%nnz, transform=ax.transAxes)
        pk = y.max()
        ipk = np.ma.argmax(y)
        fpk = freq[ipk]
        ax.text(0.05, 0.80, 'peak pow:%.0f\nch:%d\nfreq:%.0fMHz'%(pk,ipk,fpk), transform=ax.transAxes, va='top')
        #ax.set_title(title)
        ax.text(0.95, 0.90, title, transform=ax.transAxes, ha='right')
        #ax.set_ylim([-40,0])
        if (ai >= 12):
            ax.set_xlabel('freq (MHz)')
        if (ai%4 == 0):
            ax.set_ylabel('power (dB)')
    warnings.filterwarnings('default')


    ax.set_xlim(xlim)
    f1.tight_layout(rect=[0,0.03,1,0.95])
    f1.subplots_adjust(wspace=0, hspace=0)
    f1.suptitle('%s, npack=%d, nframe=%d'%(fin, npack, nframe))
    f1.savefig(png1)
    plt.close(f1)



##  plotting cross-correlation  ##

for pt in range(2):
    if (pt == 0):
        ptype = 'amp'
        y = np.ma.abs(avg_across2d)
    else:
        ptype = 'pha'
        y = np.ma.angle(avg_across2d)

    png2 = 'cross_fpga.%s.png' % ptype

    f2, sub2 = plt.subplots(4,4, figsize=(12,9), sharex=True, sharey=True)
    s2 = sub2.flatten()

    for ai in range(nAnt):
        ax = s2[ai]
        ax.plot(freq, y[ai])

        if (ai >= 12):
            ax.set_xlabel('freq (MHz)')
        if (ai%4 == 0):
            if (pt == 0):
                ax.set_ylabel('abs(cross/auto)')
            if (pt == 1):
                ax.set_ylabel('angle(cross)')

    ax.set_xlim(xlim)
    f2.suptitle(png2)
    f2.tight_layout(rect=[0,0.03,1,0.95])
    f2.subplots_adjust(wspace=0, hspace=0)
    f2.savefig(png2)
    plt.close(f2)


## plotting the scanning ##
odir = 'scan'
call('mkdir -p %s'%odir, shell=True)

nScan = len(scan_across2d)

for i in range(nScan):
    for pt in range(2):
        if (pt == 0):
            ptype = 'amp'
            y = np.ma.abs(scan_across2d[i])
        else:
            ptype = 'pha'
            y = np.ma.angle(scan_across2d[i])

        png2 = '%s/cross_fpga_%04d.%s.png' % (odir, i, ptype)

        f2, sub2 = plt.subplots(4,4, figsize=(12,9), sharex=True, sharey=True)
        s2 = sub2.flatten()

        for ai in range(nAnt):
            ax = s2[ai]
            ax.plot(freq, y[ai])

            if (ai >= 12):
                ax.set_xlabel('freq (MHz)')
            if (ai%4 == 0):
                if (pt == 0):
                    ax.set_ylabel('abs(cross/auto)')
                if (pt == 1):
                    ax.set_ylabel('angle(cross)')

        ax.set_xlim(xlim)
        f2.suptitle('%s off=%d' % (png2, offs[i]))
        f2.tight_layout(rect=[0,0.03,1,0.95])
        f2.subplots_adjust(wspace=0, hspace=0)
        f2.savefig(png2)
        plt.close(f2)


### done ###

t3 = time.time()
print('running time (sec):', t2-t1, t3-t1)

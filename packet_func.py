#!/usr/bin/env python

import sys, os.path
#from scapy.all import rdpcap
import struct
import numpy as np
import matplotlib.pyplot as plt
import time, re
from datetime import datetime
from astropy.time import Time
from astropy.stats import sigma_clip
import warnings

def toSigned(v, bits):
    mask = 1 << (bits-1)
    return -(v & mask) | (v & (mask-1))

def headerUnpack(header, order_off=0, verbose=0, hdver=1):
    if (hdver == 1):
        chk = header[5:8]
        if (chk != b'\xcc\xbb\xaa'):
            if (verbose > 0):
                print('packet read error. abort!')
            return None
        clk = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24) + (header[4]<<32)
        pko = header[8] + order_off
    elif (hdver == 2):
        chk = header[58:64]
        if (chk == b'RSTTTT' or chk == b'BURSTT'):  # updated header_2
            pass
        else:
            if (verbose > 0):
                print('packet read error. abort!')
            return None
        clk = 0
        for i in range(8):
            clk += header[7-i]
            clk *= 256
        pko = header[36] + order_off
    return clk, pko

def packetUnpack(buf, bpp, bitwidth=4, order_off=0, hdlen=64, hdver=1, unswap=False):
    header = buf[:hdlen]
    tmp = headerUnpack(header, order_off=order_off, hdver=hdver)
    if (tmp is None):
        return None
    else:
        clk, pko = tmp

    if (bitwidth==4):
        spec = np.zeros(bpp, dtype=np.complex64)
        # when reading 2 channels (= 2 bytes), 
        # the right 8-bit is even channel (e.g. ch0)
        # the left 8-bit is odd channel (e.g. ch1)
        #arr = struct.unpack('<%dH'%(bpp//2), buf[hdlen:])
        arr = struct.unpack('<%dB'%(bpp), buf[hdlen:])
        for k in range(bpp):
            #bit16 = arr[k]
            #bit8 = (bit16 & 0x00ff)         # for even channel
            bit8 = arr[k]
            bit4_i = bit8 & 0x0f
            ai = toSigned(bit4_i, 4)
            bit4_q = (bit8 & 0xf0) >> 4
            aq = toSigned(bit4_q, 4)
            if (unswap):    # use this when reading old bf data, no swapping
                spec[k] = ai + 1.j*aq
            else:           # for new bf data, swap even/odd channels
                if (k%2==0):
                    spec[k+1] = ai + 1j*aq
                else:
                    spec[k-1] = ai + 1j*aq

            #bit8 = (bit16 & 0x00ff) >> 8    # for odd channel
            #bit4_i = bit8 & 0x0f
            #ai = toSigned(bit4_i, 4)
            #bit4_q = (bit8 & 0xf0) >> 4
            #aq = toSigned(bit4_q, 4)
            #spec[2*k+1] = ai + 1.j*aq

    elif (bitwidth==16):
        #arr = struct.unpack('>%dh'%(bpp//2), buf[hdlen:])  # wrong endian?
        arr = struct.unpack('<%dh'%(bpp//2), buf[hdlen:])
        nsamp = bpp//4
        spec = np.zeros(nsamp, dtype=np.complex64)
        for k in range(bpp//4):
            #bit32 = int(arr[k])   # convert half-integer to integer (32 bits)
            #bit16_i =  bit32 & 0x0000ffff
            #bit16_q = (bit32 & 0xffff0000) >> 16
            #ai = toSigned(bit16_i, 16)
            #aq = toSigned(bit16_q, 16)
            ai = arr[2*k]
            aq = arr[2*k+1]
            spec[k] = ai + 1.j*aq

    return clk, pko, spec


def loadBatch(fh, pack0, npack, bpp, order_off=0, hdlen=64, bitwidth=4, hdver=1, meta=0, unswap=False):
    '''
    load npack blocks of binary data from the open filehandle
    starting from pack0. each block is of pack_len bytes.
    input:
        pack0: starting packet index (starting from 0)
        npack: number of packets to load
        bpp: bytes per packet (paylod only)
    optional:
        order_off: packet order offset (buggy, keep =0 for now)
        hdlen: header length in bytes
        bitwidth: data format keyword
        hdver: header format keyword (1,2)
        meta: ring buffer or file metadata length in bytes

    output:
        (masked array)
        data0, shape=(npack, bpp)
        clock, shape=(npack,)
        order, shape=(npack,)
    '''

    pack_len = hdlen + bpp
    b0 = pack0 * pack_len + meta
    fh.seek(b0)

    if (bitwidth==4):
        nsamp = bpp
    elif (bitwidth==16):
        nsamp = bpp//4

    data0 = np.ma.array(np.zeros((npack, nsamp), dtype=np.complex64), mask=False)
    clock = np.ma.array(np.zeros(npack, dtype=np.int64), mask=False)
    order = np.ma.array(np.zeros(npack, dtype=int), mask=False)
    for i in range(npack):
        buf = fh.read(pack_len)
        tmp = packetUnpack(buf, bpp, order_off=order_off, hdlen=hdlen, bitwidth=bitwidth, hdver=hdver, unswap=unswap)
        if (tmp is None):
            data0.mask[i] = True
            clock.mask[i] = True
            order.mask[i] = True
        else:
            clk, pko, spec = tmp
            data0[i] = spec
            clock[i] = clk
            order[i] = pko

    return data0, clock, order


def formSpec2(data0, bpp=8192, nAnt=16, nFPGA=16, grp=2, nChan=128, nStack=4, bitmap=None, verbose=0):
    '''
    convert packet data of bf256 to spectra
    bf256-specific settings:
        nChan=128
        nStack=4
        nFPGA=16
    for bf64, change to:
        nChan=512
        nStack=1
        nFPGA=4

    input:
        data0: shape=(nPack,bpp)
    output:
        antSpec: shape=(nFPGA,nFrame,nAnt,nChan)
    '''
    nPack = data0.shape[0]
    nChunk = nPack//nFPGA
    nFrame = nChunk * nStack
    nGrp = nChan//grp

    #antSpec = np.ma.array(np.zeros((nFPGA,nFrame,nAnt,nChan), dtype=np.complex64), mask=False)
    tmp = data0.reshape((nChunk,nFPGA,bpp)).reshape((nChunk,nFPGA,nStack,nGrp,nAnt,grp))
    antSpec = tmp.transpose((1,0,2,4,3,5)).reshape((nFPGA,nFrame,nAnt,nChan))
    antSpec = np.ma.array(antSpec, mask=False)

    tmp2 = np.tile(bitmap.reshape((1,nChunk,nFPGA)), (nStack,1,1))
    #print(tmp2.shape)
    antSpec.mask = ~tmp2.transpose((2,1,0)).reshape((nFPGA,nFrame))
    #print(antSpec)

    return antSpec


def formSpec(data0, clock, order, ppf, nAnt=16, grp=2, nChan=1024, bitmap=None, verbose=0):
    '''
    form spectra from the packet series
    input:
        data0, clock, order: as output of loadBatch
                data0.shap = (npack, bpp)
        ppf: packet per frame (e.g. 2, 4 or 8)
    optional:
        grp: number of consecutive bytes (channels) from the same antenna input 
        bitmap: if provided, should be a boolean array of shape (npack,)
                True for good packet, False for bad packet
                also implies the packets have been ordered, contiguous,
                although some packets may be invalid

                on the other hand, if the bitmap is None,
                it assumes all packets are valid, but maybe not contiguous
    
    output:
        data_tick, shape=(ntick,)
        antSpec, shape=(ntick,nAnt,nChan)   where nChan=bpp*ppf/nAnt e.g. 8192*2/16=1024
    '''
    npack, bpp = data0.shape    # e.g. 1000, 8192

    # pre-defined format, ignore input
    if (ppf == 2):      # 4bit, bf16, bf64 modes
        grp = 2
    elif (ppf == 8):    # 16bit mode
        grp = 1

    ## rearranging packets into frames, taking into account lost packets
    #clock  = np.array(clock)   # already an array

    # raw tick
    tick = clock - order        # packet of the same frame should have the same tick
                                        # i.e. tick is the frame id, delta-tick = ppf

    if (bitmap is None):
        if (verbose>1):
            print('in formSpec: bitmap is None. generate ad hoc one.')
        # new tick
        tick3 = tick - tick[0]              # tick3 starts from 0, increase by ppf (if no packet lost)
        tick3 = tick3 // ppf                # tick3 increment normalized to 1
        ntick = tick3.max() + 1             # e.g. ntick=500 for npack=1000
        ntick = tick3[-1] - tick3[0] + 1
        valid = np.ones(npack, dtype=bool)
        tick0 = tick[0]
    else:
        ntick = npack // ppf
        tmp = np.arange(npack)
        #tmp -= tmp%ppf
        tmp -= order
        tick3 = (tmp//ppf).astype(int)
        valid = bitmap
        if (verbose>1):
            print('in formSpec: npack/nvalid', npack, np.count_nonzero(valid))
        tick0 = -1
        ii = 0
        while (tick0 < 0 and ii<npack):
            if (valid[ii]):
                tick0 = tick[ii] - ii
            else:
                ii += 1
        if (tick0 < 0):
            print('error finding valid tick')
            return None
            
    if (verbose>0):
        print('ntick, ppf, bpp:', ntick, ppf, bpp)
        #print('tick3 min,max:', tick3.min(), tick3.max())
    data  = np.ma.array(np.zeros((ntick, ppf, bpp), dtype=np.complex64), mask=True)
    data_tick = np.arange(ntick) + tick0
    for i in range(npack):
        if (valid[i]):
            if (order.mask[i] == False):
                it = int(tick3[i])
                io = int(order[i])
                #print('debug:', i, it, io)
                if (io >= ppf):
                    sys.exit('incorrect ppf. abort.')
                if (it >= 0):
                    data[it, io] = data0[i]
                    data[it, io].mask = False

    pack_flag = data.mask.all(axis=2)
    empty_frame = data.mask.all(axis=(1,2))
    fault_frame = data.mask.any(axis=(1,2))
    empty_pack  = data.mask.all(axis=(0,2))

    if (verbose>1): # debug info
        #print('nframes:', nframe, 'ntick:', ntick)
        print('ntick:', ntick)
        print('nempty:', np.count_nonzero(empty_frame)) # frames that are totally empty
        print('nfault:', np.count_nonzero(fault_frame)) # frames that are partially empty

    #across2d = across.reshape((-1,nAnt,grp)).transpose((1,0,2)).reshape((nAnt,-1))
    if (ppf==2 or ppf==1):      # 4bit version
        antSpec = data.reshape((ntick,-1,nAnt,grp)).transpose((0,2,1,3)).reshape((ntick,nAnt,-1))
    elif (ppf==8):    # 16bit version
        #antSpec = data.reshape((ntick,-1,nAnt,grp)).transpose((0,2,1,3)).reshape((ntick,nAnt,-1))
        evntmp = data[:,0:4].reshape((ntick,-1,nAnt)).transpose((0,2,1))
        oddtmp = data[:,4:8].reshape((ntick,-1,nAnt)).transpose((0,2,1))
        #evntmp = data[:,0:4].reshape((ntick,4,-1,nAnt)).transpose((0,3,1,2)).reshape((ntick,nAnt,-1))
        #oddtmp = data[:,4:8].reshape((ntick,4,-1,nAnt)).transpose((0,3,1,2)).reshape((ntick,nAnt,-1))
        antSpec = np.ma.array(np.zeros((ntick,nAnt,nChan),dtype=complex), mask=True)
        antSpec[:,:,0::2] = evntmp
        antSpec[:,:,1::2] = oddtmp

    return data_tick, antSpec


def loadSpec(fh, pack0, npack, bpp=8192, ppf=2, order_off=0, nAnt=16, grp=2, hdlen=64, bitmap=None, nBlock=0, verbose=0, bitwidth=4, hdver=1, meta=0, blocklen=128000, unswap=False):
    '''
    a wrapper of loadBatch + formSpec
    input:
        fh: open binary file handle
        pack0: starting packet index (starting from 0)
        npack: number packets to load
        bpp: bytes per packet (payload length, e.g. 8192)
        ppf: packet per frame (e.g. 2)

    optional:
        nAnt: number of antenna inputs in the packet
        grp: number of consecutive channels (bytes) of the same antenna
        hdlen: header length in bytes
        order_off: (buggy) shift the packet order around, kepp =0 for now
        bitmap: boolean array, shape=(npack,); True for a good packet, False for bad
        nBlock: int; number of Block in the open file (fh)
                if bitmap is None and nBlock>0, the bitmap is automatically loaded
        verbose: int; pass to formSpec
        bitwidth: size of the I/Q data (4 or 16)
                (automatically determines the grp and ppf)
        hdver: header version (1, 2)
        meta: ring buffer or file metadata length in bytes

    output:
        data_tick, shape=(ntick,)
        antSpec, shape=(ntick, nAnt, nChan)
    '''
    #print('debug: meta', meta)
    if (meta == 64):    # assuming a valid file header
        # override default nBlock and blocklen
        mdict = metaRead(fh)
        blocklen = mdict['packet_number']
        nBlock = mdict['block_number']
        # also override hdver
        hdver = 2
        if (verbose):
            print('nBlock', nBlock, 'blocklen', blocklen)

    if (bitmap is None and nBlock>0):
        BM = loadFullbitmap(fh, nBlock, blocklen=blocklen, meta=meta)
        bitmap = BM[pack0:pack0+npack]

    if (bitmap is not None):
        nvalid = np.count_nonzero(bitmap)
        fvalid = float(nvalid/npack)
        if (verbose>0):
            print('pack0, npack, nvalid:', pack0, npack, nvalid)
        if (nvalid == 0):
            print('no valid packet!')
            return None
    else:
        if (verbose>0):
            print('pack0, npack:', pack0, npack, 'bitmap N/A')

    if (bitwidth==16):
        grp = 1
        ppf = 8

    data0, clock, order = loadBatch(fh, pack0, npack, bpp, order_off=order_off, hdlen=hdlen, bitwidth=bitwidth, hdver=hdver, meta=meta, unswap=unswap)
    tmp = formSpec(data0, clock, order, ppf, nAnt=nAnt, grp=grp, bitmap=bitmap, verbose=verbose)

    if (tmp is not None):
        data_tick, antSpec = tmp
    else:
        return None

    return data_tick, antSpec


def loadNode(fh, pack0, npack, bpp=8192, ppf=1, order_off=0, nAnt=16, grp=2, hdlen=64, bitmap=None, nBlock=0, verbose=0, bitwidth=4, hdver=2, meta=64, nFPGA=4, no_bitmap=False, get_order=False, unswap=False):
    '''
    load the ring buffer file of one node

    input:
        fh:: an open file handle

    output:
        antspec:: shape (nFPGA, nFrame, nAnt, nChan)
    '''

    if (nFPGA == 4):
        nChan = 512
        nStack = 1
    elif (nFPGA == 16):
        nChan = 128
        nStack = 4

    mdict = metaRead(fh)
    blocklen = mdict['packet_number']
    nBlock = mdict['block_number']
    #print('debug:', nBlock, blocklen, meta)
    BM = loadFullbitmap(fh, nBlock, blocklen=blocklen, meta=meta)
    bitmap = BM[pack0:pack0+npack]
    if (no_bitmap):
        bitmap = np.ones(npack, dtype=bool)
    #print('debug:', bitmap[2])


    data0, clock0, order0 = loadBatch(fh, pack0, npack, bpp, order_off=order_off, hdlen=hdlen, bitwidth=bitwidth, hdver=hdver, meta=meta, unswap=unswap)
    #print('debug:', data0.shape)
    #print('debug:', data0[2])
    antSpec = formSpec2(data0, nFPGA=nFPGA, nChan=nChan, nStack=nStack, bitmap=bitmap)

    #data1 = data0.reshape((-1,nFPGA,bpp))
    #clock1 = clock0.reshape((-1,nFPGA))
    #order1 = order0.reshape((-1,nFPGA))
    #bitmap1 = bitmap.reshape((-1,nFPGA))

    #antSpec = []
    #for i in range(nFPGA):
    #    tmp = formSpec(data1[:,i], clock1[:,i], order1[:,i], ppf, nChan=nChan, nAnt=nAnt, grp=grp, bitmap=bitmap1[:,i], verbose=verbose)
    #    antSpec.append(tmp[1])
    #return np.array(antSpec)    # shape = (nFPGA, nFrame, nAnt, nChan)

    out_order = np.median(order0.flatten()).astype(int)

    if (get_order):
        return antSpec, out_order
    else:
        return antSpec


def spec2Stream(spec, shift=True):
    '''
    given a single-antenna spectra, convert to a 1D time stream

    input:
        spec, shape=(nFrame, nChan)
                    note:: the spectral channels are assumed to be monotonically increasing

    optional:
        shift == True:: fftshift the input spectrum and perform complex inverse FFT
                        (this is the default)

        shift == False:: treat the input spectrum as positive freq half spectrum
                        and perform a inverse rFFT

    output:
        stream, shape=(nStream,); nStream=nFrame*nChan

    '''
    nFrame, nChan = spec.shape
    nStream = nFrame * nChan

    if (shift):
        # shift the spectrum before ifft
        tmp = np.fft.fftshift(spec, axes=1)
        stream2d = np.fft.ifft(tmp, axis=1)
    else:
        stream2d = np.fft.irfft(spec, n=nChan*2, axis=1)

    return stream2d.flatten()

def convertBitmap(buf):
    '''
    convert the input buffer to a Boolean array 
    bit == 1 --> True
    bit == 0 --> False

    input:
        buf: N bytes of binary buffer

    output:
        bitmap.shape = (nBit,), dtype=Bool
            nBit = 8*N
    '''

    bitmap = np.unpackbits(np.frombuffer(buf, dtype=np.uint8)).astype(bool)

    return bitmap


def makeBitmap(bitmap):
    arr = np.packbits(bitmap,bitorder='big')   # boolean or integers are ok
    nbyte = len(arr)
    buf = struct.pack('>%dB'%nbyte, *arr)
    return buf


def loadFullbitmap(fh, nBlock=None, bpp=8192, hdlen=64, blocklen=1000000, version=1, meta=0):
    '''
    version: 1
        the bitmap is located at the end of the buffer
        the buffer consists of nblock blocks (each block contains blocklen of packets)

        if meta==64, then nBlock and starting offset is taken from the file header

        input:
            fh: the open binary file handle
            nBlock: number blocks in the buffer

        optional:
            bpp: bytes per packet (payload only)
            hdlen: packet header lenghth in bytes
            blocklen: number of packets in a block

        output:
            Fullbitmap: boolean array, shape (nTotal,)
                nTotal = nBlock * blocklen
                value:: True: valid; False: invalid

    '''

    if (meta==64):  # assuming a valid file header
        # override default nBlock and blocklen
        mdict = metaRead(fh)
        blocklen = mdict['packet_number']
        nBlock = mdict['block_number']

    nTotal = nBlock * blocklen  # number of packets = number of bits in the bitmap
    nBytes = nTotal // 8        # bitmap length in bytes
    packlen = bpp + hdlen       # packet length in bytes
    starting = nTotal * packlen # the starting byte of the bitmap
    starting += meta

    fh.seek(starting)
    buf = fh.read(nBytes)
    Fullbitmap = convertBitmap(buf)

    return Fullbitmap


def writeSpec(fh, spec, p0=0, bitwidth=4, bitmap=None, verbose=0):
    '''
    given a spec, write the binary baseband data

    input:
        fh: a file handle, eg. fh = open(fname, 'wb')
        spec: complex, shape=(nFrame, nAnt, nChan)

    optional:
        p0: integer, add an offset to the starting packet counter

    return:
        0: on success; 1: on error
    '''

    nFrame, nAnt, nChan = spec.shape
    if (nAnt != 16):
        print('can only accept nAnt=16')
        return 1
    if (nChan != 1024):
        print('can only accept nChan=1024')
        return 1

    bpp = 8192  # bytes per packet
    if (bitwidth==4):
        ppf = 2 # packet per frame

    else:
        print('bitwidth=%d not implemented yet.'%bitwidth)
        return 1

    spec2 = spec.reshape((nFrame, nAnt, ppf, nChan//ppf))

    ipack = -1
    for i in range(nFrame):
        for pp in range(ppf):
            ipack += 1

            hd = headerPack(p0+ipack, pp)
            fh.write(hd)
            buf = dataPack(spec2[i,:,pp], bitwidth=bitwidth)
            fh.write(buf)

    return 0


def headerPack(clk, order):
    buf1 = struct.pack('<I', clk)  # 4-bytes only
    buf2 = b'\x00\xcc\xbb\xaa'
    buf3 = struct.pack('<H', order)
    hd = buf1 + buf2 + buf3 + buf3 + bytes(44) + b'BURSTTTT'

    return hd


def dataPack(data, bitwidth=4):
    '''
    pack contents of a single packet into binary format
    i.e. 512 channels for 4-bit
    128 channels for 16-bit
    input data type should be complex with the correct value range
    data.shape = (nAnt, nCh)
    '''

    nAnt, nCh0 = data.shape
    if (nAnt != 16):
        print('nAnt should be 16.')
        return None

    if (bitwidth == 4):
        nCh = 512
        grp = 2
        bpc = 1 # byte per ch
        arr = np.zeros(nAnt*nCh, dtype=np.uint8)    # unsigned Char (1-byte integer) array
    elif (bitwidth == 16):
        nCh = 128
        grp = 1
        bpc = 4 # byte per ch
        arr = np.zeros(nAnt*nCh, dtype=np.uint32)    # unsigned integer array
    else:
        print('bitwidth=%d is not implemented yet.'%bitwidth)
        return None

    if (nCh0 != nCh):
        print('inconsistent bitwidth?')
        return None

    nCk = nCh//grp  # number of chunks
    data2 = data.reshape((nAnt,nCk,grp))
    if (isinstance(data2, np.ma.MaskedArray)):
        data2 = data2.data

    i = -1
    for c in range(nCk):
        for j in range(nAnt):
            for k in range(grp):
                i += 1
                if (bitwidth == 4):
                    ai = int(data2[j,c,k].real) & 0x0f # convert to 4-bit unsigned integer
                    aq = int(data2[j,c,k].imag) & 0x0f
                    # each uint8 consists of 4-bit Q and 4-bit I
                    arr[i] = ai + (aq<<4)
                elif (bitwidth == 16):
                    ai = int(data2[j,c,k].real) & 0xffff # convert to 16-bit unsigned integer
                    aq = int(data2[j,c,k].imag) & 0xffff
                    # each uint32 consists of 16-bit Q and 16-bit I
                    arr[i] = ai& + (aq<<16)

    if (bitwidth == 4):
        buf = struct.pack('>%dB'%(nCh*nAnt), *arr)
    elif (bitwidth == 16):
        buf = struct.pack('>%dI'%(nCh*nAnt), *arr)

    return buf


def decHeader2(buf, ip=False, verbose=True):
    #if (buf[58:64]!=b'RSTTTT'):
    if (buf[58:64]==b'RSTTTT' or buf[58:64]==b'BURSTT'):
        pass
    else:
        if (verbose):
            print('invalid header encountered')
        return None

    pcnt  = struct.unpack('<Q', buf[:8])[0]
    tcnt  = struct.unpack('<Q', buf[8:16])[0]
    epoch = struct.unpack('<I', buf[16:20])[0]
    pps   = struct.unpack('<I', buf[20:24])[0]
    order = struct.unpack('<H', buf[36:38])[0]
    lip   = struct.unpack('<4B', buf[24:28])
    dip   = struct.unpack('<4B', buf[28:32])

    tmp = (pcnt, tcnt, epoch, pps, order)
    if (ip):
        tmp += ('%d.%d.%d.%d'%tuple(np.flip(lip)),)
        tmp += ('%d.%d.%d.%d'%tuple(np.flip(dip)),)
    return tmp


def filesEpoch(files, hdver=1, yr='23', tz=8, hdlen=64, meta=0, frate=400e6/1024, ppf=2):
    '''
    extract epoch time from a list of files

    if hdver==1, epoch time is extracted from the filename
    the year is probably excluded from the filename and the yr is used
    also the timezone of file system is needed

    if hdver==2, epoch time is extracted from the header as
    [old] epoch_second+pps_count+2
    [new] epoch_second + 2 + pack_cnt/(frate*ppf)
    where frate is the frame per second
    ppf is packet per frame
    '''

    if (isinstance(files, (list, np.ndarray))):
        nFile = len(files)
    elif (isinstance(files, str)):  # a scalar
        files = [files]
        nFile = len(files)
    else:
        sys.exit('type error in filesEpoch')


    if (hdver==2):
        prate = frate*ppf
        epoch = []
        for i in range(nFile):
            with open(files[i], 'rb') as fh:
                md = fh.read(meta)
                hd = fh.read(hdlen)
            tmp = decHeader2(hd)
            #ep = tmp[2]+tmp[3]+2
            ep = tmp[2] + 2 + tmp[0]/prate
            epoch.append(ep)
        epoch = np.array(epoch)

    elif (hdver==1):
        ftime0 = None
        tsec = []
        for fi in range(nFile):
            fin = files[fi]
            fbase = os.path.basename(fin)   # use 1st dir as reference

            tmp = fbase.split('.')
            ftpart = tmp[1]
            ftstr = yr+ftpart
            ftime = datetime.strptime(ftstr, '%y%m%d%H%M%S')
            if (ftime0 is None):
                ftime0 = ftime
                unix0 = Time(ftime0, format='datetime').to_value('unix')    # local time
                unix0 -= 3600.*tz                                           # convert to UTC

            dt = (ftime - ftime0).total_seconds()
            tsec.append(dt)
        epoch = np.array(tsec) + unix0

    return epoch


def beamUnpack(buf, hdlen=64, nFrame=51200, nChan=1024, verbose=0):
    '''
    unpack the beam_out data
    format is 4-bit
    also assumes the order is correct
    '''
    blockbytes = nFrame*nChan   # nChan per frame, 1 byte per channel
    nblock = len(buf) // (blockbytes+hdlen)

    srate = 400e6       # samples per sec
    prate = srate/nChan # packets per sec

    spec = np.zeros((nblock,blockbytes), dtype=np.complex64)
    epoch = np.zeros(nblock)
    pcnt = np.zeros(nblock, dtype=np.int64)
    for i in range(nblock):
        i0 = blockbytes*i
        hd = buf[i0:i0+hdlen]
        if (verbose):
            print('header:', hd)
        tmp = decHeader2(hd)
        epoch[i] = tmp[2] + 2 + tmp[0]/prate
        pcnt[i] = tmp[0]

        arr = struct.unpack('>%dB'%blockbytes, buf[i0+hdlen:i0+hdlen+blockbytes])
        for k in range(blockbytes):
            progress = 100*(k+1)/blockbytes
            if (verbose>0):
                if (progress%10==0.):
                    print('progress %d: %d bytes'%(progress, k))
            bit8 = arr[k]
            bit4_i = bit8 & 0x0f
            ai = toSigned(bit4_i, 4)
            #aq = toSigned(bit4_i, 4)
            bit4_q = (bit8 & 0xf0) >> 4
            aq = toSigned(bit4_q, 4)
            #ai = toSigned(bit4_q, 4)
            spec[i,k] = ai + 1.j*aq

    spec = spec.reshape((-1,nChan))

    return spec, pcnt, epoch


def loadBeam(fname, nStart=0, nBlock=None, nFrame=51200, nChan=1024, hdlen=64, verbose=0):
    '''
    load all data from a voltage beam file (e.g. 400 blocks)
    if nBlock is specified, load the number of blocks
    six blocks corresponds to about 1sec
    if nBlock is not specified, load all blocks

    output:
        spec    shape=(nBlock, nFrame, nChan), complex64
        pcnt    shape=(nBlock,), int64
        epoch   shape=(nBlock,), float
    '''
    blockByte = hdlen + nFrame*nChan
    if (nBlock is None):
        fs = os.path.getsize(fname)
        nBlock = fs // blockByte

    spec = np.zeros((nBlock, nFrame, nChan), dtype=np.complex64)
    pcnt = np.zeros(nBlock, dtype=np.int64)
    epoch = np.zeros(nBlock)

    with open(fname, 'rb') as fh:
        for i in range(nBlock):
            print('... block', i+nStart)
            start = blockByte*(i+nStart)
            fh.seek(start)
            buf = fh.read(blockByte)
            sp, pc, ep = beamUnpack(buf, nFrame=nFrame, nChan=nChan, hdlen=hdlen, verbose=verbose)
            spec[i] = sp
            pcnt[i] = pc[0]
            epoch[i] = ep[0]

    return spec, pcnt, epoch


def checkChain(files, nDisk=None, nFrame=51200, nChan=512, hdlen=64, verbose=False):
    '''
    check packet counters and block ordering for single-beam data split into multiple disks.
    serial blocks are written to subsequent files.
    for example, with nDisk=4:
        block0 --> file0
        block1 --> file1
        block2 --> file2
        block3 --> file3
        block4 --> file0
        block5 --> file1
        ...

    input:
        files:: (nDisk) files saved in parallel (the same filename stamp)
        nDisk:: default to the number of files
        nFrame:: number of frames per block
        nChan:: number of channels per frame saved in this ring buffer
        hdlen:: header size (bytes). there is one header per block

    output:
        fbystamp, pcnt, bidx

        fbystamp:: shape(nStamp, nDisk), files at each unique stamp
        pcnt:: shape(nStamp,), the starting packet counter
        bidx:: shape(nStamp, nTotal, 2), i.e. bidx[p][i] = [j,k] 
                for p-th timestamp,
                to get the i-th serial block, we should read the k-th block of j-th file
    '''


    byteBlock = nFrame * nChan + hdlen # 4+4i bits per channel, i.e. 1 byte/ch

    tmp = []
    for f in files:
        tmp.append(re.sub('_f\d', '', f))

    stamps = np.unique(tmp)

    pcnt = []
    bidx = []
    fbystamp = []
    for s in stamps:
        tmp = []    # files of the same stamp
        for f in files:
            if (s in f):
                tmp.append(f)
        fbystamp.append(tmp)

        with open(tmp[0], 'rb') as fh:
            buf = fh.read(hdlen)
            tup = decHeader2(buf)
            pcnt.append(tup[0])

        if (nDisk is None):
            nDisk = len(tmp)

        nTotal = 0
        for f in tmp:
            s = os.path.getsize(f)
            nBlock = s//byteBlock
            if (verbose):
                print(f, nBlock)
            nTotal += nBlock

        sbidx = []
        for i in range(nTotal):
            j = i%nDisk
            k = i//nDisk
            sbidx.append([j,k])
        bidx.append(sbidx)

    return fbystamp, pcnt, bidx


def loadChain(files, nBlock, startBlock=0, nDisk=None, nFrame=51200, nChan=1024, hdlen=64, verbose=False):
    '''
    given a list of (nDisk) files of the same unique stamp
    load contiguous nBlock of spectrum from the startBlock
    '''
    byteData  = nFrame * nChan
    byteBlock = byteData + hdlen

    fbystamp, pcstamp, bidx = checkChain(files, nDisk=nDisk, nFrame=nFrame, nChan=nChan, hdlen=hdlen, verbose=verbose)
    nStamp = len(pcstamp)
    if (verbose):
        print('loadChain: from', files)
        print('unique stamps:', nStamp)
        if (nStamp>1):
            print('warning: use only the first unique stamp')

    files0 = fbystamp[0]
    pcstamp = pcstamp[0]
    bidx = bidx[0]

    spec  = np.zeros((nBlock, nFrame, nChan), dtype=np.complex64)
    pcnt  = np.zeros(nBlock, dtype=np.int64)
    epoch = np.zeros(nBlock)
    for i in range(nBlock):
        ii = startBlock + i
        j, k = bidx[ii]

        tmp = loadBeam(files0[j], nStart=k, nBlock=1, nFrame=nFrame, nChan=nChan, hdlen=hdlen, verbose=verbose)
        spec[i]  = tmp[0][0]
        pcnt[i]  = tmp[1][0]
        epoch[i] = tmp[2][0]

    return spec, pcnt, epoch


def hdScanner(fname, meta=64, hdlen=64, sep=8256):
    '''
    scan the binary file and return the packet counter
    '''
    nTotal = os.path.getsize(fname)
    fh = open(fname, 'rb')
    pkcnt = []
    order = []
    p = meta
    while (p < nTotal):
        fh.seek(p)
        hd = fh.read(hdlen)
        tmp = decHeader2(hd)
        if (not tmp is None):
            pkcnt.append(tmp[0])
            order.append(tmp[4])
        else:
            pkcnt.append(-1)
            order.append(-1)
            print('bad header at', p)
        p += sep
    fh.close()
    return np.array(pkcnt), np.array(order)


def metaRead(fh, meta=64, dict_out=True):
    '''
    read the meta information from the file
    if dict_out is True, a dictionary is returned
    otherwise, the tuple is returned
    '''
    fh.seek(0)
    buf = fh.read(meta)

    spec = [
        ['server_id', 'H'],
        ['ver_id', 'H'],
        ['buffer_id', 'H'],
        ['packet_size', 'H'],
        ['block_number', 'H'],
        ['beam_id', 'h'],
        ['packet_number', 'I'],
        ['second_start', 'I'],
        ['second_end', 'I'],
        ['data_offset', 'Q'],
        ['bitmask_offset', 'Q'],
        ['reserved_2', 'Q'],
        ['reserved_3', 'Q'],
        ['reserved_4', 'Q']
    ]
    spec = np.array(spec)


    fmt = '<' + ''.join(spec[:,1])
    #print(spec[:,1], '-->', fmt)
    tmp = struct.unpack(fmt, buf)    # format needs to fill up 64 bytes
    #tmp = struct.unpack('<5Hh3I5Q', buf)    # format needs to fill up 64 bytes

    if (dict_out):
        names = spec[:,0]
        out = {}
        for i,n in enumerate(names):
            out[n] = tmp[i]

        return out
    else:
        #print(tmp)
        return tmp

#!/usr/bin/env python

import sys, os.path
#from scapy.all import rdpcap
import struct
import numpy as np
import matplotlib.pyplot as plt
import time
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
        chk = header[56:64]
        if (chk != b'BURSTTTT'):
            if (verbose > 0):
                print('packet read error. abort!')
            return None
        clk = 0
        for i in range(8):
            clk += header[7-i]
            clk *= 256
        pko = header[36] + order_off
    return clk, pko

def packetUnpack(buf, bpp, bitwidth=4, order_off=0, hdlen=64, hdver=1):
    header = buf[:hdlen]
    tmp = headerUnpack(header, order_off=order_off, hdver=hdver)
    if (tmp is None):
        return None
    else:
        clk, pko = tmp

    if (bitwidth==4):
        arr = struct.unpack('>%dB'%bpp, buf[hdlen:])
        spec = np.zeros(bpp, dtype=np.complex64)
        for k in range(bpp):
            bit8 = arr[k]
            bit4_i = bit8 & 0x0f
            #ai = toSigned(bit4_i, 4)
            aq = toSigned(bit4_i, 4)
            bit4_q = (bit8 & 0xf0) >> 4
            #aq = toSigned(bit4_q, 4)
            ai = toSigned(bit4_q, 4)
            spec[k] = ai + 1.j*aq

    elif (bitwidth==16):
        arr = struct.unpack('>%dh'%(bpp//2), buf[hdlen:])
        nsamp = bpp//4
        spec = np.zeros(nsamp, dtype=np.complex64)
        for k in range(bpp//4):
            #bit32 = int(arr[k])   # convert half-integer to integer (32 bits)
            #bit16_i =  bit32 & 0x0000ffff
            #bit16_q = (bit32 & 0xffff0000) >> 16
            #ai = toSigned(bit16_i, 16)
            #aq = toSigned(bit16_q, 16)
            aq = arr[2*k]
            ai = arr[2*k+1]
            spec[k] = ai + 1.j*aq

    return clk, pko, spec


def loadBatch(fh, pack0, npack, bpp, order_off=0, hdlen=64, bitwidth=4, hdver=1):
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

    output:
        (masked array)
        data0, shape=(npack, bpp)
        clock, shape=(npack,)
        order, shape=(npack,)
    '''
    pack_len = hdlen + bpp
    b0 = pack0 * pack_len
    fh.seek(b0)

    if (bitwidth==4):
        nsamp = bpp
    elif (bitwidth==16):
        nsamp = bpp//4

    data0 = np.ma.array(np.zeros((npack, nsamp), dtype=complex), mask=False)
    clock = np.ma.array(np.zeros(npack, dtype=np.int64), mask=False)
    order = np.ma.array(np.zeros(npack, dtype=int), mask=False)
    for i in range(npack):
        buf = fh.read(pack_len)
        tmp = packetUnpack(buf, bpp, order_off=order_off, hdlen=hdlen, bitwidth=bitwidth, hdver=hdver)
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
    if (ppf == 2):
        grp = 2
    elif (ppf == 8):
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
        print('tick3 min,max:', tick3.min(), tick3.max())
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
    if (ppf==2):      # 4bit version
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


def loadSpec(fh, pack0, npack, bpp=8192, ppf=2, order_off=0, nAnt=16, grp=2, hdlen=64, bitmap=None, nBlock=0, verbose=0, bitwidth=4, hdver=1):
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

    output:
        data_tick, shape=(ntick,)
        antSpec, shape=(ntick, nAnt, nChan)
    '''

    if (bitmap is None and nBlock>0):
        BM = loadFullbitmap(fh, nBlock)
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

    data0, clock, order = loadBatch(fh, pack0, npack, bpp, order_off=order_off, hdlen=hdlen, bitwidth=bitwidth, hdver=hdver)
    tmp = formSpec(data0, clock, order, ppf, nAnt=nAnt, grp=grp, bitmap=bitmap, verbose=verbose)

    if (tmp is not None):
        data_tick, antSpec = tmp
    else:
        return None

    return data_tick, antSpec


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


def loadFullbitmap(fh, nBlock, bpp=8192, hdlen=64, blocklen=1000000, version=1):
    '''
    version: 1
        the bitmap is located at the end of the buffer
        the buffer consists of nblock blocks (each block contains blocklen of packets)

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

    nTotal = nBlock * blocklen  # number of packets = number of bits in the bitmap
    nBytes = nTotal // 8        # bitmap length in bytes
    packlen = bpp + hdlen       # packet length in bytes
    starting = nTotal * packlen # the starting byte of the bitmap

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



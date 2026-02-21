#!/usr/bin/env python

from packet_func import *

rings = ['ring0', 'ring1']

for ring in rings:
    print('## ring:', ring)

    fname = '/dev/hugepages/%s'%ring
    fhuge = np.memmap(fname, '<u1')

    mdict = readRBH(fname)
    nBlock = mdict['block_number']
    nPack  = mdict['packet_number']
    block_h = mdict['head_block_id']
    bm_offset = mdict['bitmask_offset']

    p = bm_offset
    q = int(p + nBlock*nPack//8)
    buf = fhuge[p:q]
    BM = convertBitmap(buf)

    #block_t = block_h + 1 - nBlock # test including the current block
    block_t = block_h + 2 - nBlock # exclude the current block
    if (block_t >= 0):
        block_t -= nBlock

    count_tot = 0   # total bitmask
    count_val = 0   # valid bitmask
    for i in range(block_t, block_h+1): # block_h included
        bid = (i+nBlock)%nBlock

        j1 = int(bid     * nPack)
        j2 = int((bid+1) * nPack)

        count_tot += nPack
        count_val += np.count_nonzero(BM[j1:j2])

    print('total, valid:', count_tot, count_val)
    print('invalid frac:', count_val/count_tot-1)



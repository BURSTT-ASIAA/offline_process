#!/usr/bin/env python

from packet_func import *
from calibrate_func import *
import matplotlib.pyplot as plt
from loadh5 import *
from subprocess import call
from glob import glob
from astropy.time import Time
import gc

inp = sys.argv[0:]
pg  = inp.pop(0)

#nInp = 2 #16
#nBl  = nInp * (nInp-1) // 2
inpIdx = []     # list of all idx
grpIdx = {}     # idx by group (dir)
dirs   = []
nChan = 1024
flim  = [400., 800.]
nBlock = None
autoblock = True
autop0    = True
bytePayload = 8192      # packet payload size in bytes
byteHeader  = 64        # packet header size in bytes
packBlock   = 1000000   # num of packets in a block
byteBMBlock = packBlock//8  # bitmap size of a block in bytes
byteBlock   = (bytePayload+byteHeader)*packBlock + byteBMBlock  # block size in bytes
verbose     = False
bitwidth    = 4
ppf         = 2
winEigen    = -1        # do not compute eigenmodes if -1
packChunk   = 100000    # split nPack into chunks if too large

pack0 = 0               # packet offset
nPack = 1000            # num of packets to read
fout  = 'analyze.vish5'
dodiag = True
specDur = 1.e-6/(flim[1]-flim[0])*nChan    # duration of one spectrum in seconds


usage = '''
calculate cross-correlation of a single FPGA board

syntax:
    %s [options]


    options are:
    -g <input> 'idx...' 
                    # <input_dir> should contain binary packets files to be analyzed
                    # each dir is expected to contain data from one FPGA board
                    # 'idx...' is a list of array indices to correlate
                    
                    # note: <input> can be a file (.bin), which will be processed
                    # however, if <input> is a folder, all .bin files within will be processed

    -n nPack        # specify the number of packets to read and average per file
                    # defautl: %d

    --eigen winID   # specify a window ID (i.e. file ID) to compute the eigenmodes
                    # default: do not compute eigenmodes

    --16bit         # specify that data is in 16bit format
                    # (default is in 4bit format)

    --p0 pack0      # specify the pack offset
                    # disable autop0

    --nB nBlock     # specify the length of the binary file in nBlock
                    # (1 Block = 1M packets)
                    # without this parameter, the bitmap can not be loaded
                    # default is to determine the nBlock automatically

    --no-aB         # disable auto-Block

    -o output_file  # specify the output file name
                    # default: %s

    -v              # enable more message

''' % (pg, nPack, fout)

if (len(inp) < 1):
    sys.exit(usage)


while (inp):
    k = inp.pop(0)
    if (k == '-n'):
        nPack = int(inp.pop(0))
    elif (k == '-g'):
        gdir = inp.pop(0)
        tmp = inp.pop(0)
        gidx = [int(x) for x in tmp.split()]
        dirs.append(gdir)
        grpIdx[gdir] = gidx
    elif (k == '--eigen'):
        winEigen = int(inp.pop(0))
    elif (k == '--16bit'):
        bitwidth = 16
        ppf = 8
    elif (k == '--p0'):
        pack0 = int(inp.pop(0))
        autop0 = False
    elif (k == '--nB'):
        nBlock = int(inp.pop(0))
    elif (k == '--no-aB'):
        autoblock = False
    elif (k == '-o'):
        fout = inp.pop(0)
    elif (k == '-v'):
        verbose = True
    elif (k.startswith('-')):
        sys.exit('unknown option: %s' % k)
    else:
        print('unused arg:', k)

freq = np.linspace(flim[0], flim[1], nChan, endpoint=False)

cdir = fout.replace('.vish5', '.check')
if (not os.path.isdir(cdir)):
    call('mkdir -p %s'%cdir, shell=True)


winDur = nPack//2 * specDur # window duration in seconds

nGrp = len(dirs)
for i in range(nGrp):
    gdir = dirs[i]
    for x in grpIdx[gdir]:
        lab =  '%d_%d' % (i, x)
        inpIdx.append(lab)

if (len(inpIdx) == 0):
    inpIdx = [0, 1] # default to idx 0, 1 if nothing specified
nInp = len(inpIdx)
nBl  = nInp*(nInp-1)//2

info = '''
=== info ===
nInp = %d
nPack = %d
output file: %s
diagnostics: %s
winDur = %.6f sec
============
''' % (nInp, nPack, fout, cdir, winDur)
print(info)
print('idx:', inpIdx)


# misc setup
eye0 = np.identity(nInp)
eye1 = eye0.copy()
eye1[nInp-1, nInp-1] = 0
eye2 = eye1.copy()
eye2[nInp-2, nInp-2] = 0



attrs = {}
attrs['winDur'] = winDur

ftime0 = None   # the first file open time
tsec = []       # window offset time in seconds
winNFT = []
winNFTmask = []
winSpec = []
winSpecmask = []
winVar = []
winVarmask = []
winCoeff = []
winCoeffmask = []
if (winEigen>-1):
    winNFT0 = []
    winNFT0mask = []
    winSpec0 = []
    winSpec0mask = []
    winCoeff0 = []
    winCoeff0mask = []
    winNFT1 = []
    winNFT1mask = []
    winSpec1 = []
    winSpec1mask = []
    winCoeff1 = []
    winCoeff1mask = []
    winNFT2 = []
    winNFT2mask = []
    winSpec2 = []
    winSpec2mask = []
    winCoeff2 = []
    winCoeff2mask = []

grpFiles = []
for gi in range(nGrp):
    idir = dirs[gi]
    if (os.path.isdir(idir)):
        files = glob('%s/*.fpga.bin' % idir)
        files.sort()
    elif (os.path.isfile(idir)):
        files = [idir]
    grpFiles.append(files)
nFile = len(grpFiles[0])
#nFile = 2   # override when debugging



## compute eigenmodes ##
if (winEigen > -1):
    if (winEigen > nFile-1):
        sys.exit('invalid winID, greater than nFile-1: %d > %d'%(winEigen, nFile-1))
    fin = grpFiles[0][winEigen]
    fbase = os.path.basename(fin)
    ftpart = fbase.split('.')[1]

    files = [fin]
    if (nGrp > 1):
        for gi in range(1,nGrp):
            fin = None
            for x in grpFiles[gi]:
                if (ftpart in x):
                    fin = x
            if (fin is None):
                print('skip bad time', ftpart)
            else:
                files.append(fin)
    print('eigen:', ftpart, '-->', files)


    # auto-detect nBlock and invalid block
    fin = files[0]
    if (nBlock is None and autoblock):
        fsize = os.path.getsize(fin)
        nBlock = np.rint(fsize/byteBlock).astype(int)
        print('autoblock:', nBlock)
    if (nBlock == 0):
        print('bitmap info not used.')
    fh = open(fin, 'rb')
    fullBM = loadFullbitmap(fh, nBlock)
    fh.close()

    pack0_new = pack0
    ap = autop0
    while (ap):
        bitmap = fullBM[pack0_new:pack0_new+nPack]
        fvalid = float(np.count_nonzero(bitmap)/nPack)
        if (fvalid < 0.1):  # arbitrary threshold
            pack0_new += packBlock
        else:
            ap = False
    if (verbose):
        print('pack0:', pack0_new)

    ## split nPack into chunks if necessary
    nChunk = nPack // packChunk
    if (nChunk < 1):
        nChunk = 1
        nPack2 = nPack
    else:
        nPack2 = nChunk * packChunk
    if (verbose):
        print('nChunk:', nChunk, 'nPack2:', nPack2)

    chunkCov  = np.zeros((nChunk, nInp, nInp, nChan), dtype=np.complex64)
    chunkNorm = np.zeros((nChunk, nInp, nChan)) 
    for iChunk in range(nChunk):
        p0_i = packChunk * iChunk   # offset from the pack0
        #print('debug:', p0_i, nPack2, iChunk)
        if (iChunk == nChunk-1):
            nPack_i = nPack2 - p0_i
        else:
            nPack_i = packChunk

        p0 = pack0_new + p0_i

        #-- reading the raw spectrum
        # used only in this chunk, reset by next chunk
        rawspec = np.ma.array(np.zeros((nInp, nPack_i//ppf, nChan), dtype=np.complex64), mask=False)
        ai = -1
        for gi in range(nGrp):
            gidx = grpIdx[dirs[gi]]
            fin = files[gi]
            fh = open(fin, 'rb')

            t1 = time.time()
            #tick, tmpspec = loadSpec(fh, p0, nPack, nBlock=nBlock, verbose=verbose)
            tick, tmpspec = loadSpec(fh, p0, nPack_i, nBlock=nBlock, verbose=verbose, bitwidth=bitwidth)
            t2 = time.time()
            print('   chunk%d: %d packets loaded in %.2fsec'%(iChunk, nPack_i, t2-t1))
            fh.close()

            for idx in gidx:
                ai += 1
                if (verbose):
                    print(gi, dirs[gi], idx, '-->', ai)
                rawspec[ai] = tmpspec[:,idx]

            del tick, tmpspec
            gc.collect()

        #-- compute eigenmodes
        # makeCov expected input shape: (nAnt, nFrame, nChan)
        # output shape: (nAnt, nAnt, nChan), (nAnt, nChan)
        #chunkCov[iChunk], chunkNorm[iChunk] = makeCov(rawspec, scale=False)
        chunkCov[iChunk], chunkNorm[iChunk] = makeCov(rawspec, scale=False, coeff=False)
        del rawspec
        gc.collect()

    # combine chunks and obtain eigenmodes
    Cov  = chunkCov.mean(axis=0)
    norm = chunkNorm.mean(axis=0)
    W, V = Cov2Eig(Cov)

    adoneh5(fout, W, 'eigen_W')
    adoneh5(fout, V, 'eigen_V')
    eigAttrs = {'winID':winEigen, 'fileID':ftpart, 'scale':False}
    putAttrs(fout, eigAttrs, dest='eigen_W')

    png = '%s/eigenvalues_%s.png' % (cdir, ftpart)
    f1, ax1 = plt.subplots(1,1,figsize=(15,4))
    for i in range(nInp):
        modeId = nInp-1-i
        ax1.plot(freq, 10*np.log10(W[:,modeId]), label='mode-%d'%modeId)
    ax1.set_xlabel('freq (MHz)')
    ax1.set_ylabel('eigenvalue (dB)')
    ax1.set_title('fileID:%s'%ftpart)
    f1.tight_layout()
    f1.savefig(png)
    plt.close(f1)


    # define projection matrices
    Proj0 = np.zeros((nChan, nInp, nInp), dtype=np.complex64)   # remove 0 mode
    Proj1 = np.zeros((nChan, nInp, nInp), dtype=np.complex64)   # remove 1 mode
    Proj2 = np.zeros((nChan, nInp, nInp), dtype=np.complex64)   # remove 2 modes
    for ii in range(nChan):
        Vi = V[ii]
        Vih = Vi.conjugate().T
        Proj0[ii] = np.dot(Vi, np.dot(eye0, Vih))
        Proj1[ii] = np.dot(Vi, np.dot(eye1, Vih))
        Proj2[ii] = np.dot(Vi, np.dot(eye2, Vih))

    adoneh5(fout, Proj0, 'Proj0')
    adoneh5(fout, Proj1, 'Proj1')
    adoneh5(fout, Proj2, 'Proj2')



for fi in range(nFile):
    fin = grpFiles[0][fi]
    fbase = os.path.basename(fin)   # use 1st dir as reference

    tmp = fbase.split('.')
    ftpart = tmp[1]
    ftstr = '23'+ftpart # hard-coded to 2023!!
    ftime = datetime.strptime(ftstr, '%y%m%d%H%M%S')
    if (ftime0 is None):
        ftime0 = ftime
        unix0 = Time(ftime0, format='datetime').to_value('unix')    # local time
        unix0 -= 3600.*8.                                           # convert to UTC
        attrs['unix_utc_open'] = unix0

    dt = (ftime - ftime0).total_seconds()
    print('(%d/%d)'%(fi+1,nFile), fin, 'dt=%dsec'%dt)
    tsec.append(dt)

    files = [fin]   # nGrp == 1 covered
    if (nGrp > 1):
        for gi in range(1,nGrp):
            fin = None
            for x in grpFiles[gi]:
                if (ftpart in x):
                    fin = x
            if (fin is None):
                print('skip bad time', ftpart)
            else:
                files.append(fin)
    print(ftpart, '-->', files)

    # auto-detect nBlock and invalid block
    fin = files[0]
    if (nBlock is None and autoblock):
        fsize = os.path.getsize(fin)
        nBlock = np.rint(fsize/byteBlock).astype(int)
        print('autoblock:', nBlock)
    if (nBlock == 0):
        print('bitmap info not used.')
    fh = open(fin, 'rb')
    fullBM = loadFullbitmap(fh, nBlock)
    fh.close()

    pack0_new = pack0
    ap = autop0
    while (ap):
        bitmap = fullBM[pack0_new:pack0_new+nPack]
        fvalid = float(np.count_nonzero(bitmap)/nPack)
        if (fvalid < 0.1):  # arbitrary threshold
            pack0_new += packBlock
        else:
            ap = False
    if (verbose):
        print('pack0:', pack0_new)

    ## split nPack into chunks if necessary
    nChunk = nPack // packChunk
    if (nChunk < 1):
        nChunk = 1
        nPack2 = nPack
    else:
        nPack2 = nChunk * packChunk
    if (verbose):
        print('nChunk:', nChunk, 'nPack2:', nPack2)


    chunkAmpld = np.ma.array(np.zeros((nChunk, nInp, nChan)), mask=False)
    chunkCross = np.ma.array(np.zeros((nChunk, nBl, nChan), dtype=np.complex64), mask=False)
    chunkVar   = np.ma.array(np.zeros((nChunk, nBl, nChan)), mask=False)
    chunkCoeff = np.ma.array(np.zeros((nChunk, nBl, nChan), dtype=np.complex64), mask=False)
    if (winEigen>-1):
        # Proj0
        chunkAmpld0 = np.ma.array(np.zeros((nChunk, nInp, nChan)), mask=False)
        chunkCross0 = np.ma.array(np.zeros((nChunk, nBl, nChan), dtype=np.complex64), mask=False)
        chunkCoeff0 = np.ma.array(np.zeros((nChunk, nBl, nChan), dtype=np.complex64), mask=False)
        # Proj1
        chunkAmpld1 = np.ma.array(np.zeros((nChunk, nInp, nChan)), mask=False)
        chunkCross1 = np.ma.array(np.zeros((nChunk, nBl, nChan), dtype=np.complex64), mask=False)
        chunkCoeff1 = np.ma.array(np.zeros((nChunk, nBl, nChan), dtype=np.complex64), mask=False)
        # Proj2
        chunkAmpld2 = np.ma.array(np.zeros((nChunk, nInp, nChan)), mask=False)
        chunkCross2 = np.ma.array(np.zeros((nChunk, nBl, nChan), dtype=np.complex64), mask=False)
        chunkCoeff2 = np.ma.array(np.zeros((nChunk, nBl, nChan), dtype=np.complex64), mask=False)

    for iChunk in range(nChunk):
        p0_i = packChunk * iChunk   # offset from the pack0
        if (iChunk == nChunk-1):
            nPack_i = nPack2 - p0_i
        else:
            nPack_i = packChunk

        p0 = pack0_new + p0_i

        antspec = np.ma.array(np.zeros((nInp, nPack_i//ppf, nChan), dtype=np.complex64), mask=False)
        ai = -1
        for gi in range(nGrp):
            gidx = grpIdx[dirs[gi]]
            fin = files[gi]
            fh = open(fin, 'rb')

            t1 = time.time()
            #tick, tmpspec = loadSpec(fh, p0, nPack, nBlock=nBlock, verbose=verbose)
            tick, tmpspec = loadSpec(fh, p0, nPack_i, nBlock=nBlock, verbose=verbose, bitwidth=bitwidth)
            t2 = time.time()
            print('   chunk%d: %d packets loaded in %.2fsec'%(iChunk, nPack_i, t2-t1))
            fh.close()

            for idx in gidx:
                ai += 1
                if (verbose):
                    print(gi, dirs[gi], idx, '-->', ai)
                antspec[ai] = tmpspec[:,idx]

            del tick, tmpspec
            gc.collect()

        chunkAmpld[iChunk] = np.ma.abs(antspec).mean(axis=1)  # shape (nInp, nChan)

        b = -1
        for ai in range(nInp-1):
            for aj in range(ai+1, nInp):
                b += 1
                tmp  = antspec[ai] * antspec[aj].conjugate()
                norm = np.ma.abs(antspec[ai])*np.ma.abs(antspec[aj])
                norm.mask[norm==0.] = True  # mask zero auto-correlations
                chunkVar[iChunk,b] = tmp.var(axis=0)
                chunkCross[iChunk,b] = tmp.mean(axis=0)
                chunkCoeff[iChunk,b] = (tmp/norm).mean(axis=0)

        if (winEigen>-1):
            null0 = np.ma.array(np.zeros_like(antspec), mask=False) 
            null1 = np.ma.array(np.zeros_like(antspec), mask=False) 
            null2 = np.ma.array(np.zeros_like(antspec), mask=False) 
            for ii in range(nChan):
                null0[:,:,ii] = np.tensordot(Proj0[ii], antspec[:,:,ii], axes=(1,0))
                null1[:,:,ii] = np.tensordot(Proj1[ii], antspec[:,:,ii], axes=(1,0))
                null2[:,:,ii] = np.tensordot(Proj2[ii], antspec[:,:,ii], axes=(1,0))
            null0.mask = antspec.mask
            null1.mask = antspec.mask
            null2.mask = antspec.mask
            print('debug:', Proj0.shape, antspec.shape, null0.shape)
            chunkAmpld0[iChunk] = np.ma.abs(null0).mean(axis=1)
            chunkAmpld1[iChunk] = np.ma.abs(null1).mean(axis=1)
            chunkAmpld2[iChunk] = np.ma.abs(null2).mean(axis=1)

            b = -1
            for ai in range(nInp-1):
                for aj in range(ai+1, nInp):
                    b += 1
                    tmp  = null0[ai] * null0[aj].conjugate()
                    norm = np.ma.abs(null0[ai])*np.ma.abs(null0[aj])
                    norm.mask[norm==0.] = True  # mask zero auto-correlations
                    chunkCross0[iChunk,b] = tmp.mean(axis=0)
                    chunkCoeff0[iChunk,b] = (tmp/norm).mean(axis=0)

                    tmp  = null1[ai] * null1[aj].conjugate()
                    norm = np.ma.abs(null1[ai])*np.ma.abs(null1[aj])
                    norm.mask[norm==0.] = True  # mask zero auto-correlations
                    chunkCross1[iChunk,b] = tmp.mean(axis=0)
                    chunkCoeff1[iChunk,b] = (tmp/norm).mean(axis=0)

                    tmp  = null2[ai] * null2[aj].conjugate()
                    norm = np.ma.abs(null2[ai])*np.ma.abs(null2[aj])
                    norm.mask[norm==0.] = True  # mask zero auto-correlations
                    chunkCross2[iChunk,b] = tmp.mean(axis=0)
                    chunkCoeff2[iChunk,b] = (tmp/norm).mean(axis=0)


    ampld = chunkAmpld.mean(axis=0)
    cross = chunkCross.mean(axis=0)
    var   = chunkVar.mean(axis=0)
    coeff = chunkCoeff.mean(axis=0)

    winNFT.append(ampld)
    winNFTmask.append(ampld.mask)
    winVar.append(var)
    winVarmask.append(var.mask)
    winSpec.append(cross)
    winSpecmask.append(cross.mask)
    winCoeff.append(coeff)
    winCoeffmask.append(coeff.mask)

    if (winEigen>-1):
        ampld0 = chunkAmpld0.mean(axis=0)
        cross0 = chunkCross0.mean(axis=0)
        coeff0 = chunkCoeff0.mean(axis=0)
        ampld1 = chunkAmpld1.mean(axis=0)
        cross1 = chunkCross1.mean(axis=0)
        coeff1 = chunkCoeff1.mean(axis=0)
        ampld2 = chunkAmpld2.mean(axis=0)
        cross2 = chunkCross2.mean(axis=0)
        coeff2 = chunkCoeff2.mean(axis=0)

        winNFT0.append(ampld0)
        winNFT0mask.append(ampld0.mask)
        winSpec0.append(cross0)
        winSpec0mask.append(cross0.mask)
        winCoeff0.append(coeff0)
        winCoeff0mask.append(coeff0.mask)
        winNFT1.append(ampld1)
        winNFT1mask.append(ampld1.mask)
        winSpec1.append(cross1)
        winSpec1mask.append(cross1.mask)
        winCoeff1.append(coeff1)
        winCoeff1mask.append(coeff1.mask)
        winNFT2.append(ampld2)
        winNFT2mask.append(ampld2.mask)
        winSpec2.append(cross2)
        winSpec2mask.append(cross2.mask)
        winCoeff2.append(coeff2)
        winCoeff2mask.append(coeff2.mask)


    if (dodiag):    # per-window diagnostic plots
        tag = 't%07d' % dt  # window offset as tag
        # ampld, phase, and coefficient plots
        for pt in range(3):
            if (pt == 0):
                ptype = 'ampld'
                nX = nInp
                nY = nInp
                y = 10.*np.log10(np.ma.abs(cross))  # in dB
                ya = 20.*np.log10(ampld)    # in dB
                if (winEigen>-1):
                    y0 = 10.*np.log10(np.ma.abs(cross0))
                    y1 = 10.*np.log10(np.ma.abs(cross1))
                    y2 = 10.*np.log10(np.ma.abs(cross2))
                    print('debug:', ampld0.shape, ampld.shape)
                    ya0 = 20.*np.log10(ampld0)
                    ya1 = 20.*np.log10(ampld1)
                    ya2 = 20.*np.log10(ampld2)
            elif (pt == 1):
                ptype = 'phase'
                nX = nInp-1
                nY = nInp-1
                y = np.ma.angle(cross)  # in rad
                if (winEigen>-1):
                    y0 = np.ma.angle(cross0)  # in rad
                    y1 = np.ma.angle(cross1)  # in rad
                    y2 = np.ma.angle(cross2)  # in rad
            else:
                ptype = 'coeff'
                nX = nInp-1
                nY = nInp-1
                y = np.ma.abs(coeff)    # dimensionless fraction
                if (winEigen>-1):
                    y0 = np.ma.abs(coeff0)    # dimensionless fraction
                    y1 = np.ma.abs(coeff1)    # dimensionless fraction
                    y2 = np.ma.abs(coeff2)    # dimensionless fraction
            ww = nX * 4
            hh = nY * 3

            png = '%s/%s_%s.png' % (cdir, ptype, tag)
            fig, sub2d = plt.subplots(nX, nY, figsize=(ww,hh), sharex=True, sharey=True, squeeze=False)
            for i in range(1,nY):
                for j in range(i):
                    sub2d[i,j].remove()
            sub = sub2d.flatten()

            if (pt == 0):   # ampld
                for ai in range(nInp):  # the auto
                    ax = sub2d[ai,ai]
                    ax.plot(freq, ya[ai])
                    if (winEigen>-1):
                        ax.plot(freq, ya0[ai])
                        ax.plot(freq, ya1[ai])
                        ax.plot(freq, ya2[ai])
                    ax.set_xlabel('freq (MHz)')
                    ax.set_ylabel('power (dB)')
                    ax.set_title(inpIdx[ai])
                b = -1
                for ai in range(nInp-1):
                    for aj in range(ai+1,nInp):
                        b += 1
                        ax = sub2d[ai,aj]
                        ax.plot(freq, y[b])
                        if (winEigen>-1):
                            ax.plot(freq, y0[b])
                            ax.plot(freq, y1[b])
                            ax.plot(freq, y2[b])
                        ax.set_title('%s-%s'%(inpIdx[ai], inpIdx[aj]))
            else:           # phase and coeff
                #for b in range(nBl):
                b = -1
                for ai in range(nInp-1):
                    for aj in range(ai+1, nInp):
                        b += 1
                        ax = sub2d[ai,aj-1]
                        ax.plot(freq, y[b])
                        if (winEigen>-1):
                            ax.plot(freq, y0[b])
                            ax.plot(freq, y1[b])
                            ax.plot(freq, y2[b])
                        ax.set_title('%s-%s'%(inpIdx[ai], inpIdx[aj]))
                for i in range(nY):
                    ax = sub2d[i,i]
                    ax.set_xlabel('freq (MHz)')
                    if (pt == 1):
                        ax.set_ylabel('phase (rad)')
                        ax.set_ylim(-3.3, 3.3)
                    elif (pt == 2):
                        ax.set_ylabel('coeff')

            fig.tight_layout(rect=[0,0.03,1,0.95])
            #sptitle = '%s, t=%dsec, %s' % (fin, dt, ptype)
            sptitle = '%s, t=%dsec, %s\n' % (ftpart, dt, ptype)
            for gi in range(nGrp):
                sptitle += '[%d]: %s ' % (gi, dirs[gi])
            fig.suptitle(sptitle)
            fig.savefig(png)
            plt.close(fig)

    ## cleaning up memory after each file
    del ampld, tmp, norm
    del cross, coeff, var
    gc.collect()

# save results
tsec   = np.array(tsec)
winNFT = np.ma.array(winNFT, mask=winNFTmask)
winVar = np.ma.array(winVar, mask=winVarmask)
winSpec = np.ma.array(winSpec, mask=winSpecmask)
winCoeff = np.ma.array(winCoeff, mask=winCoeffmask)

adoneh5(fout, freq, 'freq')         # RF freq in MHz, shape (nChan,)
adoneh5(fout, tsec, 'winSec')       # window offset in seconds, shape (nWin,)
adoneh5(fout, winNFT, 'winNFT')     # (supposedly normalized) FT spectra, shape (nWin, nInp, nChan)
adoneh5(fout, winVar, 'winVar')     # variance in time, shape (nWin, nBl, nChan)
adoneh5(fout, winSpec, 'winSpec')   # time-avg of cross-spectra, shape (nWin, nBl, nChan)
adoneh5(fout, winCoeff, 'winCoeff') # time-avg of correlation coefficient, shape (nWin, nBl, nChan)

attrs['dirs'] = dirs
attrs['inpIdx'] = inpIdx
putAttrs(fout, attrs)

if (winEigen>-1):
    winNFT0 = np.ma.array(winNFT0, mask=winNFT0mask)
    winSpec0 = np.ma.array(winSpec0, mask=winSpec0mask)
    winCoeff0 = np.ma.array(winCoeff0, mask=winCoeff0mask)
    adoneh5(fout, winNFT0, 'winNFT0')     # (supposedly normalized) FT spectra, shape (nWin, nInp, nChan)
    adoneh5(fout, winSpec0, 'winSpec0')   # time-avg of cross-spectra, shape (nWin, nBl, nChan)
    adoneh5(fout, winCoeff0, 'winCoeff0') # time-avg of correlation coefficient, shape (nWin, nBl, nChan)

    winNFT1 = np.ma.array(winNFT1, mask=winNFT1mask)
    winSpec1 = np.ma.array(winSpec1, mask=winSpec1mask)
    winCoeff1 = np.ma.array(winCoeff1, mask=winCoeff1mask)
    adoneh5(fout, winNFT1, 'winNFT1')     # (supposedly normalized) FT spectra, shape (nWin, nInp, nChan)
    adoneh5(fout, winSpec1, 'winSpec1')   # time-avg of cross-spectra, shape (nWin, nBl, nChan)
    adoneh5(fout, winCoeff1, 'winCoeff1') # time-avg of correlation coefficient, shape (nWin, nBl, nChan)

    winNFT2 = np.ma.array(winNFT2, mask=winNFT2mask)
    winSpec2 = np.ma.array(winSpec2, mask=winSpec2mask)
    winCoeff2 = np.ma.array(winCoeff2, mask=winCoeff2mask)
    adoneh5(fout, winNFT2, 'winNFT2')     # (supposedly normalized) FT spectra, shape (nWin, nInp, nChan)
    adoneh5(fout, winSpec2, 'winSpec2')   # time-avg of cross-spectra, shape (nWin, nBl, nChan)
    adoneh5(fout, winCoeff2, 'winCoeff2') # time-avg of correlation coefficient, shape (nWin, nBl, nChan)



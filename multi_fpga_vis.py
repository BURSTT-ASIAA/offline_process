#!/usr/bin/env python

from packet_func import *
import matplotlib.pyplot as plt
from loadh5 import *
from subprocess import call
from glob import glob
from astropy.time import Time
import gc, time

inp = sys.argv[0:]
pg  = inp.pop(0)

nAnt  = 16
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
#packBlock   = 1000000   # num of packets in a block
packBlock   = 102400   # num of packets in a block
verbose     = False
bitwidth    = 4
ppf         = 2
hdver       = 2
order_off   = 0
do_pcal     = False
feigs       = []
meta        = 64


pack0 = 0               # packet offset
nPack = 1000            # num of packets to read
fout  = 'analyze.vish5'
dodiag = True
no_bitmap = False


usage = '''
calculate cross-correlation of a single FPGA board

syntax:
    %s [options]

    options are:
    --pcal          # whether to perform phase correction
    -g <input> 'idx...' 
                    # <input_dir> should contain binary packets files to be analyzed
                    # each dir is expected to contain data from one FPGA board
                    # 'idx...' is a list of array indices to correlate
                    # or all for all sixteen antennas
                    #
                    # note: <input> can be a file (.bin), which will be processed
                    # however, if <input> is a folder, all .bin files within will be processed
                    #
                    # Note: if --pcal is set, a 3rd parameter is required after -g
                    # i.e. -g <input> 'idx...' <eigen.h5>
                    # with <eigen.h5> used to provide calibration info for this <input>

    -n nPack        # specify the number of packets to read and average per file
                    # defautl: %d
    --16bit         # specify that data is in 16bit format
                    # (default is in 4bit format)
    --hd VER        # header version
                    # (%d)
    --meta bytes    # the ring buffer or file metadata length in bytes
    --p0 pack0      # specify the pack offset
                    # disable autop0
    --blocklen blocklen
                    # specify the block length in packets
                    # (%d)
    --nB nBlock     # specify the length of the binary file in nBlock
                    # without this parameter, the bitmap can not be loaded
                    # default is to determine the nBlock automatically
    --no-aB         # disable auto-Block
    --ooff OFF      # set packet_order offset so that it starts from 0
    --flim fmin fmax# set the spectrum min/max freq in MHz
                    # (%.1f  %.1f)
    --no-bitmap     # ignore bitmap
    -o output_file  # specify the output file name
                    # default: %s
    -v              # enable more message

''' % (pg, nPack, hdver, packBlock, flim[0], flim[1], fout)

if (len(inp) < 1):
    sys.exit(usage)


while (inp):
    k = inp.pop(0)
    if (k == '-n'):
        nPack = int(inp.pop(0))
    elif (k == '--pcal'):
        do_pcal = True
    elif (k == '-g'):
        gdir = inp.pop(0)
        gdir = gdir.rstrip('/')
        tmp = inp.pop(0)
        if (tmp == 'all'):
            gidx = np.arange(nAnt)
        else:
            gidx = [int(x) for x in tmp.split()]
        dirs.append(gdir)
        grpIdx[gdir] = gidx
        # check do_pcal
        if (len(inp)>=1):
            if (inp[0].startswith('-')):
                do_pcal = False
        else:
            do_pcal = False
        if (do_pcal):
            feig = inp.pop(0)
            feigs.append(feig)
    elif (k == '--16bit'):
        bitwidth = 16
        ppf = 8
    elif (k == '--hd'):
        hdver = int(inp.pop(0))
    elif (k == '--meta'):
        meta = int(inp.pop(0))
    elif (k == '--p0'):
        pack0 = int(inp.pop(0))
        autop0 = False
    elif (k == '--blocklen'):
        packBlock = int(inp.pop(0))
    elif (k == '--nB'):
        nBlock = int(inp.pop(0))
    elif (k == '--no-aB'):
        autoblock = False
    elif (k == '--ooff'):
        order_off = int(inp.pop(0))
    elif (k == '--flim'):
        flim[0] = float(inp.pop(0))
        flim[1] = float(inp.pop(0))
    elif (k=='--no-bitmap'):
        no_bitmap = True
    elif (k == '-o'):
        fout = inp.pop(0)
    elif (k == '-v'):
        verbose = True
    elif (k.startswith('-')):
        sys.exit('unknown option: %s' % k)
    else:
        print('unused arg:', k)

specDur = 1.e-6/(flim[1]-flim[0])*nChan    # duration of one spectrum in seconds
freq = np.linspace(flim[0], flim[1], nChan, endpoint=False)

byteBMBlock = packBlock//8  # bitmap size of a block in bytes
byteBlock   = (bytePayload+byteHeader)*packBlock + byteBMBlock  # block size in bytes

if (fout.endswith('.vish5')):
    cdir = fout.replace('.vish5', '.check')
else:
    cdir = fout + '.check'
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
winCoeff2 = []
winCoeff2mask = []

grpFiles = []
#refV2 = []
refV2 = np.ma.array(np.zeros((nChan, nInp)), mask=False)
ii = -1
for gi in range(nGrp):
    idir = dirs[gi]
    if (os.path.isdir(idir)):
        files = glob('%s/*.bin' % idir)
        files.sort()
    elif (os.path.isfile(idir)):
        files = [idir]
    grpFiles.append(files)

    if (do_pcal):
        feig = feigs[gi]
        print('read cal:', feig)
        # eigenvectors from the scaled mode
        V2  = getData(feig, 'V2_scale') # shape: (nChan, nAnt, nMode)
        # bandpass normalization for the scaled mode
        N2  = getData(feig, 'N2_scale') # shape: (nAnt, nChan)
        N2.mask[N2==0] = True
        # take the leading eigenmode corresponding to the calibrator (at transit time)
        tmpV2  = V2[:,:,-1]         # averaged window, last mode
        tmpV2 /= np.abs(tmpV2)      # keep only the phase info

        gIdx = grpIdx[idir]
        for x in gIdx:
            ii += 1
            refV2[:,ii] = tmpV2[:,x]

nFile = len(grpFiles[0])
#nFile = 2   # override when debugging




t0 = time.time()

for fi in range(nFile):
    fin = grpFiles[0][fi]
    fbase = os.path.basename(fin)   # use 1st dir as reference

    tmp = fbase.split('.')
    ftpart = tmp[1]
    if (len(ftpart)==10):
        ftstr = '23'+ftpart # hard-coded to 2023!!
    elif (len(ftpart)==14):
        ftstr = ftpart[2:]
    ftime = datetime.strptime(ftstr, '%y%m%d%H%M%S')
    if (ftime0 is None):
        ftime0 = ftime
        unix0 = Time(ftime0, format='datetime').to_value('unix')    # local time
        unix0 -= 3600.*8.                                           # convert to UTC
        attrs['unix_utc_open'] = unix0

    dt = (ftime - ftime0).total_seconds()
    print('(%d/%d)'%(fi+1,nFile), fin, 'dt=%dsec'%dt)

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

    fin = files[0]  # any file should be the same size
    if (nBlock is None and autoblock):
        fsize = os.path.getsize(fin)
        nBlock = np.rint(fsize/byteBlock).astype(int)
        print('autoblock:', nBlock)
    if (nBlock == 0):
        print('bitmap info not used.')

    ## determine the correct p0 for all files involved
    if (autop0):
        grp_p0 = [pack0 for i in range(nGrp)]      # initialize p0 for this group of files
        concensus = False
        while (not concensus):
            for gi in range(nGrp):
                gidx = grpIdx[dirs[gi]]
                fin = files[gi]

                fh = open(fin, 'rb')
                fullBM = loadFullbitmap(fh, nBlock, blocklen=packBlock, meta=meta)
                fh.close()

                ap = True
                p0 = grp_p0[gi]
                while (ap):
                    bitmap = fullBM[p0:p0+nPack]
                    fvalid = float(np.count_nonzero(bitmap)/nPack)
                    if (fvalid < 0.1):  # arbitrary threshold
                        p0 += packBlock
                        if (p0 >= len(fullBM)):
                            grp_p0[gi] = -1
                            ap = False
                    else:
                        ap = False
                        grp_p0[gi] = p0

            print(grp_p0)
            if (np.any(np.array(grp_p0)==-1)):
                print('... autop0 failed. fallback to default pack0')
                p0 = pack0
                break

            #concensus = np.all(grp_p0)
            concensus = np.allclose(grp_p0, np.median(grp_p0))
            if (concensus):
                # satisfied with this answer
                p0 = grp_p0[0]
            else:
                # update to highest block for all files and try again
                p0 = max(grp_p0)
                grp_p0 = [p0 for i in range(nGrp)]

    else:   # autop0=False
        p0 = pack0

    if (verbose):
        print('pack0:', p0)

    antspec = np.ma.array(np.zeros((nPack//ppf, nInp, nChan), dtype=complex), mask=True)
    validGrp = True
    ai = -1
    for gi in range(nGrp):
        gidx = grpIdx[dirs[gi]]
        fin = files[gi]
        fh = open(fin, 'rb')

        t1 = time.time()
        BM = loadFullbitmap(fh, nBlock, blocklen=packBlock, meta=meta)
        bitmap = BM[p0:p0+nPack]
        if (no_bitmap):
            bitmap = np.ones(nPack, dtype=bool)
        fvalid = float(np.count_nonzero(bitmap)/nPack)
        if (fvalid < 0.1):  # arbitrary threshold
            print('no valid block. skip.')
            validGrp = False
            fh.close()
            break   # break the loop of files in the group

        print('elapsed time: %.3fsec'%(t1-t0))
        #tick, tmpspec = loadSpec(fh, p0, nPack, nBlock=nBlock, verbose=verbose)
        #tick, tmpspec = loadSpec(fh, p0, nPack, nBlock=nBlock, verbose=verbose, bitwidth=bitwidth)
        tick, tmpspec = loadSpec(fh, p0, nPack, order_off=order_off, bitmap=bitmap, verbose=verbose, bitwidth=bitwidth, hdver=hdver, meta=meta)
        t2 = time.time()
        print('   %d packets loaded in %.2fsec'%(nPack, t2-t1))
        fh.close()

        for idx in gidx:
            ai += 1
            if (verbose):
                print(gi, dirs[gi], idx, '-->', ai)
            antspec[:,ai] = tmpspec[:,idx]

        del tick, tmpspec
        gc.collect()

    if (validGrp):
        tsec.append(dt)
    else:
        del antspec
        gc.collect()
        continue    # skip the current window
    

    ## phase cal
    if (do_pcal):
        antspec /= refV2.T.reshape((1,nInp,nChan))


    #ampld = np.ma.abs(antspec).mean(axis=0)  # shape (nInp, nChan)
    ampld = np.ma.array(np.zeros((nInp, nChan)), mask=False)
    for ai in range(nInp):
        ampld[ai] = np.ma.abs(antspec[:,ai,:]).mean(axis=0)
    winNFT.append(ampld)
    winNFTmask.append(ampld.mask)

    cross = np.ma.array(np.zeros((nBl, nChan), dtype=complex), mask=False)
    var   = np.ma.array(np.zeros((nBl, nChan)), mask=False)
    coeff = np.ma.array(np.zeros((nBl, nChan), dtype=complex), mask=False)
    coeff2 = np.ma.array(np.zeros((nBl, nChan), dtype=complex), mask=False)
    b = -1
    for ai in range(nInp-1):
        for aj in range(ai+1, nInp):
            b += 1
            tmp  = antspec[:,ai,:] * antspec[:,aj,:].conjugate()
            var[b] = tmp.var(axis=0)
            cross[b] = tmp.mean(axis=0)
            # the old method, more accurate, switch back to this method
            norm2 = np.ma.abs(antspec[:,ai,:])*np.ma.abs(antspec[:,aj,:])
            coeff[b] = (tmp/norm2).mean(axis=0)
            # the new method, not good, >1 for good correlation
            norm = np.ma.abs(antspec[:,ai,:]).mean(axis=0) * np.ma.abs(antspec[:,aj,:]).mean(axis=0)
            coeff2[b] = cross[b]/norm


    winVar.append(var)
    winVarmask.append(var.mask)
    winSpec.append(cross)
    winSpecmask.append(cross.mask)
    winCoeff.append(coeff)
    winCoeffmask.append(coeff.mask)
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
            elif (pt == 1):
                ptype = 'phase'
                nX = nInp-1
                nY = nInp-1
                y = np.ma.angle(cross)  # in rad
            else:
                ptype = 'coeff'
                nX = nInp-1
                nY = nInp-1
                y = np.ma.abs(coeff)    # dimensionless fraction
            ww = nX * 4
            hh = nY * 3

            png = '%s/%s_%s.png' % (cdir, ptype, tag)
            fig, sub2d = plt.subplots(nX, nY, figsize=(ww,hh), sharex=True, sharey=True, squeeze=False)
            for i in range(1,nY):
                for j in range(i):
                    sub2d[i,j].remove()
            sub = sub2d.flatten()

            if (pt == 0):   # ampld
                y1 = 20.*np.log10(ampld)    # in dB
                for ai in range(nInp):  # the auto
                    ax = sub2d[ai,ai]
                    ax.plot(freq, y1[ai])
                    ax.set_xlabel('freq (MHz)')
                    ax.set_ylabel('power (dB)')
                    ax.set_title(inpIdx[ai])
                b = -1
                for ai in range(nInp-1):
                    for aj in range(ai+1,nInp):
                        b += 1
                        ax = sub2d[ai,aj]
                        ax.plot(freq, y[b])
                        ax.set_title('%s-%s'%(inpIdx[ai], inpIdx[aj]))
            else:           # phase and coeff
                #for b in range(nBl):
                b = -1
                for ai in range(nInp-1):
                    for aj in range(ai+1, nInp):
                        b += 1
                        ax = sub2d[ai,aj-1]
                        ax.plot(freq, y[b])
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
winCoeff2 = np.ma.array(winCoeff2, mask=winCoeff2mask)

adoneh5(fout, freq, 'freq')         # RF freq in MHz, shape (nChan,)
adoneh5(fout, tsec, 'winSec')       # window offset in seconds, shape (nWin,)
adoneh5(fout, winNFT, 'winNFT')     # (supposedly normalized) FT spectra, shape (nWin, nInp, nChan)
adoneh5(fout, winVar, 'winVar')     # variance in time, shape (nWin, nBl, nChan)
adoneh5(fout, winSpec, 'winSpec')   # time-avg of cross-spectra, shape (nWin, nBl, nChan)
adoneh5(fout, winCoeff2, 'winCoeff2') # new correlation coefficient, shape (nWin, nBl, nChan)
adoneh5(fout, winCoeff, 'winCoeff')   # time-avg of correlation coefficient, shape (nWin, nBl, nChan)
                                      # the new winCoeff will be >1 for some good baselines
                                      # so change the default winCoeff back to the original method

attrs['dirs'] = dirs
attrs['inpIdx'] = inpIdx
putAttrs(fout, attrs)

t4 = time.time()
print('finished in %.3fsec'%(t4-t0))

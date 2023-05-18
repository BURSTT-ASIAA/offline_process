#!/usr/bin/env python

from packet_func import *
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
inpIdx = []
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

pack0 = 0               # packet offset
nPack = 1000            # num of packets to read
fout  = 'analyze.vish5'
dodiag = True
specDur = 1.e-6/(flim[1]-flim[0])*nChan    # duration of one spectrum in seconds


usage = '''
calculate cross-correlation of a single FPGA board

syntax:
    %s <input_dir> [options]

    <input_dir> should contain binary packets files to be analyzed

    options are:
    --idx 'i j k ...'   # specify only a few inputs to correlate
                    # by their array index (idx number)

    -n nPack        # specify the number of packets to read and average per file
                    # defautl: %d

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
    elif (k == '--idx'):
        tmp = inp.pop(0)
        inpIdx = [int(x) for x in tmp.split()]
    elif (k == '--p0'):
        pack0 = int(inp.pop(0))
        autop0 = False
    elif (k == '--nB'):
        nBlock = int(inp.pop(0))
    elif (k == '--no-aB'):
        autoblock = False
    elif (k == '-v'):
        verbose = True
    elif (k.startswith('-')):
        sys.exit('unknown option: %s' % k)
    else:
        idir = k

freq = np.linspace(flim[0], flim[1], nChan, endpoint=False)

cdir = 'check'  # for diagnostics
call('mkdir -p %s'%cdir, shell=True)
#odir = 'output' # for results
#call('mkdir -p %s'%odir, shell=True)

#files = [fin]   # placeholder
files = glob('%s/*.fpga.bin' % idir)
files.sort()
nFile = len(files)
#nFile = 3   # override when debugging

winDur = nPack//2 * specDur # window duration in seconds

if (len(inpIdx) == 0):
    inpIdx = [0, 1] # default to idx 0, 1 if nothing specified
nInp = len(inpIdx)
nBl  = nInp*(nInp-1)//2

info = '''
=== info ===
nInp = %d
nPack = %d
nFile = %d
output file: %s
diagnostics: %s
winDur = %.6f sec
============
''' % (nInp, nPack, nFile, fout, cdir, winDur)
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
for fi in range(nFile):
    fin = files[fi]
    if (not os.path.isfile(fin)):
        print('file not found:', fin)
        continue

    if (nBlock is None and autoblock):
        fsize = os.path.getsize(fin)
        nBlock = np.rint(fsize/byteBlock).astype(int)
        print('autoblock:', nBlock)

    if (nBlock == 0):
        print('bitmap info not used.')

    fh = open(fin, 'rb')
    fullBM = loadFullbitmap(fh, nBlock)
    p0 = pack0
    ap = autop0
    while (ap):
        bitmap = fullBM[p0:p0+nPack]
        fvalid = float(np.count_nonzero(bitmap)/nPack)
        if (fvalid < 0.1):  # arbitrary threshold
            p0 += packBlock
        else:
            ap = False
    if (verbose):
        print('pack0:', p0)

    t1 = time.time()
    tick, antspec = loadSpec(fh, p0, nPack, nBlock=nBlock, verbose=verbose)
    t2 = time.time()
    print('   %d packets loaded in %.2fsec'%(nPack, t2-t1))
    fh.close()

    tmp = fin.split('.')
    ftstr = '23'+tmp[1]
    ftime = datetime.strptime(ftstr, '%y%m%d%H%M%S')
    if (ftime0 is None):
        ftime0 = ftime
        unix0 = Time(ftime0, format='datetime').to_value('unix')    # local time
        unix0 -= 3600.*8.                                           # convert to UTC
        attrs['unix_utc_open'] = unix0

    dt = (ftime - ftime0).total_seconds()
    tsec.append(dt)
    print('(%d/%d)'%(fi+1,nFile), fin, 'dt=%dsec'%dt)

    #ampld = np.ma.abs(antspec).mean(axis=0)  # shape (nInp, nChan)
    ampld = np.ma.array(np.zeros((nInp, nChan)), mask=False)
    for ai in range(nInp):
        ampld[ai] = np.ma.abs(antspec[:,ai,:]).mean(axis=0)

    cross = np.ma.array(np.zeros((nBl, nChan), dtype=complex), mask=False)
    var   = np.ma.array(np.zeros((nBl, nChan)), mask=False)
    coeff = np.ma.array(np.zeros((nBl, nChan), dtype=complex), mask=False)
    b = -1
    for ii in range(nInp-1):
        ai = inpIdx[ii]
        for jj in range(ii+1, nInp):
            aj = inpIdx[jj]
            b += 1
            tmp  = antspec[:,ai,:] * antspec[:,aj,:].conjugate()
            norm = np.ma.abs(antspec[:,ai,:])*np.ma.abs(antspec[:,aj,:])
            norm.mask[norm==0.] = True  # mask zero auto-correlations
            var[b] = tmp.var(axis=0)
            cross[b] = tmp.mean(axis=0)
            coeff[b] = (tmp/norm).mean(axis=0)


    winNFT.append(ampld)
    winNFTmask.append(ampld.mask)
    winVar.append(var)
    winVarmask.append(var.mask)
    winSpec.append(cross)
    winSpecmask.append(cross.mask)
    winCoeff.append(coeff)
    winCoeffmask.append(coeff.mask)


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
                    ax = sub2d[ai-1,ai-1]
                    ax.plot(freq, y1[ai])
                    ax.set_xlabel('freq (MHz)')
                    ax.set_ylabel('power (dB)')
                b = -1
                for ai in range(nInp-1):
                    for aj in range(ai+1,nInp):
                        b += 1
                        ax = sub2d[ai,aj]
                        ax.plot(freq, y[b])
            else:           # phase and coeff
                for b in range(nBl):
                    ax = sub[b]
                    ax.plot(freq, y[b])
                for i in range(nY):
                    for j in range(nX):
                        ax = sub2d[i,j]
                        ax.set_xlabel('freq (MHz)')
                        if (pt == 1):
                            ax.set_ylabel('phase (rad)')
                            ax.set_ylim(-3.3, 3.3)
                        elif (pt == 2):
                            ax.set_ylabel('coeff')

            fig.tight_layout(rect=[0,0.03,1,0.95])
            sptitle = '%s, t=%dsec, %s' % (fin, dt, ptype)
            fig.suptitle(sptitle)
            fig.savefig(png)
            plt.close(fig)

    ## cleaning up memory after each file
    del tick, antspec
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

putAttrs(fout, attrs)

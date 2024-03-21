#!/usr/bin/env python
## standard libs
import sys, os.path
import time, gc
from glob import glob
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.stats import sigma_clip
## ky's libs
from packet_func import *       # loadNode, 
from calibrate_func import *    # makeCov, Cov2Eig
from loadh5 import *            # getData, adoneh5, getAttrs, putAttrs
from pyplanet import *          # obsBody, loadDB

DB = loadDB()

inp = sys.argv[0:]
pg  = inp.pop(0)

flim    = [400., 800.]
nAnt    = 16
nChan   = 1024
nFPGA   = 4     # for 64-ant
nChan2  = 512   # for 64-ant
bitwidth = 4    # for 64-ant
blocklen = 256000 # for 64-ant
nBlock  = 1
autoblock = True
nPack   = 10000
p0      = 0
hdver   = 2
meta    = 64
order_off = 0
fout    = 'output.eigen.h5'
user_fout = False
no_bitmap = False
hdlen   = 64
paylen  = 8192
redo    = False
arr_config  = '16x1.0_4x0.5'    # 64-ant, 16x4, 0.5m sep between rows
do_model = True                 # always save the geometric delay to the Sun
body    = 'sun'
site    = 'fushan6'
do_scale = False
do_coeff = True

ant_flag = []
nFlag   = 0

usage   = '''
compute covariance matrix and save the eigenmodes from a FPGA binary file

syntax:
    %s <bin file(s)> [options]

options are:
    -n nPack        # read nPack from the binary file (%d)
    --p0 p0         # starting packet offset (%d)
    --blocklen blocklen
                    # change the packet number per block
                    # (default: %d)
    --nB nB         # specify the block length of the binary data
                    # (default: determined by autoblock %d)
    -o fout         # specify an output file name
                    # (default: %s)
    --flag 'ant(s)' # specify the input number (0--15) to be flagged
    --hd VER        # header version (1, 2)
                    # (default: %d)
    --meta bytes    # number of bytes in the ring buffer or file metadata
                    # ring buffer: 128 bytes
                    # file: 64 bytes
                    # (default: %d)
    --redo          # re-generate eigenmodes
                    # (default is to plot existing eigenmodes)
    --array <CONFIG>
                    # specify the array config (predefined or a config filename)
                    # (default: %s)
    --body <BODY>   # set the target to calculate geometric delay
                    # (default: %s)
    --site <SITE>   # specify the site (pre-defined sites)
                    # (default: %s)

    (special)
    --no-bitmap     # ignore the bitmap
    --4bit          # read 4-bit data
    --ooff OFF      # offset added to the packet_order

''' % (pg, nPack, p0, blocklen, nBlock, fout, hdver, meta, arr_config, body, site)

if (len(inp) < 1):
    sys.exit(usage)

files0 = []
while (inp):
    k = inp.pop(0)
    if (k == '-n'):
        nPack = int(inp.pop(0))
    elif (k == '--p0'):
        p0 = int(inp.pop(0))
    elif (k == '--blocklen'):
        blocklen = int(inp.pop(0))
    elif (k == '--nB'):
        nBlock = int(inp.pop(0))
    elif (k == '-o'):
        user_fout = True
        fout = inp.pop(0)
    elif (k == '--flag'):
        tmp = inp.pop(0).split()
        ant_flag = [int(x) for x in tmp]
        print('ant_flag:', ant_flag)
        nFlag = len(ant_flag)
    elif (k == '--no-bitmap'):
        no_bitmap = True
    elif (k == '--4bit'):
        bitwidth = 4
    elif (k == '--hd'):
        hdver = int(inp.pop(0))
    elif (k == '--meta'):
        meta = int(inp.pop(0))
    elif (k == '--ooff'):
        order_off = int(inp.pop(0))
    elif (k == '--redo'):
        redo = True
    elif (k == '--array'):
        arr_config = inp.pop(0)
    elif (k == '--body'):
        body = inp.pop(0)
    elif (k == '--site'):
        site == inp.pop(0)
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        files0.append(k)

# for autoblock
byteBlockBM = blocklen//8
byteBlock = (hdlen + paylen)*blocklen + byteBlockBM
# frequency in MHz
freq = np.linspace(flim[0], flim[1], nChan, endpoint=False)

if (arr_config == '16x1.0_4x0.5'):
    pos = np.zeros((nFPGA*nAnt,3))
    ai = -1
    for j in range(nFPGA):
        y = 0.5*j
        for i in range(nAnt):
            x = 1.0*i
            ai += 1
            pos[ai] = [x,y,0.]
else:
    if (os.path.isfile(arr_config)):
        pos = np.loadtxt(arr_config)
    else:
        sys.exit('unknown array config file: %s'%arr_config)

# for antenna-based solution, only need to calculate the geometric delay
# to a fixed reference (i.e. 0-th antenna)
BVec = pos - pos[0]


nLoop = 1
loop_files = [files0]

t00 = time.time()

for ll in range(nLoop):
    files = loop_files[ll]

    nFile = len(files)
    if (not user_fout):
        if (nFile == 1):   # override default fout if input is a single file
            fout = files[0] + '.eigen.h5'
    print('eigenmode is saved in:', fout, '...')

    if (os.path.isfile(fout) and not redo):
        attrs = getAttrs(fout)
        tsec  = getData(fout, 'winSec')
        if (do_scale):
            savN2 = getData(fout, 'N2_scale')
            savW2 = getData(fout, 'W2_scale')
            savV2 = getData(fout, 'V2_scale')
        if (do_coeff):
            savN3 = getData(fout, 'N3_coeff')
            savW3 = getData(fout, 'W3_coeff')
            savV3 = getData(fout, 'V3_coeff')

    else:   # redo or file not exist
        attrs = {}
        attrs['bitwidth'] = bitwidth
        attrs['nFPGA'] = nFPGA
        attrs['nPack'] = nPack
        attrs['p0'] = p0
        attrs['filename'] = files
        attrs['nAnt'] = nAnt
        attrs['nChan'] = nChan

        ftime0 = None
        tsec = []
        savW2 = []  # 2 for scaled
        savV2 = []
        savN2 = []  # normalization used to scale the data
        savN2mask = []  # normalization used to scale the data
        savW3 = []  # 3 for coeff
        savV3 = []
        savN3 = []  # normalization used to scale the data
        savN3mask = []  # normalization used to scale the data

        nFrame = nPack//4   # 64-ant
        # note, the shape is after transpose
        spec = np.ma.array(np.zeros((nFPGA,nAnt,nFrame,nChan), dtype=complex), mask=True)

        t0 = time.time()

        ii = -1
        for i in range(nFile):
            print(i, files[i])
            ii += 1

            fbin = files[i]

            fbase = os.path.basename(fbin)   # use 1st dir as reference
            tmp = fbase.split('.')
            ftpart = tmp[1]
            if (len(ftpart)==10):
                ftstr = '23'+ftpart # hard-coded to 2023!!
            elif (len(ftpart)==14):
                ftstr = ftpart[2:]  # strip leading 20
            ftime = datetime.strptime(ftstr, '%y%m%d%H%M%S')
            if (ftime0 is None):
                ftime0 = ftime
                unix0 = Time(ftime0, format='datetime').to_value('unix')    # local time
                unix0 -= 3600.*8.                                           # convert to UTC
                attrs['unix_utc_open'] = unix0

            dt = (ftime - ftime0).total_seconds()
            print('(%d/%d)'%(ii+1,nFile), fbin, 'dt=%dsec'%dt)

            if (i==0):
                ooff = -2
                tsec.append(dt)
            elif (i==1):
                ooff = -3
            fh = open(fbin, 'rb')
            # spec0.shape = (nFPGA, nFrame, nAnt, nChan)
            spec0 = loadNode(fh, p0, nPack, order_off=ooff, blocklen=blocklen, bitwidth=bitwidth, verbose=1)
            fh.close()

            # trasposed shape = (nFPGA, nAnt, nFrame, nChan)
            spec[:,:,:,int(nChan2*i):int(nChan2*(i+1))] = spec0.transpose((0,2,1,3))
            del spec0
            gc.collect()
            t1 = time.time()
            print('... data', i, 'loaded. elapsed:', t1-t0)

        spec = spec.reshape((-1,nFrame,nChan))
        ## new shape (nFile*nAnt, nFrame, nChan), only 3 axes

        if (do_scale):
            Cov2, norm2 = makeCov(spec, scale=True, coeff=False, bandpass=True, ant_flag=ant_flag)
            #Cov2, norm2 = makeCov(spec, scale=True, coeff=False, bandpass=False, ant_flag=ant_flag)
            W2, V2 = Cov2Eig(Cov2)
            savW2.append(W2)
            savV2.append(V2)
            savN2.append(norm2)
            savN2mask.append(norm2.mask)
        if (do_coeff):
            Cov3, norm3 = makeCov(spec, scale=False, coeff=True, ant_flag=ant_flag, nPool=16)
            W3, V3 = Cov2Eig(Cov3)
            savW3.append(W3)
            savV3.append(V3)
            savN3.append(norm3)
            savN3mask.append(norm3.mask)

        #Vlast[ii] = V[:,:,nAnt-1]
        t2 = time.time()
        print('... eigenmode got. elapsed:', t2-t0)

        print('files loaded')
        tsec  = np.array(tsec)
        adoneh5(fout, tsec, 'winSec')
        putAttrs(fout, attrs)
        if (do_scale):
            savW2 = np.array(savW2).mean(axis=0)
            savV2 = np.array(savV2).mean(axis=0)
            savN2 = np.ma.array(savN2, mask=savN2mask).mean(axis=0)
            adoneh5(fout, savN2, 'N2_scale')    # shape (nAnt, nChan)
            adoneh5(fout, savW2, 'W2_scale')    # shape (nChan, nMode)
            adoneh5(fout, savV2, 'V2_scale')    # shape (nChan, nAnt, nMode)
        if(do_coeff):
            savW3 = np.array(savW3).mean(axis=0)
            savV3 = np.array(savV3).mean(axis=0)
            savN3 = np.ma.array(savN3, mask=savN3mask).mean(axis=0)
            adoneh5(fout, savN3, 'N3_coeff')
            adoneh5(fout, savW3, 'W3_coeff')
            adoneh5(fout, savV3, 'V3_coeff')



    if (do_model):
        ## calculate geometric delay
        uttime = tsec + attrs['unix_utc_open']
        dtime = Time(uttime, format='unix').to_datetime()
        b, obs = obsBody(body, time=dtime[0], site=site, retOBS=True, DB=DB)

        az = []
        el = []
        for ti in dtime:
            obs.date = ti
            b.compute(obs)
            az.append(b.az)
            el.append(b.alt)
        az = np.array(az)
        el = np.array(el)
        phi    = np.pi/2. - az
        theta  = np.pi/2. - el
        unitVec  = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], ndmin=2).T
        unitVec *= -1

        # BVec.shape = (nFPGA*nAnt, 3)
        # unitVec.shape = (nTime, 3)
        tauGeo = np.tensordot(BVec, unitVec, axes=(1,1))  # delay in meters, shape (nFPGA*nAnt, nTime)
        tauGeo /= 2.998e8   # delay in sec.

        adoneh5(fout, tauGeo, 'tauGeo')
        attrs2 = {'body':body, 'site':site}
        putAttrs(fout, attrs2, dest='tauGeo')


    (nChan3, nAnt3, nMode3) = savV3.shape
    nCol = nAnt3//nAnt

    ## plot normalization and eigenvalues
    fig, sub = plt.subplots(3,1,figsize=(15,15),sharex=True)

    # normalization
    ax = sub[0]
    for ai in range(nAnt3):
        ax.plot(freq, savN3[ai], label='Ant%d'%ai)
    ax.set_yscale('log')
    ax.set_ylabel('voltage normalization')
    ax.legend(ncols=nCol)

    # eigenvalues
    ax = sub[1]
    for ai in range(nMode3):  # nMode = nAnt
        if (ai < nFlag):    # skip first n (weakest) modes correspondsng to the flagged antennas
            continue
        y = savW3[:,ai]
        y2 = sigma_clip(10.*np.log10(y), sigma=10)
        #ax.plot(freq, 10.*np.log10(y), label='Mode%d'%ai)
        ax.plot(freq, y2, label='Mode%d'%ai)
    ax.set_ylabel('power (dB)')
    ax.legend(ncols=nCol)

    # eigenvector of leading mode, phase
    ax = sub[2]
    for ai in range(nAnt3):
        if (ai < nFlag):
            continue
        y = savV3[:,ai,-1]
        ax.plot(freq, np.ma.angle(y), label='Ant%d'%ai)
    ax.set_ylabel('phase (rad)')
    ax.legend(ncols=nCol)
    ax.set_xlabel('freq (MHz)')
    ax.set_xlim(flim[0], flim[1]*1.25)

    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.suptitle(fout)
    fig.savefig('%s.png'%fout)
    plt.close(fig)


    t3 = time.time()
    print('... all done. elapsed:', t3-t00)

#!/usr/bin/env python

import sys, os.path
import time, gc
from glob import glob
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.stats import sigma_clip

from packet_func import *
from calibrate_func import *
from loadh5 import *
from pyplanet import *
from delay_func2 import *

DB = loadDB()

inp = sys.argv[0:]
pg  = inp.pop(0)

bitwidth = 16
nAnt    = 16
nChan   = 1024
flim    = [400., 800.]
blocklen = 128000
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
combine = False
redo    = False
nPool   = 4
do_model = True
body    = 'sun'
site    = 'fushan6'
arr_config = '16x1.0y0.5'
rows    = None
aref    = None
theta_rot = None


ant_flag = []
nFlag   = 0

usage   = '''
compute covariance matrix and save the eigenmodes from a FPGA binary file
if multiple files are given, two modes are available:
    the default mode is to process each file (16-element) independently
    the combine mode will group the files into a larger array 
        (nFile*nAnt-element) for covariance computation

syntax:
    %s <bin file(s)> [options]

options are:
    -n nPack    # read nPack from the binary file (%d)

    --p0 p0     # starting packet offset (%d)

    --blocklen blocklen
                # change the packet number per block
                # (default: %d)
    --nB nB     # specify the block length of the binary data
                # (default: determined by autoblock %d)
    -o fout     # specify an output file name
                # (default: %s)
    --flag 'ant(s)'
                # specify the input number (0--nAnt*nFPGA) to be flagged
    --hd VER    # header version (1, 2)
                # (default: %d)
    --meta bytes # number of bytes in the ring buffer or file metadata
                # ring buffer: 128 bytes
                # file: 64 bytes
                # (default: %d)
    --combine   # combine the bin files to for a larger covariance matrix
    --array <CONFIG>
                # specify the array config (predefined or a config filename)
                # (default: %s)
    --flim LOW HIGH
                # specify the frequency range in MHz
                # (%.0f %.0f)
    --rows 'rows'
                # specify the rows of the files in the --combine mode
                # quote the numbers, separate with spaces
                # default to the first nFile rows
    --aref aref # choose a reference antenna for geometric delay calc
                # default to the array origin
    --body <BODY>   # set the target to calculate geometric delay
                    # (default: %s)
    --site <SITE>   # specify the site (pre-defined sites)
                    # (default: %s)
    --rot theta_rot # the array misalignment angle in deg
                    # default to 0.0 if a config file is provided
                    # or the predefined value of the site
                    # (fushan6 default to -3.0)
                    # (longtien default to +0.5)
                    # 0.0 if it is not defined yet

    --redo      # re-generate eigenmodes
                # (default is to plot existing eigenmodes)
    --pool nPool
                # number of threads used to parallel process makeCov
                # (default: %d)

    (special)
    --no-bitmap # ignore the bitmap
    --4bit      # read 4-bit data
    --ooff OFF  # offset added to the packet_order

''' % (pg, nPack, p0, blocklen, nBlock, fout, hdver, meta, arr_config, flim[0], flim[1], body, site, nPool)

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
        hdver = 2   # override
    elif (k == '--ooff'):
        order_off = int(inp.pop(0))
    elif (k == '--combine'):
        combine = True
    elif (k == '--array'):
        arr_config = inp.pop(0)
    elif (k == '--rows'):
        tmp = inp.pop(0)
        rows = [int(x) for x in tmp.split()]
    elif (k == '--aref'):
        aref = int(inp.pop(0))
    elif (k == '--body'):
        body = inp.pop(0)
    elif (k == '--site'):
        site = inp.pop(0)
    elif (k == '--rot'):
        theta_rot = float(inp.pop(0))
    elif (k == '--flim'):
        flim[0] = float(inp.pop(0))
        flim[1] = float(inp.pop(0))
    elif (k == '--redo'):
        redo = True
    elif (k == '--pool'):
        nPool = int(inp.pop(0))
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        files0.append(k)

# for autoblock
byteBlockBM = blocklen//8
byteBlock = (hdlen + paylen)*blocklen + byteBlockBM
# frequency in MHz
freq = np.linspace(flim[0], flim[1], nChan, endpoint=False)

if (site == 'fushan6'):
    if (theta_rot is None):
        theta_rot = -3.0    # sujin's number
        #theta_rot = -1.8    # old number
elif (site == 'longtien'):
    if (arr_config == '16x1.0y0.5'):    # the default
        arr_config = '16x1.0y2.0'
        if (theta_rot is None):
            theta_rot = 0.5
if (theta_rot is None):
    theta_rot = 0.
print('using theta_rot:', theta_rot)


if (combine):
    nLoop = 1
    loop_files = [files0]
else:
    nLoop = len(files0)
    loop_files = [[x] for x in files0]


t00 = time.time()

for ll in range(nLoop):
    files = loop_files[ll]

    nFile = len(files)
    if (not user_fout):
        if (nFile == 1):   # override default fout if input is a single file
            fout = files[0] + '.eigen.h5'
    cdir = '%s.check'%fout
    if (not os.path.isdir(cdir)):
        call('mkdir %s'%cdir, shell=True)
    print('eigenmode is saved in:', fout, '...')


    pos = arrayConf(arr_config, nFile, rows=rows, theta_rot=theta_rot)
    print('debug:', 'pos.shape=',pos.shape, pos)
    if (aref is None):
        BVec = pos
    else:
        BVec = pos - pos[0]


    if (os.path.isfile(fout) and not redo):
        attrs = getAttrs(fout)
        savN2 = getData(fout, 'N2_scale')
        savW2 = getData(fout, 'W2_scale')
        savV2 = getData(fout, 'V2_scale')
        savN3 = getData(fout, 'N3_coeff')
        savW3 = getData(fout, 'W3_coeff')
        savV3 = getData(fout, 'V3_coeff')
        tsec  = getData(fout, 'winSec')
        #tauGeo = getData(fout, 'tauGeo')

    else:   # redo or file not exist
        attrs = {}
        attrs['bitwidth'] = bitwidth
        attrs['nPack'] = nPack
        attrs['p0'] = p0
        attrs['filename'] = files
        attrs['nAnt'] = nAnt
        attrs['nChan'] = nChan
        attrs['nFPGA'] = nFile

        ftime0 = None
        tsec = []
        savW2 = []  # 2 for scaled
        savV2 = []
        savN2 = []  # normalization used to scale the data
        savN2mask = []  # normalization used to scale the data
        savW3 = []  # 3 for coeff
        savV3 = []
        savN3 = []
        savN3mask = []

        if (bitwidth==4):
            nFrame = nPack // 2
        elif (bitwidth==16):
            nFrame = nPack // 8
        # note, the shape is after transpose
        spec = np.ma.array(np.zeros((nFile,nAnt,nFrame,nChan), dtype=complex), mask=True)

        t0 = time.time()

        localip = []
        dest_ip = []
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
            tsec.append(dt)

            if (autoblock):
                fsize = os.path.getsize(fbin)
                nBlock = np.rint((fsize-meta)/byteBlock).astype(int)
                print('auto block, nB:', nBlock)
                autoblock=False

            fh = open(fbin, 'rb')
            if (hdver == 2):
                fh.seek(meta)
                hbuf = fh.read(hdlen)
                tmp = decHeader2(hbuf, ip=True)
                localip.append(tmp[5])
                dest_ip.append(tmp[6])

            BM = loadFullbitmap(fh, nBlock, blocklen=blocklen, meta=meta)
            bitmap = BM[p0:p0+nPack]
            if (no_bitmap):
                bitmap = np.ones(nPack, dtype=bool)
            # spec0.shape = (nFrame, nAnt, nChan)
            tick, spec0 = loadSpec(fh, p0, nPack, bitmap=bitmap, bitwidth=bitwidth, hdver=hdver, order_off=order_off, verbose=1, meta=meta)
            #tick, spec0 = loadSpec(fh, p0, nPack, bitmap=bitmap, bitwidth=bitwidth, verbose=1)
            #tick, spec0 = loadSpec(fh, p0, nPack, nBlock=nBlock, bitwidth=bitwidth, verbose=1)
            fh.close()

            # trasposed shape = (nAnt, nFrame, nChan)
            spec[i] = spec0.transpose((1,0,2))
            del tick, spec0
            gc.collect()
            t1 = time.time()
            print('... data', i, 'loaded. elapsed:', t1-t0)

        spec = spec.reshape((-1,nFrame,nChan))
        ## new shape (nFile*nAnt, nFrame, nChan), only 3 axes
        tsec = np.array(tsec).reshape((nFile, -1)).mean(axis=0)
        
        if (len(localip)>0):
            attrs['localip'] = localip
        if (len(dest_ip)>0):
            attrs['dest_ip'] = dest_ip

        Cov2, norm2 = makeCov(spec, scale=True, coeff=False, bandpass=True, ant_flag=ant_flag, nPool=nPool)
        #Cov2, norm2 = makeCov(spec, scale=True, coeff=False, bandpass=False, ant_flag=ant_flag)
        W2, V2 = Cov2Eig(Cov2)
        savW2.append(W2)
        savV2.append(V2)
        savN2.append(norm2)
        savN2mask.append(norm2.mask)
        Cov3, norm3 = makeCov(spec, scale=False, coeff=True, ant_flag=ant_flag, nPool=nPool)
        W3, V3 = Cov2Eig(Cov3)
        savW3.append(W3)
        savV3.append(V3)
        savN3.append(norm3)
        savN3mask.append(norm3.mask)

        #Vlast[ii] = V[:,:,nAnt-1]
        t2 = time.time()
        print('... eigenmode got. elapsed:', t2-t0)

        print('files loaded')
        savW2 = np.array(savW2).mean(axis=0)
        savV2 = np.array(savV2).mean(axis=0)
        savN2 = np.ma.array(savN2, mask=savN2mask).mean(axis=0)
        savW3 = np.array(savW3).mean(axis=0)
        savV3 = np.array(savV3).mean(axis=0)
        savN3 = np.ma.array(savN3, mask=savN3mask).mean(axis=0)
        tsec  = np.array(tsec)



        adoneh5(fout, savN2, 'N2_scale')    # shape (nAnt, nChan)
        adoneh5(fout, savW2, 'W2_scale')    # shape (nChan, nMode)
        adoneh5(fout, savV2, 'V2_scale')    # shape (nChan, nAnt, nMode)
        adoneh5(fout, savN3, 'N3_coeff')    # shape (nAnt, nChan)
        adoneh5(fout, savW3, 'W3_coeff')
        adoneh5(fout, savV3, 'V3_coeff')

        adoneh5(fout, tsec, 'winSec')
        putAttrs(fout, attrs)


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
        attrs2['array'] = arr_config
        attrs2['theta_rot'] = theta_rot
        putAttrs(fout, attrs2, dest='tauGeo')




    (nChan3, nAnt3, nMode3) = savV3.shape
    nCol = nAnt3//nAnt

    ## plot normalization and eigenvalues
    fig, sub = plt.subplots(3,1,figsize=(15,15),sharex=True)

    # normalization
    ax = sub[0]
    if (not savN3 is None):
        norm = savN3
    elif (not savN2 is None):
        norm = savN2
    for ai in range(nAnt3):
        if (not ai in ant_flag):
            ax.plot(freq, norm[ai], label='Ant%d'%ai)
    ax.set_yscale('log')
    ax.set_ylabel('voltage normalization')
    #ax.legend(ncols=nCol)
    if (nAnt3<=64):
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
    ax.set_ylim(-15, 25)
    if (nAnt3<=64):
        ax.legend(ncols=nCol)

    # eigenvector of leading mode, phase
    ax = sub[2]
    for ai in range(nAnt3):
        if (ai in ant_flag):
            continue
        y = savV3[:,ai,-1]
        ax.plot(freq, np.ma.angle(y), label='Ant%d'%ai)
    ax.set_ylabel('phase (rad)')
    #ax.legend(ncols=nCol)
    if (nAnt3<=64):
        ax.legend(ncols=nCol)
    ax.set_xlabel('freq (MHz)')
    ax.set_xlim(flim[0], flim[1])
    if (nAnt3<=64):
        ax.set_xlim(flim[0], flim[1]*1.1)

    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.suptitle(fout)
    #fig.savefig('%s.png'%fout)
    fig.savefig('%s/%s.png'%(cdir, fout))
    plt.close(fig)



    ## plot phase of individual antennas
    if (nAnt3==64):
        nx = 8
        ny = 8
        ww = 16
        hh = 16
    elif (nAnt3==16):
        nx = 8
        ny = 2
        ww = 16
        hh = 4
    else:
        nx = 16
        ny = nAnt3//nx
        ww = 16
        hh = 1.0*ny

    fig, sub = plt.subplots(ny,nx,figsize=(ww,hh),sharex=True,sharey=True)
    png = '%s/%s.phases.png'%(cdir, fout)
    fig2, sub2 = plt.subplots(ny,nx,figsize=(ww,hh),sharex=True,sharey=True)
    png2 = '%s/%s.ampld.png'%(cdir, fout)

    freq2 = freq * 1e6  # to Hz
    c_arr = phiCorr(tauGeo, freq2).conjugate()

    print('savN3.shape:', savN3.shape)
    LV3  = savV3[:,:,-1] # leading mode with coeff
    LV3C = LV3 * c_arr.T.reshape((nChan, nAnt3))
    if (aref is None):
        aref = 0
    ref  = LV3[:,aref] / np.ma.abs(LV3[:,aref])
    LV3  /= ref.reshape((-1,1))
    refC  = LV3C[:,aref] / np.ma.abs(LV3C[:,aref])
    LV3C  /= refC.reshape((-1,1))

    NLV3C = savN3.T * LV3C  # shape (nChan, nAnt)
    NLV3C.fill_value = 0j

    adoneh5(fout, LV3C, 'antCal')
    fnpy = '%s/%s.antCal.npy'%(cdir, fout)
    #np.save(fnpy, LV3C)    # LV3C is phase-only
    np.save(fnpy, NLV3C.filled())    # NLV3C includes bandpass EQ

    ## version 2
    NLV3C2 = 1./savN3.T * LV3C  # shape (nChan, nAnt)
    NLV3C2.fill_value = 0j
    fnpy = '%s/%s.antCal2.npy'%(cdir, fout)
    np.save(fnpy, NLV3C2.filled())    # NLV3C includes bandpass EQ


    ai = -1
    for ii in range(ny):
        for jj in range(nx):
            ai += 1
            ax = sub[ii,jj]
            ax2 = sub2[ii,jj]
            if (not ai in ant_flag):
                ax.plot(freq, np.ma.angle(LV3[:,ai]))
                ax.plot(freq, np.ma.angle(LV3C[:,ai]))
                ax2.plot(freq, 10*np.ma.log10(np.ma.abs(LV3C[:,ai]*savN3[ai])))
            #ax.legend()
            ax.text(0.05, 0.85, 'Ant%02d'%ai, transform=ax.transAxes)
            ax2.text(0.05, 0.85, 'Ant%02d'%ai, transform=ax2.transAxes)

            if (jj==0):
                ax.set_ylabel('power (dB)')
                ax2.set_ylabel('power (dB)')
            if (ii==ny-1):
                ax.set_xlabel('freq (MHz')
                ax2.set_xlabel('freq (MHz')

    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(png)
    plt.close(fig)
    fig2.tight_layout(rect=[0,0.03,1,0.95])
    fig2.subplots_adjust(wspace=0, hspace=0)
    fig2.savefig(png2)
    plt.close(fig2)




    t3 = time.time()
    print('... all done. elapsed:', t3-t00)

#!/usr/bin/env python
## standard libs
import sys, os.path
import re
import time, gc
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import pinv
from astropy.time import Time
from astropy.stats import sigma_clip
## ky's libs
from packet_func import *       # loadNode, 
from calibrate_func import *    # makeCov, Cov2Eig
from loadh5 import *            # getData, adoneh5, getAttrs, putAttrs
from pyplanet import *          # obsBody, loadDB
from delay_func2 import *

DB = loadDB()

inp = sys.argv[0:]
pg  = inp.pop(0)

flim    = [300., 700.]
nAnt    = 16
nChan   = 1024
nFPGA   = 16     # for 256-ant
nChan2  = 128   # for 256-ant
bitwidth = 4    # for 256-ant
blocklen = 819200 # for 256-ant
nBlock  = 1
autoblock = True
nPack   = 32768
p0      = 0
hdver   = 2
meta    = 64
order_off = 0
fout    = 'recal256_<input timestamp>.eigen.h5'
user_fout = False
dcal_dir = None
no_bitmap = False
hdlen   = 64
paylen  = 8192
redo    = False
arr_config  = '16x1.0y0.5'    # 256-ant, 16x16, 0.5m sep between rows
do_model = True                 # always save the geometric delay to the Sun
body    = 'sun'
site    = 'fushan6'
do_scale = False
do_coeff = True
aref    = 0
EW_hwhm = 30.
NS_hwhm = 46.
f410    = 5e5
f610    = 7e5
chlim   = [0, nChan]
pad     = 32

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
                    # (default uses the shared input timestamp)
    --dcal DIR      # apply row phase corrections from
                    # DIR/solution_2ndCal.npz (default: None)
    --flag 'ant(s)' # specify the input number (0--255) to be flagged
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
    --aref aref     # global reference antenna index (0--255)
                    # (default: %d)

    (special)
    --no-bitmap     # ignore the bitmap
    --4bit          # read 4-bit data
    --ooff OFF      # offset added to the packet_order

''' % (pg, nPack, p0, blocklen, nBlock, hdver, meta, arr_config, body, site, aref)

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
    elif (k == '--dcal'):
        dcal_dir = inp.pop(0)
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
        site = inp.pop(0)
    elif (k == '--aref'):
        aref = int(inp.pop(0))
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        files0.append(k)

# for autoblock
byteBlockBM = blocklen//8
byteBlock = (hdlen + paylen)*blocklen + byteBlockBM

file_timestamps = {}
for fbin in files0:
    timestamp_matches = re.findall(r'(?<!\d)(\d{8}_\d{6}Z)(?!\d)', os.path.basename(fbin))
    if len(timestamp_matches) != 1:
        sys.exit('expected one YYYYMMDD_HHMMSSZ timestamp in input filename: %s'%fbin)
    file_timestamps[fbin] = timestamp_matches[0]
unique_timestamps = sorted(set(file_timestamps.values()))
if len(unique_timestamps) != 1:
    sys.exit('input files have different timestamps: %s'%', '.join(unique_timestamps))
input_timestamp = unique_timestamps[0]
if not user_fout:
    fout = 'recal256_%s.eigen.h5'%input_timestamp

# frequency in MHz
freq = np.linspace(flim[0], flim[1], nChan, endpoint=False)

theta_rot = -3.0 # Fushan
if (arr_config == '16x1.0y0.5'):
    pos = arrayConf(arr_config, nFPGA, theta_rot=theta_rot)
else:
    if (os.path.isfile(arr_config)):
        pos = np.loadtxt(arr_config)
    else:
        sys.exit('unknown array config file: %s'%arr_config)

# for antenna-based solution, only need to calculate the geometric delay
# to a fixed reference (i.e. 0-th antenna)
if (aref < 0 or aref >= nFPGA*nAnt):
    sys.exit('reference antenna must be in the range 0--%d'%(nFPGA*nAnt-1))
if aref in ant_flag:
    sys.exit('reference antenna %d cannot be flagged'%aref)
BVec = pos - pos[aref]


nLoop = 1
loop_files = [files0]

t00 = time.time()

for ll in range(nLoop):
    files = loop_files[ll]

    nFile = len(files)
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
        attrs['aref'] = aref
        attrs['filename_timestamp'] = input_timestamp

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

            if (hdver==1):
                if (len(ftpart)==10):
                    ftstr = '23'+ftpart # hard-coded to 2023!!
                elif (len(ftpart)==14):
                    ftstr = ftpart[2:]  # strip leading 20
                ftime = datetime.strptime(ftstr, '%y%m%d%H%M%S')
            elif (hdver==2):
                epoch = filesEpoch(fbin, hdver=2)
                ftime = Time(epoch[0], format='unix').to_datetime()
                ftstr = ftime.strftime('%y%m%d%H%M%S')

            if (ftime0 is None):
                ftime0 = ftime
                unix0 = Time(ftime0, format='datetime').to_value('unix')    # local time
                #unix0 -= 3600.*8.                                           # convert to UTC
                attrs['unix_utc_open'] = unix0

            dt = (ftime - ftime0).total_seconds()
            print('(%d/%d)'%(ii+1,nFile), fbin, 'dt=%dsec'%dt)

            if (i==0):
                tsec.append(dt)
            fh = open(fbin, 'rb')
            # spec0.shape = (nFPGA, nFrame, nAnt, nChan)
            spec0, order0 = loadNode(fh, p0, nPack, nFPGA=nFPGA, order_off=order_off, bitwidth=bitwidth, verbose=1,get_order=True)
            fh.close()

            # trasposed shape = (nFPGA, nAnt, nFrame, nChan)
            spec[:,:,:,int(nChan2*order0):int(nChan2*(order0+1))] = spec0.transpose((0,2,1,3))
            del spec0
            gc.collect()
            t1 = time.time()
            print('... data', i, 'loaded. elapsed:', t1-t0)

        spec = spec.reshape((-1,nFrame,nChan))
        ## new shape (nFile*nAnt, nFrame, nChan), only 3 axes

        if (do_scale):
            #Cov2, norm2 = makeCov(spec, scale=True, coeff=False, bandpass=True, ant_flag=ant_flag)
            #Cov2, norm2 = makeCov(spec, scale=True, coeff=False, bandpass=False, ant_flag=ant_flag)
            Cov2, norm2 = makeCov2(spec, scale=True, coeff=False, bandpass=True, ant_flag=ant_flag)
            W2, V2 = Cov2Eig(Cov2)
            savW2.append(W2)
            savV2.append(V2)
            savN2.append(norm2)
            savN2mask.append(norm2.mask)
        if (do_coeff):
            Cov3, norm3 = makeCov2(spec, scale=False, coeff=True, ant_flag=ant_flag, nPool=4)
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
        adoneh5(fout, freq, 'freq')
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
        phi = np.pi/2. - az
        theta = np.pi/2. - el
        unitVec = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], ndmin=2).T
        unitVec *= -1
        tauGeo = np.tensordot(BVec, unitVec, axes=(1,1)) / 2.998e8

        adoneh5(fout, tauGeo, 'tauGeo')
        attrs2 = {'body':body, 'site':site, 'array':arr_config, 'aref':aref}
        putAttrs(fout, attrs2, dest='tauGeo')

        za = np.pi/2. - el
        pntr = np.sin(za)
        pntz = np.cos(za)
        pntx = pntr * np.sin(az)
        pnty = pntr * np.cos(az)
        EWoff = np.arctan2(pntx, pntz)
        NSoff = np.arctan2(pnty, pntz)
        Eatt = np.exp(-((EWoff/np.pi*180.)**2)/(30./np.sqrt(2.*np.log(2.)))**2/2.)
        Hatt = np.exp(-((NSoff/np.pi*180.)**2)/(46./np.sqrt(2.*np.log(2.)))**2/2.)
        att0 = Eatt * Hatt
        adoneh5(fout, att0, 'atten')
        adoneh5(fout, EWoff, 'EWoff')
        adoneh5(fout, NSoff, 'NSoff')
    else:
        tauGeo = np.zeros((nFPGA*nAnt, len(tsec)))
        attrs2 = {}
        att0 = np.ones(len(tsec))
        EWoff = np.zeros(len(tsec))
        NSoff = np.zeros(len(tsec))

    (nChan3, nAnt3, nMode3) = savV3.shape
    nRow = nFPGA
    nBl = nAnt*(nAnt-1)//2
    outbase = os.path.basename(fout)
    cdir = '%s.check'%fout
    os.makedirs(cdir, exist_ok=True)

    freq2 = freq * 1e6
    c_arr = phiCorr(tauGeo, freq2).conjugate()
    row_phi_corr = None
    if dcal_dir is not None:
        f_npz = os.path.join(dcal_dir, 'solution_2ndCal.npz')
        if not os.path.isfile(f_npz):
            sys.exit('error loading row delay correction: %s'%f_npz)
        with np.load(f_npz) as dcal:
            if 'phiCorr' not in dcal.files:
                sys.exit('phiCorr is missing from %s'%f_npz)
            row_phi_corr = np.asarray(dcal['phiCorr'])
        if row_phi_corr.shape == (1,nFPGA,nChan3):
            row_phi_corr = row_phi_corr[0]
        if row_phi_corr.shape != (nFPGA,nChan3):
            sys.exit('phiCorr in %s has shape %s; expected (%d, %d)'%
                    (f_npz, row_phi_corr.shape, nFPGA, nChan3))
        antenna_phi_corr = np.broadcast_to(
                row_phi_corr[:,None,:], (nFPGA,nAnt,nChan3)).reshape((nAnt3,nChan3))
        c_arr *= antenna_phi_corr
        adoneh5(fout, row_phi_corr, 'phiCorrRow')
        putAttrs(fout, {'source':os.path.abspath(f_npz)}, dest='phiCorrRow')

    LV3 = savV3[:,:,-1].copy()
    ref = np.ma.exp(1.j*np.ma.angle(LV3[:,aref]))
    LV3 /= ref.reshape((-1,1))
    LV3C = LV3 * c_arr.T.reshape((nChan3, nAnt3))
    refC = np.ma.exp(1.j*np.ma.angle(LV3C[:,aref]))
    LV3C /= refC.reshape((-1,1))

    NLV3C = savN3.T * LV3C
    NLV3C.fill_value = 0j
    NLV3C2 = 1./savN3.T * LV3C
    NLV3C2.fill_value = 0j
    adoneh5(fout, LV3C, 'antCal')

    VrefTau = LV3C.copy()
    FTVref = np.fft.fftshift(np.fft.fft(VrefTau, n=int(nChan*pad), axis=0), axes=0)
    peak_lag = np.abs(FTVref).argmax(axis=0) - int(pad*nChan/2)
    peak_ns = peak_lag * 1e9/400e6/pad
    VrefC = VrefTau*np.exp(-2j*np.pi*peak_ns.reshape((1,-1))*freq.reshape((-1,1))*1e-3)

    med_ampld = np.ma.median(savN3, axis=0)
    rel_ampld = savN3 / med_ampld.reshape((1,-1))
    med_rel_ampld = np.ma.median(rel_ampld[:,chlim[0]:chlim[1]], axis=1)

    all_coeff = np.ma.masked_all((nRow*nBl, nChan3), dtype=complex)
    all_SEFD = np.ma.masked_all((nAnt3, nChan3))
    flux = f410 + (freq-410.)*(f610-f410)/200.
    flux *= att0[0]
    flagged = set(ant_flag)

    for row in range(nRow):
        first = row*nAnt
        last = first+nAnt
        row_vec = savV3[:,first:last,:]
        row_cov = np.einsum('cik,ck,cjk->cij', row_vec, savW3, row_vec.conjugate(), optimize=True)
        coeff_row = np.ma.masked_all((nBl,nChan3), dtype=complex)
        B = np.zeros((nBl,nAnt))
        bidx = -1
        for ai in range(nAnt-1):
            for aj in range(ai+1,nAnt):
                bidx += 1
                coeff_row[bidx] = row_cov[:,ai,aj]
                if (first+ai not in flagged and first+aj not in flagged):
                    B[bidx,ai] = 0.5
                    B[bidx,aj] = 0.5
        all_coeff[row*nBl:(row+1)*nBl] = coeff_row

        coeff_abs = np.ma.abs(coeff_row)
        sefd_bl = flux.reshape((1,nChan3))/coeff_abs*(1.-coeff_abs)
        sefd_bl = np.ma.masked_where((coeff_abs <= 0.) | (coeff_abs >= 1.), sefd_bl)
        D = np.ma.log10(sefd_bl).filled(0.)
        M = np.dot(pinv(B), D)
        row_mask = np.array([first+ai in flagged for ai in range(nAnt)]).reshape((-1,1))
        all_SEFD[first:last] = np.ma.array(10.**M, mask=np.broadcast_to(row_mask, (nAnt,nChan3)))

    del_SEFD = np.ma.masked_all(all_SEFD.shape)
    wt_SEFD = np.ma.masked_all(nAnt3)
    for row in range(nRow):
        first = row*nAnt
        last = first+nAnt
        row_slice = slice(first,last)
        med_SEFD = np.ma.median(all_SEFD[row_slice], axis=0, keepdims=True)
        del_SEFD[row_slice] = all_SEFD[row_slice] / med_SEFD
        wt_SEFD[row_slice] = 1./np.ma.median(del_SEFD[row_slice,chlim[0]:chlim[1]], axis=1)
    adoneh5(fout, all_coeff, 'coeff')
    adoneh5(fout, all_SEFD, 'SEFD400')
    adoneh5(fout, wt_SEFD, 'wt_SEFD')

    for row in range(nRow):
        first = row*nAnt
        last = first+nAnt
        rowdir = os.path.join(cdir, 'row%02d'%row)
        os.makedirs(rowdir, exist_ok=True)
        row_coeff = all_coeff[row*nBl:(row+1)*nBl]
        row_SEFD = all_SEFD[first:last]
        row_wt = wt_SEFD[first:last]
        row_med_SEFD = np.ma.median(row_SEFD[:,chlim[0]:chlim[1]], axis=1)

        fig, sub = plt.subplots(3,1,figsize=(15,15),sharex=True)
        for ai in range(first,last):
            if ai not in flagged:
                sub[0].plot(freq, savN3[ai], label='Ant%d'%ai)
        sub[0].set_yscale('log')
        sub[0].set_ylabel('voltage normalization')
        sub[0].legend(ncols=4)
        for mode in range(nMode3):
            if mode < nFlag:
                continue
            sub[1].plot(freq, sigma_clip(10.*np.log10(savW3[:,mode]), sigma=10))
        sub[1].set_ylabel('power (dB)')
        for ai in range(first,last):
            if ai not in flagged:
                sub[2].plot(freq, np.ma.angle(LV3C[:,ai]), label='Ant%d'%ai)
        sub[2].set_ylabel('phase (rad)')
        sub[2].set_xlabel('freq (MHz)')
        sub[2].set_xlim(flim[0], flim[1])
        fig.tight_layout(rect=[0,0.03,1,0.95])
        fig.suptitle('%s, row%02d'%(input_timestamp,row))
        fig.savefig(os.path.join(rowdir, outbase+'.png'))
        plt.close(fig)

        fig, sub = plt.subplots(4,4,figsize=(16,8),sharex=True,sharey=True)
        fig2, sub2 = plt.subplots(4,4,figsize=(16,8),sharex=True,sharey=True)
        fig3, sub3 = plt.subplots(4,4,figsize=(16,8),sharex=True,sharey=True)
        for local_ant in range(nAnt):
            ai = first+local_ant
            ax = sub.flat[local_ant]
            ax2 = sub2.flat[local_ant]
            ax3 = sub3.flat[local_ant]
            if ai not in flagged:
                ax.plot(freq, np.ma.angle(LV3[:,ai]), label='obs')
                ax.plot(freq, np.ma.angle(LV3C[:,ai]), label='inst.')
                ax.plot(freq, np.ma.angle(VrefC[:,ai]), color='gray', label='resid.')
                ax.text(0.55,0.85,'tau:%.2fns'%peak_ns[ai],transform=ax.transAxes,color='C1')
                ax.set_ylim(-3.5,4.5)
                ax2.plot(freq,10*np.ma.log10(np.ma.abs(LV3C[:,ai]*savN3[ai])))
                ax3.plot(freq,1./rel_ampld[ai],color='b',alpha=0.3)
                ax3.axhline(1./med_rel_ampld[ai],color='b',ls='--',label='rel_norm')
                ax3.plot(freq,np.abs(LV3C[:,ai])/0.25,color='g',label='abs(V)/0.25')
                ax3.plot(freq,1./del_SEFD[ai],color='r',alpha=0.3)
                ax3.axhline(row_wt[local_ant],color='r',ls='--',label='wt_SEFD')
                ax3.set_ylim(0,2)
            if ai == aref:
                ax.legend()
            for panel in (ax,ax2,ax3):
                panel.text(0.05,0.85,'Ant%03d'%ai,transform=panel.transAxes)
            if local_ant % 4 == 0:
                ax.set_ylabel('phase (rad)')
                ax2.set_ylabel('power (dB)')
                ax3.set_ylabel('scaling')
            if local_ant >= 12:
                ax.set_xlabel('freq (MHz)')
                ax2.set_xlabel('freq (MHz)')
                ax3.set_xlabel('freq (MHz)')
        for figx, axes, suffix, title in (
                (fig,sub,'phases','phases'),
                (fig2,sub2,'ampld','amplitude'),
                (fig3,sub3,'weight','scaling')):
            figx.tight_layout(rect=[0,0.03,1,0.95])
            figx.subplots_adjust(wspace=0,hspace=0)
            figx.suptitle('%s, row%02d, %s'%(input_timestamp,row,title))
            figx.savefig(os.path.join(rowdir,'%s.%s.png'%(outbase,suffix)))
            plt.close(figx)

        fig, s2d = plt.subplots(4,4,figsize=(12,8),sharex=True,sharey=True)
        for local_ant in range(nAnt):
            ax = s2d.flat[local_ant]
            ai = first+local_ant
            if ai not in flagged:
                ax.plot(freq,row_SEFD[local_ant]/1e6)
                ax.set_yscale('log')
                ax.set_ylim(0.02,5.00)
                ax.grid(True,which='both')
                ax.axhline(row_med_SEFD[local_ant]/1e6,color='r',ls=':')
            ax.text(0.02,0.02,'Ant%03d: %.3fMJy'%(ai,row_med_SEFD[local_ant]/1e6),color='r',transform=ax.transAxes)
            if local_ant % 4 == 0:
                ax.set_ylabel('SEFD (MJy)')
            if local_ant >= 12:
                ax.set_xlabel('freq (MHz)')
        fig.tight_layout()
        fig.subplots_adjust(wspace=0,hspace=0)
        fig.suptitle('%s, row%02d, SEFD'%(input_timestamp,row))
        fig.savefig(os.path.join(rowdir,outbase+'.ant_SEFD.png'))
        plt.close(fig)

        row_slice = slice(first,last)
        np.save(os.path.join(rowdir,outbase+'.antCal.npy'),NLV3C[:,row_slice].filled())
        np.save(os.path.join(rowdir,outbase+'.antCal2.npy'),NLV3C2[:,row_slice].filled())
        np.savez(os.path.join(rowdir,outbase+'.antCals.npz'),
                attrs=attrs,
                reference_antenna=aref,
                antenna_indices=np.arange(first,last),
                atten=att0,
                EWoff_deg=EWoff/np.pi*180.,
                NSoff_deg=NSoff/np.pi*180.,
                tauGeo_sec=tauGeo[row_slice],
                tauGeo_attrs=attrs2,
                phiCorrRow=(row_phi_corr[row] if row_phi_corr is not None else np.ones(nChan3, dtype=complex)),
                phiCorrSource=(os.path.abspath(os.path.join(dcal_dir, 'solution_2ndCal.npz')) if dcal_dir is not None else ''),
                tauI_ns=peak_ns[row_slice],
                phiCorr=c_arr[row_slice],
                freq_MHz=freq,
                winSec=tsec,
                auto=savN3[row_slice].T,
                eigenvector=LV3C[:,row_slice],
                antCal2=NLV3C2[:,row_slice].filled(),
                coeff=row_coeff,
                wt_SEFD=row_wt,
                SEFD400=row_SEFD)

    t3 = time.time()
    print('... all done. elapsed:', t3-t00)

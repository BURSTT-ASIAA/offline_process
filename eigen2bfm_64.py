#!/usr/bin/env python

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from glob import glob
import time

from packet_func import *
from loadh5 import *
from pyplanet import *
from delay_func2 import *

t0 = time.time()

nAnt  = 16
nChan = 1024
flim  = [400., 800.]
lamb0 = 2.998e8/400e6   # longest wavelength
#sep   = 1.    # meters
#aconf = 'FUSHAN6_230418.config'
#aconf = '1m'    # the default 1m spacing 1D array
aconf   = '16x1.0y0.5'
rows    = None
sep     = 1.    # separation in meters, in X-axis
theta_rot = 0.
theta_off = 0.
ant_flag = [[],[],[],[]]
## antenna selection for visibility
asel = [0, 2, 9, 15] # bl: 2, 6, 7, 9, 13, 15
#asel = [0, 1, 4, 15] # bl: 1, 3, 4, 11, 14, 15
beam0 = -16     # offset toward East
#beam0 = -7.5   # symmetric about zenith
#feig = 'fpga0.1214114800.bin.eigen.h5'
use_coeff = False
idir = None
## for 64-ant, row0--row3
nRow = 4
minRow = 0
dirname = '64ant_ver%s' % time.strftime('%y%m%d_%H%M')
medBEQ  = 64.
verbose = 0


inp = sys.argv[0:]
pg  = inp.pop(0)

usage = '''
derive beamform parameters from preprocssed eigenmodes
when multiple .eigen.h5 files are given,
the 1st is used to provide normalization (BEQ) and phase solution
and the others are used for incremental phase adjustments
if --bfm is used, get the base matrix from the BFM_DIR instead

syntax:
    %s <eigen.h5_file(s)> [options]

    options are:
    --bfm BFM_DIR       # load the base BFM for incremental adjustment
                        # (default is to start from scratch)
    --coeff             # use the coeff normalized eigenmodes
                        # default: bandpass normalized eigenmodes
    --flag Row '...'    # first specify the row number (0--3)
                        # then the antenna numbers to mask (0--15)
                        # quote multiple numbers with blank separation
    --aconf ACONF       # specify an antenna configuration file
                        # default: 1m spacing E-W array
    --sep SEP           # separation in meters along the row
                        # default: %.1f m
    --rows 'rows'       # specify the row numbers used, numbers separated by spaces
                        # default: the first nRows=4 rows
    --off theta_off     # additional offset angles of the beams in deg
                        # default: %.2f deg
    --beam0 B0          # specify the offset of the beams
                        # 0 puts the East-most beam at zenith
                        # -7.5 puts equal number of beams to the West and East
                        # -16 shifts all the beams to the East
                        #   (including the last West-most beam)
                        # default: %.1f
    --vis_sel '....'    # specify the 4 antenna numbers used for visibility
                        # default: [%s]
    --beq <med_BEQ>     # set target median value of the BFM parameters
                        # range: (0, 127), default: %.0f

    --flim fmin fmax    # set the spectrum min/max freq in MHz
                        # (default: [%.1f, %.1f] MHz )

''' % (pg, sep, theta_off, beam0, ', '.join(map(str,asel)), medBEQ, flim[0], flim[1])

if (len(inp)<1):
    sys.exit(usage)

files = []
while (inp):
    k = inp.pop(0)
    if (k == '--flag'):
        row = int(inp.pop(0))
        tmp = inp.pop(0)
        ant_flag[row] = [int(x) for x in tmp.split()]
    elif (k == '--bfm'):
        idir = inp.pop(0)
    elif (k == '--coeff'):
        use_coeff = True
    elif (k == '--beam0'):
        beam0 = float(inp.pop(0)) # changed to float
    elif (k == '--aconf'):
        aconf = inp.pop(0)
    elif (k == '--sep'):
        sep = float(inp.pop(0))
    elif (k == '--rows'):
        tmp = inp.pop(0)
        rows = [int(x) for x in tmp.split()]
    elif (k == '--rot'):
        theta_rot = float(inp.pop(0))
    elif (k == '--off'):
        theta_off = float(inp.pop(0))
    elif (k == '--beq'):
        medBEQ = float(inp.pop(0))
    elif (k == '--vis_sel'):
        tmp = inp.pop(0)
        asel = [int(x) for x in tmp.split()]
        if (len(asel) != 4):
            sys.exit('error getting the vis_sel: %s'%tmp)
    elif (k == '--flim'): 
        flim[0] = float(inp.pop(0))
        flim[1] = float(inp.pop(0))
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        files.append(k)

cwd = os.getcwd()

y = np.arange(nAnt)
ch = np.arange(nChan)
CC, YY = np.meshgrid(ch, y)

BEQ4k = medBEQ*32   # medBEQ range (0, 127); BEQ4k range (0, 4096)

N2_64 = None
V2_64 = None
if (not idir is None):
    # load the IDT BFM <-- 1/N2
    # or N2 = 1/BFM_IDT
    N2_64 = np.ma.array(np.zeros((nAnt*nRow, nChan)), mask=False)
    V2_64 = np.zeros((nChan, nAnt*nRow), dtype=complex)
    tp = 'idt'
    for i in range(nRow):
        fpga = 'fpga%d'%(i+minRow)
        ifile = '%s/%s.%s.npy'%(idir, fpga, tp)
        print('read:', ifile)
        BFM = np.load(ifile)
        #BFM = np.zeros((nBeam, nAnt, nChan, 2), dtype=np.short)
        #BFM[:,:,:,0] = BEQ.reshape((1, nAnt, -1)) * yy.real
        #BFM[:,:,:,1] = BEQ.reshape((1, nAnt, -1)) * yy.imag
        CBFM = BFM[:,:,:,0] + 1.j*BFM[:,:,:,1]
        a0 = i*nAnt
        for ai in range(nAnt):
            beq = np.abs(CBFM[ai,ai,:])
            N2_64[a0+ai,:] = 1./beq
            V2_64[:,a0+ai] = CBFM[ai,ai,:]/beq

if (rows is None):
    rows = np.arange(nRow)


# loading antenna positions
pos_64 = arrayConf(aconf, nRow, rows=rows, theta_rot=theta_rot)
#pos2 = pos.reshape((nRow,-1,3))
#sep = np.median(pos2[:,1:,0]-post[:,:-1,0])
print('sep (m):', sep)
if (verbose):
    print('pos (m):', pos_64)

# define wavelengths
fMHz = np.linspace(flim[0], flim[1], nChan, endpoint=False)  # in MHz
lamb = 2.998e8 / (fMHz * 1e6)  # in meters
#print('wavelengths in m:', lamb)
freq2 = fMHz*1e6

sin_theta_m = lamb0/sep/nAnt*(np.arange(nAnt)+beam0)    # offset toward East (HA<0)
theta_deg = np.arcsin(sin_theta_m)/np.pi*180.
theta_deg += theta_off
sin_theta_m = np.sin(theta_deg/180.*np.pi)
print('angles (deg):', theta_deg)

# define beamform matrix
## beamforming: raw, positive correction, negative correction
nBeam = len(theta_deg)
# raw beamform; 1st line below for swapped Re/Im; 2nd line below for older format
#BFM0 = np.exp(-2.j*np.pi*pos.reshape((1,nAnt,1))/lamb.reshape((1,1,nChan))*sin_theta_m.reshape((nBeam,1,1)))
# delay correction term
#TauCorr = np.exp(-2.j*np.pi*antTau.reshape((1,nAnt,1))*fMHz.reshape((1,1,nChan))*1e-3)

## identity matrix, for antenna voltages in 4bit with calibration and EQ
IDT0 = np.zeros((nBeam,nAnt,nChan), dtype=complex)
for ai in range(nAnt):
    IDT0[ai,ai,:] = 1+0j


if (use_coeff):
    tgtVec = 'V3_coeff'
    oname = 'coeff'
else:
    tgtVec = 'V2_scale'
    oname = 'scale'

for i in range(len(files)):
    feig = files[i]
    print('loading N2 and V2 from:', feig)

    # loading the phase correction
    #winSec = getData(feig, 'winSec')

    # bandpass normalization for the scaled mode
    # coeff mode use the same normalization
    # use N3 if it exists
    tmp  = getData(feig, 'N3_coeff') # shape: (nAnt, nChan)
    if (tmp is None):   # fallback to N2 if not
        tmp  = getData(feig, 'N2_scale') # shape: (nAnt, nChan)
    if (N2_64 is None):
        N2_64 = tmp
    else:
        N2_64 *= tmp

    # eigenvectors from the scaled mode
    tmp  = getData(feig, tgtVec) # shape: (nChan, nAnt, nMode)
    # take the leading eigenmode corresponding to the calibrator (at transit time)
    tmp = tmp[:,:,-1]           # shape: (nChan, nAnt)
    tref = tmp[:,0] / np.ma.abs(tmp[:,0])   # Ant0, phase-only
    tmp /= tref.reshape((-1,1))
    if (V2_64 is None):
        V2_64 = tmp
    else:
        V2_64 *= tmp

    # correct for geometric delay if available
    tauGeo = getData(feig, 'tauGeo')
    if (not tauGeo is None):
        c_arr = phiCorr(tauGeo, freq2).conjugate()
        V2_64 *= c_arr.T.reshape((nChan, nRow*nAnt))


N2_64.mask[N2_64==0] = True
# masking flagged antennas
for i in range(nRow):
    for j in ant_flag[i]:
        k = i*nAnt + j
        N2_64.mask[k] = True
BEQ_64 = 1./N2_64
## set median BEQ to 32 after scaling to 128 or 1024 after scaling to 4096
BEQ_64 *= (BEQ4k/np.ma.median(BEQ_64))
BEQ_64[BEQ_64>4096] = 4096


odir = '%s.%s.out'%(dirname, oname)
if (not os.path.isdir(odir)):
    call('mkdir %s'%odir, shell=True)


for i in range(nRow):
    fpga = 'fpga%d'%(i+minRow)
    ofile = '%s/%s.info.txt'%(odir, fpga)
    FO = open(ofile, 'w')

    print('cws =', cwd, file=FO)
    print('odir =', odir, file=FO)
    print('input_bfm =', idir, file=FO)
    for feig in files:
        print('eigen_file =', feig, file=FO)
    print('array_conf =', aconf, file=FO)
    print('sep =', sep, file=FO)
    print('beam0 =', beam0, file=FO)
    print('beam_angle =', theta_deg, file=FO)
    print('ant_flag =', ant_flag[i], file=FO)
    print('ant_sel =', asel, file=FO)

    a0 = i*nAnt

    pos3 = pos_64[a0:a0+nAnt]   # vector
    pos3 -= pos3[0]             # relative
    pos = np.sqrt(pos3[:,0]**2+pos3[:,1]**2)
    #print(pos3.shape, pos.shape)
    BFM0 = np.exp(+2.j*np.pi*pos.reshape((1,nAnt,1))/lamb.reshape((1,1,nChan))*sin_theta_m.reshape((nBeam,1,1)))

    V2 = V2_64[:,a0:a0+nAnt]
    N2 = N2_64[a0:a0+nAnt]
    BEQ = BEQ_64[a0:a0+nAnt]
    for ai in ant_flag[i]:
        N2.mask[ai] = True

    renorm = float(nAnt/(nAnt-len(ant_flag[i])))
    print('renorm =', renorm, file=FO)

    refV2  = V2[:,:]         # averaged window, last mode
    refV2 /= np.abs(refV2)      # keep only the phase info
    #print('refV2.shape:', refV2.shape)


    ## new phase correction term
    TauCorr = refV2.T.reshape((1,nAnt,nChan))
    BFM1 = BFM0 * TauCorr               # pos phase correction
    #BFM2 = BFM0 * TauCorr.conjugate()  # neg phase correction
    # BFM.shape = (nBeam, nAnt, nChan)

    IDT = IDT0 * TauCorr  # positive correction

    ## intensity for visibility
    nsel = len(asel)
    VIS = np.zeros((nBeam,nAnt,nChan), dtype=complex)
    for ii in range(nsel):
        ai = asel[ii]
        VIS[ii,ai] = 1+0j
        VIS[ii,ai] *= TauCorr[0,ai]

    b = -1
    for ii in range(nsel-1):
        ai = asel[ii]
        for jj in range(ii+1,nsel):
            aj = asel[jj]
            b += 1      # bl number
            r = 4+b*2   # row (beam) number
            VIS[r,ai] = 1+0j
            VIS[r,aj] = 1+0j
            VIS[r+1,ai] = 1+0j
            VIS[r+1,aj] = 0+1j
            VIS[r,ai]   *= TauCorr[0,ai]
            VIS[r+1,ai] *= TauCorr[0,ai]
            VIS[r,aj]   *= TauCorr[0,aj]
            VIS[r+1,aj] *= TauCorr[0,aj]


    t1 = time.time()
    print('BFMs done', ' elapsed:%.1fsec'%(t1-t0))


    # bandpass equalization
    #print('debug: N2 min/max', N2.min(), N2.max(), np.ma.median(N2, axis=1))
    #N2 = np.ma.median(N2, axis=1, keepdims=True)
    #BEQ = 1./N2     # shape: (nAnt, nChan)
    #BEQ /= np.ma.abs(BEQ).max()
    #print('debug: BEQ_64.max', BEQ_64.max())
    #BEQ /= BEQ_64.max()
    #BEQ *= 4096
    #BEQ.fill_Value = 0.
    #BEQ = BEQ.filled()

    ## renormalize to account for flagged antennas
    BEQ *= renorm

    ## set max value
    BEQ[BEQ.mask] = 0.
    #print('debug: BEQ min/max', BEQ.min(), BEQ.max())
    #print('debug: norm BEQ min/max', BEQ.min()/4096*127, BEQ.max()/4096*127)
    print('debug: norm BEQ median', np.median(BEQ)/4096*127)
    print('BEQ min,max =', BEQ.min(), BEQ.max(), file=FO)
    print('norm BEQ min,max,med =', BEQ.min()/4096*127, BEQ.max()/4096*127, np.ma.median(BEQ)/4096*127, file=FO)

    FO.close()

    ## override BEQ, no equalization
    #BEQ = np.ones((nAnt, nChan), dtype=float)*127


    #allBFM = [BFM0, BFM1, BFM2]     # phase-only beamform matrix
    #allTYP = ['raw', 'pos', 'neg']
    allBFM = [BFM1, VIS, IDT]     # phase-only beamform matrix
    allTYP = ['pos', 'vis', 'idt']
    nType = len(allTYP)

    for ii in range(nType):
        tp = allTYP[ii]
        yy = allBFM[ii]
        ofile = '%s/%s.%s.npy'%(odir, fpga, tp)
        print(ofile)

        BFM = np.zeros((nBeam, nAnt, nChan, 2), dtype=np.short)
        BFM[:,:,:,0] = BEQ.reshape((1, nAnt, -1)) * yy.real
        BFM[:,:,:,1] = BEQ.reshape((1, nAnt, -1)) * yy.imag
        np.save(ofile, BFM)

        cBFM = BFM[:,:,:,0] + 1.j*BFM[:,:,:,1]

        # diagnostic plots
        for pt in range(3):
            if (pt==0):     # amp
                ptstr = 'amp'
                z = np.abs(cBFM)
            elif (pt==1):   # pha
                ptstr = 'pha'
                z = np.angle(cBFM)
            elif (pt==2):   # profile
                ptstr = 'prf'

            fig, s2d = plt.subplots(4,4,figsize=(12,9), sharex=True, sharey=True)
            sub = s2d.flatten()

            if (pt<=1):
                for bm in range(nAnt):
                    ax = sub[bm]
                    ax.pcolormesh(CC,YY,z[bm],shading='auto')

                for ii in range(4):
                    ax = s2d[3,ii]
                    ax.set_xlabel('channel')
                    ax = s2d[ii,0]
                    ax.set_ylabel('ant_id')

                fig.suptitle('%s, BFM %s by beams'%(files[0], ptstr))

            elif (pt==2):
                for bm in range(nAnt):
                    ax = sub[bm]
                    for ai in range(nAnt):
                        ax.plot(ch, np.abs(cBFM[bm,ai,:]))

                for ii in range(4):
                    ax = s2d[3,ii]
                    ax.set_xlabel('channel')
                    ax = s2d[ii,0]
                    ax.set_ylabel('abs(BFM)')

                fig.suptitle('%s, ant weighting by beams'%(files[0],))

            fig.tight_layout(rect=[0,0.03,1,0.95])
            fig.subplots_adjust(wspace=0, hspace=0)
            png = '%s.%s.png' % (ofile, ptstr)
            fig.savefig(png)
            plt.close(fig)




        continue    # skip the bfm file
        matrix = BFM.transpose((2,1,0,3))  # output shape: (nChan, nAnt, nBeam, 2)
        matrix_size = nChan*nAnt*nBeam*2

        buf = struct.pack('<%dh'%matrix_size, *matrix.flatten())
        ofile = '%s.%s.bfm'%(files[0], tp)
        print('writing to:', ofile)
        fh = open(ofile, 'wb')
        fh.write(buf)
        fh.close()



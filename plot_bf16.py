#!/usr/bin/env python

from packet_func import *
from loadh5 import *
import sys, os.path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from glob import glob
from subprocess import call
from datetime import datetime
from astropy.time import Time
import matplotlib.colors as mcolors
import matplotlib.dates as mdates


# Import modularized masking utilities
import masking_util as mu

inp = sys.argv[0:]
pg  = inp.pop(0)


nRow    = 1
nAnt    = 16
nBeam   = nRow*nAnt
nChan0  = 1024
nFreqChan   = 1024       # for 16-ant
FreqLimit_MHz    = [400., 800.]


nFramePerBlock = 51200    # number of frames per block
nSum    = 400       # integration number

nBlock = 1 # number of blocks to read


frate   = 400e6/1024
prate   = frate*2

nBlockToIntegrate = 1        # number of blocks per window to integrate
BlockSampleStepSize = 400      # separation between windows in blocks

verbose = False

odir   = 'intensity.plots'
read_raw = False
zlim    = None
pts     = [1]

time_zone_hr      = 8   # Hilo time offset from UTC in hours
use_local_time = True # 

mask_file = None
freq_mask_bands = []
show_MAD_hist = False
time_mask_pow_thresh_dB = 0 # Default: 0 (disabled)
time_mask_str = ""

prefix  = 'intensity_'

usage   = '''
plot amplitude of the spectrum as a function of frequency and time

syntax:
    %s -f <fpga_id> <dir> [options]

    example: the data are named 'intensity_fpga2_sum400_yymmdd_HHMMSS'
    then <fpga_id> = 2, i.e. -f 2

options are:
    --sum SUM       # the integration for the intensity data
                    # (default: %d)
    --frame FRAME   # number of frames per block
                    # (default: %d)
    --sep BlockSampleStepSize  # separation between window in blocks (default: %d)
    --win nBlockToIntegrate  # integration window size in blocks (default: %d) 
    -o <odir>       # specify an output dir
    --redo          # force reading raw data
    --zlim zmin zmax# set the min/max color scale
    -v              # verbose
    --raw           # plot the raw intensity
                    # (default is to plot the normalized intensity)
    --both          # plot both raw and normalized intensities
    --flim fmin fmax    # set the spectrum min/max freq in MHz
                    # (default: [%.1f, %.1f] MHz )
    --fmask <file>  #  text file with freq ranges to mask (lo hi) in MHz
    --tmask pow_thresh_dB    # enable time masking excluding window with median absolute deviation (MAD) power >  (>0 to enable) dB from the median spectrum
    --tz hr     # timezone offset from UTC
                # e.g. Taipei = +8; Hilo = -10
                # (default, %.1f hour)

    --utc       # show times in UTC
    --prefix <name> # prefix of file name (default: %s)                      
''' % (pg, nSum, nFramePerBlock, BlockSampleStepSize, nBlockToIntegrate, FreqLimit_MHz[0], FreqLimit_MHz[1], time_zone_hr, prefix)


if (len(inp)<1):
    sys.exit(usage)

dirs  = []
rings = []
while (inp):
    k = inp.pop(0)
    if (k=='-o'):
        odir = inp.pop(0)
    elif (k=='--redo'):
        read_raw = True
    elif (k=='--zlim'):
        zmin = float(inp.pop(0))
        zmax = float(inp.pop(0))
        zlim = [zmin, zmax]
    elif (k=='--sep'):
        BlockSampleStepSize = int(inp.pop(0))
    elif (k=='--win'):
        nBlockToIntegrate = int(inp.pop(0))
    elif (k=='--sum'): # unlikely to be changed
        nSum = int(inp.pop(0))
    elif (k=='--frame'):
        nFramePerBlock = int(inp.pop(0))
    elif (k=='-f'):
        ring_id = int(inp.pop(0))
        idir = inp.pop(0)
        dirs.append(idir)
        rings.append(ring_id)
    elif (k=='-v'):
        verbose=True
        show_MAD_hist = True
    elif (k=='--raw'): ### to be implemented
        pts = [0]
    elif (k=='--both'):### to be implemented
        pts = [0, 1]
    elif (k == '--flim'): 
        FreqLimit_MHz[0] = float(inp.pop(0))
        FreqLimit_MHz[1] = float(inp.pop(0))
    elif (k=='--fmask'): #  Parse mask file
        mask_file = inp.pop(0)
    elif (k=='--tmask'): 
        time_mask_pow_thresh_dB = float(inp.pop(0))
        time_mask_str = " time mask: MAD > %.2f dB"%time_mask_pow_thresh_dB
        print('enable time masking excluding samples with MAD > %.2f sigma'%time_mask_pow_thresh_dB )
    elif (k == '--tz'):
        time_zone_hr = float(inp.pop(0))
        print('use time zone: %.1f hr'%time_zone_hr)
    elif (k == '--utc'):
        use_local_time = False
        time_zone_hr = 0
    elif (k == '--prefix'):
        prefix = inp.pop(0)
        print('prefix of file name [%s]'%prefix )
    elif (k == '--nblock'):
        nBlock = int(inp.pop(0))        
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        sys.exit('extra argument: %s'%k)

# for loading intensity data
# intensity data is from integrating baseband data by factor of nSum;  1 baseband block has nFramePerBlock frames
# 4+4 bit complex baseband sample  -> 16-bit intenstiy sample
SampleSize_Byte = 2

nTime   = nFramePerBlock//nSum       
nElem   = nTime*nFreqChan*nBeam
nBytePerIntensityBlock   = nElem * SampleSize_Byte    

print('Assuming intensity data has (%d frames per block) * (%d frequency channel) * (%d beam) = %d elements per block = %d Byte per block (%d Byte per sample)'%(nTime, nFreqChan, nBeam, nElem, nBytePerIntensityBlock, SampleSize_Byte) )

nDir = len(dirs)

# define here for adjustable range
FreqArray   = np.linspace(FreqLimit_MHz[0], FreqLimit_MHz[1], nChan0, endpoint=False)




# Load global frequency mask once
freq_mask_bands = mu.load_freq_band(mask_file)

freq_mask_str = ""
if freq_mask_bands:
    freq_mask_str = "frequency mask: " + ", ".join([f"[{r[0]}, {r[1]}]" for r in freq_mask_bands])
    freq_mask_str += " MHz"

    print('Apply frequency mask: ', freq_mask_str)

# for multiple ring buffers / FGPAs
arrInts  = []
arrNInts = []
freqs    = []
tsecs    = []
arrDateTimes   = []

if (not os.path.isdir(odir)):
    call('mkdir %s'%odir, shell=True)

for j in range(nDir):
    idir = dirs[j]
    ring_id = rings[j]
    #ring_name = 'ring%d'%ring_id
    ring_name = 'fpga%d'%ring_id

    
    #i0 = nFreqChan*ring_id   
    i0 = 0  # rudp16: common freq range
    freq = FreqArray[i0:i0+nFreqChan]
    freqs.append(freq)

    if (ring_name in idir):
        #ofile = idir + '.inth5' # SH: this causes hidden and unnamed output fpga0/.inth5
        ofile = '%s/%s.inth5'%(odir, ring_name)
    else:
        ofile = '%s/%s.inth5'%(odir, ring_name)

    print('output intensity to:', ofile)

    # sub-folder per ring buffer
    odir_plot = '%s.plots'%ofile

    adoneh5(ofile, freq, 'freq')

    files   = glob('%s/%s%s_*'%(idir, prefix, ring_name)) #intensity_
    files.sort()
    nFile   = len(files)
    if (False):
        nFile = 3
        files = files[:nFile]
    print('nFile:', nFile)

    nTimeWindow = 0
    fnWins = []
    for f in files:
        s = os.path.getsize(f)
        nBlockInFile = s // nBytePerIntensityBlock # num of blocks in this file
        
        nTimeWindowInFile = nBlockInFile // BlockSampleStepSize # number of windows in this file

        if verbose:
            print(' %d blocks in [%s], sampled every %d block = %d time windows'%(nBlockInFile, f, BlockSampleStepSize, nTimeWindowInFile))

        if (nTimeWindowInFile == 0):
            nTimeWindowInFile = 1
        fnWins.append(nTimeWindowInFile)
        nTimeWindow += nTimeWindowInFile
        #print(f, nBlockInFile, nTimeWindowInFile)

    attributes   = {}
    attributes['idir'] = idir
    attributes['ring_id'] = ring_id
    attributes['nChan'] = nFreqChan
    attributes['nRow'] = nRow
    attributes['nFile'] = nFile
    attributes['nWin'] = nTimeWindow
    attributes['sepBlock'] = BlockSampleStepSize
    attributes['winBlock'] = nBlockToIntegrate
    attributes['nFrame'] = nFramePerBlock
    attributes['nSum'] = nSum
    attributes['nTime'] = nTime

    # intensity array
    if (os.path.isfile(ofile)):
        tmp = getData(ofile, 'intensity')
        if (tmp is None):
            read_raw = True
        else:
            if (not read_raw):
                IntensityArray = tmp
                NormalizedIntensityArray = getData(ofile, 'norm.intensity')
                WindowElapsedTime_s = getData(ofile, 'winSec')
                attributes = getAttrs(ofile)
                StartEpochTime = attributes.get('unix_utc_open')
    else:
        read_raw = True

    if (read_raw): 

        print("process raw intensity data...")
        #IntensityArray  = np.ma.array(np.zeros((nFile, nRow, nAnt, nFreqChan)), mask=False)  # unmasked by default
        IntensityArray  = np.ma.array(np.zeros((nTimeWindow, nFreqChan, nRow, nAnt)), mask=False)  # unmasked by default
        #fepoch  = filesEpoch(files, hdver=2, meta=0)
        #StartEpochTime  = fepoch[0] # UTC
        #attributes['unix_utc_open'] = StartEpochTime
        #WindowElapsedTime_s  = fepoch - StartEpochTime
        WindowEpochTime = np.zeros(nTimeWindow)
        iWin = -1
        for i in range(nFile):
            fbin = files[i]
            print('reading: %s (%d/%d)'%(fbin, i+1,nFile))
            fh = open(fbin, 'rb')


            BlockHeader_Byte = 64
            for j in range(fnWins[i]):
                iWin += 1
                print('file:',i,'win:',j,'win2:',iWin)

                # average intenisty over N blocks for each time window
                for k in range(nBlockToIntegrate):
                    ### to-do replace by general function
                    p = (BlockHeader_Byte + nBytePerIntensityBlock) *(j*BlockSampleStepSize + k)
                    fh.seek(p)
                    hd = fh.read(BlockHeader_Byte)

                    if (k == 0):
                        tmp = decHeader2(hd)
                        ### to-do replace by general function
                        WindowEpochTime[iWin] = tmp[2] + 2 + (tmp[0]-tmp[4])/prate # UTC
                    buf = fh.read(nBytePerIntensityBlock)
                    data = np.frombuffer(buf, dtype=np.float16).reshape((nTime,nFreqChan,nRow,nAnt))
                    IntensityArray[iWin] += data.mean(axis=0)
                IntensityArray[iWin] /= nBlockToIntegrate

            fh.close()

        StartEpochTime = WindowEpochTime[0]
        attributes['unix_utc_open'] = StartEpochTime
        WindowElapsedTime_s = WindowEpochTime - StartEpochTime
        

        adoneh5(ofile, IntensityArray, 'intensity')
        adoneh5(ofile, WindowElapsedTime_s, 'winSec')

        putAttrs(ofile, attributes)

        ## bandpass normalization
        NormalizedIntensityArray = IntensityArray / np.median(IntensityArray,axis=0) # median over time
        adoneh5(ofile, NormalizedIntensityArray, 'norm.intensity')

    print('Starting epoch time:', StartEpochTime)


    StartTime = StartEpochTime
    if use_local_time: 
        StartTime += 3600 * time_zone_hr

    dt = Time(StartTime + WindowElapsedTime_s, format='unix').to_datetime()
    arrDateTimes.append(dt) 


    arrInts.append(IntensityArray)
    arrNInts.append(NormalizedIntensityArray)
    print("Intensity array shape of row %d (time, frequency, row, antenna): "%j, IntensityArray.shape, NormalizedIntensityArray.shape)
    tsecs.append(WindowElapsedTime_s)

#IntensityArray = np.concatenate(arrInts, axis=3)
#NormalizedIntensityArray = np.concatenate(arrNInts, axis=3)
#freq = np.concatenate(freqs, axis=0)
#print('combine', IntensityArray.shape, NormalizedIntensityArray.shape)


#WindowDateTime = Time(StartTime+WindowElapsedTime_s, format='unix').to_datetime()
#matDT = mdates.date2num(WindowDateTime)
#X, Y = np.meshgrid(WindowElapsedTime_s, freq, indexing='xy')
#X, Y = np.meshgrid(WindowDateTime, freq, indexing='xy')
#X = WindowDateTime
#Y = freq
#print('X:', X)


# if single FPGA or ring buffer. save plots to subfolder
if (nDir==1): 
    odir = odir_plot

if (not os.path.isdir(odir)):
    call('mkdir %s'%odir, shell=True)

##  16 beams in one plot
png = '%s/norm-intensity-16beams.png' % (odir)

#fig, sub = plt.subplots(4,16,figsize=(32,8), sharex=True, sharey=True)
#fig, tmp = plt.subplots(8,16,figsize=(32,12), sharex=True, height_ratios=[2,1,2,1,2,1,2,1])
#fig, tmp = plt.subplots(2,16,figsize=(32,3), sharex=True, height_ratios=[2,1])
fig, tmp = plt.subplots(4,8,figsize=(16,6), sharex=True, height_ratios=[2,1,2,1])
sub  = tmp[0::2].flatten()
sub2 = tmp[1::2].flatten()


all_profiles = []

# Define a custom color for shaded masks (Time/Freq)
shade_color = mcolors.to_rgba('gray', alpha=0.4)

for kk in range(nDir):

    
    NormalizedIntensityArray = arrNInts[kk]
    freq = freqs[kk]

    WindowDateTime = arrDateTimes[kk] 
    #WindowElapsedTime_s = tsecs[kk]
    #WindowDateTime = Time(StartTime+WindowElapsedTime_s, format='unix').to_datetime()
    X = WindowDateTime
    Y = freq

    # Use modularized frequency mask loader
    freq_mask = mu.load_freq_mask(mask_file, freq)


    if (zlim is None):
        vmin = NormalizedIntensityArray.min()
        vmax = NormalizedIntensityArray.max()
    else:
        vmin = zlim[0]
        vmax = zlim[1]
    print('zmin,zmax:', vmin, vmax)

    b = -1

    ProfileArray = np.ma.zeros((nTimeWindow, nRow, nAnt))
 
    for row_idx in range(nRow):
        for ant_idx in range(nAnt):
            b += 1
            ax = sub[b] #sub[nRow-1-row_idx, ant_idx]

            data_2d = NormalizedIntensityArray[:, :, row_idx, ant_idx].copy()

            print('row %d  beam %d: '%(row_idx, ant_idx), data_2d.shape)
            
            invalid_masked_data = mu.mask_invalid_data(data_2d, label=f"FPGA {kk}, beam {ant_idx}", verbose=verbose)
            # Use modularized time masking (saturation/erratic detection)
            #time_mask = mu.get_saturation_mask(data_2d, var_factor=5) 
            # Use raw data for reference spectrum calculation if possible, 
            # otherwise it defaults to data_2d median in get_saturation_mask.
            if time_mask_pow_thresh_dB > 0:
                time_mask = mu.get_saturation_mask(
                    data_2d, 
                    MAD_thresh_dB=time_mask_pow_thresh_dB,
                    show_hist=show_MAD_hist    #debug 
                   
                ) ##IntensityArray[:, :, row_idx, ant_idx]
            else:
                # Disabled
                time_mask = np.zeros(data_2d.shape[0], dtype=bool)
            
            
            
            # Apply masks
            # Use modularized plotting function
            # This handles blanks for NaNs and shading for masks
            final_masked_data = mu.plot_masked_pcolormesh(
                sub[b], WindowDateTime, freq, data_2d, 
                freq_mask=freq_mask, time_mask=time_mask, 
                vmin=(zlim[0] if zlim else None), vmax=(zlim[1] if zlim else None)
            )
          
            #Calculate profile using the masked data
            # average over freq, each node separately

            IntensityProfile = np.ma.median(final_masked_data, axis=1)  
            ProfileArray[:, row_idx, ant_idx] = IntensityProfile
            #IntensityProfile = np.ma.median(NormalizedIntensityArray[:,:,row_idx,ant_idx], axis=1)

            ax2 = sub2[b] #sub2[nRow-1-row_idx, ant_idx]
            ax2.plot(WindowDateTime, IntensityProfile)
            ax2.set_ylim(vmin, vmax)

            # HIGHLIGHT: Add gray shaded vertical bands for masked time regions
            if np.any(time_mask):
                # Identify continuous blocks of True in time_mask for efficient shading
                diff = np.diff(time_mask.astype(int))
                starts = np.where(diff == 1)[0] + 1
                ends = np.where(diff == -1)[0] + 1
                
                # Handle edge cases
                if time_mask[0]: starts = np.insert(starts, 0, 0)
                if time_mask[-1]: ends = np.append(ends, len(time_mask) - 1)
                
                for s, e in zip(starts, ends):
                    sub2[b].axvspan(WindowDateTime[s], WindowDateTime[e], color='gray', alpha=0.7, lw=0)


    all_profiles.append(ProfileArray)


mask_str = freq_mask_str + time_mask_str

fig.autofmt_xdate()
fig.tight_layout(rect=[0,0.03,1,0.95])
fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle("%s : %s"%(ring_name, mask_str) if mask_str else ring_name)
fig.savefig(png)
plt.close(fig)


## another figure for spec.avg intensity vs time
print('plotting intensity profile vs time of alls beams...')
fig_prof,ax_prof = plt.subplots(1,1,figsize=(12,4))


#for row_idx in range(nRow):
#    for ant_idx in range(nAnt):
#          
#        IntensityProfile = np.ma.median(#NormalizedIntensityArray[:,:,row_idx,ant_idx], axis=1) # avg in freq, each node separately
#        ax_prof.plot(WindowDateTime, IntensityProfile, label='%d'%ant_idx)


# Get all beam colors in a single call (using nAnt since nBeam=nAnt in this case); Get color and darken it by scaling RGB components (alpha remains 1)
beam_colors = mu.get_beam_colors(nAnt)

base_cmap = plt.get_cmap('gist_rainbow') 
for kk in range(nDir):
    WindowDateTime = arrDateTimes[kk] # Use pre-calculated DT
    ProfileArray = all_profiles[kk]

    for row_idx in range(nRow):
        for ant_idx in range(nAnt):
  
            # Plot and capture curve
            curve, = ax_prof.plot(WindowDateTime, ProfileArray[:, row_idx, ant_idx], label='beam %d'%(ant_idx), color=beam_colors[ant_idx])

            # Use generalized labeling function passing only the line object

            mu.label_curve_at_peak(ax_prof, curve)


ax_prof.set_xlabel('time (UTC%+d)'%time_zone_hr)
ax_prof.set_ylabel('normalized intensity')
#ax_prof.legend()  # use text label instead

ax_prof.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=np.arange(0,60,5)))
ax_prof.grid(which='both')
ax_prof.tick_params(which='major', length=8)

# Set minor ticks every 5 minutes
ax_prof.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=np.arange(0, 60, 5)))
# Format major ticks: month-day hour:minute
ax_prof.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

# Set major tick appearance
ax_prof.tick_params(which='major', length=8, labelsize=9)

# Adjust rotation for better readability
plt.setp(ax_prof.get_xticklabels(), rotation=15, ha="right")


fig_prof.autofmt_xdate()
fig_prof.tight_layout(rect=[0,0.03,1,0.95])
fig_prof.suptitle("%s : %s"%(ring_name, mask_str) if mask_str else ring_name)
fig_prof.savefig('%s/prof_vs_time.png'%odir)
plt.close(fig_prof)

sys.exit('finished')

# unused
## plot each row separately
'''
for row_idx in range(nRow):
    fig, s2d = plt.subplots(8,4,figsize=(16,12), sharex=True, height_ratios=[2,1,2,1,2,1,2,1])
    sub = s2d[0::2,:].flatten()
    sub2 = s2d[1::2,:].flatten()
    for ii in range(4):
        s2d[ii*2,0].set_ylabel('freq (MHz)')
        s2d[ii*2+1,0].set_ylabel('amp')
        s2d[7,ii].set_xlabel('time (sec)')

    for kk in range(nDir):
        IntensityArray = arrInts[kk]
        NormalizedIntensityArray = arrNInts[kk]
        freq = freqs[kk]
        WindowElapsedTime_s = tsecs[kk]
        WindowDateTime = Time(StartTime+WindowElapsedTime_s, format='unix').to_datetime()
        X = WindowDateTime
        Y = freq

        ## plotting
        #for pt in range(1,2):
        for pt in pts:
            if (pt==0):
                png = '%s/raw.row%d.png' % (odir,row_idx)
                arr = IntensityArray[:,:,row_idx]
                #print(IntensityArray.shape, arr.shape)
            elif (pt==1):
                png = '%s/norm.row%d.png' % (odir,row_idx)
                arr = NormalizedIntensityArray[:,:,row_idx]
            print('plotting:', png, '...')

            if (zlim is None):
                vmin = arr.min()
                vmax = arr.max()
            else:
                vmin = zlim[0]
                vmax = zlim[1]
            print('zmin,zmax:', vmin, vmax)


            for ant_idx in range(nAnt):
                ax = sub[ant_idx]
                #print(arr[:,:,ant_idx].shape)
                #ax.imshow(IntensityArray[:,ant_idx].T, origin='lower', aspect='auto')
                ax.pcolormesh(X,Y,arr[:,:,ant_idx].T, vmin=vmin, vmax=vmax, shading='auto')

                ax2 = sub2[ant_idx]
                y2d = arr[:,:,ant_idx].T
                y1d = y2d.mean(axis=0)
                ax2.plot(WindowDateTime, y1d, marker='.')
                ax2.set_ylim(vmin, vmax)
                ax2.grid()


    fig.autofmt_xdate()
    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(png)
    plt.close(fig)

'''


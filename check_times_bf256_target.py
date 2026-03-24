#!/usr/bin/env python

import numpy as np
import sys, os

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

import astropy.units as u
from astropy.time import Time

from datetime import datetime, timedelta
from astropy.coordinates import SkyCoord, get_sun
from astropy.coordinates import AltAz, EarthLocation

import matplotlib.dates as mdates

import ephem
import pytz
from glob import glob

from  getStationConfig import *

cal_atten = True

inp = sys.argv[0:]
pg  = inp.pop(0)

UTCOffset_hr = 8  #Taiwan time (UTC+8)

theta_off_deg_1d = 0 # deg

fmin = 400
fmax = 800
nChan = 1024
chlim = [0, nChan]

sep1 = 1.0
sep2 = 0.5
sitename = 'FUS'

calstr = datetime.now().strftime('%Y%m%d')
theta_rot_deg = 0.0
tlim = [-2, 2]    # plot range in hours from transit time

#scale = 1 #unused

# attenuation
# MobileMark LPDA;  Ehwhm, Hhwhm= (40, 35) for MED
Antenna = {
    "MMLPDA"    : {"Ehwhm": 30., "Hhwhm": 60.},
    "ASIAALPDA" : {"Ehwhm": 30., "Hhwhm": 45.},
    "MED"       : {"Ehwhm": 40., "Hhwhm": 35.}
}

antenna_type = 'MMLPDA'

Ehwhm   = Antenna[antenna_type]['Ehwhm'] #30 # E-W beam half width (deg) at half max
Hhwhm   = Antenna[antenna_type]['Hhwhm'] #60 # N-S beam half width (deg) at half max


# for tilted antenna 
Ecent   = 0. # antenna pointing center in EW (deg)
Ncent   = 0. # antenna pointing center in NS (deg)

## 1st beam0 convention
# sign=1, the original method
# sign=-1, the more consistent method
sign=-1
    
# arbitrary beam numbers
nBeam1 = None
nBeam2 = None

# arbitrary nAnt, nRow
man_nAnt = 0
man_nRow = 0

# whether to scale the intensity by SEFD, which is proportion to freq^2
scale_snr = True

# whether to save the profiles in an npz file
saveNPZ = False


def atten(x, hwhm):
    sig = hwhm/np.sqrt(2.*np.log(2.))
    return np.exp(-x**2/2./sig**2)

# resolution
t_samp_s = 60    # 60 seconds

# normalize intensity
norm_int = False

'''
#old ver
    {pg} <nAnt> <nRow> <beam0_BFM1d> <beam0_BFM2d> <target> <date> [options]
    
    <nAnt> is the number of antennas in the 1D array
    <nRow> is the number of rows'''
usage = f'''
compute the equitorial coordinates of beams and save to npy file

syntax:
    {pg} <beam0_BFM1d> <beam0_BFM2d> <target> <date> [options]
    
 
    <beam0_BFM1d> is the number of beams to offset in E-W baseline.
                  0 means the westernmost beam is toward the zenith, 
                  1 means shifting the westernmost beam by 1 beam east.
    <beam0_BFM2d> is the number of beams to offset in N-S baseline.
                  0 means the southernmost beam is toward the zenith,
                  1 means shifting the southernmost beam by 1 beam north.
    <target> is the CDS name of the target.
             Both uppercase and lowercase letters are allowed.
             e.g., psr_b0329+54, sun, Crab, Cyg_A, Cas_A, 3c48, ...
    <date> is YYYYMMSS format. It automatically calculate the transit and plot +/- 4 hours.
    
e.g.:
    {pg} 16 16 -6 -7.5 psr_b0329+54 20240819    for bf64 at Fushan with beam0(EW) = -6, beam0(NS) = -7.5
    {pg} 16 4 -7.5 -1.5 PSR_B0329+54 20250314 --site LTN --angle '-10.14 -5.23 7.24 35.24'

options are:
    --nant nAnt         # override nAnt from site config
    --nrow nRow         # override nRow from site config
    --off deg            # additional offset angles of the 1D beams in deg
                        # (default: {theta_off_deg_1d:.1f} deg)
    --site SITE         # predefined site name (e.g., fus, ltn)
                        # (default: {sitename})
    --s_ew EW_sep_m     # antenna spacing in EW direction in meters
                        # (default: {sep1:.1f} m)
    --s_ns NS_sep_m     # antenna spacing in NS direction in meters
                        # (default: {sep2:.1f} m)
                        # For Nantou (--site ltn), default = 2 m
    --caldate YYMMDD    # the date when the calibration data was taken
                        # (default: {calstr})
    --rot DEG           # array misalignment angle in degree
                        # (default: Fushan = -3; Nantou = +0.5; Green Island: = +1.7)
    --noatten           # do not consider antenna attenuation
    --acenter E N       # antenna pointing center in EW and NS (deg)
    --hwhm EW NS        # change the E-W and N-S beam half width at half maximum of antenna , assuming Gaussian beam (in deg)
                        # default: {Ehwhm} {Hhwhm}
    --ant antenna_type  # 'MMLPDA', 'ASIAALPDA', 'MED'  (default: {antenna_type})
    --angle 'deg0 deg1 deg2 ...'
                        # fixed angles for the 2nd beamforming matrix
                        # The number of degrees should be the same as <nRow>
                        # (default: automatically calculated using <beam0_BFM2d>)
    --utcoff UTCOffset_hr  # UTC offset in hour for time zone: affect the time axis and the time of output file (default: {UTCOffset_hr})
    --start HHMMSS      # start time for data recording in HHMMSS or yyyymmdd_HHMMSS format
                        # Use YYYYMMDD_HHMMSS if the start date differs from <date>
                        # (default: 4 hours behind the transit time)
    --end HHMMSS        # end time for data recording in HHMMSS or yyyymmdd_HHMMSS format
                        # Use YYYYMMDD_HHMMSS if the end date differs from <date>.
                        # (default: 4 hours ahead of the transit time)
    --tlim tr_low tr_hi # plot time range in hours from transit time
                        # If --start or --end is given, --tlim will be adjusted.
                        # (default: {tlim[0]:.1f} {tlim[1]:.1f})
    --sign S            # BFM1 convention (1 or -1)
                        # (default: {sign})
    --tsamp t_samp_s    # Time resolution in seconds.
                        # The smaller the t_samp_s, the longer the script takes.
                        # (default: {t_samp_s} seconds)
    --n1 nBeam1         # number of beams in 1st beamform (default: nAnt)
    --n2 nBeam2         # number of beams in 2nd beamform (default: nRow)
    --flim fmin fmax    # Frequency range in MHz
                        # (default: {fmin} {fmax})
    --norm              # Plot intensity after normalization to the max(intensity)
                        # (default: no normalization)
''' 

if (len(inp)<1):
    sys.exit(usage)

while(inp):
    k = inp.pop(0)
    if (k == '--s_ew'):
        sep1 = float(inp.pop(0))
    elif (k == '--s_ns'):
        sep2 = float(inp.pop(0))
    elif (k == '--off'):
        theta_off_deg_1d = float(inp.pop(0))
    elif (k == '--site'):
        sitename = inp.pop(0)
    elif (k == '--caldate'):
        calstr = inp.pop(0)
    elif (k == '--rot'):
        theta_rot_deg = float(inp.pop(0))
    elif (k == '--hwhm'):
        Ehwhm = float(inp.pop(0))   # EW beam width
        Hhwhm = float(inp.pop(0))   # NS beam width
        antenna_type = 'user_defined' 
    elif (k == '--ant'):
        antenna_type = inp.pop(0)
        if antenna_type not in Antenna:
            print(f"Warning: '{antenna_type}' not found in pre-defined antennas. Use default one  with HWHM:  {Ehwhm:.1f} x {Hhwhm:1f} deg")
        else:
            Ehwhm = Antenna[antenna_type]['Ehwhm']
            Hhwhm = Antenna[antenna_type]['Hhwhm']
    elif (k == '--noatten'):
        cal_atten = False
        attinfo = 'noatten'
        print('No antenna attenuation considered')
    elif (k == '--acenter'):
        Ecent = float(inp.pop(0))
        Ncent = float(inp.pop(0))
    elif (k == '--angle'):
        angle = np.array(inp.pop(0).split()).astype(float)
    elif (k == '--tlim'):
        tlim[0] = float(inp.pop(0))
        tlim[1] = float(inp.pop(0))
    elif (k == '--utcoff'):
        UTCOffset_hr = int(inp.pop(0))
    elif (k == '--start'):
        t_rec_start = str(inp.pop(0))
        if len(t_rec_start) not in (15, 6): sys.exit('The format of --start is wrong.')
    elif (k == '--end'):
        t_rec_end = str(inp.pop(0))
        if len(t_rec_end) not in (15, 6): sys.exit('The format of --end is wrong')
    elif (k == '--sign'):
        sign = int(inp.pop(0))
    elif (k == '--tsamp'):
        t_samp_s = int(inp.pop(0))
    elif (k == '--n1'):
        nBeam1 = int(inp.pop(0))
    elif (k == '--n2'):
        nBeam2 = int(inp.pop(0))
    elif (k == '--nant'):
        man_nAnt = int(inp.pop(0))
    elif (k == '--nrow'):
        man_nRow = int(inp.pop(0))
    elif (k == '--flim'):
        fmin = float(inp.pop(0))
        fmax = float(inp.pop(0))
    elif (k == '--norm'):
        norm_int = True
    elif (k == '--no_freq2'):
        scale_snr = False
    elif (k == '--chlim'):
        chlim[0] = int(inp.pop(0))
        chlim[1] = int(inp.pop(0))
    elif (k == '--save'):
        saveNPZ = True
    #elif (k.startswith('-')): #this confuses with negative numbers
    elif (k.startswith('--')):
        sys.exit('unknown option: %s'%k)
    else:
        #nAnt = int(k) # 16
        #nRow = int(inp.pop(0)) # 16
        beam01 = float(k) #float(inp.pop(0)) # -6
        beam02 = float(inp.pop(0)) # -7.5
        targetname = str(inp.pop(0))
        obsdate = str(inp.pop(0))
        obsdt = datetime.strptime(obsdate, '%Y%m%d') - timedelta(hours=UTCOffset_hr)
        calstr = obsdate

dur_hr = tlim[1] - tlim[0] # duration in hours

print(f'UTC offset: {UTCOffset_hr} hour')



# Overwrite current setting if station name is found
station_config = GetStationConfig(sitename)
print(f'Station Config for {sitename}: ', station_config)
if station_config:
    sitename = station_config["Name"]
    theta_rot_deg = station_config["Rotate_deg"]
    # overwrite the arguments
    nAnt    = station_config["NAntPerRow"]
    nRow    = station_config["NRow"]
else:
    sys.exit('Invalid site name: %s' % sitename)

if 'angle' in locals():
    if (len(angle) != nRow):
        sys.exit('Check angle: len(angle) != nRow')

# antenna radiation pattern
attinfo = 'att%dn%d' % (Ehwhm, Hhwhm)

# override nAnt, nRow if manually set
if (man_nAnt > 0):
    nAnt = man_nAnt
if (man_nRow > 0):
    nRow = man_nRow

if (nBeam1 is None):
    nBeam1 = nAnt
if (nBeam2 is None):
    nBeam2 = nRow


#print('scale =', scale)
    
#if scale==0:
#    print('!!! Confirm that SCALE = 0')


freq_MHz = np.linspace(fmin, fmax, nChan, endpoint=False) 
freq_Hz = freq_MHz * 1e6
lamb = 2.998e8 / (freq_MHz * 1e6)
lamb0 = lamb[0]



#if len(tmp.keys()) == 0:
#    sys.exit('check sitename: %s' % sitename)


obs = ephem.Observer()
obs.date = obsdt
obs.epoch = '2000'

# use general config file
obs.lon = station_config['Longitude']
obs.lat = station_config['Latitude']
obs.elevation = float(station_config['Elevation'])

obs.pressure = 1013 # mBar
obs.temp = 25 # Celsius

site = obs

# original beam_sep
sin_theta_m1_ori = lamb0/sep1/nAnt*(np.arange(nBeam1)+beam01)
sin_theta_m2_ori = lamb0/sep2/nRow*(np.arange(nBeam2)+beam02)

# angle = np.array([-1.68, 9.45, 30.95, 35.24])

if 'angle' in locals():
    sin_theta_m2_ori = np.sin(angle/180.*np.pi)
    print(angle)

if theta_off_deg_1d != 0:
    tmp = np.arcsin(sin_theta_m1_ori)/np.pi*180.
    tmp += theta_off_deg_1d
    sin_theta_m1_ori = np.sin(tmp/180.*np.pi)

beam_angle_1d_deg_ori = np.arcsin(sin_theta_m1_ori)*180/np.pi
beam_angle_2d_deg_ori = np.arcsin(sin_theta_m2_ori)*180/np.pi


# apply calibration using Sun
caldt = datetime.strptime(calstr, '%Y%m%d')
site.date = caldt
suncal = ephem.Sun()
obs.date = site.next_transit(suncal)

suncal.compute(site)

max_el = suncal.alt

zenith_angle = np.pi/2. - float(max_el)

theta_rot = theta_rot_deg/180.*np.pi
theta_off = theta_rot * zenith_angle
theta_off_deg = theta_off / np.pi*180.
#print('misalignment angle: %.2f deg'%theta_rot_deg)
#print('central beam offset: %.2f deg' % (theta_off_deg))

beam_angle_1d_deg = beam_angle_1d_deg_ori + theta_off_deg
beam_angle_1d_sin = np.sin(beam_angle_1d_deg/180*np.pi)

beam_angle_2d_deg = beam_angle_2d_deg_ori+ theta_off_deg
beam_angle_2d_sin = np.sin(beam_angle_2d_deg/180*np.pi)

#print('beam_angle_1d_deg', beam_angle_1d_deg)
#print('beam_angle_2d_deg', beam_angle_2d_deg)

def rot(pos, theta):
    mat = np.zeros((3,3))
    mat[0,0] = np.cos(theta)
    mat[0,1] = -np.sin(theta)
    mat[1,0] = np.sin(theta)
    mat[1,1] = mat[0,0]
    mat[2,2] = 1.

    return np.dot(pos,mat.T)

pos0 = np.zeros((nAnt*nRow, 3)) 
pos0[:,0] = np.concatenate(([np.arange(nAnt)*sep1]*nRow))
pos0[:,1] = np.concatenate((np.stack([np.arange(nRow)*sep2]*nAnt, axis=1)))
pos = rot(pos0, theta_rot) 
pos = pos.reshape(len(pos)//nAnt, nAnt,3)

pos0 = pos0.reshape(len(pos0)//nAnt, nAnt,3)

beam_angle_1d_az = np.zeros_like(beam_angle_1d_deg)

beam_angle_1d_az[np.where(beam_angle_1d_deg>=0)] = np.radians(270)
beam_angle_1d_az[np.where(beam_angle_1d_deg<0)] = np.radians(90)

sin_theta_m = sin_theta_m1_ori
BFM1 = np.zeros((nBeam2,nBeam1,nAnt,nChan), dtype=complex)


#for i in range(4):
for i in range(nRow):
    BFM1[i] = np.exp(sign*2.j*np.pi*pos0[i,:,0].reshape((1,nAnt,1))*sin_theta_m.reshape((nBeam1,1,1)) / 2.998e8 * freq_Hz.reshape((1, 1, nChan)))

#BFM1 *= 127/scale ## ? not sure
BFM1 /= nAnt
if (sign == -1):
    BFM1 = np.flip(BFM1, axis=1)

#sin_theta_m = sin_theta_m2_ori * (sep2/0.5)
sin_theta_m = sin_theta_m2_ori

BFM2 = np.zeros((nBeam2,nRow,nChan), dtype=complex)

BTau_s = pos0[:,0,1].reshape((1,nRow,1))*sin_theta_m.reshape((nBeam2,1,1))/2.998e8
BFM2 = np.exp(-1.j*2.*np.pi*BTau_s*freq_Hz.reshape((1,1,nChan)))
#BFM2 *= scale
BFM2 /= nRow
BFM2 = BFM2.transpose((2,1,0))

del sin_theta_m, BTau_s


# calculate beam pattern

nSky = int((dur_hr*3600+120)/t_samp_s)

inVolt_all = np.zeros((nRow,nAnt,nSky,nChan), dtype=complex)

obs.date = obsdt

if targetname.lower() == 'sun':
    body = ephem.Sun()

else:
    target_coord = SkyCoord.from_name(targetname)

    ra_h, ra_m, ra_s = target_coord.ra.hms
    ra_str = f"{int(ra_h):02}:{int(ra_m):02}:{ra_s:05.2f}"
    dec_d, dec_m, dec_s = target_coord.dec.dms
    dec_str = f"{int(dec_d):+03}:{int(dec_m):02}:{dec_s:05.2f}"
    print('target coord', ra_str, dec_str)

    body = ephem.FixedBody()
    body._ra = ephem.hours(ra_str)
    body._dec = ephem.degrees(dec_str)
    body._epoch = '2000'

    body.compute(obs)

t_trans = obs.next_transit(body) #UTC Date
#t_trans = ephem.localtime(t_trans)
#t_trans_utc = datetime.utcfromtimestamp(t_trans.timestamp())

# UTC datetime
t_trans_utc = t_trans.datetime()

# local datetime (manual offset)
t_trans_local = t_trans_utc + timedelta(hours=UTCOffset_hr)

print(f'{targetname} transit at {t_trans_local} (local time UTC{UTCOffset_hr:+d} )')



#t_start = t_trans_utc + timedelta(hours=tlim[0], minutes=0)
#t_end   = t_trans_utc + timedelta(hours=tlim[1], minutes=0)
#t_start = t_trans_utc - timedelta(hours=4)
t_start = (t_trans_utc + timedelta(hours=tlim[0])).replace(second=0, microsecond=0)
t_end = (t_trans_utc + timedelta(hours=tlim[1])).replace(second=0, microsecond=0)
#t_end   = t_trans_utc + timedelta(hours=4)


ut2 = np.linspace(t_start.timestamp(), (t_end+timedelta(minutes=2)).timestamp(), nSky, endpoint=False) # +2min to cover entire x axis
ut2 = [datetime.fromtimestamp(ts) for ts in ut2]; ut2 = np.array(ut2)
#ut_epoch = Time(ut2, 'datetime').to_value('unix')

ha = []
alt = []
az  = []
for i in range(nSky):
    obs.date = ut2[i]
    body.compute(obs)
    tmp = body.ha
    if (tmp > np.pi):
        tmp -= 2.*np.pi
    ha.append(tmp)
    alt.append(body.alt)
    az.append(body.az)

ha = np.array(ha)
ha_deg = ha/np.pi*180.
el = np.array(alt)
az = np.array(az)

za = np.pi/2 - el   # zenith angle, theta
z = np.cos(za)
r = np.sin(za)
x = r*np.sin(az)
y = r*np.cos(az)

unitVec = np.array([x,y,z])
#Tau_m = np.dot(pos, unitVec)
#Tau_s = Tau_m / 2.

inVolt = np.zeros((nRow,nAnt,nSky,nChan), dtype=complex)

for r in range(nRow):
    Tau_m = np.dot(pos[r], unitVec)
    Tau_s = Tau_m / 2.998e8
    phi = 2.*np.pi*Tau_s.reshape((nAnt,nSky,1))*freq_Hz.reshape((1,1,nChan))
    inVolt[r] = np.exp(1.j*phi)

delaz = az.copy()
delaz[az < np.pi] -= np.radians(90)
delaz[az >= np.pi] -= np.radians(270)
delalt = za
tmp1 = delalt * np.sin(delaz) / np.tan(delaz)
tmp2 = delalt * np.sin(delaz)
tmp1 *= (180 / np.pi)
tmp2 *= (180 / np.pi)

EWang = np.arctan(x/z)/np.pi*180.
NSang = np.arctan(y/z)/np.pi*180.

if cal_atten:
    print('Consider antenna attenuation')
    #print('debug: EWang, Ecent, Ehwhm', EWang, Ecent, Ehwhm)
    #print('debug: NSang, Ncent, Hhwhm', NSang, Ncent, Hhwhm)
    Eatt = atten(EWang-Ecent, Ehwhm)
    Hatt = atten(NSang-Ncent, Hhwhm)
    #print('debug: max(Eatt)', Eatt.max())
    #print('debug: max(Hatt)', Hatt.max())
    att0 = Eatt*Hatt
    inVolt *= att0[np.newaxis,np.newaxis,:,np.newaxis]

outVolt = np.zeros((nBeam2, nBeam1, nSky, nChan), dtype=complex)

for ch in range(nChan):
    tmp = np.empty((nRow,nBeam1,nSky), dtype=complex)
    for r in range(nRow):
        tmp[r] = np.dot(BFM1[r,:,:,ch], inVolt[r,:,:,ch])

    for t in range(nSky):
        #outVolt[:,:,t,ch] = tmp[:,:,t] # 1D only
        outVolt[:,:,t,ch] = np.dot(BFM2[ch].T, tmp[:,:,t]) # 2D

outInt = np.ma.array((np.abs(outVolt)**2).mean(axis=3), mask=False)
outInt = np.flip(outInt, axis=1)

outInt1 = np.abs(outVolt)**2; outInt1 = np.flip(outInt1, axis=1)

tmp = np.abs(outVolt)**2
if (scale_snr):
    tmp /= (freq_MHz/400.)**2
#outInt = np.ma.array(tmp.mean(axis=3), mask=False); outInt = np.flip(outInt, axis=1)
outInt = np.ma.array(tmp[:,:,:,chlim[0]:chlim[1]].mean(axis=3), mask=False)
outInt = np.flip(outInt, axis=1)
outInt1 = np.flip(tmp, axis=1)

beam_int = outInt.copy()
        
del inVolt, outVolt, outInt1, outInt

if norm_int:
    max_val = np.max(beam_int)
    beam_int = beam_int / max_val

a1 = beam_int.copy()

b1 = np.zeros((len(el)))
c1 = np.zeros((len(el),2)) # beamID

for i in range(len(el)):
    b1[i] = np.max(a1[:,:,i])
    c1[i] = np.unravel_index(np.argmax(a1[:,:,i]), a1[:,:,i].shape)


colors = cm.tab20(np.linspace(0,1,nRow))
fig, ax = plt.subplots(1, 1, figsize=(14,4))

for i in range(nBeam1):
    for r in range(nBeam2):
        ax.plot(ut2 + timedelta(hours=UTCOffset_hr), a1[r,i,:], ls='-', color=colors[r])
    
    max_bid = np.argmax(np.max(a1[:,i], axis=1))
    #print(i, max_bid)
    tmp = np.where(a1[max_bid,i,:] == np.max(a1[max_bid,i,:]))[0][0]
    ax.text((ut2+timedelta(hours=UTCOffset_hr))[tmp], a1[max_bid,i,:][tmp], i)



# setting time axis
t_start += timedelta(hours=UTCOffset_hr)
t_end += timedelta(hours=UTCOffset_hr)
x_min = t_trans_utc + timedelta(hours=tlim[0]+UTCOffset_hr)
x_max = t_trans_utc + timedelta(hours=tlim[1]+UTCOffset_hr)

if 't_rec_start' in globals():
    if len(t_rec_start) == 15:
        t_start = datetime.strptime(t_rec_start, '%Y%m%d-%H%M%S')
    if len(t_rec_start) == 6:
        t_start = datetime.strptime(obsdate+'-'+t_rec_start, '%Y%m%d-%H%M%S')
    x_min = t_start
if 't_rec_end' in globals():
    if len(t_rec_end) == 15:
        t_end = datetime.strptime(t_rec_end, '%Y%m%d-%H%M%S')
    if len(t_rec_end) == 6:
        t_end = datetime.strptime(obsdate+'-'+t_rec_end, '%Y%m%d-%H%M%S')
    x_max = t_end

# png, txt, npz files
OutFilePrefix = 'times_%s_%s_%s_%dx%d_b0_%.2f_%.2f_utc_%d' % (targetname, int(obsdate), sitename, nBeam1, nBeam2, beam01, beam02, UTCOffset_hr)

f_out_txt = '%s.txt' % (OutFilePrefix)
with open(f_out_txt, 'w') as f:
    f.write('#%s %s\ntransit at %s (local time UTC%+d)\n#\n' % (targetname, sitename, t_trans_local, UTCOffset_hr))
    f.write('#start_time end_time beam_id\n')

b_start_arr = []
b_end_arr = []
max_bid_arr = []

i = nBeam1-1
max_bid = np.argmax(np.max(a1[:,i], axis=1))
#print('i=%d'%nBeam1, max_bid, a1[max_bid,i].max())
j = np.where(a1[max_bid,i,:] > 0.15*a1[max_bid,i].max())[0][0]
b_start = (ut2+timedelta(hours=UTCOffset_hr))[j]
#.strftime('%Y%m%d_%H%M%S')#.strftime('%Y%m%d %H:%M')

# SH: why #0 beam is treated separately?
for i in range(nBeam1-1, 0, -1):
    max_bid = np.argmax(np.max(a1[:,i], axis=1))
    y1 = a1[max_bid,i]
    y2 = a1[max_bid,i-1]
    j1 = np.ma.argmax(y1)
    j2 = np.ma.argmax(y2)
    y1.mask = True; y1.mask[j1:j2] = False
    y2.mask = True; y2.mask[j1:j2] = False
    j = np.ma.argmin(np.ma.abs(y1-y2))
    
    b_end = (ut2+timedelta(hours=UTCOffset_hr))[j]
    #.strftime('%Y%m%d_%H%M%S')#.strftime('%Y%m%d %H:%M')

    # ensure the time is in range for observation scheduling
    if (b_end <= t_start) or (b_start >= t_end): 
        b_start = b_end
        continue
    elif (b_start < t_start): b_start = t_start
    elif (b_end > t_end): b_end = t_end

    b_start_str = b_start.strftime('%Y%m%d_%H%M%S')
    b_end_str = b_end.strftime('%Y%m%d_%H%M%S')

    print('Beam %03d (row %d, beam %02d): %s - %s' % (max_bid*nBeam1+i, max_bid, i, b_start_str, b_end_str))
    #with open(f_out_txt, 'a') as f:
    #    f.write('Beam %03d (row %d, beam %02d): %s - %s\n' % (max_bid*16+i, max_bid, i, b_start, b_end))
    #print('%s %s %d' % (b_start, b_end, max_bid*16+i))
    with open(f_out_txt, 'a') as f:
        f.write('%s %s %d\n' % (b_start_str, b_end_str, max_bid*nBeam1+i))

    # NPZ output
    b_start_arr.append(b_start_str)
    b_end_arr.append(b_end_str)
    max_bid_arr.append(max_bid*nBeam1+i)

    b_start = b_end

#i = 0
for i in [0]:
    max_bid = np.argmax(np.max(a1[:,i], axis=1))
    j = np.where(a1[max_bid,i,:] > 0.15*a1[max_bid,i].max())[0][-1]
    b_end = ((ut2+timedelta(hours=UTCOffset_hr))[j])
    #.strftime('%Y%m%d_%H%M%S')#.strftime('%Y%m%d %H:%M')
    if (b_end <= t_start) or (b_start >= t_end): continue
    elif (b_start < t_start): b_start = t_start
    elif (b_end > t_end): b_end = t_end

    b_start_str = b_start.strftime('%Y%m%d_%H%M%S')
    b_end_str = b_end.strftime('%Y%m%d_%H%M%S')
    print('Beam %03d (row %02d, beam %02d): %s - %s' % (max_bid*nBeam1+i, max_bid, i, b_start_str, b_end_str))
    #with open(f_out_txt, 'a') as f:
    #    f.write('Beam %03d (row %d, beam %02d): %s - %s' % (max_bid*16+i, max_bid, i, b_start, b_end))
    #print('%s %s %d' % (b_start, b_end, max_bid*16+i))
    with open(f_out_txt, 'a') as f:
        f.write('%s %s %d' % (b_start_str, b_end_str, max_bid*nBeam1+i))

    b_start_arr.append(b_start) # last beam
    b_end_arr.append(b_end)
    max_bid_arr.append(max_bid*nBeam1+i)


b_start_arr = np.array(b_start_arr)
b_end_arr = np.array(b_end_arr)
max_bid_arr = np.array(max_bid_arr)

#ax.set_xlim([t_trans_utc + timedelta(hours=tlim[0]+ UTCOffset_hr), t_trans_utc + timedelta(hours=tlim[1]+8)])
ax.set_xlim(x_min, x_max)

# label E-W beams
for i in range(nBeam1):
    max_bid = np.argmax(np.max(a1[:,i], axis=1))
    tmp = np.where(a1[max_bid,i,:] == np.max(a1[max_bid,i,:]))[0][0]
    label_loc = (ut2+timedelta(hours=UTCOffset_hr))[tmp]
    if x_min <= label_loc <= x_max:
        ax.text(label_loc, a1[max_bid,i,:][tmp], i)


ax.axvline(x=t_trans_local, color='limegreen', lw=10, zorder=0, alpha=0.3)
lines = [Line2D([0], [0], linewidth=2, color=x) for x in colors]
labels = ['row %d' % x for x in range(nRow)]
labels.extend(['transit'])
lines.append(Line2D([0.4, 0.5], [0, 0], linewidth=6, color='limegreen'))
ax.legend(lines, labels[:len(lines)], handlelength=2.0, handletextpad=0.5, ncols=2, fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
ax.set_xlabel(f'Local time (UTC{UTCOffset_hr:+d})')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=np.arange(0,60,5)))
ax.grid(which='both')

# common title with Latex for symbols
title1 = (
    rf"{targetname} at {sitename} "
    rf"($\theta_{{\mathrm{{rot}}}} = {theta_rot_deg:.2f}^\circ$), "
    rf"transit at {t_trans_local:%Y-%m-%d %H:%M:%S}, "
)

if 'angle' in locals(): 
    angle_str = ', '.join([f"{x}^\\circ" for x in angle])
    title1 += rf"$\theta_{{\mathrm{{2nd}}}} = ({angle_str})$, "
    

title2 = (
    rf"beam0: (1st={beam01:.2f}$^\circ$, 2nd={beam02:.2f}$^\circ$), " 
    rf"$\theta_{{\mathrm{{off,EW}}}} = {theta_off_deg_1d:.2f}^\circ$, " 
    rf"antenna HWHM = ${Ehwhm}^\circ \times {Hhwhm}^\circ$"
)
# \n not working in raw string 'r'
ax.set_title(title1 + "\n" + title2)
'''
if ('angle' in locals()): 
    ax.set_title(rf'{targetname}, {sitename} ($\theta_\mathrm{{rot}}$ = {theta_rot_deg:.2f}$\circ$), theta_2nd = {angle} deg \ntransit at {t_trans_local.strftime('%Y-%m-%d %H:%M:%S')}, 1st_beam0 = {beam01:.2f}, 2nd_beam0 = {beam02:.2f}$\circ$, $\theta_{off,EW}$ = {theta_off_deg_1d:.2f} \circ$, {attinfo}'
                 
else: 
    ax.set_title('%s, %s (theta_rot = %.2f deg)\ntransit at %s, 1st_beam0 = %.2f, 2nd_beam0 = %.2f, theta_off_EW = %.2f deg, %s' % (targetname, sitename, theta_rot_deg, t_trans_local.strftime('%Y-%m-%d %H:%M:%S'), beam01, beam02, theta_off_deg_1d, attinfo))
'''

ax.tick_params(which='major', length=UTCOffset_hr)
fig.tight_layout()
fig.savefig('%s.png' % (OutFilePrefix))
plt.show()


## save the profiles
if (saveNPZ):
    f_out_npz = '%s.npz' % f_out_name
    # note: np.load(fnpz, allow_pickle=True)
    attr = {
            'src':targetname,
            'site':sitename,
            'theta_rot_deg=':theta_rot_deg,
            'nAnt':nAnt,
            'nRow':nRow,
            'beam01':beam01,
            'beam02':beam02,
            'nBeam1':nBeam1,
            'nBeam2':nBeam2,
            'sep1':sep1,
            'sep2':sep2,
            'EWhwhm':Ehwhm,
            'NShwhm':Hhwhm,
            'EWcent':Ecent,
            'NScent':Ncent
            }

    np.savez(f_out_npz,
            attr=attr,
            beam_prof=a1,       # shape (BeamNS, BeamEW, time)
            datetime_ut=ut2,    # datetime array, shaep (time,)
            az=az,
            el=el,
            EWang=EWang,
            NSang=NSang,
            atten=att0,
            b_start=b_start_arr,    # estimated beam switching on time, shape (BeamNS,)
            b_end=b_end_arr,        # estimated beam switching off time, shape (BeamNS,)
            max_bid=max_bid_arr     # beam id of the peak intensity, shape (BeamNS,)
            )

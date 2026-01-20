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

cal_atten = True

inp = sys.argv[0:]
pg  = inp.pop(0)

theta_off_deg_1d = 0 # deg

nChan = 1024

sep1 = 1.0
sep2 = 0.5
sitename = 'FUS'

calstr = datetime.now().strftime('%Y%m%d')
theta_rot_deg = None
tlim = [-2, 2]    # plot range in hours from transit time

scale = 1

# attenuation
Ehwhm   = 30 # E-W beam half width (deg) at half max
Hhwhm   = 60 # N-S beam half width (deg) at half max
Ecent   = 0. # antenna pointing center in EW (deg)
Ncent   = 0. # antenna pointing center in NS (deg)

## 1st beam0 convention
# sign=1, the original method
# sign=-1, the more consistent method
sign=-1
    
# arbitrary beam numbers
nBeam1 = None
nBeam2 = None

# whether to scale the intensity by SEFD, which is proportion to freq^2
scale_snr = True


def atten(x, hwhm):
    sig = hwhm/np.sqrt(2.*np.log(2.))
    return np.exp(-x**2/2./sig**2)

usage = '''
compute the equitorial coordinates of beams and save to npy file

syntax:
    %s <nAnt> <nRow> <beam0_BFM1d> <beam0_BFM2d> <target> <date> [options]
    
    <nAnt> is the number of antennas in the 1D array
    <nRow> is the number of rows
    <beam0_BFM1d> is the number of beams to offset in E-W baseline.
                  0 means the Southern most beam is 0 deg N, 
                  1 means shifting the Southern most beam by 1 beam north.
    <beam0_BFM2d> is the number of beams to offset in E-W baseline.
    <target> is the CDS name of the target.
             e.g., psr_b0329+54, sun, ..
    <date> is YYYYMMSS format. It automatically calculate the transit and plot +/- 4 hours.
    
e.g.:
    %s 16 16 -6 -7.5 psr_b0329+54 20240819    for bf64 at Fushan with beam0(EW) = -6, beam0(NS) = -7.5

options are:
    -off deg            # additional offset angles of the 1D beams in deg
                        # (default: %.1f deg)
    -s_ew EW_sep_m      # antenna spacing in EW direction in meters
                        # (default: %.1f m)
    -s_ns NS_sep_m      # antenna spacing in NS direction in meters
                        # (default = %.1f m)
    --site SITE         # predefined site name (e.g., fus, ltn)
                        # (default: %s)
    --caldate YYMMDD    # the date when the calibration data was taken
                        # (default: %s)
    --rot DEG           # array misalignment angle in degree
                        # (default: Fushan = -3; Nantou = +0.5)
    --scale SCALE       # scale factor
                        # (default: 1)
    --noatten           # do not consider antenna attenuation
    --acenter E N       # antenna pointing center in EW and NS (deg)
    --hwhm EW NS        # change the E-W and N-S beam half width at half maximum (in deg)
                        # default: %d %d
    --angle 'deg0 deg1 deg2 ...'
                        # fixed angles for the 2nd beamforming matrix
                        # The number of degrees should be the same as <nRow>
                        # (default: automatically calculated using <beam0_BFM2d>)
    --tlim tr_low tr_hi # plot time range in hours from transit time
                        # (%.1f %.1f)
    --sign S            # BFM1 convention (1 or -1)
                        # (%d)
    --n1 nBeam1         # number of beams in 1st beamform (default: nAnt)
    --n2 nBeam2         # number of beams in 2nd beamform (default: nRow)
    --no_freq2          # disable the freq^2 scaling of SNR
''' % (pg, pg, theta_off_deg_1d, sep1, sep2, sitename, calstr, Ehwhm, Hhwhm, tlim[0], tlim[1], sign)

if (len(inp)<1):
    sys.exit(usage)

while(inp):
    k = inp.pop(0)
    if (k == '-s_ew'):
        sep1 = float(inp.pop(0))
    elif (k == '-s_ns'):
        sep2 = float(inp.pop(0))
    elif (k == '-off'):
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
    elif (k == '--sign'):
        sign = int(inp.pop(0))
    elif (k == '--n1'):
        nBeam1 = int(inp.pop(0))
    elif (k == '--n2'):
        nBeam2 = int(inp.pop(0))
    elif (k == '--no_freq2'):
        scale_snr = False
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        nAnt = int(k) # 16
        nRow = int(inp.pop(0)) # 16
        beam01 = float(inp.pop(0)) # -6
        beam02 = float(inp.pop(0)) # -7.5
        targetname = str(inp.pop(0))
        obsdate = str(inp.pop(0))
        obsdt = datetime.strptime(obsdate, '%Y%m%d') - timedelta(hours=8)
        calstr = obsdate

if 'angle' in locals():
    if (len(angle) != nRow):
        sys.exit('Check angle: len(angle) != nRow')

attinfo = 'att%dn%d' % (Ehwhm, Hhwhm)
if (nBeam1 is None):
    nBeam1 = nAnt
if (nBeam2 is None):
    nBeam2 = nRow

if sitename.lower() in ['fus', 'fushan']:
    sitename = 'Fushan'
if sitename.lower() in ['ltn', 'longtien', 'nantou']:
    sitename = 'Longtien'
if sitename.lower() in ['grn', 'lyudao']:
    sitename = 'Lyudao'

if (theta_rot_deg is None):
    if (sitename.lower() == 'fushan'):
        theta_rot_deg = -3.0
    elif (sitename.lower() == 'longtien'):
        theta_rot_deg = +0.5
    elif (sitename.lower() == 'lyudao'):
        theta_rot_deg = 0.
    else:
        theta_rot_deg = 0.


print('scale =', scale)
    
if scale==0:
    print('!!! Confirm that SCALE = 0')


#fMHz = np.linspace(400,800, nChan, endpoint=False) 
fMHz = np.linspace(400,800, nChan, endpoint=False) 
freq = fMHz * 1e6
lamb = 2.998e8 / (fMHz * 1e6)
lamb0 = lamb[0]

# Read coordinates of Fushan and Nantou
tmp = {}
if sitename.lower() == 'fushan':
    tmp['LAT'] = '24:45:23.41411'
    tmp['LON'] = '121:34:53.93382'
    tmp['ELV'] = 642.9882

if sitename.lower() == 'longtien':
    tmp['LAT'] = '23:42:52.49877'
    tmp['LON'] = '120:49:27.77502'
    tmp['ELV'] = 878.7997

if sitename.lower() == 'lyudao':
    tmp['LAT'] = '+22.6750'
    tmp['LON'] = '121.5007'
    tmp['ELV'] = 15.0

if len(tmp.keys()) == 0:
    sys.exit('check sitename: %s' % sitename)

if sitename.lower() in ['ltn', 'longtien', 'nantou']:
    sep2 = 2
    print('change sep2 to 2')

obs = ephem.Observer()
obs.date = obsdt
obs.epoch = '2000'
obs.lon = tmp['LON']
obs.lat = tmp['LAT']
obs.elevation = float(tmp['ELV'])

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
BFM1 = np.zeros((nRow,nBeam1,nAnt,nChan), dtype=complex)


#for i in range(4):
for i in range(nRow):
    BFM1[i] = np.exp(sign*2.j*np.pi*pos0[i,:,0].reshape((1,nAnt,1))*sin_theta_m.reshape((nBeam1,1,1)) / 2.998e8 * freq.reshape((1, 1, nChan)))

#BFM1 *= 127/scale ## ? not sure
BFM1 /= nAnt
if (sign==-1):
    BFM1 = np.flip(BFM1, axis=1)

#sin_theta_m = sin_theta_m2_ori * (sep2/0.5)
sin_theta_m = sin_theta_m2_ori

BFM2 = np.zeros((nBeam2,nRow,nChan), dtype=complex)

BTau_s = pos0[:,0,1].reshape((1,nRow,1))*sin_theta_m.reshape((nBeam2,1,1))/2.998e8
BFM2 = np.exp(-1.j*2.*np.pi*BTau_s*freq.reshape((1,1,nChan)))
#BFM2 *= scale
BFM2 /= nRow
BFM2 = BFM2.transpose((2,1,0))

del sin_theta_m, BTau_s


# calculate beam pattern

nSky = 960

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

t_trans = obs.next_transit(body)
t_trans = ephem.localtime(t_trans)
print(targetname, 'transit at', t_trans, '(local time)')
t_trans_utc = datetime.utcfromtimestamp(t_trans.timestamp())
t_start = t_trans_utc + timedelta(hours=tlim[0], minutes=0)
t_end   = t_trans_utc + timedelta(hours=tlim[1], minutes=0)
#t_start = t_trans_utc - timedelta(hours=4)
#t_end   = t_trans_utc + timedelta(hours=4)

ut2 = np.linspace(t_start.timestamp(), t_end.timestamp(), nSky)
ut2 = [datetime.fromtimestamp(ts) for ts in ut2]; ut2 = np.array(ut2)

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
Tau_m = np.dot(pos, unitVec)
Tau_s = Tau_m / 2.

inVolt = np.zeros((nRow,nAnt,nSky,nChan), dtype=complex)

for r in range(nRow):
    Tau_m = np.dot(pos[r], unitVec)
    Tau_s = Tau_m / 2.998e8
    phi = 2.*np.pi*Tau_s.reshape((nAnt,nSky,1))*freq.reshape((1,1,nChan))
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

outVolt = np.zeros((nRow, nBeam1, nSky, nChan), dtype=complex)

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
    tmp /= (fMHz/400.)**2
outInt = np.ma.array(tmp.mean(axis=3), mask=False); outInt = np.flip(outInt, axis=1)
outInt1 = np.flip(tmp, axis=1)

beam_int = outInt.copy()
        
del inVolt, outVolt, outInt1, outInt

max_val = np.max(beam_int)
#beam_int = beam_int / max_val

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
        ax.plot(ut2 + timedelta(hours=8), a1[r,i,:], ls='-', color=colors[r])
    
    max_bid = np.argmax(np.max(a1[:,i], axis=1))
    #print(i, max_bid)
    tmp = np.where(a1[max_bid,i,:] == np.max(a1[max_bid,i,:]))[0][0]
    ax.text((ut2+timedelta(hours=8))[tmp], a1[max_bid,i,:][tmp], i)


f_out_txt = 'times_%s_%s_%s_%dx%d_b0_%.1f.txt' % (targetname, int(obsdate), sitename, nBeam1, nBeam2, beam01)
with open(f_out_txt, 'w') as f:
    f.write('#%s transit at %s (local time)\n#\n' % (targetname, t_trans))
    f.write('#start_time end_time beam_id\n')

i = nBeam1-1
max_bid = np.argmax(np.max(a1[:,i], axis=1))
print('i=%d'%nBeam1, max_bid, a1[max_bid,i].max())
j = np.where(a1[max_bid,i,:] > 0.15*a1[max_bid,i].max())[0][0]
b_start = ((ut2+timedelta(hours=8))[j]).strftime('%Y%m%d_%H%M%S')#.strftime('%Y%m%d %H:%M')

for i in range(nBeam1-1, 0, -1):
    max_bid = np.argmax(np.max(a1[:,i], axis=1))
    y1 = a1[max_bid,i]
    y2 = a1[max_bid,i-1]
    j1 = np.ma.argmax(y1)
    j2 = np.ma.argmax(y2)
    y1.mask = True; y1.mask[j1:j2] = False
    y2.mask = True; y2.mask[j1:j2] = False
    j = np.ma.argmin(np.ma.abs(y1-y2))
    b_end = ((ut2+timedelta(hours=8))[j]).strftime('%Y%m%d_%H%M%S')#.strftime('%Y%m%d %H:%M')
    print('Beam %03d (row %d, beam %02d): %s - %s' % (max_bid*nBeam1+i, max_bid, i, b_start, b_end))
    #with open(f_out_txt, 'a') as f:
    #    f.write('Beam %03d (row %d, beam %02d): %s - %s\n' % (max_bid*16+i, max_bid, i, b_start, b_end))
    #print('%s %s %d' % (b_start, b_end, max_bid*16+i))
    with open(f_out_txt, 'a') as f:
        f.write('%s %s %d\n' % (b_start, b_end, max_bid*nBeam1+i))
    b_start = b_end

i = 0
max_bid = np.argmax(np.max(a1[:,i], axis=1))
j = np.where(a1[max_bid,i,:] > 0.15*a1[max_bid,i].max())[0][-1]
b_end = ((ut2+timedelta(hours=8))[j]).strftime('%Y%m%d_%H%M%S')#.strftime('%Y%m%d %H:%M')
print('Beam %03d (row %d, beam %02d): %s - %s' % (max_bid*nBeam1+i, max_bid, i, b_start, b_end))
#with open(f_out_txt, 'a') as f:
#    f.write('Beam %03d (row %d, beam %02d): %s - %s' % (max_bid*16+i, max_bid, i, b_start, b_end))
#print('%s %s %d' % (b_start, b_end, max_bid*16+i))
with open(f_out_txt, 'a') as f:
    f.write('%s %s %d' % (b_start, b_end, max_bid*nBeam1+i))


ax.set_xlim([t_trans_utc + timedelta(hours=tlim[0]+8), t_trans_utc + timedelta(hours=tlim[1]+8)])
ax.axvline(x=t_trans, color='limegreen', lw=10, zorder=0, alpha=0.3)
lines = [Line2D([0], [0], linewidth=2, color=x) for x in colors]
labels = ['row %d' % x for x in range(nRow)]
labels.extend(['transit'])
lines.append(Line2D([0.4, 0.5], [0, 0], linewidth=6, color='limegreen'))
ax.legend(lines, labels[:len(lines)], handlelength=2.0, handletextpad=0.5, ncols=2, fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
ax.set_xlabel('Local time')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=np.arange(0,60,5)))
ax.grid(which='both')
if ('angle' in locals()):
    ax.set_title('%s, %s (theta_rot = %.2f deg), theta_2nd = %s deg\ntransit at %s, 1st_beam0 = %.2f, 2nd_beam0 = %.2f, theta_off_EW = %.2f deg, %s' % (targetname, sitename, theta_rot_deg, angle, t_trans.strftime('%Y-%m-%d %H:%M:%S'), beam01, beam02, theta_off_deg_1d, attinfo))
else: 
    ax.set_title('%s, %s (theta_rot = %.2f deg)\ntransit at %s, 1st_beam0 = %.2f, 2nd_beam0 = %.2f, theta_off_EW = %.2f deg, %s' % (targetname, sitename, theta_rot_deg, t_trans.strftime('%Y-%m-%d %H:%M:%S'), beam01, beam02, theta_off_deg_1d, attinfo))
ax.tick_params(which='major', length=8)
fig.tight_layout()
fig.savefig('times_%s_%s_%s_%dx%d_b0_%.1f.png' % (targetname, int(obsdate), sitename, nBeam1, nBeam2, beam01))
plt.show()


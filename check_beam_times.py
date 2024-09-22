#!/usr/bin/env python

from loadh5 import *
from pyplanet import *
from astropy.time import Time
import matplotlib.pyplot as plt

DB = loadDB()

beam0 = -9
nAnt = 16
sep = 1.    # meters
lamb0 = 2.998e8/400e6   # longest wavelength
#sin_theta_m = lamb0/sep/nAnt*(np.arange(nAnt)-nAnt//2)  # centered above
nChan = 1024
fMHz = np.linspace(400., 800., nChan, endpoint=False)

cal = 'sun'         # the default calibrator
caldt = datetime.utcnow()
calstr = caldt.strftime('%y%m%d')
site = 'fushan6'
tz = 8.             # time zone in hours from utc
theta_rot_deg = None  # array misalignment angle in deg (>0 for North-toward-West)
theta_off_inp = 0.  # offset of central beam when calibrated to actual Sun transit time data
flim = [0.,0.]
use_cal = False     # whether to use theta_off according to calibration

inp = sys.argv[0:]
pg  = inp.pop(0)

usage = '''
plot the intensity beam profiles and compare them to expected time

syntax:
    %s <src> <t1> <t2> [options]

    <t1>, <t2> are UTC in the format of 'yymmdd_HHM'

options are
    -n nAnt     # number of antennas in the 1D array
                # (default: %d)
    -s sep      # antenna spacing in meter
                # (default: %.1f m)
    -b beam0    # specify the position of the easternmost beam
                # in unit of 400MHz native (FFT) beam spacings
                # (default: %.1f)
    --site SITE # predefined site name
                # (default: %s)
    --flim FMIN FMAX
                # set min/max freq in MHz
    --rot DEG   # array misalignment angle in degree
                # default for: Fushan = -3.0; Nantou = +0.5
                # (otherwise: 0.0 deg)
    --off DEG   # additional beam offset of the beams in deg
                # (if --caldate is not set, will default to theta_off=0)
                # (if --caldate is set, the prediction will be used to counter the calibrator offset)
    --caldate YYMMDD
                # the date when the calibration data was taken
                # (default: today)
''' % (pg, nAnt, sep, beam0, site)

if (len(inp)<1):
    sys.exit(usage)

while(inp):
    k = inp.pop(0)
    if (k == '-n'):
        nAnt = int(inp.pop(0))
    elif (k == '-s'):
        sep = float(inp.pop(0))
    elif (k == '-b'):
        beam0 = float(inp.pop(0))
    elif (k == '--site'):
        site = inp.pop(0)
    elif (k == '--flim'):
        flim[0] = float(inp.pop(0))
        flim[1] = float(inp.pop(0))
    elif (k == '--rot'):
        theta_rot_deg = float(inp.pop(0))
    elif (k == '--off'):
        theta_off_inp = float(inp.pop(0))
    elif (k == '--caldate'):
        calstr = inp.pop(0)
        caldt = datetime.strptime(calstr, '%y%m%d')
        use_cal = True
    else:
        src = k
        t1str = inp.pop(0)
        t2str = inp.pop(0)


if (site.lower() == 'fushan6' and theta_rot_deg is None):
    theta_rot_deg = -3.0
elif (site.lower() == 'longtien' and theta_rot_deg is None):
    theta_rot_deg = 0.5
elif (theta_rot_deg is None):
    theta_rot_deg = 0.


## define beam angles
nBeam = nAnt
sin_theta_m = lamb0/sep/nAnt*(np.arange(nAnt)+beam0)    # offset toward East (HA<0)
theta_deg = np.arcsin(sin_theta_m)/np.pi*180.
theta_rad = np.arcsin(sin_theta_m)

## estimate central beam offset angle
cb, cobs = obsBody(cal, time=caldt, site=site, retOBS=True, DB=DB)
cobs.pressure = 1013     # mbar
cobs.temperature = 25    # deg C
decl = cb.dec
t_trans = cobs.next_transit(cb)
cobs.date = t_trans
cb.compute(cobs)
max_el = cb.alt

theta_rot = theta_rot_deg/180.*np.pi
print('misalignment angle: %.2f deg'%theta_rot_deg)


## modify the beam angles
#if (theta_off_inp is not None):
if (not use_cal):
    theta_off_deg = theta_off_inp
else:
    zenith_angle = np.pi/2. - float(max_el)
    theta_off = theta_rot * zenith_angle
    theta_off_deg = theta_off / np.pi*180.
    t_off = theta_off_deg*60./(15.*np.cos(decl))
    print('calibrator (%s): decl %s  max_el %s'%(cal, decl, max_el))
    print('central beam offset: %.2f deg, %.2f min' % (theta_off_deg, t_off))
print('beam offset adopted: %.2f deg'%theta_off_deg)
theta_deg += theta_off_deg
theta_rad = theta_deg / 180. * np.pi
sin_theta_m = np.sin(theta_rad)



## produce beam profiles
dt1 = Time.strptime(t1str, '%y%m%d_%H%M')
dt2 = Time.strptime(t2str, '%y%m%d_%H%M')
dur = ((dt2-dt1)*86400).value   # in sec
nSky = 500
tsec = np.linspace(0, dur, nSky)
ut0 = dt1.to_value('unix')
ut2 = Time(ut0 + tsec, format='unix').to_datetime()
#print('debug:', dt1, dt2, dur, ut0)
lt2 = Time(ut0 + tsec + 3600*tz, format='unix').to_datetime()

b, obs = obsBody(src, time=ut2[0], site=site, retOBS=True, DB=DB)
#print(ut[0], b.ha, b.az, b.alt)
obs.pressure = 1013     # mbar
obs.temperature = 25    # deg C

ha = []
alt = []
az  = []
bra  = []
bdec = []
for i in range(nSky):
    obs.date = ut2[i]
    b.compute(obs)
    tmp = b.ha
    if (tmp > np.pi):
        tmp -= 2.*np.pi
    ha.append(tmp)
    alt.append(b.alt)
    az.append(b.az)
    #print(tmp, b.alt, b.az)
    bra.append(b.ra)
    bdec.append(b.dec)
ha = np.array(ha)
ha_deg = ha/np.pi*180.
#print(ha)
#print(alt, az)
el = np.array(alt)
az = np.array(az)
bra = np.array(bra)
bdec = np.array(bdec)

za = np.pi/2 - el   # zenith angle, theta
z = np.cos(za)
r = np.sin(za)
x = r*np.sin(az)
y = r*np.cos(az)

def rot(pos, theta):
    '''
    input:
        pos.shape = (n, 3), antenna (x,y,z) in meters
        theta = rotation angle in radian

    output:
        pos2 = new antenna positions (x,y,z) in meters
    '''
    mat = np.zeros((3,3))
    mat[0,0] = np.cos(theta)
    mat[0,1] = -np.sin(theta)
    mat[1,0] = np.sin(theta)
    mat[1,1] = mat[0,0]
    mat[2,2] = 1.

    return np.dot(pos,mat.T)

# generate model
pos0 = np.zeros((nAnt, 3))                   # X, Y, Z coordinate in m
pos0[:,0] = np.arange(nAnt)*sep              # 1D array in X direction (E-W)
pos = rot(pos0, theta_rot) 
#print(pos)
unitVec = np.array([x,y,z])                 # shape (3, nSky)
Tau_m = np.dot(pos, unitVec)                # shape (nAnt, nSky)
#print('Tau_m.shape', Tau_m.shape)
Tau_s = Tau_m / 2.998e8
freq = fMHz * 1e6
phi = 2.*np.pi*Tau_s.reshape((nAnt,nSky,1))*freq.reshape((1,1,nChan))
inVolt = np.exp(-1.j*phi)                      # shape (nAnt, nSky, nChan)

# beamform
BTau_s = pos[:,0].reshape((1,nAnt,1))*sin_theta_m.reshape((nBeam,1,1))/2.998e8
                                            # shape (nBeam, nAnt, 1)
BFM0 = np.exp(-1.j*2.*np.pi*BTau_s*freq.reshape((1,1,nChan)))
                                            # shape (nBeam, nAnt, nChan)

outVolt = np.zeros((nBeam, nSky, nChan), dtype=complex)
for ch in range(nChan):
    outVolt[:,:,ch] = np.dot(BFM0[:,:,ch], inVolt[:,:,ch])

outInt = np.ma.array((np.abs(outVolt)**2).mean(axis=2), mask=False)
#outInt -= outInt.min()
outInt /= outInt.max()
outInt = np.flip(outInt, axis=0)

ofile = '%s.UT%s.%s.switch.times'%(site, t1str, src)
fo = open(ofile, 'w')
print('# beam switching time: UT, local_time', file=fo)
Tsw = []
for i in range(nBeam-1, 0, -1):
    y1 = outInt[i]
    #y1.mask[y1<0.11]=True
    y2 = outInt[i-1]
    #y2.mask[y2<0.11]=True
    j1 = np.ma.argmax(y1)
    j2 = np.ma.argmax(y2)
    y1.mask = True
    y1.mask[j1:j2] = False
    y2.mask = True
    y2.mask[j1:j2] = False
    j = np.ma.argmin(np.ma.abs(y1-y2))
    Tsw.append(ut2[j])
    #print('beam%d-->%d:'%(i,i-1), ut2[j], y1[j], y2[j])
    #print('beam %02d-->%02d:'%(i,i-1), ut2[j].strftime('%y%m%d %H:%M:%S'), file=fo)
    print('beam %02d-->%02d:'%(i,i-1), ut2[j].strftime('%y%m%d %H:%M:%S'), lt2[j].strftime('%y%m%d %H:%M:%S'), file=fo)
fo.close()
call('cat %s'%ofile, shell=True)


import matplotlib.dates as mdates

for ri in range(1):
    fig, ax = plt.subplots(1,1,figsize=(15,5))
    ax.set_xlabel('time (UT)')
    ax.set_ylabel('intensity')
    ax.grid()
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    for i in range(nBeam):
        #j = 15-i
        cc = 'C%d'%i
        y = outInt[i].data
        ax.plot(ut2, y, color=cc, ls=':', label='model')
        imax = y.argmax()
        xmax = ut2[imax]
        ymax = y[imax]
        label = 'beam-%d'%i
        ax.text(xmax, ymax+0.05, label, color=cc, ha='center')

    ax.set_xlim(ut2[0], ut2[-1])
    ax.set_ylim(-0.05, 1.10)

    fig.text(0.02, 0.95, 'theta_rot: %.2f deg, theta_off: %.2f deg'%(theta_rot_deg, theta_off_deg))
    #fig.text(0.98, 0.95, 'cal src, date: %s, %s'%(cal, calstr), ha='right')
    fig.text(0.98, 0.95, '%s ra,dec: (%s, %s)'%(src, hours(bra.mean()), degrees(bdec.mean())), ha='right')

    fig.autofmt_xdate()
    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.suptitle('%s @ %s, %s, beam0=%d'%(src,site,t1str, beam0))
    png = '%s.UT%s.%s.angles.png'%(site, t1str, src)
    fig.savefig(png)
    plt.close(fig)

    print('figure saved:', png)



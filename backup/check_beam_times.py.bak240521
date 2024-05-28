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

src = 'sun'
site = 'fushan6'
theta_rot_deg = 0.  # array misalignment angle in deg (>0 for North-toward-West)
theta_off_deg = 0.  # offset of central beam when calibrated to actual Sun transit time data
flim = [0.,0.]

inp = sys.argv[0:]
pg  = inp.pop(0)

usage = '''
plot the intensity beam profiles and compare them to expected time

syntax:
    %s <src> <t1> <t2> [options]

    <t1>, <t2> are in the format of 'yymmdd_HHM'

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
                # e.g. Fushan = -3; Nantou = +1
                # (default: %.1f)
    --off DEG   # central beam offset to account for Sun transit calibration
                # (default: %.1f)
''' % (pg, nAnt, sep, beam0, site, theta_rot_deg, theta_off_deg)

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
        theta_off_deg = float(inp.pop(0))
    else:
        src = k
        t1str = inp.pop(0)
        t2str = inp.pop(0)

nBeam = nAnt
sin_theta_m = lamb0/sep/nAnt*(np.arange(nAnt)+beam0)    # offset toward East (HA<0)
theta_deg = np.arcsin(sin_theta_m)/np.pi*180.
theta_rad = np.arcsin(sin_theta_m)
theta_deg += theta_off_deg
theta_rad = theta_deg / 180. * np.pi
sin_theta_m = np.sin(theta_rad)

theta_rot = theta_rot_deg/180.*np.pi


dt1 = Time.strptime(t1str, '%y%m%d_%H%M')
dt2 = Time.strptime(t2str, '%y%m%d_%H%M')
dur = ((dt2-dt1)*86400).value   # in sec
nSky = 500
tsec = np.linspace(0, dur, nSky)
ut0 = dt1.to_value('unix')
ut2 = Time(ut0 + tsec, format='unix').to_datetime()
print(dt1, dt2, dur, ut0)

b, obs = obsBody(src, time=ut2[0], site=site, retOBS=True, DB=DB)
#print(ut[0], b.ha, b.az, b.alt)
obs.pressure = 1013     # mbar
obs.temperature = 25    # deg C
decl = b.dec
print('source decl:', decl, float(decl))
t_trans = obs.next_transit(b)
obs.date = t_trans
b.compute(obs)
max_el = b.alt
print('source max el.:', max_el, float(max_el))
zenith_angle = np.pi/2. - float(max_el)
theta_off = theta_rot * zenith_angle
theta_off_deg = theta_off / np.pi*180.
t_off = theta_off_deg*60./(15.*np.cos(decl))
print('central beam offset:', theta_off, 'rad', theta_off_deg, 'deg', t_off,'min')

#theta_rad += theta_off
#sin_theta_m = np.sin(theta_rad)
#theta_deg = theta_rad / np.pi * 180.
#print('theta_off_deg', theta_off_deg)





ha = []
alt = []
az  = []
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
ha = np.array(ha)
ha_deg = ha/np.pi*180.
#print(ha)
#print(alt, az)
el = np.array(alt)
az = np.array(az)

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
print('Tau_m.shape', Tau_m.shape)
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

Tsw = []
for i in range(nBeam-1):
    y1 = outInt[i]
    #y1.mask[y1<0.11]=True
    y2 = outInt[i+1]
    #y2.mask[y2<0.11]=True
    j1 = np.ma.argmax(y1)
    j2 = np.ma.argmax(y2)
    y1.mask = True
    y1.mask[j1:j2] = False
    y2.mask = True
    y2.mask[j1:j2] = False
    j = np.ma.argmin(np.ma.abs(y1-y2))
    Tsw.append(ut2[j])
    print('beam%d-->%d:'%(i,i+1), ut2[j], y1[j], y2[j])



import matplotlib.dates as mdates

for ri in range(1):
    fig, ax = plt.subplots(1,1,figsize=(15,5))
    ax.set_xlabel('time (UT)')
    ax.set_ylabel('intensity')
    ax.grid()
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    for i in range(nBeam):
        j = 15-i
        cc = 'C%d'%i
        ax.plot(ut2, outInt[i].data, color=cc, ls=':', label='model')

    ax.set_xlim(ut2[0], ut2[-1])

    fig.autofmt_xdate()
    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.suptitle('%s @ %s'%(src,site))
    fig.savefig('%s.UT%s.%s.angles.png'%(site, t1str, src))
    plt.close(fig)


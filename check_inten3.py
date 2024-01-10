#!/usr/bin/env python

from loadh5 import *
from pyplanet import *
from astropy.time import Time
import matplotlib.pyplot as plt

nAnt = 16
sep = 1.    # meters
lamb0 = 2.998e8/400e6   # longest wavelength
#sin_theta_m = lamb0/sep/nAnt*(np.arange(nAnt)-nAnt//2)  # centered above
sin_theta_m = lamb0/sep/nAnt*(np.arange(nAnt)-nAnt)    # offset toward East (HA<0)
theta_deg = np.arcsin(sin_theta_m)/np.pi*180.
theta_rad = np.arcsin(sin_theta_m)

src = 'sun'
site = 'fushan6'
#fint = 'data_230916_inten0.inth5'

inp = sys.argv[0:]
pg  = inp.pop(0)

usage = '''
plot the intensity beam profiles and compare them to expected time

syntax:
    %s <.inth5> [options]

''' % (pg,)

if (len(inp)<1):
    sys.exit(usage)

while(inp):
    k = inp.pop(0)
    fint = k


attrs = getAttrs(fint)
epoch0 = attrs['unix_utc_open']
#epoch0 += 3600*8 # convert to local time
sec = getData(fint, 'winSec')
ut  = Time(epoch0+sec, format='unix').to_datetime()
nInten = getData(fint, 'norm.intensity')
nProf = nInten.mean(axis=2)



b, obs = obsBody(src, time=ut[0], site=site, retOBS=True)
#print(ut[0], b.ha, b.az, b.alt)
obs.pressure = 1013     # mbar
obs.temperature = 25    # deg C

ha = []
alt = []
az  = []
for i in range(len(ut)):
    obs.date = ut[i]
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

# method 1
phi1 = 0
NSoff = np.arcsin(np.sin(el)*np.sin(phi1)+np.cos(el)*np.cos(phi1)*np.cos(az))
EWoff = np.arcsin((-1)*np.sin(az)*np.cos(el)/np.cos(NSoff))
EWdeg = EWoff/np.pi*180.

# method 2
za = np.pi/2 - el   # zenith angle, theta
z = np.cos(za)
r = np.sin(za)
x = r*np.sin(az)
y = r*np.cos(az)
EWoff2 = np.arctan2(-x,z)   # East negative, similar to hour angle
NSoff2 = np.arctan2(y,z)    # South negative
EWdeg2 = EWoff2/np.pi*180.

nn = len(az)
for i in range(nn):
    #print('%+.6f  %+.6f;  %+.6f  %+.6f'%(EWoff[i], NSoff[i], EWoff2[i], NSoff2[i]))
    #print(az[i], el[i], x[i], y[i], z[i])



fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.set_xlabel('angle (deg)')
ax.set_ylabel('intensity')

for i in range(nAnt):
    j = 15-i
    cc = 'C%d'%i
    ax.plot(EWdeg2, nProf[:,j], color=cc)
    ax.plot(ha_deg, nProf[:,j], color=cc, linestyle=':')
    ax.axvline(theta_deg[i], color=cc, linestyle='--')
    hai = np.argmax(nProf[:,j])
    ha_peak = ha_deg[hai]
    #print(i, 'ha diff: %.3fdeg'%(ha_peak-theta_deg[i]))

fig.tight_layout(rect=[0,0.03,1,0.95])
fig.suptitle('%s'%fint)
fig.savefig('%s.angles.png'%fint)
plt.close(fig)


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
    print(tmp, b.alt, b.az)
ha = np.array(ha)
#print(ha)
#print(alt, az)

ha_sec = []
for i in range(nAnt):
    peak_rad = theta_rad[i]
    j = np.abs(ha - peak_rad).argmin()
    if (ha[j] < peak_rad):
        x0 = peak_rad - ha[j]
        x1 = ha[j+1] - peak_rad
        s = ha[j+1] - ha[j]
        isec = x1/s*sec[j] + x0/s*sec[j+1]
    else:
        x0 = ha[j] - peak_rad
        x1 = peak_rad - ha[j-1]
        s = ha[j] - ha[j-1]
        isec = x1/s*sec[j] + x0/s*sec[j-1]
    ha_sec.append(isec)
ha_sec = np.array(ha_sec)
ha_ut = Time(epoch0+ha_sec, format='unix').to_datetime()
#print(ha_sec, ha_ut)


fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.set_xlabel('time (UT)')
ax.set_ylabel('intensity')

for i in range(nAnt):
    j = 15-i
    cc = 'C%d'%i
    ax.plot(ut, nProf[:,j], color=cc)
    ax.axvline(ha_ut[i], color=cc, linestyle='--')

fig.autofmt_xdate()
fig.tight_layout(rect=[0,0.03,1,0.95])
fig.suptitle('%s'%fint)
fig.savefig('%s.times.png'%fint)
plt.close(fig)


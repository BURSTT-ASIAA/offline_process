#!/usr/bin/env python
# calculates the apparent (date) coordinates of planets
# for Saturn, also can calculate the ring inclination (earth_tilt)
# 2011/06/02

from ephem import *
from extract_source import *
import numpy as np
import sys, os.path
from datetime import datetime


#from astropy.constants import c, h, k_B
c   = 299792458.0       # speed of light SI
h   = 6.62606957e-34    # Planck constant SI
k_B = 1.3806488e-23     # Boltzmann constant SI

import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation, Galactic
from astropy.utils import iers
# the default IERS url is not working: http://maia.usno.navy.mil/ser7/finals2000A.all
#iers.conf.iers_auto_url = 'ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all'

DB = loadDB()

def LSRvel(ra, dec, obstime, loc='taroge4'):
    '''
    calculate the velocity correction between MLO and the LSR
    toward the RA,DEC (epoch) direction at time_str

    give the coordinates in J2000!
    ra should be a string in hourangle
    dec should be a string in DMS
    obstime should be a string in 'yyyy-mm-dd HH:MM:SS'
        or can be a datetime object, or astropy.time.Time object

    return the velocity correction in km/s
    positive velocity means the observed freq is higher than rest freq (blue shifted)
    negative velocity means the observed freq is lower than rest freq (red shifted)
    '''

    if (loc == 'mlo'):
        loc = EarthLocation.from_geodetic(lat=19.5363*u.deg, lon=-155.5753*u.deg, height=3426.*u.m)
    elif (loc == 'atca'):
        loc = EarthLocation.from_geodetic(lat=-30.3128846*u.deg, lon=149.5489798*u.deg, height=236.87*u.m)
    elif (loc == 'taroge4'):
        #loc = EarthLocation.from_geodetic(lat=121.7785*u.deg, lon=24.4066*u.deg, height=300.0*u.m)  # arbitrary placeholder near Nanao
        loc = EarthLocation.from_geodetic(lat=121.777984392128*u.deg, lon=24.3827607578448*u.deg, height=707.86*u.m)  # exact number from Yaocheng

    sc = SkyCoord(ra=ra, dec=dec, unit=('hourangle', 'degree'), obstime=obstime, location=loc)
    vbsr = sc.radial_velocity_correction().to('km/s')
    
    scg = sc.transform_to(Galactic)
    l, b = scg.l.to(u.rad).value, scg.b.to(u.rad).value

    bary2lsr = 11.1 * np.cos(l)*np.cos(b) + 12.24 * np.sin(l)*np.cos(b) + 7.25 * np.sin(b)
    #bary2lsr =  9.0 * np.cos(l)*np.cos(b) + 12.00 * np.sin(l)*np.cos(b) + 7.00 * np.sin(b)

    vlsr = vbsr.value + bary2lsr

    return vlsr


def obsBody(body, site='taroge4', time=datetime.utcnow(), epoch='2000', DB={}, retOBS=False, **kwargs):
        ## define planet to calculate
        if   (body.lower() in 'jupiter'):
            b = Jupiter()
        elif (body.lower() in 'saturn' ):
            b = Saturn()
        elif (body.lower() in 'mars'   ):
            b = Mars()
        elif (body.lower() in 'neptune'):
            b = Neptune()
        elif (body.lower() in 'uranus'):
            b = Uranus()
        elif (body.lower() in 'venus'  ):
            b = Venus()
        elif (body.lower() in 'pluto'  ):
            b = Pluto()
        elif (body.lower() in 'mercury'):
            b = Mercury()
        elif (body.lower() in 'moon'   ):
            b = Moon()
        elif (body.lower() in 'sun'    ):
            b = Sun()
        elif (DB.get(body)):
            tmp = DB[body]
            b = FixedBody()
            b._ra  = hours(tmp['RAs'])
            b._dec = degrees(tmp['DECs'])
            b._epoch = '2000'
        else:
            print('unknown PLANET =', body)
            sys.exit()


        ## define observer
        obs = obsSite(site)
        obs.date        = time      # this is variable now
        # can omit epoch if using J2000 (is the default)
        obs.epoch       = epoch
        #mlo.epoch      = '2000/1/1.5'
        if (kwargs.get('elim')):
            obs.horizon = kwargs['elim']

        # Note:
        # epoch matters only if output is astrometric coordinates (not apparent coordinates)
        # astrometric coordinates = a_ra, a_dec (will use epoch setting) --> geocentric
        # apparent geocentric     = g_ra, g_dec (will ignore epoch setting)
        # apparent topocentric    = ra, dec     (also will ignore epoch settting)


        ## start calculation for each line
        b.compute(obs)

        # (b.ra, b.dec) = apparent topocentric
        # (b.a_ra, b.a_dec) = astrometric
        # b.size = angular diameter
        # b.earth_tilt = ring opening angle toward earth (for Saturn only?)

        if (retOBS):
            return b, obs
        else:
            return b


def obsSite(site='fushan6'):
        ## define observer
        # Mauna Loa Observatory site information taken from AMiBA Wiki
        obs = Observer()

        # by default, set temp and pressure to 0 to ignore atmosphere correction
        obs.temp        = 0
        obs.pressure    = 0
        # can be overriden per site

        if (site == 'mlo'):
            obs.long        = '-155.5753'
            obs.lat         = '+19.5363'
            obs.elevation   = 3426.0
        elif (site == 'taroge4'):
            obs.long        = '121.7785'
            obs.lat         = '+24.4066'
            obs.elevation   = 300.0
        elif (site == 'cafe' or site == 'FUG'):
            obs.long        = '121.5380'
            obs.lat         = '+25.2980'
            obs.elevation   = 10.0
        elif (site == 'fushan1'):   # site 1, 220310
            obs.long        = '121.5807'
            obs.lat         = '+24.7589'
            obs.elevation   = 650.0
        elif (site == 'fushan6' or site == 'FUS'):   # site 6, 230418
            #obs.long        = '121.5817'   # old
            #obs.lat         = '+24.7564'   # old
            obs.long        = '121.58164005833' # new
            obs.lat         = '+24.75649957777' # new
            obs.elevation   = 640.162
            #lon = '121:34:53.90421'
            #lat = '24:45:23.39848'
            #height = 640.162*u.m
        elif (site == 'longtien' or site == 'LTN'): # DGPS measurement at 230721
            obs.long        = '120.82450942'    #'120.8244'
            obs.lat         = '+23.71464178' #'+23.7147'
            obs.elevation   = 879.814       #850.0
        elif (site == 'lyudao' or site == 'GRN'):
            # old ver
            #obs.long        = '121.5007'
            #obs.lat         = '+22.6750'
            #obs.elevation   = 15.0
            # new ver: 2025-03 GNSS measurement, GCP 1 next to antenna 0
            obs.long        = '121.50121981'
            obs.lat         = '+22.67561324'
            obs.elevation   = 32.677
        elif (site == 'kinmen' or site == 'KMN'): # SH: from app 'GPS test'  2025-03
            obs.long         = '118.34253917'
            obs.lat         = '24.43985611'
            obs.elevation   = 35.
        elif (site == 'dongsha'):
            obs.long        = '116.7274'
            obs.lat         = '+20.10752'
            obs.elevation   = 5.0
        elif (site == 'iaa'):
            obs.long        = '121.5377'
            obs.lat         = '+25.0212'
            obs.elevation   = 50.0
        elif (site == 'pahala' or site == 'PHL'):
            obs.long        = '-155.4686'
            obs.lat         = '+19.2497'
            obs.elevation   = 542.5
        elif (site == 'tnro'):
            obs.long         = '99.216560'
            obs.lat         = '+18.863546'
            obs.elevation   = 390.0
        elif (site == 'ogasawara' or site == 'OGW'): # SH: 2024/12/23  GCP #2
            obs.long         = '142.216932'
            obs.lat         = '+27.092032'
            obs.elevation   = 261.459
        elif (site == 'gbd'):
            obs.long        = '77.42808'
            obs.lat         = '+13.60300'
            obs.elevation   = 935.0
        else:
            obs = None

        #obs.date       = time      # this is variable now
        # can omit epoch if using J2000 (is the default)
        #obs.epoch      = epoch
        #obs.epoch      = '2000/1/1.5'

        # Note:
        # epoch matters only if output is astrometric coordinates (not apparent coordinates)
        # astrometric coordinates = a_ra, a_dec (will use epoch setting) --> geocentric
        # apparent geocentric     = g_ra, g_dec (will ignore epoch setting)
        # apparent topocentric    = ra, dec     (also will ignore epoch settting)

        return obs


def solbody(body):
        ## canonicalize the body name
        if   (body.lower() in 'jupiter'):
            name = 'jupiter'
        elif (body.lower() in 'saturn' ):
            name = 'saturn'
        elif (body.lower() in 'mars'   ):
            name = 'mars'
        elif (body.lower() in 'neptune'):
            name = 'neptune'
        elif (body.lower() in 'uranus'):
            name = 'uranus'
        elif (body.lower() in 'venus'  ):
            name = 'venus'
        elif (body.lower() in 'pluto'  ):
            name = 'pluto'
        elif (body.lower() in 'mercury'):
            name = 'mercury'
        elif (body.lower() in 'moon'   ):
            name = 'moon'
        elif (body.lower() in 'sun'    ):
            name = 'sun'
        else:
            print('unknown PLANET =', body)
            name = None

        return name


def coordSite(site, lon=None, lat=None, height=None):
    tmp = obsSite(site)
    if (not tmp is None):
        #print(tmp, tmp.long, tmp.lat)
        lon = tmp.long*u.radian
        lat = tmp.lat*u.radian
        height = tmp.elevation*u.m
        #print(lon, lat, height)

    #if (site.lower() == 'fushan6'):
    #    lon = '121:34:53.90421'
    #    lat = '24:45:23.39848'
    #    height = 640.162*u.m
    #elif (site.lower() == 'longtien'):
    #    lon = 120.82450942*u.deg
    #    lat = 23.71464178*u.deg
    #    height = 879.814*u.m 
    if (site.lower() == 'aro'):
        lon = -78.072776*u.deg
        lat = 45.955555*u.deg
        height = 260.4*u.m

    if (lon is None or lat is None):    # site must be defined in astropy
        coord = EarthLocation.of_site(site)
    else:   # use geodetic location
        if (height is None):
            height = 0.*u.m
        coord = EarthLocation.from_geodetic(lon=lon, lat=lat, height=height, ellipsoid='WGS84')
    
    return coord    # ECEF coordinates in meters


def vecBaseline(site1, site2):
    '''
    compute the baseline vector from site1 (reference) toward site2
    in ECEF coordinates in meters
    '''
    v1 = site1.value
    v2 = site2.value
    BL = np.zeros(3)
    for i in range(3):
        BL[i] = v2[i] - v1[i]
    return BL


def tauBaseline(RA_rad, DEC_rad, BLvec, t_arr, t0=None):
    '''
    input:
        RA_rad:: target RA in radian
        DEC_rad:: target DEC in radian
        BLvec:: should be a baseline vector in the ECEF system in meters, originated from a reference site
        t_arr:: if t0 is None, t_arr should a Time object array
                if t0 is a Time object, then t_arr is expected to be time offset in seconds

    output:
        tau:: delay in meters
    '''
    if (not t0 is None):
        t_arr = t0 + TimeDelta(t_arr, format='sec')
    nTime = len(t_arr)

    ST = t_arr.sidereal_time('apparent', 'greenwich')
    dRA_rad = RA_rad - ST.radian
    #print('debug:', RA_rad, DEC_rad, dRA_rad)

    unit_vec = np.zeros(3)
    tauGeo = np.zeros(nTime)
    for i in range(nTime):
        r = np.cos(DEC_rad)
        unit_vec[0] = r*np.cos(dRA_rad[i])
        unit_vec[1] = r*np.sin(dRA_rad[i])
        unit_vec[2] = np.sin(DEC_rad)
        tauGeo[i] = np.dot(unit_vec, BLvec)

    return tauGeo


def arrayConf(arr_config, nRow, rows=None, theta_rot=0.):
    '''
    specify an antenna config or a file listing the antenna positions
    and the number of 'rows' to return
    if a list of rows is given, then override the 'nRow'
    theta_rot (in deg, CCW=+, CW=-) will rotate the arrays by the given angle
    
    return antenna positions (X,Y,Z) in metrs
    shape = (nTotal, 3)
    where nTotal = nRow*nAnt 
    or the number of entries in the file
    '''
    pos = None
    if (arr_config == '16x1.0y0.5'):
        nAnt = 16
        xsep = 1.0
        ysep = 0.5
    elif (arr_config == '16x1.0y2.0'):
        nAnt = 16
        xsep = 1.0
        ysep = 2.0
    elif (arr_config == '2x8'):
        nAnt = 8
        xsep = 1.0
        ysep = 1.0
        rows = np.arange(2)
    else:
        if (os.path.isfile(arr_config)):
            pos = np.loadtxt(arr_config, usecols=(1,2,3))
        else:
            print('error opening aconf file: %s'%arr_config)
            return None

    if (rows is None):
        rows = np.arange(nRow)
    else:
        nRow = len(rows)

    if (pos is None):
        pos = np.zeros((nRow*nAnt,3))
        ai = -1
        for j in rows:
            y = ysep*j
            for i in range(nAnt):
                x = xsep*i
                ai += 1
                pos[ai] = [x,y,0.]

    if (theta_rot != 0.):   # rotation along Z
        theta = theta_rot /180.*np.pi   # in radian
        mat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
            ])
        tmp = np.tensordot(mat, pos[:,:2], axes=(1,1))
        pos[:,:2] = tmp.T


    return pos


def get_tauGeo(dtarr, pos, body='sun', site='fushan6', aref=0):
    '''
    calculate the geometric delay of antennas toward a astro body

    input:
        dtarr:: array of datetime obj, shape=(nTime,)
        pos:: array of (X,Y,Z) in meters, shape=(nAnt, 3)
        body:: solar body or a source defined in YTLA_CAL.csv
        site:: one of the sites defined in obsSite() of pyplanet.py
        aref:: antenna id to be considered the reference antenna
                the delay will be relative to this reference
                Note: set aref=None if relative to the coordinates origin is preferred

    output:
        tauGeo:: geometric delay in sec, shape=(nAnt, nTime)
    '''

    az, el = getAzEl(dtarr, body=body, site=site)

    phi    = np.pi/2. - az
    theta  = np.pi/2. - el
    unitVec  = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], ndmin=2).T
    unitVec *= -1

    # BVec.shape = (nAnt, 3)
    # unitVec.shape = (nTime, 3)
    if (aref is None):  # relative to the coordinates origin
        BVec = pos
    else:               # relative to the ref ant
        BVec = pos - pos[aref]
    tauGeo = np.tensordot(BVec, unitVec, axes=(1,1))  # delay in meters, shape (nFPGA*nAnt, nTime)
    tauGeo /= 2.998e8   # delay in sec.

    return tauGeo


def getAzEl(dtarr, body='sun', site='fushan6'):
    '''
    calculate az,el of a source at a given site

    input:
        dtarr:: array of datetime obj, shape=(nTime,)
        pos:: array of (X,Y,Z) in meters, shape=(nAnt, 3)
        body:: solar body or a source defined in YTLA_CAL.csv
        site:: one of the sites defined in obsSite() of pyplanet.py

    output:
        az, el:: array in radian, shape=(nTime,)
    '''

    b, obs = obsBody(body, time=dtarr[0], site=site, retOBS=True, DB=DB)

    az = []
    el = []
    for ti in dtarr:
        obs.date = ti
        b.compute(obs)
        az.append(b.az)
        el.append(b.alt)
    az = np.array(az)
    el = np.array(el)

    return az, el


if (__name__ == '__main__'):

    DB0 = loadDB

    body = 'sun'
    b = obsBody(body, DB=DB0, site='mlo')
    print(b.ra, b.dec)
 

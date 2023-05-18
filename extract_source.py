#!/usr/bin/env python
import ephem
import numpy as np
import sys, os.path
from datetime import datetime

host = os.environ.get('HOST')
hostname = os.environ.get('HOSTNAME')
home = os.environ.get('HOME')

#bindir = '/tiara/home/kylin/local/bin'
if (host == 'coma18'):
    bindir = '%s/analysis_burstt/local/bin' % home
else:
    bindir = '%s/local/bin' % home
if (hostname == 'frblab3' or hostname=='frblab1'):
    bindir = '/data/kylin/bin'
fcat0 = '%s/YTLA_CAL.csv' % bindir
#fcat0 = 'YTLA_CAL.csv'

def loadDB(fcat=fcat0):
    try:
        cat = np.loadtxt(fcat, skiprows=1, delimiter=',', dtype='U')
    except:
        print('error opening catalog:', fcat)
        return None

    #print 'debug: in loadDB, n=', len(cat)

    DB = {}
    for src in cat:
        name = src[0].strip()
        RAH = src[1]
        RAM = src[2]
        RAS = src[3]
        DECD = src[4]
        DECM = src[5]
        DECS = src[6]
        flux = src[7]

        RAs  = '%s:%s:%s' % (RAH, RAM, RAS)
        DECs = '%s:%s:%s' % (DECD, DECM, DECS)

        RAr  = np.pi / 180. * 15. * (float(RAH) + float(RAM)/60. + float(RAS)/3600.)
        if (DECD.startswith('-')):
            DECr = np.pi / 180. * (float(DECD) - float(DECM)/60. - float(DECS)/3600.)
        else:
            DECr = np.pi / 180. * (float(DECD) + float(DECM)/60. + float(DECS)/3600.)

        #DB[name] = {'RAs':RAs, 'DECs':DECs, 'RAr':RAr, 'DECr':DECr, 'flux':flux}
        DB[name.lower()] = {'RAs':RAs, 'DECs':DECs, 'RAr':RAr, 'DECr':DECr, 'flux':flux}

    return DB


def matchpointing(ftab, DB={}):
    '''
    a new pointing and source matching function to replace the perl version (group_patch_extend.pl)
    '''
    try:
        (sid, uid, t1, t2, spol) = np.loadtxt(ftab, usecols=(0,1,4,5,10), unpack=True)
        (s1, s2, ra1, dec1) = np.loadtxt(ftab, usecols=(2,3,8,9), dtype='U', unpack=True)
    except:
        print('error accessing tab file:', ftab)
        return None

    nunit = len(uid)
    # t1,t2 did not record the decimal seconds
    # convert the string timestamps s1,s2 to seconds instead
    ts1 = np.zeros_like(t1)
    ts2 = np.zeros_like(t2)
    dt0 = datetime.strptime('00:00:00', '%H:%M:%S')
    for i in range(nunit):
        ddt = datetime.strptime(s1[i], '%H:%M:%S.%f') - dt0
        ts1[i] = ddt.total_seconds()
        ddt = datetime.strptime(s2[i], '%H:%M:%S.%f') - dt0
        ts2[i] = ddt.total_seconds()
        #print s1[i], s2[i], t1[i], t2[i], ts1[i], ts2[i]

    dlog = os.path.dirname(ftab)
    obsdate = os.path.basename(ftab).split('.')[0]
    #(y4, mm, dd) = obsdate.split('-')
    #yy = y4.lstrip('20')
    #fvalog = '%s%s%s.valog' % (yy, mm, dd)
    short = datetime.strptime(obsdate, '%Y-%m-%d').strftime('%y%m%d')
    fvalog = short + '.valog'
    #print fvalog

    cols = list(range(5,13))
    cols.insert(0, 1)
    try:
        #(tsec, rah, decd, azd, eld, hpol, opol, skpol, osf) = np.loadtxt(fvalog, usecols=cols, unpack=True)
        pnt = np.loadtxt(fvalog, usecols=cols)
    except:
        print('error accessing valog file:', fvalog)
    tsec = pnt[:,0]
    #print pnt.shape, tsec.shape

    fout = obsdate + '.pointing'
    out = open(fout, 'w')
    print('# 1: sid, order of sched of the day', file=out)
    print('# 2: uid, order of unit in each sched', file=out)
    print('# 3: t1, unit begin time (UT HH:MM:SS)', file=out)
    print('# 4: t2, unit end   time (UT HH:MM:SS)', file=out)
    print('# 5: t1, unit begin time (UT second-of-day)', file=out)
    print('# 6: t2, unit end   time (UT second-of-day)', file=out)
    print('# 7: avg RA (hr)', file=out)
    print('# 8: avg DEC (deg)', file=out)
    print('# 9: avg Az (deg)', file=out)
    print('#10: avg El (deg)', file=out)
    print('#11: avg HexPol (deg)', file=out)
    print('#12: avg ObsPol (deg)', file=out)
    print('#13: avg SkyPol (deg)', file=out)
    print('#14: target', file=out)




    for u in range(nunit):
        w = np.logical_and(tsec>=ts1[u], tsec<=ts2[u])
        #print pnt[w].shape
        pnt_avg = pnt[w].mean(axis=0)
        # take care of periodicity for RA and Az
        # perform vector avg of the pointings
        rar = pnt[w][:,1] * 15. / 180. * np.pi
        dcr = pnt[w][:,2] / 180. * np.pi
        avg_rra, avg_rdec = vecavg(rar, dcr)
        tgt, dist = nearsrc(avg_rra, avg_rdec, DB)
        if (avg_rra < 0.):
            avg_rra += 2. * np.pi
        avg_hra  = avg_rra * 12. / np.pi
        avg_ddec = avg_rdec * 180. / np.pi
        
        

def vecavg(a, d):
    '''for a series of spherical angles (a, d) -- following the convention of RA/DEC
    calculate the vector average of the vectors and return the angle of the avg vector
    a and d should have unit of radian'''

    x = np.cos(d) * np.cos(a)
    y = np.cos(d) * np.sin(a)
    z = np.sin(d)

    avg_a = np.arctan2(y.mean(), x.mean())
    avg_d = np.arcsin(z.mean())

    return avg_a, avg_d
        


if (__name__ == '__main__'):

    #fcat = '/home/amibausr/kylin/bin/ATNF_CAL_1JY.csv'
    ftab = '2017-08-16.tcs_ctrl.tab'


    DB = loadDB()
    #print(DB)

    for k in list(DB.keys()):
        print(k, DB[k])



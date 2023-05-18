#!/usr/bin/env python

from ephem import *
import sys
from datetime import datetime
from pyplanet import *

#fcat = '/home/amibausr/kylin/bin/YTLA_CAL.csv'
#DB = loadDB(fcat)
DB  = loadDB()

## preparation
inp     = sys.argv[0:]
pg      = inp.pop(0)
now     = datetime.utcnow()    # force UTC
#time   = now  # default time
date    = now.strftime('%y%m%d')
epoch   = '2000'
elim    = '0.'
skey    = 't_set'
bodies  = []
tz      = 8   # Hilo time offset from UTC in hours
site    = 'cafe'
use_local = True # set to False to use UTC


## user inputs
usage = '''
usage: %s TARGET [options]

    [options]:
    -d YYMMDD   # specify the date to check
                # default to today

    --site Site # specify the site name
                # default CAFE

    -e epoch    # '2000', '1950', or something
                # default to '2000'

    -l elim     # set the elevation limit in deg
                # default to 0.

    -s KEY      # change the sort key:
                # ra, dec, t_trans, t_rise, t_set
                # default to t_set

    --local     # show times in local time
                # (the default)

    --tz hr     # timezone offset from UTC
                # e.g. Taipei = +8; Hilo = -10
                # (default, Taipei)

    --utc       # show times in UTC

''' % pg

if (len(inp) < 1):
    print(usage)
    sys.exit()

while (inp):
    arg = inp.pop(0)
    if (arg == '-d'):
        date = inp.pop(0)
    elif (arg == '-e'):
        epoch = inp.pop(0)
    elif (arg == '-l'):
        elim = inp.pop(0)
    elif (arg == '-s'):
        skey = inp.pop(0)
    elif (arg == '--site'):
        site = inp.pop(0)
    elif (arg == '--local'):
        use_local = True
    elif (arg == '--tz'):
        tz = float(inp.pop(0))
    elif (arg == '--utc'):
        use_local = False
    elif (arg.startswith('-')):
        sys.exit('unknown option: %s' % arg)
    else:
        bodies.append(arg)


time = datetime.strptime(date, '%y%m%d')    # check next transit time from UTC 0h
print('=======')
print(' SITE:', site)
print(' DATE: ', time.strftime('%Y-%m-%d'))
print(' Limit: ', elim, ' deg')
print(' sort by:', skey)
if (use_local):
    print(' * times are shown in local time (tz:%d) *' % tz)
else:
    print(' * times are shown in UT *')
print('=======')
#print '%-10s   %-11s   %-11s   %-6s   %-8s   %-8s   %-8s' % ('#NAME', 'RA_2000', 'DEC_2000', 'El_deg', 'T_trans', 'T_rise', 'T_set')

RES = {}
for body in bodies:
    #b, mlo = mloplanet(body.lower(), time, epoch, DB=DB, retMLO=True, elim=elim)
    b, obs = obsBody(body.lower(), time=time, site=site, DB=DB, retOBS=True, elim=elim)
    t_rise  = obs.next_rising(b)
    t_trans = obs.next_transit(b)
    t_set   = obs.next_setting(b)
    obs.date = t_trans
    b.compute(obs)

    if (use_local):
        t_riseH  = ephem.Date(t_rise + tz*ephem.hour)
        t_transH = ephem.Date(t_trans + tz*ephem.hour)
        t_setH   = ephem.Date(t_set + tz*ephem.hour)

    #print '%-10s   %11s   %11s   % 6.2f   %8s   %8s   %8s' % (body, b.a_ra, b.a_dec, b.alt/np.pi*180., t_trans.datetime().strftime('%H:%M:%S'), t_rise.datetime().strftime('%H:%M:%S'), t_set.datetime().strftime('%H:%M:%S'))

    obj = {}
    obj['ra']  = b.a_ra
    obj['dec'] = b.a_dec
    obj['el_trans'] = b.alt/np.pi*180.
    obj['t_trans'] = t_trans.datetime().strftime('%H:%M:%S')
    obj['t_rise']  = t_rise.datetime().strftime('%H:%M:%S')
    obj['t_set']   = t_set.datetime().strftime('%H:%M:%S')
    if (use_local):
        obj['t_transH'] = t_transH.datetime().strftime('%H:%M:%S')
        obj['t_riseH']  = t_riseH.datetime().strftime('%H:%M:%S')
        obj['t_setH']   = t_setH.datetime().strftime('%H:%M:%S')

    RES[body] = obj

## try to order the results by ascending RA
#list_sort = sorted(RES.items(), key=lambda d: d[1]['ra'])
list_sort = sorted(list(RES.items()), key=lambda d: d[1][skey])
## the sort key will use the UT times (avoid midnight crossing)


print('%-10s   %-11s   %-11s   %-6s   %-8s   %-8s   %-8s' % ('#NAME', 'RA_2000', 'DEC_2000', 'El_deg', 'T_trans', 'T_rise', 'T_set'))
for obj in list_sort:
    body = obj[0]
    prop = obj[1]
    
    if (use_local):
        print('%-10s   %11s   %11s   % 6.2f   %8s   %8s   %8s' % (body, prop['ra'], prop['dec'], prop['el_trans'], prop['t_transH'], prop['t_riseH'], prop['t_setH']))
    else:
        print('%-10s   %11s   %11s   % 6.2f   %8s   %8s   %8s' % (body, prop['ra'], prop['dec'], prop['el_trans'], prop['t_trans'], prop['t_rise'], prop['t_set']))


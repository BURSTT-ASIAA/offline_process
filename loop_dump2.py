#!/usr/bin/env python

import sys, os.path
import time
from subprocess import call


#cmd0 = '~/work/c/sendsocket 127.0.0.1 8000'
cmd0 = '/home/ubuntu/work/c/outrigger/sendsocket'      # no need to specify port anymore, fixed
#cmd0 = '/home/ubuntu/work/c/thread_2/sendsocket'      # no need to specify port anymore, fixed

inp = sys.argv[0:]
pg  = inp.pop(0)

tWait = 300     # wait interval in sec
nBlock = 2      # number of Block (1 Block = 1M packets) to output
tBlock = 1.   # the nominal time for one Block in sec
host    = '127.0.0.1'
odir    = '/disk1/data'

usage = '''
repeat sendsocket to save packets at a regular interval
(the listensocket is now a daemon and always running)
an instance of the rudp2huge program must be running

new version (loop_dump2.py) to work with new sendsocket command
that saves a selected fraction of ring buffer instead of all

syntax:
    %s DUR [options]

    DUR is the duration to run in seconds

    options are:
    -w tWait    # specify the wait interval between outputs in seconds
                # (default: %d)
    -n nBlock   # specify how many Blocks to output (1 Block = 1M packets)
                # (default: %d)
    -o out_dir  # output directory to save the binary data
                # (default: %s)
    -h host     # host address
                # (default: %s)

''' % (pg, tWait, nBlock, odir, host)

if (len(inp) < 1):
    sys.exit(usage)

while (inp):
    k = inp.pop(0)
    if (k == '-w'):
        tWait = float(inp.pop(0))
    elif (k == '-n'):
        nBlock = int(inp.pop(0))
    elif (k == '-o'):
        odir = inp.pop(0)
    elif (k == '-h'):
        host = inp.pop(0)
    elif (k.startswith('-')):
        sys.exit('unknown option: %s'%k)
    else:
        dur = float(k)

cmd0 += ' -host %s' % host

cmd = cmd0 + ' -cmd 1'           # start receiving packets
print('starting script:', time.asctime())
call(cmd, shell=True)

tSleep = tBlock*nBlock*3    # wait 3 cycles before first output 
print('pre-run wait: %.2f sec' % tSleep)
time.sleep(tSleep)

t0 = time.time()
t1 = time.time()
idx = 0
while ((t1-t0)<=dur):
    cmd = cmd0 + ' -cmd 4 -from -%d -to -1 -out %s' % (nBlock, odir)
    print('output_%04d:'%idx, time.asctime())
    call(cmd, shell=True)

    time.sleep(tWait)            # wait before next loop
    idx += 1
    t1 = time.time()


print('scrip finished:', time.asctime())
if (False):
    # stop the packets
    cmd = cmd0 + ' 0'
    print('stopping packets:', time.asctime())
    call(cmd, shell=True)



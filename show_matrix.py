#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sys, os.path


inp = sys.argv[0:]
pg = inp.pop(0)

usage = '''
plot out the 0-ch beamform matrix
usage:
    %s <matrix_file>

''' % pg

if (len(inp) < 1):
    sys.exit(usage)

#ifile = 'bfm_4vis.bin'
ifile = inp.pop(0)
png = ifile + '.map0.png'

nChan = 1024
nAnt = 16
nBeam = 16
nPart = 2
nSize = nChan*nBeam*nAnt  # number of elements
nByte = nSize * 2 * 2   # real/imag * short (2 bytes)

arr = np.fromfile(ifile, dtype=np.int16, count=nSize*2).reshape((nChan,nBeam,nAnt,nPart))
#print(arr[0])
mat = arr[:,:,:,0] + 1.j*arr[:,:,:,1]
mmin = arr.min()
mmax = arr.max()
amin = np.abs(mat).min()
amax = np.abs(mat).max()


x = np.arange(nAnt)
y = np.arange(nBeam)
X,Y = np.meshgrid(x, y, indexing='xy')

fig, sub = plt.subplots(2,2,figsize=(15,9))

# real part
ax = sub[0,0]
ax.pcolormesh(X,Y,mat[0].real, vmin=mmin, vmax=mmax, shading='nearest')
ax.set_title('real, ch=0')
ax.set_xlabel('Ant')
ax.set_ylabel('Beam')
# imag part
ax = sub[0,1]
ax.pcolormesh(X,Y,mat[0].imag, vmin=mmin, vmax=mmax, shading='nearest')
ax.set_title('imag, ch=0')
ax.set_xlabel('Ant')
ax.set_ylabel('Beam')
# ampld part
ax = sub[1,0]
ax.pcolormesh(X,Y,np.abs(mat[0]), vmin=amin, vmax=amax, shading='nearest')
ax.set_title('ampld, ch=0')
ax.set_xlabel('Ant')
ax.set_ylabel('Beam')
# phase part
ax = sub[1,1]
ax.pcolormesh(X,Y,np.angle(mat[0]), vmin=-3.2, vmax=3.2, shading='nearest')
ax.set_title('phase, ch=0')
ax.set_xlabel('Ant')
ax.set_ylabel('Beam')

sptitle = 'beamform matrix, ch=0 ' + ifile
fig.tight_layout(rect=[0,0.03,1,0.95])
fig.suptitle(sptitle)
fig.savefig(png)
plt.close(fig)



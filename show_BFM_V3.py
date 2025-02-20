#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from loadh5 import *

#files = glob('*.npy')
#files = glob('../row_*/fpga*.h5')
#files.sort()

medamp = np.zeros((16,16))
for i in range(16):
    #f = files[i]
    #dat = np.load(f)    # shape (1024,16), complex eigenvector
    pat = '../b1%d/fpga%d.*.eigen.h5'%(i//4+1, i%4)
    tmp = glob('%s'%(pat,))
    if (len(tmp)>0):
        f = tmp[0]
        print('%s --> %s'%(pat, f))
    else:
        print('skip:', pat)
        continue
    V3 = getData(f, 'V3_coeff')
    dat = V3[:,:,-1]
    medamp[i] = np.median(np.abs(dat), axis=0)

fig, ax = plt.subplots(1,1,figsize=(8,6))
s=ax.pcolormesh(medamp)
cb = plt.colorbar(s, ax=ax)
ax.set_xlabel('ant id')
ax.set_ylabel('row id')
fig.tight_layout()
fig.savefig('medamp_V3.png')
plt.close(fig)

N_eff_V3 = ((medamp**2).sum())**2/((medamp**4).sum())
print('V3 N_eff:', N_eff_V3)

medamp = np.zeros((16,16))
for i in range(16):
    #f = files[i]
    #dat = np.load(f)    # shape (1024,16), complex eigenvector
    pat = '../b1%d/fpga%d.*.eigen.h5'%(i//4+1, i%4)
    tmp = glob('%s'%(pat,))
    if (len(tmp)>0):
        f = tmp[0]
        print('%s --> %s'%(pat, f))
    else:
        print('skip:', pat)
        continue
    N3 = getData(f, 'N3_coeff')
    dat = N3[:,:]
    medamp[i] = np.median(np.abs(dat), axis=1)

fig, ax = plt.subplots(1,1,figsize=(8,6))
s=ax.pcolormesh(medamp)
cb = plt.colorbar(s, ax=ax)
ax.set_xlabel('ant id')
ax.set_ylabel('row id')
fig.tight_layout()
fig.savefig('medamp_N3.png')
plt.close(fig)


#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from glob import glob

#files = glob('*.npy')
#files.sort()

medamp = np.zeros((16,16))
for i in range(16):
    #f = files[i]
    pat = 'row%02d'%(i+1)
    tmp = glob('*_%s.*.npy'%pat)
    if (len(tmp)>0):
        f = tmp[0]
        print('%s --> %s'%(pat, f))
    else:
        print('skip:', pat)
        continue
    dat = np.load(f)    # shape (1024,16), complex eigenvector
    medamp[i] = np.median(np.abs(dat), axis=0)

fig, ax = plt.subplots(1,1,figsize=(8,6))
s=ax.pcolormesh(medamp)
cb = plt.colorbar(s, ax=ax)
ax.set_xlabel('ant id')
ax.set_ylabel('row id')
fig.tight_layout()
fig.savefig('medamp.png')
plt.close(fig)


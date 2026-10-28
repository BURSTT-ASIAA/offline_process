#!/usr/bin/env python

import sys

import numpy as np


def print_value(name, value, indent=0):
    prefix = ' '*indent
    if isinstance(value, np.ndarray):
        if value.ndim > 1:
            print('%s%s: shape=%s dtype=%s' % (prefix, name, value.shape, value.dtype))
            return
        if value.ndim == 0:
            value = value.item()
        else:
            print('%s%s = %s' % (prefix, name, value))
            return

    if isinstance(value, dict):
        print('%s%s:' % (prefix, name))
        for key in sorted(value, key=str):
            print_value(key, value[key], indent+2)
    else:
        print('%s%s = %s' % (prefix, name, value))


usage = '''
usage: %s <npz file(s)>
''' % sys.argv[0]

files = sys.argv[1:]
if not files:
    sys.exit(usage)

for filename in files:
    try:
        archive = np.load(filename, allow_pickle=True)
    except (OSError, ValueError) as error:
        sys.exit('error opening %s: %s' % (filename, error))

    print('=====')
    print('INFO for', filename)
    print('=====')
    for key in sorted(archive.files):
        print_value(key, archive[key])
    archive.close()
    print('')


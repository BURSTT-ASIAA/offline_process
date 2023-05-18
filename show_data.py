#!/usr/bin/env python

from loadh5 import *


inp = sys.argv[0:]
pg  = inp.pop(0)

obj = '/'
mode    = None


usage   = '''
display the content of dataset or attrributes from the h5 files

usage:
    %s <-a|-d> <KEY> <h5 files> [options]

    -a <KEY>        # show the <KEY> in attrs of the designated object
                    # default to the root group '/'

    -d <KEY>        # show the <KEY> dataset
                    # <KEY> position is relative to the designated object (default: '/')

    options are:
    -g <OBJ>        # change the designated object (default = '/')

''' % pg


if (len(inp) < 3):
    sys.exit(usage)


files   = []
while (inp):
    k = inp.pop(0)

    if (k == '-a'):
        key = inp.pop(0)
        mode = 'attrs'
    elif (k == '-d'):
        key = inp.pop(0)
        mode = 'data'
    elif (k == '-g'):
        obj = inp.pop(0)
    elif (k.startswith('-')):
        sys.exit('unknown option: %s' % k)
    else:
        files.append(k)

print(files)

for fh5 in files:
    print('<%s>' % fh5)
    if (not os.path.isfile(fh5)):
        print('... not found. skip.')
        continue

    try:
        fh = h5py.File(fh5, 'r')
        fh.close()
    except:
        print('... h5py error. skip')
        continue

    if (mode == 'data'):
        dname = '%s/%s' % (obj, key)
        dobj = getData(fh5, dname)
        if (dobj is None):
            print('... data is empty?', dname)
            continue
        else:
            print('... %s =' % dname , dobj)

    elif (mode == 'attrs'):
        attrs = getAttrs(fh5, dest=obj)
        if (attrs is None):
            print('... attrs is empty?', obj)
            continue
        if (not attrs.get(key) is None):
            aobj = attrs[key]
            print('... %s =' % key, aobj)
        else:
            print('... %s not found in %s.' % (key, obj))




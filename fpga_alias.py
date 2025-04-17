#!/usr/bin/env python

fpga_alias = {
    '192.168.40.223':'FUS_row01',
    '192.168.40.224':'FUS_row02',
    '192.168.40.222':'FUS_row03',
    '192.168.40.229':'FUS_row04',
    '192.168.40.252':'FUS_row05',
    '192.168.40.250':'FUS_row06',
    '192.168.40.244':'FUS_row07',
    '192.168.40.247':'FUS_row08',
    '192.168.40.107':'FUS_row09',
    '192.168.40.105':'FUS_row10',
    '192.168.40.110':'FUS_row11',
    '192.168.40.248':'FUS_row12',
    '192.168.40.116':'FUS_row13',   # was 111
    '192.168.40.108':'FUS_row14',
    '192.168.40.109':'FUS_row15',
    '192.168.40.104':'FUS_row16',
    '192.168.40.249':'LTN_row01',
    '192.168.40.246':'LTN_row02',
    '192.168.40.245':'LTN_row03',
    '192.168.40.220':'LTN_row04',
    #'192.168.40.115':'LUD_row01',
    #'192.168.40.114':'LUD_row02',
    '192.168.40.118':'LUD_row01',
    '192.168.40.119':'LUD_row02',
    '192.168.40.106':'LUD_row03',
    '192.168.40.113':'LUD_row04'
}

site_ips = {}
fpga_id = {}
for k in fpga_alias.keys():
    v = fpga_alias[k]
    s = v[:3]
    if (site_ips.get(s) is None):
        site_ips[s] = [k]
    else:
        site_ips[s].append(k)

    f = k.split('.')[-1]
    fpga_id[v] = f
#print(site_ips)
#print(fpga_id)



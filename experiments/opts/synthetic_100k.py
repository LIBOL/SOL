#! /usr/bin/env python
#################################################################################
#     File Name           :     synthetic_100k.py
#     Created By          :     yuewu
#     Creation Date       :     [2016-10-25 11:21]
#     Last Modified       :     [2016-12-06 21:34]
#     Description         :
#################################################################################

import numpy as  np
import collections

const_eta_search = np.logspace(-5, 5, 11, base=2)
eta_search = np.logspace(-2, 8, 11, base=2)
delta_search = np.logspace(-5, 5,11, base=2)
r_search = np.logspace(-5, 5, 11, base=2)
delta_ofs_search = np.logspace(-5, 5, 11, base=2) / 100.0

fs_num = [50,60,70,80,90,100, 120,140,160,180,200]

fs_opts = collections.OrderedDict()

fs_opts['SOFS'] = {
    'params':{'norm':'L2'},
    'cv':{'r': r_search},
    'lambda': fs_num
}
fs_opts['PET'] = {
    'params':{'norm':'L2', 'power_t':'0'},
    'cv':{'eta':eta_search},
    'lambda': fs_num
}
fs_opts['FOFS'] = {
    'params':{'norm':'L2'},
    'cv':{'eta': const_eta_search, 'lambda': delta_ofs_search},
    'lambda': fs_num
}
fs_opts['FGM'] = {
    'lambda': fs_num
}

fs_opts['liblinear'] = {
    'lambda': [0.0001,0.00015, 0.0002, 0.00025,0.0005,0.001,0.01, 0.018, 0.02, 0.022, 0.023, 0.024, 0.025]
}

fs_opts['mRMR'] = {
    'params':{'binary_thresh':0},
    'lambda': fs_num
}
fs_opts['GPU-mRMR'] = {
    'params':{'binary_thresh':0},
    'lambda': fs_num
}
#fs_opts['AROW'] = {
#    'params':{'norm':'L2'},
#    'cv':{'r': r_search},
#    'lambda': [-1]
#}
#fs_opts['OGD'] = {
#    'params':{'norm':'L2'},
#    'cv':{'eta':eta_search},
#    'lambda': [-1]
#}

draw_opts = {
    'accu':{
        'xlim':[49, 200],
        'ylim':[0.7, 1]
    },
    'time': {
        'logy': True,
        'xlim':[49, 200],
        'ylim':[2, 1000],
        'legend_order':100,
    }
}

#! /usr/bin/env python
#################################################################################
#     File Name           :     synthetic_100k.py
#     Created By          :     yuewu
#     Creation Date       :     [2016-10-25 11:21]
#     Last Modified       :     [2016-12-07 10:41]
#     Description         :
#################################################################################

import numpy as  np
import collections

const_eta_search = np.logspace(-5, 5, 11, base=2)
eta_search = np.logspace(-2, 8, 11, base=2)
delta_search = np.logspace(-5, 5,11, base=2)
r_search = np.logspace(-5, 5, 11, base=2)
delta_ofs_search = np.logspace(-5, 5, 11, base=2) / 100.0

fs_num = [150,160,170,180,190,200, 220,240,260,280,300]

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
    'lambda': [0.0002,0.0003,0.0004,0.0008,0.01,0.015,0.016,0.017,0.018,0.019,0.02]
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
        'xlim':[149, 300],
        #'ylim':[0.73, 1]
    },
    'time': {
        'logy': True,
        'xlim':[149, 300],
        'ylim':[4, 3000]
    }
}

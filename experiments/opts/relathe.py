#! /usr/bin/env python
#################################################################################
#     File Name           :     synthetic_100k.py
#     Created By          :     yuewu
#     Creation Date       :     [2016-10-25 11:21]
#     Last Modified       :     [2016-12-05 19:45]
#     Description         :
#################################################################################

import numpy as  np
import collections

const_eta_search = np.logspace(-5, 5, 11, base=2)
eta_search = np.logspace(-2, 8, 11, base=2)
delta_search = np.logspace(-5, 5,11, base=2)
r_search = np.logspace(-5, 5, 11, base=2)
delta_ofs_search = np.logspace(-5, 5, 11, base=2) / 100.0
norm_search = ['L2', 'None']

dim = 4322
fs_num = (np.array([0.05, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]) * dim).astype(np.int)

fs_opts = collections.OrderedDict()

fs_opts['SOFS'] = {
    'cv':{'r': r_search, 'norm':norm_search},
    'lambda': fs_num
}
fs_opts['PET'] = {
    'params':{'power_t':'0'},
    'cv':{'eta':eta_search, 'norm':norm_search},
    'lambda': fs_num
}
fs_opts['FOFS'] = {
    'cv':{'eta': const_eta_search, 'lambda': delta_ofs_search, 'norm':norm_search},
    'lambda': fs_num
}
fs_opts['FGM'] = {
    'lambda': fs_num
}
fs_opts['liblinear'] = {
    'lambda': [0.015625,0.03125,0.0625,0.125,64,128,512,1024,2048,4096,9182]
}
fs_opts['mRMR'] = {
    'lambda': fs_num
}
fs_opts['GPU-mRMR'] = {
    'lambda': fs_num
}

draw_opts = {
    'accu':{
        'ylim':[0.73,0.9],
    },
    'time': {
        'logy': True,
    }
}

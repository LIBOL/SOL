#! /usr/bin/env python
#################################################################################
#     File Name           :     synthetic_100k.py
#     Created By          :     yuewu
#     Creation Date       :     [2016-10-25 11:21]
#     Last Modified       :     [2016-11-19 20:54]
#     Description         :
#################################################################################

import numpy as  np

const_eta_search = np.logspace(-5, 5, 11, base=2)
eta_search = np.logspace(-2, 8, 11, base=2)
delta_search = np.logspace(-5, 5,11, base=2)
r_search = np.logspace(-2, 8, 11, base=2)
delta_ofs_search = np.logspace(-5, 5, 11, base=2) / 100.0

fs_num = [50,60,70,80,90,100, 120,140,160,180,200]
fs_opts = {
    'pet': {
        'params':{'norm':'L2', 'power_t':'0'},
        'cv':{'eta':eta_search},
        'lambda': fs_num
    },
    'fofs': {
        'params':{'norm':'L2'},
        'cv':{'eta': const_eta_search, 'lambda': delta_ofs_search},
        'lambda': fs_num
    },
    'sofs': {
        'params':{'norm':'L2'},
        'cv':{'r': r_search},
        'lambda': fs_num
    },
    'FGM':  {
        'lambda': fs_num
    },
    'mRMR': {
        'params':{'binary_thresh':0},
        'lambda': fs_num
    },
    'liblinear': {
        'lambda': [0.0001,0.00015, 0.0002, 0.00025,0.0005,0.001,0.01, 0.018, 0.02, 0.022, 0.023, 0.024, 0.025]
    }
}


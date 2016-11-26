#! /usr/bin/env python
#################################################################################
#     File Name           :     synthetic_100k.py
#     Created By          :     yuewu
#     Creation Date       :     [2016-10-25 11:21]
#     Last Modified       :     [2016-11-20 17:46]
#     Description         :
#################################################################################

import numpy as  np

const_eta_search = np.logspace(-5, 5, 11, base=2)
eta_search = np.logspace(-2, 8, 11, base=2)
delta_search = np.logspace(-5, 5,11, base=2)
r_search = np.logspace(-2, 8, 11, base=2)
delta_ofs_search = np.logspace(-5, 5, 11, base=2) / 100.0

fs_num = [150,160,170,180,190,200, 220,240,260,280,300]
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
        'lambda': [0.0002,0.0003,0.0004,0.0008,0.01,0.015,0.016,0.017,0.018,0.019,0.02]
    }
}


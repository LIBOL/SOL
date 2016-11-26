#! /usr/bin/env python
#################################################################################
#     File Name           :     synthetic_100k.py
#     Created By          :     yuewu
#     Creation Date       :     [2016-10-25 11:21]
#     Last Modified       :     [2016-11-25 14:20]
#     Description         :      
#################################################################################

import numpy as  np

const_eta_search = np.logspace(-5, 5, 11, base=2)
eta_search = np.logspace(-2, 8, 11, base=2)
delta_search = np.logspace(-5, 5,11, base=2)
r_search = np.logspace(-2, 8, 11, base=2)
delta_ofs_search = np.logspace(-5, 5, 11, base=2) / 100.0
norm_search = ['L2', 'None']

dim = 20958
fs_num = (np.array([0.025,0.05, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]) * dim).astype(np.int)
fs_opts = {
    'pet': {
        'params':{'power_t':'0'},
        'cv':{'eta':eta_search, 'norm':norm_search},
        'lambda': fs_num
    },
    'fofs': {
        'cv':{'eta': const_eta_search, 'lambda': delta_ofs_search, 'norm':norm_search},
        'lambda': fs_num
    },
    'sofs': {
        'cv':{'r': r_search, 'norm':norm_search},
        'lambda': fs_num
    },
    'FGM':  {
        'lambda': fs_num
    },
    'mRMR': {
        'params':{'binary_thresh':0.5},
        'lambda': fs_num
    },
    'liblinear': {
        'lambda': [0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2,
                   64,512,1024,2048,8192,16384,131072]
    }
}

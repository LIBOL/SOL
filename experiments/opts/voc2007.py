#! /usr/bin/env python
#################################################################################
#     File Name           :     voc2007.py
#     Created By          :     yuewu
#     Creation Date       :     [2016-11-29 12:35]
#     Last Modified       :     [2016-12-06 22:07]
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

dim = 8192
passes = 1
fs_num = (np.array([0.01, 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]) * dim).astype(np.int)
#fs_num = (np.array([0.1,0.2,0.3,0.4,0.5,0.6]) * dim).astype(np.int)

fs_opts = collections.OrderedDict()

fs_opts['SOFS'] = {
    'params':{'norm':'L2'},
    'cv':{'r': r_search},
    'lambda': fs_num
}
fs_opts['PET'] = {
    'params':{'power_t':0, 'norm':'L2'},
    'cv':{'eta':const_eta_search},
    'lambda': fs_num
}
fs_opts['FOFS'] = {
    'params':{'norm':'L2'},
    'cv':{'eta': const_eta_search, 'lambda': delta_ofs_search},
    'lambda': fs_num
}
fs_opts['mRMR'] = {
    'params': {
        'binary_thresh':0.5,
        'ol_model_params': {'power_t':0, 'norm':'L2'},
    },
    'lambda': fs_num
}
fs_opts['GPU-mRMR'] = {
    'params': {
        'binary_thresh':0.5,
        'ol_model_params': {'power_t':0, 'norm':'L2'},
    },
    'lambda': fs_num
}
#
#    'liblinear': {
#        'lambda': [0.015625,0.03125,0.0625,0.125, 0.5,2,128,512,2048,4096,16384,131072,262144,524288]

draw_opts = {
    'accu':{
        'ylim':[0.88,0.91],
        'xlim':[500,8000],
        'clip_on':True,
        'legend_loc':'lower right',
    },
    'time': {
        'logy': True,
        'ylim': [3,2000],
        'xlim':[500,8000],
        'legend_order':100,
    }
}

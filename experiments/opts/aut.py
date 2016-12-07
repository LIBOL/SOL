#! /usr/bin/env python
#################################################################################
#     File Name           :     aut.py
#     Created By          :     yuewu
#     Creation Date       :     [2016-10-25 11:21]
#     Last Modified       :     [2016-12-05 19:56]
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

dim = 20072
fs_num = (np.array([0.025,0.05, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]) * dim).astype(np.int)

fs_opts = collections.OrderedDict()

fs_opts['SOFS'] =  {
    'cv':{'r': r_search, 'norm':norm_search},
    'lambda': fs_num
}

fs_opts['PET'] =  {
    'params':{'power_t':'0'},
    'cv':{'eta':eta_search, 'norm':norm_search},
    'lambda': fs_num
}
fs_opts['FOFS'] =  {
    'cv':{'eta': const_eta_search, 'lambda': delta_ofs_search, 'norm':norm_search},
    'lambda': fs_num
}
fs_opts['FGM'] =   {
    'lambda': fs_num
}
fs_opts['liblinear'] =  {
    'lambda': [0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2,512,1024,2048,4096,8192,16384]
}
fs_opts['mRMR'] =  {
    'params':{'binary_thresh':0.5},
    'lambda': fs_num
}
fs_opts['GPU-mRMR'] = {
    'params':{'binary_thresh':0.5},
    'lambda': fs_num
}

draw_opts = {
    'accu':{
        'clip_on':True,
        'ylim':[0.93,0.985],
        'legend_loc':'lower right',
        'bbox_to_anchor':(1,0.15),
    },
    'time': {
        'logy': True,
        #'xlim':[0,25000],
        'legend_loc':'center',
        'bbox_to_anchor':(0.7,0.65),
        'legend_order':100,
    }
}

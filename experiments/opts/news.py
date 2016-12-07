#! /usr/bin/env python
#################################################################################
#     File Name           :     synthetic_100k.py
#     Created By          :     yuewu
#     Creation Date       :     [2016-10-25 11:21]
#     Last Modified       :     [2016-12-06 16:35]
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

dim = 1355191
fs_num = (np.array([0.005,0.05, 0.1,0.2,0.3,0.4,0.5]) * dim).astype(np.int)

fs_opts = collections.OrderedDict()

fs_opts['SOFS'] = {
    'params':{'norm':'L2'},
    'cv':{'r': r_search},
    'lambda': fs_num
}
fs_opts['PET'] = {
    'params':{'power_t':'0', 'norm':'L2'},
    'cv':{'eta':const_eta_search},
    'lambda': fs_num
}

fs_opts['liblinear'] = {
    'lambda': [5000,10000,20000,40000,80000,160000]
}

fs_opts['FGM'] = {
    'lambda': fs_num
}
draw_opts = {
    'accu':{
    },
    'time': {
        'logy': True,
        'legend_loc':'center',
        'bbox_to_anchor':(0.7,0.65),
        #'legend_order':100,
    }
}

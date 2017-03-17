#! /usr/bin/env python
#################################################################################
#     File Name           :     synthetic_100k.py
#     Created By          :     yuewu
#     Creation Date       :     [2016-10-25 11:21]
#     Last Modified       :     [2016-12-07 18:08]
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

dim = 47152
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

fs_opts['FGM'] = {
    'lambda': fs_num
#    }
#    'liblinear': {
#        'lambda': [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0008,0.0007,0.0009,0.001,0.01,0.012,0.013,0.014,0.015,0.016,0.018,0.02]
}

draw_opts = {
    'accu':{
        'ylim':[0.9,0.98],
        'clip_on': True,
    },
    'time': {
        'logy': True,
        'legend_loc':'center',
        'bbox_to_anchor':(0.7,0.65),
        #'legend_order':100,
    }
}

ol_opts = {}
ol_opts['ada-fobos'] = {
    'cv':{'eta':const_eta_search, 'delta':delta_search}
}
ol_opts['ada-rda'] = {
    'cv':{'eta':const_eta_search, 'delta':delta_search}
}
ol_opts['alma2'] = {
    'cv':{'alpha': np.linspace(0.1, 1, 10)}
}
ol_opts['arow'] = {
    'cv':{'r':r_search}
}
ol_opts['cw'] = {
    'cv':{'a': np.logspace(-4, 0, 5, base=2), 'phi':np.linspace(0, 2, 9)}
}
ol_opts['eccw'] = {
    'cv':{'a': np.logspace(-4, 0, 5, base=2), 'phi':np.linspace(0, 2, 9)}
}
ol_opts['ogd'] = {
    'cv':{'eta':eta_search}
}
ol_opts['pa'] = {}
ol_opts['pa1'] = {
    'cv':{'C':np.logspace(-4, 4, 9, base=2)}
}
ol_opts['pa2'] = {
    'cv':{'C':np.logspace(-4, 4, 9, base=2)}
}
ol_opts['perceptron'] = {}
ol_opts['sop'] = {
    'cv':{'a':np.logspace(-4, 4, 9, base=2)}
}
ol_opts['rda'] = {}
ol_opts['erda-l1'] = {}

for k,v in ol_opts.iteritems():
    if 'params' not in v:
        v['params'] = {}
    v['params']['step_show'] = 50000

ol_opts['liblinear'] = {
    'cv': {'C': np.logspace(-5,7,13, base=2)}
}
ol_opts['vw'] = {
    'cv':{'learning_rate':np.logspace(-4,7,12,base=2)}
}

sol_opts = {}
sol_opts['stg'] = {
    'params':{'k':10},
    'cv':{'eta':eta_search},
    'lambda': np.logspace(-6,-1,10,base=10)
}
#sol_opts['fobos-l1'] = {
#    'cv':{'eta':eta_search},
#    'lambda': np.logspace(-6,-1,10,base=10)
#}
#sol_opts['rda-l1'] = {
#    'lambda': np.logspace(-6,-1,10,base=10)
#}
#
#sol_opts['erda-l1'] = {
#    'params':{'rou':0.001},
#    'lambda': np.logspace(-6,-1,10,base=10)
#}
#sol_opts['ada-fobos-l1'] = {
#    'cv':{'eta':const_eta_search, 'delta':delta_search},
#    'lambda': np.logspace(-6,-1,10,base=10)
#}
#sol_opts['ada-rda-l1'] = {
#    'cv':{'eta':const_eta_search, 'delta':delta_search},
#    'lambda': np.logspace(-7,-2,10,base=10)
#}
#sol_opts['liblinear'] = {
#    'lambda':np.logspace(-5,7,13, base=2)
#}
#
#sol_opts['vw'] = {
#    'cv':'vw',
#    'lambda':np.logspace(-6,-2,10, base=10)
#}

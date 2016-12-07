#! /usr/bin/env python
#################################################################################
#     File Name           :     webspam.py
#     Created By          :     yuewu
#     Description         :      
#################################################################################
import numpy as np

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
    v['params']['step_show'] = 20000

#ol_opts['liblinear'] = {
#    'cv': {'C': np.logspace(-5,7,13, base=2)}
#}
#ol_opts['vw'] = {
#    'cv':{'learning_rate':np.logspace(-4,7,12,base=2)}
#}

sol_opts = {}
sol_opts['stg'] = {
    'params':{'k':10},
    'cv':{'eta':eta_search},
    'lambda': np.logspace(-6,-1,10,base=10)
}
sol_opts['fobos-l1'] = {
    'cv':{'eta':eta_search},
    'lambda': np.logspace(-6,-1,10,base=10)
}
sol_opts['rda-l1'] = {
    'lambda': np.logspace(-6,-1,10,base=10)
}

sol_opts['erda-l1'] = {
    'params':{'rou':0.001},
    'lambda': np.logspace(-6,-1,10,base=10)
}
sol_opts['ada-fobos-l1'] = {
    'cv':{'eta':const_eta_search, 'delta':delta_search},
    'lambda': np.logspace(-6,-1,10,base=10)
}
sol_opts['ada-rda-l1'] = {
    'cv':{'eta':const_eta_search, 'delta':delta_search},
    'lambda': np.logspace(-7,-2,10,base=10)
}
#sol_opts['liblinear'] = {
#    'lambda':np.logspace(-5,7,13, base=2)
#}
#
#sol_opts['vw'] = {
#    'cv':'vw',
#    'lambda':np.logspace(-6,-2,10, base=10)
#}

#! /usr/bin/env python
#################################################################################
#     File Name           :     webspam.py
#     Created By          :     yuewu
#     Description         :      
#################################################################################
import numpy as np

ol_opts = {}
ol_opts['ada-fobos'] = {'algo':'ada-fobos', 'cv':['eta=0.0625:2:128', 'delta=0.0625:2:16']}
ol_opts['ada-rda'] = {'algo':'ada-rda', 'cv':['eta=0.0625:2:128', 'delta=0.0625:2:16']}
ol_opts['alma2'] = {'algo':'alma2', 'cv':['alpha=0.1:+0.1:1']}
ol_opts['arow'] = {'algo':'arow', 'cv':['r=0.0625:2:16']}
ol_opts['cw'] = {'algo':'cw', 'cv':['a=0.0625:2:1', 'phi=0:+0.25:2']}
ol_opts['eccw'] = {'algo':'eccw', 'cv':['a=0.0625:2:1', 'phi=0:+0.25:2']}
ol_opts['ogd'] = {'algo':'ogd', 'cv':['eta=0.0625:2:128']}
ol_opts['pa'] = {'algo':'pa'}
ol_opts['pa1'] = {'algo':'pa1', 'cv':['C=0.0625:2:16']}
ol_opts['pa2'] = {'algo':'pa2', 'cv':['C=0.0625:2:16']}
ol_opts['perceptron'] = {'algo':'perceptron'}
ol_opts['sop'] = {'algo':'sop', 'cv':['a=0.0625:2:16']}
ol_opts['rda'] = {'algo':'rda'}
ol_opts['erda'] = {'algo':'erda-l1'}

for k,v in ol_opts.iteritems():
	ol_opts[k]['params'] = ['step_show=20000']

#ol_opts['vw'] = {'algo':'vw'}
#ol_opts['liblinear'] = {'algo':'liblinear'}

sol_opts = {}
sol_opts['stg'] = {'algo':'stg', 'cv':'ogd', 'params':['k=10'], 'lambda': np.logspace(-6,-1,10,base=10) }
sol_opts['fobos-l1'] = {'algo':'fobos-l1', 'cv':'ogd', 'lambda': np.logspace(-6,-1,10,base=10) }
sol_opts['rda-l1'] = {'algo':'rda-l1', 'lambda': np.logspace(-6,-1,10,base=10) }
sol_opts['erda-l1'] = {'algo':'erda-l1', 'params':['rou=0.001'], 'lambda': np.logspace(-6,-1,10,base=10) }
sol_opts['ada-fobos-l1'] = {'algo':'ada-fobos-l1', 'cv':'ada-fobos', 'lambda': np.logspace(-6,-1,10,base=10) }
sol_opts['ada-rda-l1'] = {'algo':'ada-rda-l1', 'cv':'ada-rda', 'lambda': np.logspace(-7,-2,10,base=10) }
#sol_opts['vw'] = {'algo':'vw', 'lambda':np.logspace(-6,-2,10, base=10)}

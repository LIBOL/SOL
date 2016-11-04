#! /usr/bin/env python
#################################################################################
#     File Name           :     real-sim.py
#     Created By          :     
#     Creation Date       :     [2016-10-25 10:22]
#     Last Modified       :     [2016-11-03 23:00]
#     Description         :      
#################################################################################

import numpy as np

const_eta_search = '0.03125:2.0:32'
eta_search = '0.25:2.0:256'
delta_search = '0.03125:2.0:32'
r_search = '0.25:2.0:256'
delta_ofs_search = '0.0003125:2:0.32'

fs_opts = {}
#fs_opts['pet'] = {'algo':'pet', 'cv':[('eta','0.0625:2:128')], 'B': np.linspace(0.1,1,10) }
#fs_opts['fofs'] = {'algo':'fofs', 'cv':[('eta', '0.03125:2.0:32'), ('lambda', '0.03125:2.0:32')], 'B': np.linspace(0.1,1,10) }
fs_opts['sofs'] = {'algo':'sofs', 'cv':[('r', '0.0625:2:16')], 'B': np.linspace(0.1,1,10) }
#fs_opts['liblinear'] = {'algo':'liblinear', 'lambda': np.logspace(-10,10,21, base=2) }
fs_opts['fgm'] = {'algo':'fgm', 'B': [2, 3, 4, 5, 6, 7, 8, 9, 10]}

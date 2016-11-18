#! /usr/bin/env python
#################################################################################
#     File Name           :     synthetic_100k.py
#     Created By          :     yuewu
#     Creation Date       :     [2016-10-25 11:21]
#     Last Modified       :     [2016-11-14 19:04]
#     Description         :      
#################################################################################

const_eta_search = '0.03125:2.0:32'
eta_search = '0.25:2.0:256'
delta_search = '0.03125:2.0:32'
r_search = '0.25:2.0:256'
delta_ofs_search = '0.0003125:2:0.32'

fs_opts = {}
fs_opts['pet'] = {'algo':'pet', 'cv':[('eta',eta_search)], 'B': [50] }
fs_opts['sofs'] = {'algo':'sofs', 'cv':[('r', r_search)], 'B': [50]}

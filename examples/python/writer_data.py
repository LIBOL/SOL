#! /usr/bin/env python
#################################################################################
#     File Name           :     writer_data.py
#     Created By          :     yuewu
#     Creation Date       :     [2017-05-08 22:16]
#     Last Modified       :     [2017-05-08 22:33]
#     Description         :     writer data to different formats
#################################################################################

import os.path as osp
import sys
from sklearn.datasets import load_svmlight_file

from pysol import SOLDataWriter

curr_path = osp.abspath(osp.dirname(__file__))
data_path = osp.join(curr_path, '../../data/a1a')
print data_path

#write to svm
out_path = 'a1a.svm'
writer = SOLDataWriter(out_path, 'svm')
writer.write(data_path, 'svm')

#write to bin
out_path = 'a1a.bin'
writer = SOLDataWriter(out_path, 'bin')
writer.write(data_path, 'svm')

#load into svmlight
data = load_svmlight_file(data_path)
X = data[0]
y = data[1]

#write to svm
out_path = 'a1a2.svm'
writer = SOLDataWriter(out_path, 'svm')
writer.write(X, y)

#write to bin
out_path = 'a1a2.bin'
writer = SOLDataWriter(out_path, 'bin')
writer.write(X, y)

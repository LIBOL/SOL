#! /usr/bin/env python
#################################################################################
#     File Name           :     test.py
#     Created By          :     yuewu
#     Description         :
#################################################################################

import pylsol
from sklearn.datasets import load_svmlight_file
import numpy as np
import sys


def iterate_process(data_num, iter_num, update_num, err_rate):
    print data_num, iter_num, update_num, err_rate


data = load_svmlight_file('../data/a1a')
tdata = load_svmlight_file('../data/a1a.t')

params = {'lambda':0.01}
m1 = pylsol.LSOL("ada-rda-l1", 2, verbose=True, **params)
m2 = pylsol.LSOL("ada-rda-l1", 2, verbose=True, **params)

m1.fit('../data/a1a', 'svm', 4)
m2.fit('../data/a1a', 'svm', 1)
m2.fit('../data/a1a', 'svm', 1)
m2.fit('../data/a1a', 'svm', 1)
m2.fit('../data/a1a', 'svm', 1)

print 'test if train multiple times are the same\nm1\tm2'
print m1.score(tdata[0], tdata[1]), m2.score(tdata[0], tdata[1])
print m1.sparsity, m2.sparsity
sys.exit()

print 'test if all formats are the same'

m1 = pylsol.LSOL("cw", 2, phi=1)
m2 = pylsol.LSOL("cw", 2, phi=1)
m3 = pylsol.LSOL("cw", 2, phi=1)

#m3.inspect_learning(iterate_process)

print 'train from :\nfile \ttarray\tsparse'
print m1.fit('../data/a1a', 'svm'), m2.fit(data[0].todense(), data[1]), m3.fit(
    data[0], data[1])

print 'test from array:\nm1\tm2\tm3'
print m1.score(tdata[0].todense(), tdata[1]), m2.score(
    tdata[0].todense(), tdata[1]), m3.score(tdata[0].todense(), tdata[1])

print 'test from sparse:\nm1\tm2\tm3'
print m1.score(tdata[0], tdata[1]), m2.score(tdata[0], tdata[1]), m3.score(
    tdata[0], tdata[1])

print 'predict from file:\nm1\tm2\tm3'
s1 = 1 - np.sum(m1.predict('../data/a1a.t', 'svm') == tdata[1], dtype=np.float64) / tdata[1].shape[0]
s2 = 1 - np.sum(m2.predict('../data/a1a.t', 'svm') == tdata[1], dtype=np.float64) / tdata[1].shape[0]
s3 = 1 - np.sum(m3.predict('../data/a1a.t', 'svm') == tdata[1], dtype=np.float64) / tdata[1].shape[0]
print s1, s2, s3

print 'predict from array:\nm1\tm2\tm3'
s1 = 1 - np.sum(m1.predict(tdata[0].todense()) == tdata[1], dtype=np.float64) / tdata[1].shape[0]
s2 = 1 - np.sum(m2.predict(tdata[0].todense()) == tdata[1], dtype=np.float64) / tdata[1].shape[0]
s3 = 1 - np.sum(m3.predict(tdata[0].todense()) == tdata[1], dtype=np.float64) / tdata[1].shape[0]
print s1, s2, s3

print 'predict from sparse:\nm1\tm2\tm3'
s1 = 1 - np.sum(m1.predict(tdata[0]) == tdata[1], dtype=np.float64) / tdata[1].shape[0]
s2 = 1 - np.sum(m2.predict(tdata[0]) == tdata[1], dtype=np.float64) / tdata[1].shape[0]
s3 = 1 - np.sum(m3.predict(tdata[0]) == tdata[1], dtype=np.float64) / tdata[1].shape[0]
print s1, s2, s3

print 'decision function from file:\nm1\tm2\tm3'
s1 = 1 - np.sum(np.sign(m1.decision_function('../data/a1a.t', 'svm')) == tdata[1], dtype=np.float64) / tdata[1].shape[0]
s2 = 1 - np.sum(np.sign(m2.decision_function('../data/a1a.t', 'svm')) == tdata[1], dtype=np.float64) / tdata[1].shape[0]
s3 = 1 - np.sum(np.sign(m3.decision_function('../data/a1a.t', 'svm')) == tdata[1], dtype=np.float64) / tdata[1].shape[0]
print s1, s2, s3

print m1.sparsity

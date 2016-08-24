#! /usr/bin/env python
#################################################################################
#     File Name           :     test_sklearn.py
#     Created By          :     yuewu
#     Description         :      
#################################################################################

import numpy as np
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
from sklearn import svm, grid_search
from sklearn.utils import check_array
import ipdb

from lsol import LSOL

data = load_svmlight_file('../data/a1a')
tdata = load_svmlight_file('../data/a1a.t')

#clf.fit(data[0], data[1])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(data[0], data[1], test_size=0.4, random_state = 0)

#print X_train.flags['C_CONTIGUOUS']

clf = LSOL("arow", 2)
clf.fit(X_train, y_train)
print clf.score(X_test, y_test)

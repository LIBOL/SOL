#! /usr/bin/env python
#################################################################################
#     File Name           :     gen_gaussian.py
#     Created By          :     yuewu
#     Creation Date       :     [2017-05-10 23:25]
#     Last Modified       :     [2017-05-11 15:07]
#     Description         :     Geneate Gaussian distribution
#################################################################################

import numpy as np
from sklearn.datasets import dump_svmlight_file

dim = 3
mean=np.random.random((dim,))
#conv=np.eye(dim)
conv=np.random.random((dim,1))
conv = np.eye(dim) + np.matmul(conv,conv.T)

N=500*1024
x=np.random.multivariate_normal(mean, conv,N).astype(np.float32)
y=np.zeros((N,))

print 'mean'
mean = np.mean(x,axis=0)
print mean.dtype
print mean
print conv

dump_svmlight_file(x,y,'gaussian.svm', zero_based=False)


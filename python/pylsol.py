#! /usr/bin/env python
#################################################################################
#     File Name           :     pylsol.py
#     Created By          :     yuewu
#     Description         :
#################################################################################

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
from sklearn.utils.validation import check_is_fitted

import lsol

class LSOL(BaseEstimator):
    """LIBSOL Scikit-learn Base Estimator wrapper"""

    params = dict()

    def __init__(self, algo = None, cls_num=2, batch_size = 256, buf_size = 2, **args):
        """
        Parameters
        ----------
        algo: string
            name of the algorithm to use
        cls_num: int
            number of classes
        batch_size: int
            size of mini-batches in processing
        buf_size: int
            number of mini-batches for bufferring
        args: dict
            dictionary of model parameter keys and values

        Returns
        -------
        (BaseEstimator): Returns self
        """

        #clear estimator attributes
        if hasattr(self, 'fit_'):
            del self.fit_
        if hasattr(self, 'lsol_'):
            del self.lsol_
        if hasattr(self, 'algo'):
            del self.algo
        if hasattr(self, 'cls_num'):
            del self.cls_num

        self.algo = algo
        self.cls_num = cls_num
        self.batch_size = batch_size
        self.buf_size = buf_size

        for k,v in args.iteritems():
            if k != 'self' and k != '__classes__' and v is not None:
                self.params[k] = v

        self.fit_ = False
        self.lsol_ = None

        super(LSOLBaseEstimator, self).__init__()

    def get_lsol(self):
        """Create a new lsol instance from dll

        Returns
        -------
        lsol.CLSOL instance
        """
        if self.lsol_ is None:
            self.lsol_ = lsol.CLSOL(self.algo, self.cls_num, self.batch_size, self.buf_size, self.params)
        return self.lsol_

    def fit(self, param1, param2, pass_num = 1):
        """Fit the model according to the given training data

        Parameters
        ---------
        param1: 1. {array like or sparse matrix}, shape = [n_samples, n_features]
                    Training vector, where n_samples is the number of samples
                    and n_features is the number of features
                2. string path to the training data file
        param2: 1. array-like, shape=[n_samples]
                    Target label vector relative to X
                2. string type of the training data
        pass_num: number of passes to train through the training data

        Returns
        ------
        return self
        """
        model = self.get_lsol()

        if self.fit_ == False:
            self.lsol_.begin_train()

        if isinstance(param1, np.ndarray) and isinstance(param2, np.ndarray):
            self.lsol_.learn_from_data(param1, param2, 0, pass_num)
        elif isinstance(param1, csr_matrix) and isinstance(param2, np.ndarray):
            self.lsol_.learn_from_data(param1, param2, 1, pass_num)
        elif isinstance(param1, str) and isinstance(param2, str):
            self.lsol_.learn_from_file(param1, param2, pass_num)
        else:
            raise NotImplementedError("ndarray and csr_matrix not supported yet")

        self.fit_ = True
        return self

    def predict(self, X):
        """Predict class labels for samples in X

        Parameters
        ----------
        X:  {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples

        Returns
        -------
        C: array, predicted class label per sample
        """
        raise NotImplementedError("")

    def score(self, param1, param2):
        """Returns the mean accuracy on the given test data and labels

        Parameters
        ---------
        param1: 1. {array like or sparse matrix}, shape = [n_samples, n_features]
                    Training vector, where n_samples is the number of samples
                    and n_features is the number of features
                2. string path to the training data file
        param2: 1. array-like, shape=[n_samples]
                    Target label vector relative to X
                2. string type of the training data

        Returns
        ------
        score: float
            Mean accuracy
        """
        model = self.get_lsol()

        self.lsol_.end_train()

        if isinstance(param1, (np.ndarray, csr_matrix)) and isinstance(param2, np.ndarray):
            raise NotImplementedError("ndarray and csr_matrix not supported yet")
        elif isinstance(param1, str) and isinstance(param2, str):
            self.lsol_.learn(param1, param2, pass_num)
        else:
            raise NotImplementedError("ndarray and csr_matrix not supported yet")

        self.fit_ = True
        return self


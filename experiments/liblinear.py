#This script is to run experiment automatically to test the performance of the algorithm

import os
import sys
import os.path as osp
import logging
import time
from  sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
import numpy as np
from scipy.sparse import csr_matrix

def train_test_l2(dtrain, dtest, C,
                  fold_num = 5,
                  retrain = False,
                  verbose=False):
    """Train and Test L2-SVM with Liblinear

    Parameters
    ----------
    dtrain: DatsSet
        training dataset
    dtest: DataSet
        test dataset
    C: float or list
        l2 penalty, or list of values for cross validation
    fold_num: int
        number of folds to do cross validation
    retrain: bool
        whether to retrain the model and cross validation
    verbose: bool
        wheter to print the detailed information

    Return
    ------
    tuple (test accuracy, test time, train accuracy, train time)
    """

    if dtrain.dtype != 'svm':
        raise Exception("liblinear only supports svm type data")

    dual = True if dtrain.data_num < dtrain.dim else False

    if isinstance(C, list):
        cv_output_path  = osp.join(dtrain.work_dir, 'cv-liblinear.txt')
        if os.path.exists(cv_output_path) and retrain == False:
            with open(cv_output_path, 'r') as fh:
                line = fh.readline()
            C = float(line.split('=')[1])
        else:
            #cross validation
            x_train, y_train = datasets.load_svmlight_file(dtrain.rand_path())
            svc = svm.LinearSVC(penalty='l2', dual=dual)
            clf = GridSearchCV(estimator=svc, param_grid=dict(C=C),
                    n_jobs=4, cv=fold_num, verbose=verbose)
            clf.fit(x_train, y_train)
            C = clf.best_estimator_.C

            #write the cross validation results
            with open(cv_output_path, 'w') as fh:
                fh.write('Best Result: C=%f' %(C))

        logging.info('cross validation parameters: C=%f' %(C))

    clf = svm.LinearSVC(penalty='l2', C=C, dual=dual)

    start_time = time.time()

    #load dataset
    logging.info("loading training data %s..." %(dtrain.name))
    x_train, y_train = datasets.load_svmlight_file(dtrain.data_path)

    logging.info("train liblinear with C=%f..." %(C))
    clf.fit(x_train, y_train)

    train_time = time.time() - start_time
    logging.info("training time of liblinear: %.4f sec" % (train_time))

    start_time = time.time()

    train_accu = clf.score(x_train, y_train)

    #load dataset
    logging.info("loading test data %s..." %(dtest.name))
    x_test, y_test = datasets.load_svmlight_file(dtest.data_path)
    #check dimensions
    if x_test.shape[1] < x_train.shape[1]:
        x_test = x_test.toarray()
        pad = np.zeros((x_test.shape[0],x_train.shape[1] - x_train.shape[1]))
        x_test = csr_matrix(np.concatenate((x_test, pad), axis=1))
    elif x_test.shape[1] > x_train.shape[1]:
        x_test = x_test[:,0:x_train.shape[1]]

    logging.info("test liblinear with C=%f..." %(C))
    test_accu = clf.score(x_test, y_test)

    test_time = time.time() - start_time

    logging.info("test accuracy: %.4f" %(test_accu))
    logging.info("test time: %.4f sec" %(test_time))

    return test_accu, test_time, train_accu, train_time

def train_test_l1(dtrain, dtest, C):
    """Train and Test L1-SVM with Liblinear

    Parameters
    ----------
    dtrain: DatsSet
        training dataset
    dtest: DataSet
        test dataset
    C: float
        l1 penalty

    Return
    ------
    tuple (feat_num, test accuracy, test time, train accuracy, train time)
    """

    if dtrain.dtype != 'svm':
        raise Exception("liblinear only supports svm type data")

    clf = svm.LinearSVC(penalty='l1', C=C, dual=False)

    start_time = time.time()

    #load dataset
    logging.info("loading training data %s..." %(dtrain.name))
    x_train, y_train = datasets.load_svmlight_file(dtrain.data_path)

    logging.info("train liblinear with C=%f..." %(C))
    clf.fit(x_train, y_train)

    train_time = time.time() - start_time
    logging.info("training time of liblinear: %.4f sec" % (train_time))


    train_accu = clf.score(x_train, y_train)

    #load dataset
    logging.info("loading test data %s..." %(dtest.name))
    x_test, y_test = datasets.load_svmlight_file(dtest.data_path)

    #check dimensions
    if x_test.shape[1] < x_train.shape[1]:
        x_test = x_test.toarray()
        pad = np.zeros((x_test.shape[0],x_train.shape[1] - x_test.shape[1]))
        x_test = csr_matrix(np.concatenate((x_test, pad), axis=1))
    elif x_test.shape[1] > x_train.shape[1]:
        x_test = x_test[:,0:x_train.shape[1]]


    logging.info("test liblinear with C=%f..." %(C))
    start_time = time.time()
    test_accu = clf.score(x_test, y_test)

    test_time = time.time() - start_time

    if len(clf.coef_.shape) == 2:
        feat_num = np.count_nonzero(clf.coef_) / float(clf.coef_.shape[0])
    else:
        feat_num = np.count_nonzero(clf.coef_)

    logging.info("test accuracy: %.4f" %(test_accu))
    logging.info("test time: %.4f sec" %(test_time))

    return feat_num, test_accu, test_time, train_accu, train_time

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'Usage dt_name train_file test_file'
        sys.exit()

    from sol.dataset import  DataSet

    dtrain = DataSet(sys.argv[1], sys.argv[2], 'svm')
    dtest = DataSet(sys.argv[1], sys.argv[3], 'svm')

    C_list = [0.001, 0.01, 0.1, 1]

    print 'train test l1-svm'
    for C in C_list:
        print train_test_l1(dtrain, dtest, C=C)

    print 'train test l2-svm'
    for C in C_list:
        print train_test_l2(dtrain, dtest, C=C)

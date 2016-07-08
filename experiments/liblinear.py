#This script is to run experiment automatically to test the performance of the algorithm

import os
import sys
import os.path as osp
import logging
import time
from  sklearn import svm, datasets
from sklearn.grid_search import GridSearchCV
import numpy as np

def run(dtrain, dtest, opts, retrain=False, fold_num = 5):
    if opts['algo'] != 'liblinear':
        raise Exception('wrong method %s called for liblinear script' %(opts['algo']))

    if dtrain.dtype != 'svm':
        raise Exception("liblinear only supports svm type data")

    penalty = 'l2'
    C = 1.0
    if 'params' in opts:
        if 'C' in opts['params']:
            C = float(opts['params']['C'])
        if 'penalty' in opts['params']:
            penalty = opts['params']['penalty']

    if penalty == 'l1':
        #l1-svm
        test_accu_list = []
        sparsity_list = []
        x_train, y_train = datasets.load_svmlight_file(dtrain.data_path)
        x_test, y_test = datasets.load_svmlight_file(dtest.data_path)
        for l1 in opts['lambda']:
            C = 1 / l1
            clf = svm.LinearSVC(penalty=penalty, C=C, dual=False)

            logging.info("train model...")
            clf.fit(x_train, y_train)

            logging.info("test model...")
            test_accu = clf.score(x_test, y_test)
            sparsity = 1 - np.count_nonzero(clf.coef_) / float(clf.coef_.size)

            test_accu_list.append(test_accu)
            sparsity_list.append(sparsity)

            logging.info("test accuracy: %.4f" %(test_accu))
            logging.info("model sparsity: %.4f" %(sparsity))

        return test_accu_list, sparsity_list
    else:
        if 'cv' in opts:
            cv_output_path  = osp.join(dtrain.work_dir, 'cv-liblinear.txt')
            if os.path.exists(cv_output_path) and retrain == False:
                with open(cv_output_path, 'r') as fh:
                    line = fh.readline()
                C = float(line.split('=')[1])
            else:
                #cross validation
                x_train, y_train = datasets.load_svmlight_file(dtrain.rand_path())
                cv_params = opts['cv']
                svc = svm.LinearSVC(penalty=penalty)
                clf = GridSearchCV(estimator=svc, param_grid=dict(C=opts['cv']),
                        n_jobs=4, cv=fold_num, verbose=True)
                clf.fit(x_train, y_train)
                C = clf.best_estimator_.C
                with open(cv_output_path, 'w') as fh:
                    fh.write('Best Result: C=%f' %(C))

            logging.info('cross validation parameters: C=%f' %(C))

        #l2-svm
        clf = svm.LinearSVC(penalty=penalty, C=C)

        logging.info("train model...")
        start_time = time.time()
        x_train, y_train = datasets.load_svmlight_file(dtrain.data_path)
        clf.fit(x_train, y_train)
        end_time = time.time()
        train_accu = clf.score(x_train, y_train)
        train_time = end_time - start_time

        logging.info("training accuracy: %.4f" %(train_accu))
        logging.info("training time: %.4f seconds" %(train_time))

        logging.info("test model...")
        start_time = time.time()
        x_test, y_test = datasets.load_svmlight_file(dtest.data_path)
        test_accu = clf.score(x_test, y_test)
        end_time = time.time()
        test_time = end_time - start_time

        logging.info("test accuracy: %.4f" %(test_accu))
        logging.info("test time: %.4f seconds" %(test_time))

        return train_accu, train_time, test_accu, test_time

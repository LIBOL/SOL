#This script is to run experiment automatically to test the performance of the algorithm

import os
import sys
import os.path as osp
import logging
import time
import numpy as np
from vowpalwabbit.sklearn_vw import VWClassifier
from sklearn.cross_validation import train_test_split

def vw_exe():
    if sys.platform == 'win32':
        return 'vw.exe'
    else:
        return'vw'

def convert_to_vw(input_path, output_path):
    """convert data into vw format
    """
    with open(input_path,'r') as rfh:
        lines = rfh.readlines()

    wfh = open(output_path, 'w') as wfh:
        for line in lines:
            if len(line.strip()) == 0:
                break
            pos = 0
            while line[pos] != ' ' and line[pos] != '\t':
                pos = pos + 1

            wfh.write('%s |%s' %(line[0:pos], line[pos:]))

    return None

def calc_accuracy(test_path, predict_path):
    """calculate prediction accuracy 
    """
    labels = []
    with open(test_path,'r') as rfh:
        lines = rfh.readlines()

    for line in lines:
        if len(line.strip()) == 0:
            break
        pos = 0
        while line[pos] != ' ' and line[pos] != '\t':
            pos = pos + 1
        labels.append(int(line[:pos]))

    with open(predict_path,'r') as rfh:
        lines = rfh.readlines()
    predicts = [int(v) for v in filter(None, [l.strip() for l in lines])]
    assert len(labels) == len(predicts)
    return float(np.sum(np.array(labels) == np.array(predicts))) / len(labels)


def train(dtrain, model_params, cache=False):
    """train vw model"""
    vw_data_path = dtrain.rand_path('svm') + '.vw'
    if osp.exists(vw_data_path) == False:
        convert_to_vw(dtrain_rand_path('svm'), vw_data_path)

    model_path = osp.join(dtrain.work_dir, 'vw.model')

    cmd = vw_exe() + ' -f \"%s\"' %(model_path)
    if cache == True:
        cmd += ' --cache_file \"%s\"' %(vw_data_path + '.cache')
    for k,v in model_params:
        cmd += ' --%s \"%s\"' %(k,str(v))

    print cmd
    start_time = time.time()
    if os.system(cmd) != 0:
        logging.error('call vw failed, vw in path?')
        sys.exit()
    return time.time() - start_time

def test(dtest):
    """test vw model"""
    assert dtest.dtype == 'svm'
    vw_data_path = dtest.data_path + '.vw'
    if osp.exists(vw_data_path) == False:
        convert_to_vw(dtest.data_path, vw_data_path)

    model_path = osp.join(dtest.work_dir, 'vw.model')
    predict_path = osp.join(dtest.work_dir, 'vw.predict')

    cmd = vw_exe() + ' -t -i \"%s\" -p \"%s\"' %(model_path, predict_path)

    print cmd
    start_time = time.time()
    if os.system(cmd) != 0:
        logging.error('call vw failed, vw in path?')
        sys.exit()
    test_time = time.time() - start_time
    return calc_accuracy(dtest.data_path, predict_path), test_time


def run(dtrain, dtest, opts, retrain=False, fold_num = 5):
    if dtrain.dtype != 'svm':
        raise Exception("vw only supports svm type data")

    if 'cv' in opts:
        cv_output_path  = osp.join(dtrain.work_dir, 'cv-vw.txt')
        if os.path.exists(cv_output_path) and retrain == False:
            with open(cv_output_path, 'r') as fh:
                line = fh.readline()
            C = float(line.split('=')[1])
        else:
            #cross validation
            x_train, y_train = datasets.load_svmlight_file(dtrain.rand_path())
            cv_params = opts['cv']
            model = VWClassifier()
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

#This script is to run experiment automatically to test the performance of the algorithm

import os
import sys
import os.path as osp
import logging
import time
import re
import numpy as np
from vowpalwabbit.sklearn_vw import VWClassifier
from sklearn.grid_search import GridSearchCV
from  sklearn import datasets
from operator import itemgetter

#pylsol_dir = osp.join(osp.dirname(osp.dirname(osp.abspath(osp.expanduser(__file__)))), 'python')
#sys.path.insert(0, pylsol_dir)
#from dataset import DataSet

def vw_exe():
    if sys.platform == 'win32':
        return 'vw.exe'
    else:
        return'vw'

def convert_to_vw(input_path, output_path):
    """convert data into vw format
    """
    logging.info('convert %s to %s' %(input_path, output_path))
    with open(input_path,'r') as rfh:
        lines = rfh.readlines()

    with open(output_path, 'w') as wfh:
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
        val = int(line[:pos])
        if val == 0:
            val = -1
        labels.append(val)

    with open(predict_path,'r') as rfh:
        lines = rfh.readlines()
    predicts = [1 if float(v) > 0 else -1 for v in filter(None, [l.strip() for l in lines])]
    assert len(labels) == len(predicts)
    return float(np.sum(np.array(labels) == np.array(predicts))) / len(labels)

def parse_sparsity(dt, model_path):
    with open(model_path, 'r') as fh:
        lines = fh.readlines()
    valid_dim = float(len(lines) - 12)
    if valid_dim < 0:
        valid_dim = 0
    return  1 - valid_dim / dt.dim

def test(dtest, cache=False):
    """test vw model"""
    assert dtest.dtype == 'svm'
    vw_data_path = dtest.data_path + '.vw'
    if osp.exists(vw_data_path) == False:
        convert_to_vw(dtest.data_path, vw_data_path)

    model_path = osp.join(dtest.work_dir, 'vw.model')
    predict_path = osp.join(dtest.work_dir, 'vw.predict')

    cmd = vw_exe() + ' -t -i \"%s\" -p \"%s\"' %(model_path, predict_path)

    if cache == True:
        cache_path = vw_data_path +  '.cache'
        cmd += ' --cache_file \"%s\"' %(cache_path)

    cmd += ' \"%s\"' %(vw_data_path)

    print cmd
    start_time = time.time()
    if os.system(cmd) != 0:
        logging.error('call vw failed, vw in path?')
        sys.exit()
    test_time = time.time() - start_time
    return calc_accuracy(dtest.data_path, predict_path), test_time

def train(dtrain, model_params=[], cache=False, readable_path = None):
    """train vw model"""
    vw_data_path = dtrain.rand_path('svm') + '.vw'

    model_path = osp.join(dtrain.work_dir, 'vw.model')

    cmd = vw_exe() + ' -f \"%s\"' %(model_path)

    if cache == True:
        cache_path = vw_data_path +  '.cache'
        cmd += ' --cache_file \"%s\"' %(cache_path)

    if readable_path != None:
        cmd += ' --readable_model \"%s\"' %(readable_path)

    for k,v in model_params:
        cmd += ' --%s \"%s\"' %(k,str(v))
    cmd += ' \"%s\"' %(vw_data_path)

    print cmd
    start_time = time.time()
    if os.system(cmd) != 0:
        logging.error('call vw failed, vw in path?')
        sys.exit()
    train_time = time.time() - start_time
    train_accu, train_test_time = test(dtrain, cache)
    return train_accu, train_time


def run(dtrain, dtest, opts, retrain=False, fold_num = 5):
    if dtrain.dtype != 'svm':
        raise Exception("vw only supports svm type data")

    best_l = 1.0
    if 'cv' in opts:
        cv_output_path  = osp.join(dtrain.work_dir, 'cv-vw.txt')
        if os.path.exists(cv_output_path) and retrain == False:
            with open(cv_output_path, 'r') as fh:
                line = fh.readline()
            C = float(line.split('=')[1])
        else:
            #cross validation
            logging.info("cross validation")
            logging.info("loading %s" %(dtrain.rand_path()))
            x_train, y_train = datasets.load_svmlight_file(dtrain.rand_path())
            logging.info("loading %s" %(dtest.data_path))
            x_test, y_test = datasets.load_svmlight_file(dtest.data_path)

            clf = GridSearchCV(estimator=VWClassifier(), param_grid=opts['cv'],
                        n_jobs=4, cv=fold_num, verbose=True)
            clf.fit(x_train, y_train)

            best_l = max(clf.grid_scores_, key=itemgetter(1)).parameters['l']
            with open(cv_output_path, 'w') as fh:
                fh.write('Best Result: l=%f' %(best_l))

        logging.info('cross validation parameters: l=%f' %(best_l))

    vw_data_path = dtrain.rand_path('svm') + '.vw'
    convert_to_vw(dtrain.rand_path('svm'), vw_data_path)
    cache_path = vw_data_path +  '.cache'
    if osp.exists(cache_path):
        os.remove(cache_path)

    if 'lambda' in opts:
        #sol
        sparsity_list = []
        test_accu_list = []
        for l1 in opts['lambda']:
            readable_model_path = osp.join(dtrain.work_dir, 'vw.r.model')
            train_accu, train_time = train(dtrain, model_params=[('learning_rate', best_l), ('l1',l1)], cache=True, readable_path = readable_model_path)
            logging.info("training accuracy: %.4f" %(train_accu))
            logging.info("training time: %.4f seconds" %(train_time))

            test_accu, test_time = test(dtest, cache=True)
            logging.info("test accuracy: %.4f" %(test_accu))
            logging.info("test time: %.4f seconds" %(test_time))
            #parse sparsity
            sparsity = parse_sparsity(dtrain, readable_model_path)
            sparsity_list.append(sparsity)
            test_accu_list.append(test_accu)

        return sparsity_list, test_accu_list

    else:
        train_accu, train_time = train(dtrain, model_params=[('learning_rate', best_l)])
        logging.info("training accuracy: %.4f" %(train_accu))
        logging.info("training time: %.4f seconds" %(train_time))

        test_accu, test_time = test(dtest)
        logging.info("test accuracy: %.4f" %(test_accu))
        logging.info("test time: %.4f seconds" %(test_time))
        return train_accu, train_time, test_accu, test_time

#if __name__ == '__main__':
#    data_dir = osp.join(osp.dirname(osp.dirname(osp.abspath(osp.expanduser(__file__)))), 'data')
#    dtrain = DataSet('a1a', osp.join(data_dir, 'a1a'), 'svm')
#    dtest = DataSet('a1a', osp.join(data_dir, 'a1a.t'), 'svm')
#
#    opts = {'algo':'vw', 'cv':{'l':np.logspace(-2,2,5,base=10)}}
#    print run(dtrain, dtest, opts)

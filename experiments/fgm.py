#! /usr/bin/env python
#################################################################################
#     File Name           :     fgm.py
#     Created By          :     yuewu
#     Creation Date       :     [2016-11-03 17:39]
#     Last Modified       :     [2016-11-03 22:56]
#     Description         :      helper script to run FGM
#################################################################################

import os
import sys
import os.path as osp
import logging
import time
import re

def fgm_exe():
    if sys.platform == 'win32':
        return 'FGM.exe'
    else:
        return 'FGM'

def fgm_predict_exe():
    if sys.platform == 'win32':
        return 'Predict.exe'
    else:
        return 'Predict'


def parse_feature_num(path):
    with open(path, 'r') as fh:
        lines = fh.read()

    pattern=re.compile('w (\d+)')
    res = pattern.findall(lines)
    assert len(res) == 1
    return int(res[0])

def parse_accuracy(path):
    with open(path, 'r') as fh:
        lines = fh.read()

    pattern=re.compile('Accuracy = (\d+\.\d+)\s*\%.*')
    res = pattern.findall(lines)
    assert len(res) == 1
    return float(res[0]) / 100.0


def test(dtest, model_path):
    """test FGM model"""
    assert dtest.dtype == 'svm'

    predict_path = osp.join(dtest.work_dir, 'fgm.predict')
    out_path = osp.join(dtest.work_dir, 'fgm.out')

    cmd = fgm_predict_exe() + ' \"%s\" \"%s\" \"%s\" > \"%s\" | type \"%s\"' %(dtest.data_path,
                                                        model_path,
                                                        predict_path,
                                                        out_path, out_path)

    print cmd
    start_time = time.time()
    if os.system(cmd) != 0:
        logging.error('call fgm failed, fgm in path?')
        sys.exit()
    test_time = time.time() - start_time
    return parse_accuracy(out_path), test_time

def train(dtrain, model_path, model_params = []):
    """train FGM model"""
    assert dtrain.dtype == 'svm'
    if dtrain.class_num != 2:
        raise Exception("FGM only supports binary classification")

    cmd = fgm_exe()
    for k,v in model_params:
        cmd += ' -%s %s' %(k,str(v))

    cmd += ' \"%s\" \"%s\"' %(dtrain.data_path, model_path)

    print cmd
    start_time = time.time()
    if os.system(cmd) != 0:
        raise Exception('call fgm failed, fgm in path?')
    train_time = time.time() - start_time
    return parse_feature_num(model_path), train_time


def run(dtrain, dtest, opts):
    if dtrain.dtype != 'svm':
        raise Exception("vw only supports svm type data")

    feat_num_list = []
    test_accu_list = []
    train_time_list = []
    for B in opts['B']:
        model_path = osp.join(dtrain.work_dir, 'fgm.model')
        logging.info("train FGM with B=%d" %(B))
        feat_num, train_time = train(dtrain, model_path,
                                     model_params=[('s', 12), ('c', 10), ('t',0), ('B', B)])

        train_time_list.append(train_time)
        logging.info("training time: %.4f seconds" %(train_time))

        feat_num_list.append(feat_num)
        logging.info("non-zero feature number : %d" %(feat_num))

        logging.info("test FGM with B=%d" %(B))
        test_accu, test_time = test(dtest, model_path)
        logging.info("test accuracy: %.4f" %(test_accu))
        logging.info("test time: %.4f seconds" %(test_time))
        test_accu_list.append(test_accu)

    return feat_num_list, test_accu_list, train_time_list


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'Usage dt_name train_file test_file'
        sys.exit()

    from sol.dataset import  DataSet

    dtrain = DataSet(sys.argv[1], sys.argv[2], 'svm')
    dtest = DataSet(sys.argv[1], sys.argv[3], 'svm')

    opts={'B':[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 26, 30, 32, 35, 38, 40, 42, 45, 48, 50, 55, 60]}
    opts={'B':[2, 3]}
    print run(dtrain, dtest, opts)

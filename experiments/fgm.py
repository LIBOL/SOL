#! /usr/bin/env python
#################################################################################
#     File Name           :     fgm.py
#     Created By          :     yuewu
#     Creation Date       :     [2016-11-03 17:39]
#     Last Modified       :     [2016-11-17 11:22]
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

def parse_accuracy(path):
    with open(path, 'r') as fh:
        lines = fh.read()

    pattern=re.compile('Accuracy = (\d+\.*\d*)\s*\%.*')
    res = pattern.findall(lines)
    assert len(res) == 3
    return float(res[-1]) / 100.0


def test(dtest, model_path):
    """test FGM model"""
    assert dtest.dtype == 'svm'

    predict_path = osp.join(dtest.work_dir, 'fgm.predict')
    out_path = osp.join(dtest.work_dir, 'fgm.out')

    cmd = fgm_predict_exe() + ' \"%s\" \"%s\" \"%s\" > \"%s\" ' %(dtest.data_path,
                                                        model_path,
                                                        predict_path,
                                                        out_path)

    logging.info(cmd)
    start_time = time.time()
    if os.system(cmd) != 0:
        logging.error('call fgm failed, fgm in path?')
        sys.exit()
    test_time = time.time() - start_time
    return parse_accuracy(out_path), test_time

def train_test(dtrain, dtest, B, s = 12, c=10):

    if dtrain.dtype != 'svm':
        raise Exception("FGM only supports svm type data")
    if dtrain.class_num != 2:
        raise Exception("FGM only supports binary classification")

    #training
    logging.info("train FGM with B=%d" %(B))
    model_path = osp.join(dtrain.work_dir, 'fgm.model')
    cmd = '%s -s %d -c %d -t 1 -B %d ' %(fgm_exe(), s, c, B)
    cmd += ' \"%s\" \"%s\"' %(dtrain.data_path, model_path)

    logging.info(cmd)
    start_time = time.time()

    if os.system(cmd) != 0:
        raise Exception('call fgm failed, fgm in path?')

    train_time = time.time() - start_time
    train_accu = test(dtrain, model_path)[0]
    feat_num = B

    logging.info("training accuracy of fgm: %.4f" % (train_accu))
    logging.info("training time of fgm: %.4f sec" % (train_time))

    logging.info("test FGM with B=%d" %(B))

    test_accu, test_time = test(dtest, model_path)
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

    B_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 26, 30, 32, 35, 38, 40, 42, 45, 48, 50, 55, 60]
    B_list = [2, 3]

    for B in B_list:
        print train_test(dtrain, dtest, B=B)

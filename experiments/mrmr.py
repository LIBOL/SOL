#! /usr/bin/env python
#################################################################################
#     File Name           :     mrmr.py
#     Created By          :     yuewu
#     Creation Date       :     [2016-11-06 20:53]
#     Last Modified       :     [2016-11-06 21:10]
#     Description         :      
#################################################################################

import os
import sys
import os.path as osp
import logging
import time
import re

def mrmr_exe():
    if sys.platform == 'win32':
        return 'mrmr_win32.exe'
    else:
        return 'mrmr'

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

    predict_path = osp.join(dtest.work_dir, 'mrmr.predict')
    out_path = osp.join(dtest.work_dir, 'mrmr.out')

    cmd = mrmr_predict_exe() + ' \"%s\" \"%s\" \"%s\" > \"%s\" | type \"%s\"' %(dtest.data_path,
                                                        model_path,
                                                        predict_path,
                                                        out_path, out_path)

    print cmd
    start_time = time.time()
    if os.system(cmd) != 0:
        logging.error('call mrmr failed, mrmr in path?')
        sys.exit()
    test_time = time.time() - start_time
    return parse_accuracy(out_path), test_time

def convert_model_file(model_file, readable_file, train_time):
    logging.info('parse mRMR model file %s to %s\n' %(model_file,
                                                      readable_file))
    c_feat = []
    pattern = re.compile(r'(\S*)\s*')
    is_begin = False
    try:
        file_handler = open(model_file,'r')
        while True:
            line = file_handler.readline()
            line = line.strip()
            if is_begin == True and len(line) == 0:
                break
            if line == '*** mRMR features ***':
                line = file_handler.readline()
                is_begin = True
                continue
            if (is_begin == False):
                continue
            result_list = pattern.findall(line)
            c_feat.append(int(result_list[1]))
    except IOError as e:
        print "I/O error ({0}): {1}".format(e.errno,e.strerror)
        sys.exit()
    else:
        file_handler.close()
        print 'feature number %d' %(len(c_feat))
    #write c_feat into file
    try:
        file_handler = open(parse_file,'wb')
    
        for k in range(0,len(c_feat)):
            file_handler.write('%d\n' %c_feat[k])
    except IOError as e:
        print "I/O error ({0}): {1}".format(e.errno,e.strerror)
        sys.exit()
    else:
        file_handler.close()
    return c_feat

def train(dtrain, model_path, model_params):
    """train FGM model"""
    if dtrain.class_num != 2:
        raise Exception("FGM only supports binary classification")

    cmd = mrmr_exe()
    for k,v in model_params.iteritems():
        cmd += ' -%s %s' %(k,str(v))

    cmd += ' -i \"%s\" > \"%s\"' %(dtrain.rand_path('csv'), model_path)

    print cmd
    start_time = time.time()
    if os.system(cmd) != 0:
        raise Exception('call mrmr failed, mrmr in path?')
    train_time = time.time() - start_time
    return model_params['v'], train_time


def run(dtrain, dtest, opts):
    feat_num_list = []
    test_accu_list = []
    train_time_list = []
    for B in opts['B']:
        model_path = osp.join(dtrain.work_dir, 'mrmr-%d.model' %(B))
        readable_path = osp.join(dtrain.work_dir, 'mrmr-%d.readable.model' %(B))
        if osp.exists(readable_path) == False:
            logging.info("train mrmr with B=%d" %(B))
            params = opts['params']
            params['v'] = dtrain.dim
            params['n'] = B
            feat_num, train_time = train(dtrain, model_path, params)
            convert_model_file(model_path, readable_path, train_time)
        else:
            pass

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

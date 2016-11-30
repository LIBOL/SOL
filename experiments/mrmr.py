#! /usr/bin/env python
#################################################################################
#     File Name           :     mrmr.py
#     Created By          :     yuewu
#     Creation Date       :     [2016-11-06 20:53]
#     Last Modified       :     [2016-11-30 13:54]
#     Description         :
#################################################################################

import os
import sys
import os.path as osp
import logging
import time
import re
from sol import sol_train

def mrmr_exe(use_gpu):
    if use_gpu == False:
        if sys.platform == 'win32':
            return 'fast-mrmr.exe'
        else:
            return 'fast-mrmr'
    else:
        if sys.platform == 'win32':
            return 'gpu-mrmr.exe'
        else:
            return 'gpu-mrmr'

def convert_model_file(model_path, readable_path, train_time):
    logging.info('parse mRMR model file %s to %s\n' %(model_path,
                                                      readable_path))
    c_feat = []
    with open(model_path, 'r') as fh:
        c_feat = [int(v) for v in filter(None, fh.readline().strip().split(','))]

    logging.info('feature number %d' %(len(c_feat)))
    #write c_feat into file
    try:
        file_handler = open(readable_path,'wb')

        file_handler.write('#train time: %f\n' %train_time)
        for val in c_feat:
            file_handler.write('%d\n' %val)
    except IOError as e:
        logging.error("I/O error ({0}): {1}".format(e.errno,e.strerror))
        sys.exit()
    else:
        file_handler.close()
    return c_feat

def train_test(dtrain, dtest, use_gpu, B,
               binary_thresh=None,
               ol_algo = 'ogd',
               ol_model_params = {},
               ol_cv_params = None):
    """train and test mrmr models

    Parameters
    ----------
    dtrain: DatsSet
        training dataset
    dtest: DataSet
        test dataset
    use_gpu: bool
        whether use gpu
    B: int
        number of features to select
    t: float
        binary threshold
    ol_algo: str
        online algorithm for further process
    ol_model_params: dict
        online model parameters
    ol_cv_params: dict
        online cross validaton parameters

    Return
    ------
    tuple (feat_num, test accuracy, test time, train accuracy, train time)

    """

    if use_gpu == True:
        model_path = osp.join(dtrain.work_dir, 'mrmr-gpu-%d.model' %(B))
        readable_path = osp.join(dtrain.work_dir, 'mrmr-gpu-%d.readable.model' %(B))
    else:
        model_path = osp.join(dtrain.work_dir, 'mrmr-%d.model' %(B))
        readable_path = osp.join(dtrain.work_dir, 'mrmr-%d.readable.model' %(B))

    if osp.exists(readable_path) == False:
        logging.info("train mrmr with B=%d" %(B))

        mrmr_path = osp.join(dtrain.work_dir, '%s.mrmr' %(dtrain.name))

        if osp.exists(mrmr_path) == False:
            if binary_thresh == None:
                csv_path = dtrain.convert('csv')
            else:
                csv_path = dtrain.binarize('csv', binary_thresh)

            cmd = 'mrmr-reader \"%s\" \"%s\"' %(csv_path, mrmr_path)
            logging.info("convert %s to %s" %(csv_path, mrmr_path))
            if os.system(cmd) != 0:
                raise Exception('call mrmr-reader failed, mrmr-reader in path?')
            os.remove(csv_path)

        cmd = mrmr_exe(use_gpu)
        cmd += ' -a %d -f \"%s\" > \"%s\"' %( B, mrmr_path, model_path)

        logging.info(cmd)
        start_time = time.time()
        if os.system(cmd) != 0:
            raise Exception('call mrmr failed, mrmr in path?')
        train_time1 = time.time() - start_time

        convert_model_file(model_path, readable_path, train_time1)
    else:
        #load train time
        with open(readable_path, 'r') as fh:
            line = fh.readline()

        train_time1 = float(line.split(':')[1])

    logging.info("mrmr time cost : %.4f sec" %(train_time1))

    feat_num =  B

    ol_model_params["filter"] = readable_path

    test_accu, test_time, train_accu, train_time, m = sol_train.train_test(
        dtrain, dtest,
        model_name = ol_algo,
        model_params = ol_model_params,
        cv_params = ol_cv_params)

    train_time += train_time1

    return feat_num, test_accu, test_time, train_accu, train_time

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'Usage dt_name train_file test_file'
        sys.exit()

    logger = logging.getLogger('')

    numeric_level = getattr(logging, "INFO", None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: " + args.log)
    logger.setLevel(numeric_level)

    from sol.dataset import  DataSet

    dtrain = DataSet(sys.argv[1], sys.argv[2], 'svm')
    dtest = DataSet(sys.argv[1], sys.argv[3], 'svm')

    opts={'B':[20, 30, 200], 'params':{'binary_thresh':0.5, 'ol_model_params':{'verbose':True},
                                  'ol_cv_params':{'eta':[0.01,0.1, 1]}}}

    for B in opts['B']:
        print train_test(dtrain, dtest, B, **opts['params'])

#! /usr/bin/env python

import os.path as osp
import sys
curr_path = osp.dirname(osp.abspath(osp.expanduser(__file__)))
sys.path.append(osp.join(curr_path, 'opts'))

import argparse
import logging
import time
import importlib

import cPickle
import numpy as np

from sol.dataset import DataSet
from sol.cv import CV
from pysol import SOL
import liblinear
import fgm
import fig

DESCRIPTION='Feature Selection  Experiment Scripts'

def cv(dtrain, dtest, algo, model_params, cv_params, fold_num = 5,
       retrain=False):
    cv_output_path  = osp.join(dtrain.work_dir, 'cv-%s.txt' %(algo))
    if osp.exists(cv_output_path) and retrain == False:
        best_params = CV.load_results(cv_output_path)
    else:
        #cross validation
        logging.info("cross validation on dataset %s with parameters %s" %(dtrain.name, str(cv_params)))
        cv = CV(dtrain, fold_num, cv_params, model_params)
        cv.train_val(algo)
        best_params = cv.get_best_param()[0]
        cv.save_results(cv_output_path)

    logging.info('cross validation parameters: %s' %(str(best_params)))
    return best_params

def run_fs(dtrain, dtest, opts, fold_num = 5, retrain=False):
    logging.info('run sol: %s' %(opts['algo']))
    if opts['algo'] == 'liblinear':
        if 'params' not in opts:
            opts['params'] = {}
        opts['params']['penalty'] = 'l1'

        sparsity_list, test_accu_list, train_time_list = liblinear.run(dt_train, dt_test, opts)
        feat_num_list = [((1-val) * dtrain.dim) for val in sparsity_list]
        return np.array(feat_num_list), np.array(test_accu_list), np.array(train_time_list)
    elif opts['algo'] == 'fgm':
        return fgm.run(dtrain, dtest, opts)

    model_params = {}
    if 'params' in opts:
        model_params = opts['params']

    if 'cv' in opts:
        best_params = cv(dtrain, dtest, opts['algo'], model_params, opts['cv'], fold_num, retrain)
        for k,v in best_params:
            model_params[k] = v

    feat_num_list = []
    test_accu_list = []
    train_time_list = []
    print opts['B']
    for B in opts['B']:
        model_params['B'] = B
        m = SOL(algo=opts['algo'], class_num = dtrain.class_num, **model_params)

        logging.info("train %s on %s with B=%d ..." %(opts['algo'],
                                                      dtrain.name, B))

        start_time = time.time()
        train_accu = m.fit(dtrain.rand_path('svm'), 'svm')
        end_time = time.time()

        feat_num_list.append((1-m.sparsity) * dtrain.dim)
        train_time_list.append(end_time - start_time)

        logging.info("training accuracy: %.4f" %(train_accu))
        logging.info("training time: %.4f seconds" %(end_time - start_time))
        logging.info("model sparsity: %.4f seconds" %(m.sparsity))

        logging.info("test %s on %s with B=%d ..." %(opts['algo'], dtrain.name,
                                                    B))
        start_time = time.time()
        test_accu = m.score(dtest.rand_path('svm'), 'svm')
        end_time = time.time()

        logging.info("test accuracy: %.4f" %(test_accu))
        logging.info("test time: %.4f seconds" %(end_time - start_time))

        test_accu_list.append(test_accu)

    return np.array(feat_num_list), np.array(test_accu_list), np.array(train_time_list)

def exp_fs(args, dt_train, dt_test, opts, cache_data_path):
    res = {}
    if args.retrain == True or osp.exists(cache_data_path) == False:
        for algo, opt in opts.iteritems():
            res_len = 0
            if 'B' in opt:
                opt['B'] = [int(v * dt_train.dim) if v <= 1 else v for v in opt['B']]
                res_len = len(opt['B'])
            elif 'lambda' in opt:
                res_len = len(opt['lambda'])

            res[algo] = [np.zeros((args.shuffle, res_len)),
                         np.zeros((args.shuffle, res_len)),
                         np.zeros((args.shuffle, res_len))]

        for rid in xrange(args.shuffle):
            logging.info('random pass %d' %(rid))
            rand_path = dt_train.rand_path(tgt_type='svm', force=True)
            for algo, opt in opts.iteritems():
                algo_res = run_fs(dt_train, dt_test, opt)
                res[algo][0][rid,:] = algo_res[0]
                res[algo][1][rid,:] = algo_res[1]
                res[algo][2][rid,:] = algo_res[2]

    #load & save results
    if osp.exists(cache_data_path) == False:
        with open(cache_data_path, 'wb') as fh:
            cPickle.dump(res, fh)
    elif args.retrain == False:
        with open(cache_data_path,'rb') as fh:
            res = cPickle.load( fh)

    algo_list = []
    ave_feat_num_list = []
    ave_test_accu_list = []
    ave_train_time_list = []
    for algo, vals in res.iteritems():
        algo_list.append(algo)
        ave_feat_num = np.average(vals[0], axis=0)
        ave_test_accu = np.average(vals[1], axis=0)
        ave_train_time = np.average(vals[2], axis=0)
        ave_feat_num_list.append(ave_feat_num)
        ave_test_accu_list.append(ave_test_accu)
        ave_train_time_list.append(ave_train_time)

    print ave_feat_num_list
    print ave_test_accu_list
    #draw sparsity vs test accuracy
    fig.plot(ave_feat_num_list,
             algo_list,
             ave_test_accu_list,
             'Selected Features',
             'Test Accuracy',
             dt_train.name + '-selected-feature-test-accuracy.pdf')

    fig.plot(ave_feat_num_list,
             algo_list,
             ave_train_time_list,
             'Selected Features',
             'Training Time',
             dt_train.name + '-selected-feature-train-time.pdf')


def set_logging(args):
    logger = logging.getLogger('')

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: " + args.log)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter("%(threadName)s: %(asctime)s- %(levelname)s: %(message)s")

    #file handler
    fileHandler = logging.FileHandler(args.log)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    #stream handler (write to stderr)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

def getargs():
    """ Parse program arguments.
    """

    parser = argparse.ArgumentParser(description=DESCRIPTION,
            formatter_class=
            argparse.RawTextHelpFormatter)

    #input output
    parser.add_argument('--retrain', action='store_true',  help='whether retrain model, ignoring existing cache')
    parser.add_argument('--shuffle', type=int,  default=1, help='number of times to shuffle the data')
    parser.add_argument('-f', '--fold_num', type=int, default=5, help='number of folds in cross validation')
    parser.add_argument('-o', '--output', type=str, default=None,
            help='output file to save the results')
    parser.add_argument('-c', '--cache', type=str, default=None,
            help='cache file of learning results')
    parser.add_argument('dtname', type=str, help='dataset name')
    parser.add_argument('train_file', type=str, help='path to training data')
    parser.add_argument('test_file', type=str, help='path to test data')
    parser.add_argument('dtype', type=str, nargs='?', default='svm', help='path to test data')

    #log related settings
    parser.add_argument("--log_level", type=str, default="INFO", help="log level")
    parser.add_argument("--log", type=str, default="log.log", help="log file")

    args= parser.parse_args()
    set_logging(args)

    return args

if __name__ == '__main__':
    args = getargs()
    if args.output == None:
        args.output = args.dtname + '-result.txt'

    try:
        dt_opts = importlib.import_module(args.dtname)
    except ImportError as e:
        print e
        print 'make sure that your have <dtname>.py, refer to rcv1.py for an example'
        sys.exit()

    assert ('fs_opts' in dt_opts.__dict__)

    dt_train = DataSet(args.dtname,args.train_file, args.dtype)
    dt_test = DataSet(args.dtname,args.test_file, args.dtype)

    if args.cache == None:
        args.cache = args.dtname + '-fs-cache.pkl'

    #remove liblinear if not svm format
    if args.dtype != 'svm':
        if 'fs_opts' in dt_opts.__dict__:
            dt_opts.fs_opts.pop('liblinear', None)

    exp_fs(args, dt_train, dt_test, dt_opts.fs_opts, args.cache)

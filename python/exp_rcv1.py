#!/usr/bin/env python

import os
import sys
import argparse
import logging
import time
import numpy as np

from dataset import DataSet
from lsol_core import Model
from cv import CV

DESCRIPTION='Large Scale Online Learning Test Scripts'

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
    parser.add_argument('-o', '--output', type=str, default='result.txt',
            help='output file to save the results')
    parser.add_argument('train_file', type=str, help='path to training data')
    parser.add_argument('test_file', type=str, help='path to test data')
    parser.add_argument('dtype', type=str, nargs='?', default='svm', help='path to test data')

    #log related settings
    parser.add_argument("--log_level", type=str, default="INFO", help="log level")
    parser.add_argument("--log", type=str, default="log.log", help="log file")

    args= parser.parse_args()
    set_logging(args)
    return args

def run_algo(dtrain, dtest, opts, retrain=False, fold_num = 5):
    model_params = []

    model_params = []
    if 'params' in opts:
        model_params = [item.split('=') for item in opts['params']]

    if 'cv' in opts:
        cv_output_path  = os.path.join(dtrain.work_dir, 'cv-%s.txt' %(opts['algo']))
        if os.path.exists(cv_output_path) and retrain == False:
            best_params = CV.load_results(cv_output_path)
        else:
            #cross validation
            cv_params = [item.split('=') for item in opts['cv']]
            cv = CV(dtrain, fold_num, cv_params, model_params)
            cv.train_val(opts['algo'])
            best_params = cv.get_best_param()[0]
            cv.save_results(cv_output_path)

        logging.info('cross validation parameters: %s' %(str(best_params)))
        for k,v in best_params:
            model_params.append([k,v])

    with Model(model_name = opts['algo'], class_num = 2, params = model_params) as m:
        output_path = os.path.join(dtrain.work_dir, opts['algo'] + '.model')

        logging.info("train model...")
        start_time = time.time()
        train_accu = 1 - m.train(dtrain.rand_path(), dtrain.dtype,
                model_path=output_path)
        end_time = time.time()
        train_time = end_time - start_time

        logging.info("training accuracy: %.4f" %(train_accu))
        logging.info("training time: %.4f seconds" %(train_time))

        logging.info("test model...")
        start_time = time.time()
        test_accu = 1 - m.test(dtest.data_path,dtest.dtype)
        end_time = time.time()
        test_time = end_time - start_time

        logging.info("test accuracy: %.4f" %(test_accu))
        logging.info("test time: %.4f seconds" %(test_time))

    return train_accu, train_time, test_accu, test_time

if __name__ == '__main__':
    args = getargs()

    opts = {}
    opts['ada-fobos'] = {'algo':'ada-fobos',
            'cv':['eta=0.0625:2:128', 'delta=0.0625:2:16']}
    opts['ada-rda'] = {'algo':'ada-rda',
            'cv':['eta=0.0625:2:128', 'delta=0.0625:2:16']}
    opts['alma2'] = {'algo':'alma2', 'cv':['alpha=0.1:+0.1:1']}
    opts['arow'] = {'algo':'arow', 'cv':['r=0.0625:2:16']}
    opts['cw'] = {'algo':'cw', 'cv':['a=0.0625:2:1', 'phi=0:+0.25:2']}
    opts['eccw'] = {'algo':'eccw', 'cv':['a=0.0625:2:1', 'phi=0:+0.25:2']}
    opts['ogd'] = {'algo':'ogd', 'cv':['eta=0.0625:2:128']}
    opts['pa'] = {'algo':'pa'}
    opts['pa1'] = {'algo':'pa1', 'cv':['C=0.0625:2:16']}
    opts['pa2'] = {'algo':'pa2', 'cv':['C=0.0625:2:16']}
    opts['perceptron'] = {'algo':'perceptron'}
    opts['rda'] = {'algo':'rda'}
    opts['sop'] = {'algo':'sop', 'cv':['a=0.0625:2:16']}

    dt_train = DataSet('rcv1',args.train_file, args.dtype)
    dt_test = DataSet('rcv1',args.test_file, args.dtype)

    res = {}
    for algo, opt in opts.iteritems():
        res[algo] = np.zeros((args.shuffle, 4))

    for rid in xrange(args.shuffle):
        logging.info('random pass %d' %(rid))
        rand_path = dt_train.rand_path(force=True)
        for algo, opt in opts.iteritems():
            res[algo][rid, :] = run_algo(dt_train, dt_test, opt, args.retrain, args.fold_num)

    out_file = open(args.output, 'w')
    line = '{0: <12}{1:<16}{1:<16}{2:<16}{2:<15}'.format('algorithm', 'train', 'test')
    print line
    out_file.write('%s\n' %(line))
    line = '{0: <12}{1:<16}{2:<16}{1:<16}{2:<15}'.format('', 'accuracy', 'time(s)')
    print line
    out_file.write('%s\n' %(line))
    ddof = 1 if args.shuffle > 1 else 0
    for algo, vals in res.iteritems():
        ave_vals = np.average(vals, axis=0)
        std_vals = np.std(vals, axis=0, ddof=ddof)
        line = '{0: <12}{1:.4f}+/-{2:.4f} {3:.4f}+/-{4:.4f} {5:.4f}+/-{6:.4f} {7:.4f}+/-{8:.4f}'.format(algo, ave_vals[0], std_vals[0], ave_vals[1],
        std_vals[1], ave_vals[2], std_vals[2], ave_vals[3], std_vals[3])
        print line
        out_file.write('%s\n' %(line))

#! /usr/bin/env python

import argparse
import logging
import os.path as osp
import sys

import importlib

import cPickle
import numpy as np

from sol.dataset import DataSet
from sol.cv import CV
from sol import sol_train
import liblinear
import fgm
import fig
import mrmr

sys.path.append(osp.join(osp.dirname(osp.abspath(osp.expanduser(__file__))), 'opts'))
DESCRIPTION = 'Feature Selection  Experiment Scripts'

def run_fs(dtrain, dtest, algo, opts, cv_process_num):
    """
    Run Feature Selection Algorithm

    Parameter
    ---------
    dtrain: DataSet
        training dataset
    dtest: DataSet
        test dataset
    algo: str
        name of the algorithm to use
    opts: dict
        options to train the model
    cv_process_num: int
        number of processes to do cross validaton
    """

    logging.info('run sol: %s', algo)

    model_params = opts['params'] if 'params' in opts else {}
    cv_params = opts['cv'] if 'cv' in opts else None
    feat_num_list = []
    test_accu_list = []
    train_time_list = []

    for val in opts['lambda']:
        if algo == 'liblinear':
            model_params['C'] = val
            feat_num, test_accu, test_time, train_accu, train_time = \
                    liblinear.train_test_l1(dtrain, dtest, **model_params)
        elif algo == 'fgm':
            model_params['B'] = val
            feat_num, test_accu, test_time, train_accu, train_time = \
                    fgm.train_test(dtrain, dtest, **model_params)
        elif algo == 'mrmr':
            model_params['B'] = val
            feat_num, test_accu, test_time, train_accu, train_time = \
                    mrmr.train_test(dtrain, dtest, False, **model_params)
        elif algo == 'gpu-mrmr':
            model_params['B'] = val
            feat_num, test_accu, test_time, train_accu, train_time = \
                    mrmr.train_test(dtrain, dtest, True, **model_params)
        else:
            if val > 0:
                model_params['B'] = val
            test_accu, test_time, train_accu, train_time, m = sol_train.train_test(
                dtrain, dtest,
                model_name=algo,
                model_params=model_params,
                cv_params=cv_params,
                cv_process_num= cv_process_num)

            feat_num = int((1-m.sparsity) * dtrain.dim)

        logging.info("non-zero feature number: %d", feat_num)

        feat_num_list.append(feat_num)
        test_accu_list.append(test_accu)
        train_time_list.append(train_time)

    return np.array(feat_num_list), \
            np.array(test_accu_list), \
            np.array(train_time_list)

def exp_fs(dtrain, dtest,
           opts,
           output_path,
           repeat=1,
           retrains=None,
           cv_process_num=1,
           draw_opts = {}):
    """
    Experiment to run all algorithms

    Parameters
    ----------
    dtrain: DataSet
        traning dataset
    dtest: DataSet
        test dataset
    opts: dict
        options for each algorithm
    output_path: str
        output path to save the results
    repeat: int
        number of repeats to run the algorithms
    retrains: list[str]
        which algorithm should be retrained, even it has been trained before
    cv_process_num: int
        number of processes to do cross validaton
    """

    if osp.exists(output_path) is True:
        with open(output_path, 'rb') as rfh:
            res = cPickle.load(rfh)
    else:
        res = {}

    retrains = [] if retrains is None else retrains
    if len(retrains) == 1 and retrains[0].lower() == 'all':
        retrains = opts.keys()
    retrains = [v.lower() for v in retrains]

    for algo_ori, opt in opts.iteritems():
        algo = algo_ori.lower()
        if algo_ori in res and algo_ori != algo:
            res[algo] = res[algo_ori]
            del res[algo_ori]

        if (algo in res) and (algo not in retrains):
            continue

        res_len = len(opt['lambda'])

        res[algo] = [np.zeros((repeat, res_len)) for i in xrange(3)]

        for rid in xrange(repeat):
            if rid > 0 and (algo == 'liblinear' or algo == 'fgm'):
                for i in xrange(3):
                    res[algo][i][rid, :] = res[algo][i][rid - 1, :]
            else:
                logging.info('random pass %d', rid)
                dtrain.rand_path(tgt_type='svm', force=True)
                algo_res = run_fs(dtrain, dtest, algo.lower(), opt, cv_process_num)
                for i in xrange(3):
                    res[algo][i][rid, :] = algo_res[i]

    #save results
    with open(output_path, 'wb') as wfh:
        cPickle.dump(res, wfh)

    algo_list = []
    ave_feat_nums = []
    ave_test_accuracy = []
    for algo, opt in opts.iteritems():
        if algo.lower() == 'gpu-mrmr':
            continue
        algo_list.append(algo)
        vals = res[algo.lower()]
        ave_feat_nums.append(np.average(vals[0], axis=0))
        ave_test_accuracy.append(np.average(vals[1], axis=0))

    #draw sparsity vs test accuracy
    fig.plot(ave_feat_nums,
             ave_test_accuracy,
             '#Selected Features',
             'Test Accuracy (%)',
             algo_list,
             osp.join(dtrain.work_dir, dtrain.name.replace('_', '-') + '-test-accuracy.pdf'),
             **(draw_opts['accu']))

    algo_list = []
    ave_feat_nums = []
    ave_train_time = []
    for algo, opt in opts.iteritems():
        algo_list.append(algo)
        vals = res[algo.lower()]
        ave_feat_nums.append(np.average(vals[0], axis=0))
        ave_train_time.append(np.average(vals[2], axis=0))


    fig.plot(ave_feat_nums,
             ave_train_time,
             '#Selected Features',
             'Training Time (s)',
             algo_list,
             osp.join(dtrain.work_dir, dtrain.name.replace('_', '-') + '-train-time.pdf'),
             **(draw_opts['time']))


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
    parser.add_argument('--retrain', type=str, nargs='+',  help='retrian the algorithms')
    parser.add_argument('--repeat', type=int,  default=1, help='number of times to shuffle and repeat training the  data')
    parser.add_argument('-f', '--fold_num', type=int, default=5, help='number of folds in cross validation')
    parser.add_argument('-o', '--output', type=str, default=None, help='output file to save the results')
    parser.add_argument('-p', '--process_num', type=int, default=1, help='number of processes to do cross validation')
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

    try:
        dt_opts = importlib.import_module(args.dtname)
    except ImportError as e:
        print e
        print 'make sure that your have <dtname>.py, refer to rcv1.py for an example'
        sys.exit()

    assert 'fs_opts' in dt_opts.__dict__

    passes = 1
    if 'passes' in dt_opts.__dict__:
        passes = dt_opts.passes

    dt_train = DataSet(args.dtname, args.train_file, args.dtype, passes)
    dt_test = DataSet(args.dtname, args.test_file, args.dtype)

    if args.output == None:
        args.output = osp.join(dt_train.work_dir, args.dtname + '-fs-cache.pkl')

    #remove liblinear if not svm format
    if args.dtype != 'svm':
        if 'fs_opts' in dt_opts.__dict__:
            dt_opts.fs_opts.pop('liblinear', None)
            dt_opts.fs_opts.pop('fgm', None)

    if 'draw_opts' not in dt_opts.__dict__:
        draw_opts = {'accu':{}, 'time':{}}
    else:
        draw_opts = dt_opts.draw_opts

    exp_fs(dt_train, dt_test,
           dt_opts.fs_opts,
           args.output,
           repeat=args.repeat,
           retrains=args.retrain,
           cv_process_num=args.process_num,
           draw_opts=draw_opts)

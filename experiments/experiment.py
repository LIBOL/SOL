#!/usr/bin/env python

import os.path as osp
import sys

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
import fig
vw = None

sys.path.append(osp.join(osp.dirname(osp.abspath(osp.expanduser(__file__))), 'opts'))
DESCRIPTION='Large Scale Online Learning Experiment Scripts'

def run_ol(dtrain,
           dtest,
           algo,
           opts,
           fold_num=5,
           cv_process_num=1):
    """
    Run Online Learning Algorithm

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
    fold_num: int
        number of folds to do cross validation
    cv_process_num: int
        number of processes to do cross validaton
    """

    logging.info('run ol: %s', algo)

    model_params = opts['params'] if 'params' in opts else {}
    cv_params = opts['cv'] if 'cv' in opts else None

    if algo == 'liblinear':
        params = model_params.copy()
        params.update(cv_params)
        return liblinear.train_test_l2(dtrain, dtest,
                                       fold_num=fold_num,
                                       **params)
    elif algo == 'vw':
        return vw.train_test(dtrain, dtest,
                             model_params=model_params,
                             cv_params=cv_params,
                             fold_num=fold_num,
                             cv_process_num=cv_process_num)

    #cross validation
    if cv_params != None:
        cv_output_path = osp.join(dtrain.work_dir, 'cv-%s.txt' % (algo))
        if osp.exists(cv_output_path):
            best_params = CV.load_results(cv_output_path)
        else:
            cv_ = CV(dtrain, fold_num, cv_params, model_params, process_num=cv_process_num)
            cv_.train_val(algo)
            best_params = cv_.get_best_param()[0]
            cv_.save_results(cv_output_path)

        logging.info('cross validation results: %s', str(best_params))

        model_params.update(best_params)

    logging.info("learn model with %s algorithm on %s ...", algo, dtrain.name)
    logging.info("parameter settings: %s", model_params)

    model = SOL(algo, dtrain.class_num, **model_params)

    #record update number and learning rate
    train_log = []
    def record_training_process(data_num, iter_num, update_num, err_rate):
        """closure logging function"""
        train_log.append([data_num, iter_num, update_num, err_rate])

    model.inspect_learning(record_training_process)

    #training
    start_time = time.time()
    train_accu = model.fit(dtrain.rand_path(), dtrain.dtype)
    end_time = time.time()
    train_time = end_time - start_time

    logging.info("training accuracy: %.4f", train_accu)
    logging.info("training time: %.4f seconds", train_time)

    #test
    logging.info("test %s on %s...", algo, dtest.name)
    start_time = time.time()
    test_accu = model.score(dtest.data_path, dtest.dtype)
    end_time = time.time()
    test_time = end_time - start_time

    logging.info("test accuracy: %.4f", test_accu)
    logging.info("test time: %.4f seconds", test_time)

    return test_accu, test_time, train_accu, train_time, np.array(train_log)

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


def exp_ol(dtrain, dtest,
           opts,
           output_path,
           repeat=1,
           retrains=None,
           fold_num=5,
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
    fold_num: int
        number of folds to do cross validation
    cv_process_num: int
        number of processes to do cross validaton
    """

    if osp.exists(output_path) is True:
        with open(output_path, 'rb') as rfh:
            save_obj = cPickle.load(rfh)
    else:
        save_obj = {'res':{}, 'train_log':{}}

    res = save_obj['res']
    train_log = save_obj['train_log']

    retrains = [] if retrains is None else retrains
    if len(retrains) == 1 and retrains[0].lower() == 'all':
        retrains = opts.keys()
    retrains = [v.lower() for v in retrains]

    for algo, opt in opts.iteritems():
        algo = algo.lower()
        if algo in res and algo not in retrains:
            continue

        if algo not in retrains:
            retrains.append(algo)

        res[algo] = np.zeros((repeat, 4))

        if algo != 'liblinear' and algo != 'vw':
            train_log[algo] = [None for i in xrange(repeat)]

    for rid in xrange(repeat):
        logging.info('random pass %d', rid)
        dtrain.rand_path(force=True)
        for algo in retrains:
            opt = opts[algo]
            if (algo in res) and (algo not in retrains):
                continue
            algo_res = run_ol(dtrain, dtest,
                              algo,
                              opt,
                              fold_num=fold_num,
                              cv_process_num=cv_process_num)
            res[algo][rid, :] = algo_res[0:4]
            if algo != 'liblinear' and algo != 'vw':
                train_log[algo][rid] = algo_res[4]

    #save results
    save_obj['res'] = res
    save_obj['train_log'] = train_log
    with open(output_path, 'wb') as wfh:
        cPickle.dump(save_obj, wfh)

    #print train and test results
    line = '{0: <12}{1:<16}{1:<16}{2:<16}{2:<15}'.format('algorithm', 'test', 'train')
    print line
    line = '{0: <12}{1:<16}{2:<16}{1:<16}{2:<15}'.format('', 'accuracy', 'time(s)')
    print line
    ddof = 1 if repeat > 1 else 0
    for algo, opt in opts.iteritems():
        vals = res[algo.lower()]
        ave_vals = np.average(vals, axis=0)
        std_vals = np.std(vals, axis=0, ddof=ddof)
        line = '{0: <12}{1:.4f}+/-{2:.4f} \
                {3:.4f}+/-{4:.4f} \
                {5:.4f}+/-{6:.4f} \
                {7:.4f}+/-{8:.4f}'.format(algo,
                                          ave_vals[0], std_vals[0],
                                          ave_vals[1], std_vals[1],
                                          ave_vals[2], std_vals[2],
                                          ave_vals[3], std_vals[3])
        print line

    #draw training log
    data_nums = []
    update_nums = []
    error_rates = []
    algo_list = []

    for algo, opt in opts.iteritems():
        if algo == 'liblinear' or algo == 'vw':
            continue
        log = train_log[algo.lower()]
        algo_list.append(algo)
        data_nums.append(log[0][:,0].astype(np.int))
        ave_update_nums = np.zeros(log[0][:,2].shape)
        ave_error_rates = np.zeros(log[0][:,3].shape)
        for rid in xrange(repeat):
            ave_update_nums = ave_update_nums + log[rid][:,2]
            ave_error_rates = ave_error_rates + log[rid][:,3]
        error_rates.append(ave_error_rates / repeat)
        update_nums.append(ave_update_nums / repeat)

    fig.plot(data_nums,
             error_rates,
             'Number of samples',
             'Cumulative Error Rate',
             algo_list,
             osp.join(dtrain.work_dir, dtrain.name.replace('_', '-') + '-error-rate.pdf'),
             **(draw_opts['train-error']))

    fig.plot(data_nums,
             update_nums,
             'Number of samples',
             'Cumulative Number of Updates',
             algo_list,
             osp.join(dtrain.work_dir, dtrain.name.replace('_', '-') + '-update-num.pdf'),
             **(draw_opts['update-num']))

def getargs():
    """ Parse program arguments.
    """

    parser = argparse.ArgumentParser(description=DESCRIPTION,
            formatter_class=
            argparse.RawTextHelpFormatter)

    #input output
    parser.add_argument('--retrain', type=str, nargs='+', help='retrian the algorithms')
    parser.add_argument('--repeat', type=int, default=1, help='number of times to shuffle and repeat training the  data')
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

    args = parser.parse_args()
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

    passes = 1
    if 'passes' in dt_opts.__dict__:
        passes = dt_opts.passes

    dtrain = DataSet(args.dtname, args.train_file, args.dtype, passes)
    dtest = DataSet(args.dtname, args.test_file, args.dtype)

    if args.output == None:
        args.output = osp.join(dtrain.work_dir, args.dtname + '-ol-cache.pkl')

    if ('ol_opts' in dt_opts.__dict__  and 'vw' in dt_opts.ol_opts):
        vw = importlib.import_module('vw')

    if 'draw_opts' not in dt_opts.__dict__:
        draw_opts = {'train-error':{}, 'update-num':{}}
    else:
        draw_opts = dt_opts.draw_opts

    exp_ol(dtrain, dtest,
           dt_opts.ol_opts,
           args.output,
           repeat=args.repeat,
           retrains=args.retrain,
           cv_process_num=args.process_num,
           fold_num=args.fold_num,
           draw_opts=draw_opts)

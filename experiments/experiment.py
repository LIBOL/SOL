#!/usr/bin/env python

import os.path as osp
import sys
pylibsol_dir = osp.join(osp.dirname(osp.dirname(osp.abspath(osp.expanduser(__file__)))), 'python')
sys.path.insert(0, pylibsol_dir)

import argparse
import logging
import time
import importlib

import cPickle
import numpy as np

from dataset import DataSet
from pylsol import LSOL
from cv import CV
import liblinear
import fig
vw = None

DESCRIPTION='Large Scale Online Learning Experiment Scripts'

def run_ol(dtrain, dtest, opts, retrain=False, fold_num = 5):
    logging.info('run ol: %s' %(opts['algo']))
    if opts['algo'] == 'liblinear':
        return liblinear.run(dt_train, dt_test, opts, retrain, fold_num)
    elif opts['algo'] == 'vw':
        return vw.run(dt_train, dt_test, opts, retrain, fold_num)

    model_params = []
    if 'params' in opts:
        model_params = [item.split('=') for item in opts['params']]

    if 'cv' in opts:
        cv_output_path  = osp.join(dtrain.work_dir, 'cv-%s.txt' %(opts['algo']))
        if osp.exists(cv_output_path) and retrain == False:
            best_params = CV.load_results(cv_output_path)
        else:
            #cross validation
            logging.info("cross validation on dataset %s with parameters %s" %(dtrain.name, str(opts['cv'])))
            cv_params = [item.split('=') for item in opts['cv']]
            cv = CV(dtrain, fold_num, cv_params, model_params)
            cv.train_val(opts['algo'])
            best_params = cv.get_best_param()[0]
            cv.save_results(cv_output_path)

        logging.info('cross validation parameters: %s' %(str(best_params)))
        for k,v in best_params:
            model_params.append([k,v])

    model_params = dict(model_params)
    m = LSOL(algo=opts['algo'], class_num = dtrain.class_num, **model_params)
    train_log = []
    def record_training_process(data_num, iter_num, update_num, err_rate, stat=train_log):
        train_log.append([data_num, iter_num, update_num, err_rate])
    m.inspect_learning(record_training_process)

    output_path = osp.join(dtrain.work_dir, opts['algo'] + '.model')

    logging.info("train %s on %s..." %(opts['algo'], dtrain.name))
    start_time = time.time()
    train_accu = m.fit(dtrain.rand_path(), dtrain.dtype)
    end_time = time.time()
    train_time = end_time - start_time

    logging.info("training accuracy: %.4f" %(train_accu))
    logging.info("training time: %.4f seconds" %(train_time))

    logging.info("test %s on %s..." %(opts['algo'], dtrain.name))
    start_time = time.time()
    test_accu = m.score(dtest.data_path,dtest.dtype)
    end_time = time.time()
    test_time = end_time - start_time

    logging.info("test accuracy: %.4f" %(test_accu))
    logging.info("test time: %.4f seconds" %(test_time))

    return train_accu, train_time, test_accu, test_time, np.array(train_log)

def run_sol(dtrain, dtest, opts):
    logging.info('run sol: %s' %(opts['algo']))
    if opts['algo'] == 'liblinear':
        return liblinear.run(dt_train, dt_test, opts)
    elif opts['algo'] == 'vw':
        return vw.run(dt_train, dt_test, opts)

    model_params = []
    if 'params' in opts:
        model_params = [item.split('=') for item in opts['params']]

    if 'cv' in opts:
        cv_output_path  = osp.join(dtrain.work_dir, 'cv-%s.txt' %(opts['cv']))
        if osp.exists(cv_output_path) :
            best_params = CV.load_results(cv_output_path)
        else:
            raise Exception('%s does not exist!' %(cv_output_path))

        logging.info('cross validation parameters: %s' %(str(best_params)))
        for k,v in best_params:
            model_params.append([k,v])

    model_params = dict(model_params)
    sparsity_list = []
    test_accu_list = []
    for l1 in opts['lambda']:
        model_params['lambda'] = l1
        m = LSOL(algo=opts['algo'], class_num = dtrain.class_num, **model_params)

        logging.info("train %s on %s with l1=%f ..." %(opts['algo'], dtrain.name, l1))

        start_time = time.time()
        train_accu = m.fit(dtrain.rand_path('bin'), 'bin')
        end_time = time.time()

        sparsity_list.append(m.sparsity)

        logging.info("training accuracy: %.4f" %(train_accu))
        logging.info("training time: %.4f seconds" %(end_time - start_time))
        logging.info("model sparsity: %.4f seconds" %(m.sparsity))

        logging.info("test %s on %s with l1=%f ..." %(opts['algo'], dtrain.name, l1))
        start_time = time.time()
        test_accu = m.score(dtest.rand_path('bin'), 'bin')
        end_time = time.time()

        logging.info("test accuracy: %.4f" %(test_accu))
        logging.info("test time: %.4f seconds" %(end_time - start_time))

        test_accu_list.append(test_accu)

    return np.array(sparsity_list), np.array(test_accu_list)

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
    parser.add_argument('--ol_cache', type=str, default=None,
            help='cache file of online learning results')
    parser.add_argument('--sol_cache', type=str, default=None,
            help='cache file of sparses online learning results')
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


def exp_online(args, dt_train, dt_test, opts, cache_data_path):
    if args.retrain == True or osp.exists(cache_data_path) == False:
        res = {}
        res_log = {}
        for algo, opt in opts.iteritems():
            res[algo] = np.zeros((args.shuffle, 4))
            if algo != 'liblinear' and algo != 'vw':
                res_log[algo] = [None for i in xrange(args.shuffle)]

        for rid in xrange(args.shuffle):
            logging.info('random pass %d' %(rid))
            rand_path = dt_train.rand_path(force=True)
            for algo, opt in opts.iteritems():
                algo_res = run_ol(dt_train, dt_test, opt, args.retrain, args.fold_num)
                res[algo][rid, :] = algo_res[0:4]
                if algo != 'liblinear' and algo != 'vw':
                    res_log[algo][rid] = algo_res[4]

    if osp.exists(cache_data_path) == False:
        cache_data = {'res': res, 'res_log': res_log}
        with open(cache_data_path, 'wb') as fh:
            cPickle.dump(cache_data, fh)
    elif args.retrain == False:
        with open(cache_data_path,'rb') as fh:
            cache_data = cPickle.load( fh)
        res = cache_data['res']
        res_log = cache_data['res_log']

    #save accuracy and time cost
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

    #draw training log

    xs = []
    error_rates = []
    update_nums = []
    algo_list = []
    for algo, log in res_log.iteritems():
        algo_list.append(algo)
        xs.append(log[0][:,0].astype(np.int))
        ave_update_nums = np.zeros(log[0][:,2].shape)
        ave_error_rates = np.zeros(log[0][:,3].shape)
        for rid in xrange(args.shuffle):
            ave_update_nums = ave_update_nums + log[rid][:,2]
            ave_error_rates = ave_error_rates + log[rid][:,3]
        error_rates.append(ave_error_rates / args.shuffle)
        update_nums.append(ave_update_nums / args.shuffle)

    fig.plot(xs,algo_list,
             error_rates,
             'Number of samples',
             'Cumulative Error Rate',
             dt_train.name + '-error-rate.pdf')
    fig.plot(xs,algo_list,
             update_nums,
             'Number of samples',
             'Cumulative Number of Updates',
             dt_train.name + '-update-num.pdf')

def exp_sol(args, dt_train, dt_test, opts, cache_data_path):
    res = {}
    if args.retrain == True or osp.exists(cache_data_path) == False:
        for algo, opt in opts.iteritems():
            if 'lambda' in opt and algo != 'liblinear':
                opt['lambda'] = np.hstack(([0],opt['lambda']))
            res[algo] = [np.zeros((args.shuffle, len(opts[algo]['lambda']))),
                    np.zeros((args.shuffle, len(opts[algo]['lambda'])))]

        for rid in xrange(args.shuffle):
            logging.info('random pass %d' %(rid))
            rand_path = dt_train.rand_path(force=True)
            rand_path = dt_train.rand_path(tgt_type='bin', force=True)
            for algo, opt in opts.iteritems():
                algo_res = run_sol(dt_train, dt_test, opt)
                res[algo][0][rid,:] = algo_res[0]
                res[algo][1][rid,:] = algo_res[1]

    if osp.exists(cache_data_path) == False:
        with open(cache_data_path, 'wb') as fh:
            cPickle.dump(res, fh)
    elif args.retrain == False:
        with open(cache_data_path,'rb') as fh:
            res = cPickle.load( fh)

    #save sparsity and test ccuracy
    ddof = 1 if args.shuffle > 1 else 0

    algo_list = []
    ave_sparsity_list = []
    ave_test_accu_list = []
    for algo, vals in res.iteritems():
        algo_list.append(algo)
        ave_sparsity = np.average(vals[0], axis=0)
        ave_test_accu = np.average(vals[1], axis=0)
        ave_sparsity_list.append(ave_sparsity)
        ave_test_accu_list.append(ave_test_accu)

    #draw sparsity vs test accuracy
    fig.plot(ave_sparsity_list,
             algo_list,
             ave_test_accu_list,
             'Sparsity',
             'Test Accuracy',
             dt_train.name + '-sparsity-test-error.pdf')

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

    dt_train = DataSet(args.dtname,args.train_file, args.dtype)
    dt_test = DataSet(args.dtname,args.test_file, args.dtype)

    if args.ol_cache == None:
        args.ol_cache = args.dtname + '-ol-cache.pkl'
    if args.sol_cache == None:
        args.sol_cache = args.dtname + '-sol-cache.pkl'

    #remove liblinear and vw if not svm format
    if args.dtype != 'svm':
        if 'ol_opts' in dt_opts.__dict__:
            dt_opts.ol_opts.pop('liblinear', None)
            dt_opts.ol_opts.pop('vw', None)
        if 'sol_opts' in dt_opts.__dict__:
            dt_opts.sol_opts.pop('liblinear', None)
            dt_opts.sol_opts.pop('vw', None)

    if ('ol_opts' in dt_opts.__dict__  and 'vw' in dt_opts.ol_opts) or ('sol_opts' in dt_opts.__dict__  and 'vw' in dt_opts.sol_opts):
        vw = importlib.import_module('vw')


    if 'ol_opts' in dt_opts.__dict__ and len(dt_opts.ol_opts) > 0:
        exp_online(args, dt_train, dt_test, dt_opts.ol_opts, args.ol_cache)
    if 'sol_opts' in dt_opts.__dict__ and len(dt_opts.sol_opts) > 0:
        exp_sol(args, dt_train, dt_test, dt_opts.sol_opts, args.sol_cache)

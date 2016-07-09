#!/usr/bin/env python

import os
import os.path as osp
import sys
pylsol_dir = osp.join(osp.dirname(osp.dirname(osp.abspath(osp.expanduser(__file__)))), 'python')
sys.path.insert(0, pylsol_dir)

import argparse
import logging
import time
import cPickle
import numpy as np
import matplotlib.pyplot as plt

from dataset import DataSet
from lsol_core import Model
from cv import CV

import liblinear
import vw
import fig

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
    parser.add_argument('--ol_cache', type=str, default='ol_cache.pkl',
            help='cache file of online learning results')
    parser.add_argument('--sol_cache', type=str, default='sol_cache.pkl',
            help='cache file of sparses online learning results')
    parser.add_argument('train_file', type=str, help='path to training data')
    parser.add_argument('test_file', type=str, help='path to test data')
    parser.add_argument('dtype', type=str, nargs='?', default='svm', help='path to test data')

    #log related settings
    parser.add_argument("--log_level", type=str, default="INFO", help="log level")
    parser.add_argument("--log", type=str, default="log.log", help="log file")

    args= parser.parse_args()
    set_logging(args)
    return args

def run_ol(dtrain, dtest, opts, retrain=False, fold_num = 5):
    logging.info('run ol: %s' %(opts['algo']))
    if opts['algo'] == 'liblinear':
        return liblinear.run(dt_train, dt_test, opts, retrain, fold_num)
    elif opts['algo'] == 'vw':
        return vw.run(dt_train, dt_test, opts, retrain, fold_num)

    model_params = []
    if 'params' in opts:
        model_params = [item.split('=') for item in opts['params']]
    model_params.append(('step_show','50000'))

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

        train_log = m.train_log()

        logging.info("test model...")
        start_time = time.time()
        test_accu = 1 - m.test(dtest.data_path,dtest.dtype)
        end_time = time.time()
        test_time = end_time - start_time

        logging.info("test accuracy: %.4f" %(test_accu))
        logging.info("test time: %.4f seconds" %(test_time))

    return train_accu, train_time, test_accu, test_time, train_log

def exp_online(args, dt_train, dt_test, cache_data_path=None):
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
    opts['sop'] = {'algo':'sop', 'cv':['a=0.0625:2:16']}
    opts['rda'] = {'algo':'rda'}
    opts['erda'] = {'algo':'erda-l1'}

    if args.dtype == 'svm':
        opts['liblinear'] = {'algo':'liblinear', 'cv':np.logspace(-5,7,13, base=2)}
        opts['vw'] = {'algo':'vw', 'cv':{'l':np.logspace(-4,7,12,base=2)}}
        #opts['vw'] = {'algo':'vw'}

    if cache_data_path == None or osp.exists(cache_data_path) == False:
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

    if cache_data_path != None:
        if osp.exists(cache_data_path) == False:
            cache_data = {'res': res, 'res_log': res_log}
            with open(cache_data_path, 'wb') as fh:
                cPickle.dump(cache_data, fh)
        else:
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
        ave_error_rates = np.zeros(log[0][:,1].shape)
        ave_update_nums = np.zeros(log[0][:,2].shape)
        for rid in xrange(args.shuffle):
            ave_error_rates = ave_error_rates + log[rid][:,1]
            ave_update_nums = ave_update_nums + log[rid][:,2]
        error_rates.append(ave_error_rates / args.shuffle)
        update_nums.append(ave_update_nums / args.shuffle)

    fig.plot(xs,algo_list, error_rates, 'Number of samples', 'Cumulative Error Rate', 'error_rate.pdf', draw_legend=False)
    fig.plot(xs,algo_list, update_nums, 'Number of samples', 'Cumulative Number of Updates', 'update_num.pdf')

def run_sol(dtrain, dtest, opts):
    logging.info('run sol: %s' %(opts['algo']))
    if opts['algo'] == 'liblinear':
        return liblinear.run(dt_train, dt_test, opts)
    elif opts['algo'] == 'vw':
        return vw.run(dt_train, dt_test, opts)

    model_params = []
    if 'params' in opts:
        model_params = [item.split('=') for item in opts['params']]
    model_params.append(('step_show','50000'))

    if 'cv' in opts:
        cv_output_path  = os.path.join(dtrain.work_dir, 'cv-%s.txt' %(opts['cv']))
        if os.path.exists(cv_output_path) :
            best_params = CV.load_results(cv_output_path)
        else:
            raise Exception('%s does not exist!' %(cv_output_path))

        logging.info('cross validation parameters: %s' %(str(best_params)))
        for k,v in best_params:
            model_params.append([k,v])

    sparsity_list = []
    test_accu_list = []
    for l1 in opts['lambda']:
        with Model(model_name = opts['algo'], class_num = 2, params = model_params + [('lambda', l1)]) as m:
            logging.info("train model...")
            start_time = time.time()
            train_accu = 1 - m.train(dtrain.rand_path('bin'), 'bin')
            end_time = time.time()
            train_time = end_time - start_time

            logging.info("training accuracy: %.4f" %(train_accu))
            logging.info("training time: %.4f seconds" %(train_time))

            logging.info("test model...")
            start_time = time.time()
            test_accu = 1 - m.test(dtest.rand_path('bin'), 'bin')
            end_time = time.time()
            test_time = end_time - start_time

            logging.info("test accuracy: %.4f" %(test_accu))
            logging.info("test time: %.4f seconds" %(test_time))

            sparsity_list.append(m.sparsity())
            test_accu_list.append(test_accu)

    return np.array(sparsity_list), np.array(test_accu_list)

def exp_sol(args, dt_train, dt_test, cache_data_path):
    opts = {}
    opts['stg'] = {'algo':'stg', 'cv':'ogd', 'params':['k=10'], 'lambda': np.logspace(-6,-1,10,base=10) }
    opts['fobos-l1'] = {'algo':'fobos-l1', 'cv':'ogd', 'lambda': np.logspace(-6,-1,10,base=10) }
    opts['rda-l1'] = {'algo':'rda-l1', 'lambda': np.logspace(-6,-1,10,base=10) }
    opts['erda-l1'] = {'algo':'erda-l1', 'params':['rou=0.001'], 'lambda': np.logspace(-6,-1,10,base=10) }
    opts['ada-fobos-l1'] = {'algo':'ada-fobos-l1', 'cv':'ada-fobos', 'lambda': np.logspace(-6,-1,10,base=10) }
    opts['ada-rda-l1'] = {'algo':'ada-rda-l1', 'cv':'ada-rda', 'lambda': np.logspace(-7,-2,10,base=10) }

    if args.dtype == 'svm':
        opts['liblinear'] = {'algo':'liblinear', 'params':{'penalty':'l1'}, 'lambda':np.logspace(-5,7,13, base=2)}
        opts['vw'] = {'algo':'vw','cv':'vw', 'lambda':np.logspace(-6,-2,10, base=10)}

    #opts['stg'] = {'algo':'stg', 'params':['k=10'], 'lambda': np.logspace(-6,-1,10,base=10) }
    res = {}
    if cache_data_path == None or osp.exists(cache_data_path) == False:
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

    if cache_data_path != None:
        if osp.exists(cache_data_path) == False:
            with open(cache_data_path, 'wb') as fh:
                cPickle.dump(res, fh)
        else:
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
    fig.plot(ave_sparsity_list,algo_list, ave_test_accu_list, 'Sparsity', 'Test Error Rate', 'sparsity_test_error.pdf')

if __name__ == '__main__':
    args = getargs()

    dt_train = DataSet('rcv1',args.train_file, args.dtype)
    dt_test = DataSet('rcv1',args.test_file, args.dtype)

    exp_online(args, dt_train, dt_test, args.ol_cache)
    exp_sol(args, dt_train, dt_test, args.sol_cache)

#!/usr/bin/env python

import os
import os.path as osp
import sys
import logging
import argparse
import time
import ast

from sol.dataset import DataSet
from sol.cv import CV
from pysol import SOL

DESCRIPTION = 'Large Scale Online Learning Training Scripts'

def train(dt, model_name,
          model_params={},
          output_path= None,
          fold_num= 5,
          cv_params=None,
          retrain=False,
          cv_process_num=1):
    """
    train a SOL model

    Parameter
    ---------
    dt: DataSet
        the dataset used to train the model
    model_name: str
        name of the algorithm to use
    model_params: dict{param, val}
        model parameters
    output_path: str
        path to save the model
    fold_num: int
        number of folds to do cross validation
    cv_params: dict{param, range}
        cross validation parameters
        cv_process_num: int
        number of processes to do cross validation
    retrain: bool
        whether to re-do the cross validation

    Return
    ------
    tuple(train accuracy, train time, model)
    """

    if cv_params != None:
        cv_output_path = osp.join(dt.work_dir, 'cv-%s.txt' % (model_name))
        if osp.exists(cv_output_path) and retrain == False:
            best_params = CV.load_results(cv_output_path)
        else:
            #cross validation
            param_B = None
            param_lambda = None
            if 'B' in model_params:
                param_B  = model_params['B']
                del model_params['B']
            if 'lambda' in model_params:
                param_lambda  = model_params['lambda']
                del model_params['lambda']

            cv = CV(dt, fold_num, cv_params, model_params, process_num=cv_process_num)
            cv.train_val(model_name)
            best_params = cv.get_best_param()[0]
            cv.save_results(cv_output_path)

            if param_B is not None:
                model_params['B'] = param_B
            if param_lambda is not None:
                model_params['lambda'] = param_lambda

        logging.info('cross validation results: %s' % (str(best_params)))

        model_params.update(best_params)

    logging.info("learn model with %s algorithm on %s ..." % (model_name,
                                                           dt.name))
    logging.info("parameter settings: %s" % (model_params))

    start_time = time.time()
    m = SOL(model_name, dt.class_num, **model_params)
    train_accu = m.fit(dt.data_path, dt.dtype, dt.pass_num)
    train_time = time.time() - start_time

    logging.info("training accuracy of %s: %.4f" % (model_name, train_accu))
    logging.info("training time of %s: %.4f sec" % (model_name, train_time))

    if output_path != None:
        logging.info("save model of %s to %s" % (model_name, output_path))
        m.save(output_path)

    return train_accu, train_time, m

def train_test(dtrain, dtest,
               model_name,
               model_params={},
               output_path=None,
               fold_num=5,
               cv_params=None,
               retrain=False,
               cv_process_num=1):
    train_accu, train_time, m = train(dtrain, model_name, model_params,
                                      output_path, fold_num, cv_params,
                                      retrain, cv_process_num)

    logging.info("test %s on %s..." % (model_name, dtrain.name))

    start_time = time.time()
    test_accu = m.score(dtest.data_path, dtest.dtype)
    test_time = time.time() - start_time

    logging.info("test accuracy: %.4f" %(test_accu))
    logging.info("test time: %.4f sec" %(test_time))

    return test_accu, test_time, train_accu, train_time, m

def finetune(dt, model_path,
             model_params = {},
             output_path = None):
    """Finetune from an existing model

    Parameter
    --------
    dt: DataSet
        the dataset used to train the model
    model_path: str
        path to exisitng model
    model_params: dict{param, val}
        model parameters
    output_path: str
        path to save the model

    Return
    ------
    tuple(train accuracy, train time)
    """

    logging.info("finetnue model from %s ..." % (model_path))
    logging.info("parameter settings: %s" % (model_params))

    init_params = {}
    if 'batch_size' in model_params:
        init_params['batch_size'] = model_params['batch_size']
        del model_params['batch_size']
    if 'buf_size' in model_params:
        init_params['buf_size'] = model_params['buf_size']
        del model_params['buf_size']
    if 'verbose' in model_params:
        init_params['verbose'] = model_params['verbose']
        del model_params['verbose']

    m = SOL(**init_params)
    m.load(model_path)
    algo = m.name
    m.set_params(**model_params)

    start_time = time.time()
    train_accu = m.fit(dt.data_path, dt.dtype, dt.pass_num)
    train_time = time.time() - start_time

    logging.info("training accuracy of %s: %.4f" % (algo, train_accu))
    logging.info("training time of %s: %.4f sec" % (algo, train_time))

    if output_path != None:
        logging.info("save model of %s to %s" % (algo, output_path))
        m.save(output_path)

    return train_accu, train_time, m

def main():
    args = getargs()
    dt_name = osp.basename(args.input)
    dt = DataSet(dt_name, args.input, args.data_type, args.passes)

    model_params = {'verbose': args.verbose,
                    'batch_size': args.batch_size,
                    'buf_size': args.buf_size,
                    'norm': args.norm}

    if args.model == None and args.algo != None:
        if args.params != None:
            for item in args.params:
                parts = item.split('=')
                model_params[parts[0]] = parts[1]
        cv_params = None
        if args.cv != None:
            cv_params = {}
            for item in args.cv:
                parts = item.split('=')
                cv_params[parts[0]] = ast.literal_eval(parts[1])

        train(dt, model_name = args.algo,
              model_params = model_params,
              output_path = args.output,
              fold_num = args.fold_num,
              cv_params = cv_params,
              retrain = args.retrain)
    elif args.model != None and args.algo == None:
        finetune(dt, model_path = args.model, model_params = model_params,
                 output_path = args.output)
    else:
        raise Exception("either model or algo should be specified")

def set_logging(args):
    logger = logging.getLogger('')

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: " + args.log)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter(
        "%(threadName)s: %(asctime)s- %(levelname)s: %(message)s")

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

    parser = argparse.ArgumentParser(
        description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter)

    #input output
    parser.add_argument('input',
                        type=str,
                        help='path to training data')

    parser.add_argument(
        'output',
        type=str,
        nargs='?',
        help='path to save the generated model')

    parser.add_argument(
        '-a',
        '--algo',
        type=str,
        default=None,
        help='name of the algorithm to use')

    parser.add_argument(
        '-t',
        '--data_type',
        type=str,
        default='svm',
        choices=['svm', 'bin', 'csv'],
        help='training data type')

    parser.add_argument(
        '-m',
        '--model',
        type=str,
        help='existing pre-trained model')

    #data related settings
    parser.add_argument(
        '-p',
        '--passes',
        type=int,
        default=1,
        help='number of passes to go through the training data')

    parser.add_argument(
        '--norm',
        type=str,
        default='None',
        choices=['None', 'L1', 'L2'],
        help='normalization method of data')

    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=256,
        help='mini-batch size ')

    parser.add_argument(
        '--buf_size',
        type=int,
        default=2,
        help='number of mini-batches in buffer')

    #model related settings
    parser.add_argument(
        '--cv',
        type=str,
        nargs='+',
        help='parameters waiting for cross validation, in the format "param=[v1,..,vn]"')

    parser.add_argument(
        '-f',
        '--fold_num',
        type=int,
        default=5,
        help='number of folds in cross validation')

    parser.add_argument(
        '--params',
        type=str,
        nargs='+',
        help='parameters for the model, in the format "param=val"')

    parser.add_argument(
        '--retrain',
        action='store_true',
        help='whether retrain model, ignoring existing cross validation paramters')

    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='whether show detailed information')

    #log related settings
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="log level")
    parser.add_argument("--log", type=str, default="log.log", help="log file")

    args = parser.parse_args()
    set_logging(args)
    return args

if __name__ == '__main__':
    main()

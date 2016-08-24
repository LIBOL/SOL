#!/usr/bin/env python

import os
import sys
import argparse
import logging
import time

from lsol.dataset import DataSet
from lsol.cv import CV
from pylsol import LSOL

DESCRIPTION = 'Large Scale Online Learning Training Scripts'


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
    parser.add_argument('input', type=str, help='path to training data')
    parser.add_argument(
        'output', type=str, nargs='?', help='path to save the generated model')
    parser.add_argument(
        '-a',
        '--algo',
        type=str,
        default='ogd',
        help='name of the algorithm to use')
    parser.add_argument(
        '-t',
        '--data_type',
        type=str,
        default='svm',
        choices=['svm', 'bin', 'csv'],
        help='training data type')
    parser.add_argument(
        '-m', '--model', type=str, help='existing pre-trained model')

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
        default='none',
        choices=['none', 'L1', 'L2'],
        help='normalization method of data')
    parser.add_argument(
        '-b', '--batch_size', type=int, default=256, help='mini-batch size ')
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
        help='parameters waiting for cross validation, in the format "param=start_val:step_val:end_val"')
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
        "--log_level", type=str, default="INFO", help="log level")
    parser.add_argument("--log", type=str, default="log.log", help="log file")

    args = parser.parse_args()
    set_logging(args)
    return args


if __name__ == '__main__':
    args = getargs()
    try:
        dt_name = os.path.basename(args.input)
        dt = DataSet(dt_name, args.input, args.data_type)
        model_params = [('verbose', args.verbose)]
        if args.params != None:
            model_params = [item.split('=') for item in args.cv]

        if args.cv != None:
            cv_output_path = os.path.join(dt.work_dir,
                                          'cv-%s.txt' % (args.algo))
            if os.path.exists(cv_output_path) and args.retrain == False:
                best_params = CV.load_results(cv_output_path)
            else:
                #cross validation
                cv_params = [item.split('=') for item in args.cv]
                cv = CV(dt, args.fold_num, cv_params, model_params)
                cv.train_val(args.algo)
                best_params = cv.get_best_param()[0]
                cv.save_results(cv_output_path)
            logging.info('cross validation parameters: %s' %
                         (str(best_params)))
            for k, v in best_params:
                model_params.append((k, v))

        model_params = dict(model_params)

        start_time = time.time()
        m = LSOL(
            args.algo,
            dt.class_num,
            batch_size=args.batch_size,
            buf_size=args.buf_size,
            **model_params)
        logging.info("learn model with %s algorithm..." % (args.algo))
        accu = m.fit(dt.data_path, dt.dtype, args.passes)
        logging.info("training accuracy of %s: %.4f" % (args.algo, accu))
        logging.info("training time of %s: %.4f seconds" %
                     (args.algo, time.time() - start_time))

        if args.output != None:
            logging.info("save model of %s to %s" % (args.algo, args.output))
            m.save(args.output)

    except Exception as err:
        print 'train failed: %s' % (err.message)

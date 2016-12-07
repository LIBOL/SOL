#!/usr/bin/env python

import os
import os.path as osp
import sys
import argparse
import logging
import time
import numpy as np

from sol.dataset import DataSet
from sol.cv import CV
from pysol import SOL

DESCRIPTION = 'Large Scale Online Learning Test Scripts'


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
    parser.add_argument('model',
                        type=str,
                        help='existing pre-trained model')
    parser.add_argument('input',
                        type=str,
                        help='path to test data')
    parser.add_argument(
        'output',
        type=str,
        nargs='?',
        help='path to save the predicted results')
    parser.add_argument(
        '-t',
        '--data_type',
        type=str,
        default='svm',
        choices=['svm', 'bin', 'csv'],
        help='training data type')

    #data related settings
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=256,
        help='mini-batch size')
    parser.add_argument(
        '--buf_size',
        type=int,
        default=2,
        help='number of mini-battches in buffer')

    #log related settings
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="log level")
    parser.add_argument("--log",
                        type=str,
                        default="log.log",
                        help="log file")

    args = parser.parse_args()
    set_logging(args)
    return args


def main():
    args = getargs()
    dt_name = osp.basename(args.input)
    dt = DataSet(dt_name, args.input, args.data_type)

    m = SOL(batch_size=args.batch_size, buf_size=args.buf_size)
    m.load(args.model)

    algo = m.name
    logging.info("testing algorithm %s ..." % (algo))
    start_time = time.time()
    if args.output == None:
        accu = m.score(dt.data_path, dt.dtype)
    else:
        scores, predicts, labels = m.decision_function(dt.data_path, dt.dtype,get_labels=True)
        accu = np.sum(predicts == labels, dtype=np.float64) / predicts.shape[0]
    test_time = time.time() - start_time

    logging.info("test accuracy of %s: %.4f" % (algo, accu))
    logging.info("test time of %s: %.4f sec" % (algo, test_time))

    if args.output != None:
        logging.info("write prediction results to %s" %(args.output))
        with open(args.output, 'w') as fh:
            if m.n_classes == 2:
                for i in xrange(scores.shape[0]):
                    fh.write('%d\t%d\t%f\n' %(int(labels[i]), int(predicts[i]), scores[i]))
            else:
                for i in xrange(scores.shape[0]):
                    fh.write('%d\t%d\t%s\n' %(int(labels[i]), int(predicts[i]), '\t'.join([str(v) for v in scores[i,:]])))

if __name__ == '__main__':
    main()

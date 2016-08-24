#!/usr/bin/env python

import os
import sys
import argparse
import logging
import time

from dataset import DataSet
from cv import CV
from pylsol import LSOL

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
    parser.add_argument('model', type=str, help='existing pre-trained model')
    parser.add_argument('input', type=str, help='path to test data')
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
        '-b', '--batch_size', type=int, default=256, help='mini-batch size')
    parser.add_argument(
        '--buf_size',
        type=int,
        default=2,
        help='number of mini-battches in buffer')

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

        start_time = time.time()
        m = LSOL(batch_size=args.batch_size, buf_size=args.buf_size)
        m.load(args.model)

        algo = m.name
        logging.info("testing algorithm %s ..." % (algo))
        accu = m.score(dt.data_path, dt.dtype)
        logging.info("test accuracy of %s: %.4f" % (algo, accu))
        logging.info("test time of %s: %.4f seconds" %
                     (algo, time.time() - start_time))
    except Exception as err:
        print 'test failed %s' % (err.message)

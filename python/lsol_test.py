#!/usr/bin/env python

import os
import sys
import argparse
import logging

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
    parser.add_argument('-i', '--input_path', type=str, required=True, help='path to training data')
    parser.add_argument('-t', '--data_type', type=str, default='svm', choices=['svm', 'bin', 'csv'], help='training data type')
    parser.add_argument('-m', '--model', type=str, help='existing pre-trained model')
    parser.add_argument('-o', '--output', type=str, help='path to save the predicted results')

    #data related settings
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='mini-batch size')
    parser.add_argument('--buf_size', type=int, default=2, help='number of mini-battches in buffer')

    #log related settings
    parser.add_argument("--log_level", type=str, default="INFO", help="log level")
    parser.add_argument("--log", type=str, default="log.log", help="log file")

    args= parser.parse_args()
    set_logging(args)
    return args

if __name__ == '__main__':
    args = getargs()

    dname = os.path.splitext(os.path.basename(args.input_path))[0]
    try:
        dt = DataSet(dname,args.input_path, args.data_type)

        with Model(model_path = args.model, batch_size = args.batch_size, buf_size = args.buf_size) as m:
            if args.output != None and not os.path.isabs(args.output):
                args.output = os.path.join(dt.work_dir, args.output)
            logging.info("predicting...")
            accu = m.test(dt.data_path,dt.dtype, args.output)
            logging.info("test accuracy: %f" %(accu))
    except Exception as err:
        print 'test failed %s' %(err.message)

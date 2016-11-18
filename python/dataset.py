#!/usr/bin/env python
"""dataset list"""

import sys
import os
import re
import random
import logging
import pysol

class DataSet(object):
    def __init__(self, name, data_path = '', dtype = 'svm', pass_num = 1):
        self.name = name
        self.dtype = dtype
        self.slice_type = dtype

        if data_path == '':
            data_path = '{0}{1}{0}_train'.format(name, os.sep)
        self.data_path = data_path
        if os.path.exists(self.data_path) == False:
            raise Exception('file %s not found' %(self.data_path))
        self.data_name = os.path.splitext(os.path.basename(self.data_path))[0]

        self.work_dir = os.path.join('./cache/', self.name)
        if os.path.exists(self.work_dir) == False:
            os.makedirs(self.work_dir)

        self.pass_num = pass_num

        #prepare the dataset
        self.__analyze_dataset()

    def __analyze_dataset(self):
        """analyze the dataset to obtain dim and class number
        """
        info_file = os.path.join(self.work_dir,  self.data_name + '_info.txt')

        #if not analyzed before, analyze
        if os.path.exists(info_file) == False :
            logging.info('analyze data %s' %(self.data_path))
            if pysol.analyze_data(self.data_path, self.dtype, info_file) != 0:
                sys.exit()

        #parse data num
        pattern = re.compile(r'data number\s*:\s*(\d+)')
        result_list = pattern.findall(open(info_file,'r').read())
        if len(result_list) != 1:
            logging.error('parse data number failed, result_list is %s' %(str(result_list)))
            sys.exit()

        self.data_num = (int)(result_list[0])

        #parse dimension
        pattern = re.compile(r'dimension\s*:\s*(\d+)')
        result_list = pattern.findall(open(info_file,'r').read())
        if len(result_list) != 1:
            print result_list
            print 'parse dimension failed'
            sys.exit()

        self.dim = (int)(result_list[0])

        #parse class number
        pattern = re.compile(r'class num\s*:\s*(\d+)')
        result_list = pattern.findall(open(info_file,'r').read())
        if len(result_list) != 1:
            logging.error('parse class num failed, result_list is %s' %(str(result_list)))
            sys.exit()

        self.class_num = (int)(result_list[0])


    def split_path(self, split_id):
        path =  os.path.join(self.work_dir, self.data_name + '.split.%d.%s' %(split_id, self.slice_type))
        if os.path.exists(path) == False:
            raise Exception("slice path %s not found, called split_file already?" %(path))
        return path

    def cache_path(self):
        if self.dtype == 'bin':
            return self.data_path
        else:
            cache_path = os.path.join(self.work_dir, self.data_name + '.bin')
            if os.path.exists(cache_path):
                return cache_path
            logging.info('convert data %s to ' %(self.data_path, cache_path))
            if pysol.convert_data(self.data_path, self.dtype, cache_path, 'bin') != 0:
                sys.exit()
            return cache_path

    def convert(self, dtype):
        if self.dtype == dtype:
            return self.data_path
        else:
            dst_path = os.path.join(self.work_dir, self.data_name + '.' + dtype)
            if os.path.exists(dst_path):
                return dst_path
            logging.info('convert data %s to %s' %(self.data_path, dst_path))
            if pysol.convert_data(self.data_path, self.dtype, dst_path, dtype) != 0:
                sys.exit()
            return dst_path

    def rand_path(self, tgt_type = None, force=False):
        tgt_type = self.dtype if tgt_type == None else tgt_type
        output_path = os.path.join(self.work_dir, self.data_name + '.shuffle.' + tgt_type)
        if os.path.exists(output_path) and force == False:
            return output_path
        logging.info('convert data %s to %s' %(self.data_path, output_path))
        if pysol.shuffle_data(self.data_path, self.dtype, output_path, tgt_type) != 0:
            sys.exit()
        return output_path

    def split_file(self, split_num, tgt_type = None):
        """Split file into slices
        Parameters:
        tgt_type: string
            splited file type
        split_num: int
            number of splits
        """
        self.slice_type = self.dtype if tgt_type == None else tgt_type
        output_prefix = os.path.join(self.work_dir, self.data_name + ".split.")
        is_exist = True
        for i in xrange(split_num):
            if not os.path.exists('%s%d.%s' %(output_prefix, i, tgt_type)):
                is_exist = False

        extra_file = '%s%d.%s' %(output_prefix, split_num, tgt_type)
        if os.path.exists(extra_file):
            is_exist = False
            #remove this file
            os.remove(extra_file)

        if is_exist == True:
            return None

        logging.info('split %s to %d folds' %(self.data_path, split_num))
        if pysol.split_data(self.data_path, self.dtype, split_num, output_prefix, self.slice_type, True) != 0:
            sys.exit()

if __name__ == '__main__':
    a1a = DataSet('a1a', data_path = 'a1a')
    print a1a.data_path
    print a1a.cache_path()
    print a1a.rand_path("bin")
    a1a.split_file(4, "bin")
    print a1a.split_path(2)

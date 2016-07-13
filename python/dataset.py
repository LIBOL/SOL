#!/usr/bin/env python
"""dataset list"""

import sys
import os
import re
import random
import ConfigParser
import logging

class DataSet(object):
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    config = ConfigParser.ConfigParser()
    config_path = os.path.join(curr_path, 'defaults.cfg') 

    env_paths = os.environ['PATH']
    env_sep = ';' if sys.platform == 'win32' else ':'
    if os.path.exists(config_path):
        config.read(config_path)
        bin_dir = config.get('env', 'BIN_DIR', '.')
        cache_dir = config.get('data', 'CACHE_DIR', 'cache')
    else:
        bin_dir = os.path.join(curr_path, '../dist/bin')
        cache_dir = os.path.join(curr_path, '../cache')

    if bin_dir not in env_paths.split('env_sep'):
        os.environ['PATH'] = bin_dir + env_sep + env_paths;

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

        self.work_dir = os.path.join(self.cache_dir, self.name)
        if os.path.exists(self.work_dir) == False:
            os.makedirs(self.work_dir)

        self.pass_num = pass_num

        #prepare the dataset
        self.__analyze_dataset()

    def __get_cmd_path(self, cmd_name):
        if sys.platform == 'win32':
            cmd_name += ".exe"
        return cmd_name

    def __analyze_dataset(self):
        """analyze the dataset to obtain dim and class number
        """
        info_file = os.path.join(self.work_dir,  self.data_name + '_info.txt')

        #if not analyzed before, analyze
        if os.path.exists(info_file) == False :
            exe_path = self.__get_cmd_path('analyze')
            logging.info('analyze dataset of %s' %self.name)
            cmd = '{0} -i \"{1}\" -s {2} -o {3} '.format(exe_path, self.data_path, self.dtype, info_file)
            print cmd
            if os.system(cmd) != 0:
                raise Exception('analysis of %s failed' %(self.data_path))

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
            exe_path = self.__get_cmd_path('converter')
            logging.info('cache file  %s to %s' %(self.data_path, cache_path))
            cmd = '{0} -i \"{1}\" -s {2} -o \"{3}\" -d bin'.format(exe_path, self.data_path, self.dtype, cache_path)
            print cmd
            if os.system(cmd) != 0:
                raise Exception('convert data %s to %s format failed' %(self.data_path, self.dtype))
            return cache_path

    def rand_path(self, tgt_type = None, force=False):
        tgt_type = self.dtype if tgt_type == None else tgt_type
        output_path = os.path.join(self.work_dir, self.data_name + '.shuffle.' + tgt_type)
        if os.path.exists(output_path) and force == False:
            return output_path
        exe_path = self.__get_cmd_path('shuffle')
        logging.info('shuffle file  %s to %s' %(self.data_path, output_path))
        cmd = '{0} -i \"{1}\" -s {2} -o \"{3}\" -d {4}'.format(exe_path, self.data_path, self.dtype, output_path, tgt_type)
        print cmd
        if os.system(cmd) != 0:
            raise Exception('shuffle data %s failed' %(self.data_path))
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

        exe_path = self.__get_cmd_path('split')
        logging.info('split file  %s to %d slices' %(self.data_path, split_num))
        cmd = '{0} -i \"{1}\" -s {2} -n {3} -o \"{4}\" -d {5} -r'.format(exe_path, self.data_path, self.dtype, split_num, output_prefix, self.slice_type)
        if os.system(cmd) != 0:
            raise Exception('split data %s failed' %(self.data_path))

if __name__ == '__main__':
    a1a = DataSet('a1a', data_path = 'a1a')
    print a1a.data_path
    print a1a.cache_path()
    print a1a.rand_path("bin")
    a1a.split_file(4, "bin")
    print a1a.split_path(2)

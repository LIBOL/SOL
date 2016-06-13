#!/usr/bin/env python
"""dataset list"""

import sys
import os
import re
import random
import ConfigParser

class DataSet(object):
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    config = ConfigParser.ConfigParser()
    config.read(os.path.join(curr_path, 'defaults.cfg'))
    bin_dir = config.get('env', 'BIN_DIR', '.')
    env_paths = os.environ['path']
    env_sep = ';' if sys.platform == 'win32' else ':'
    if bin_dir not in env_paths.split('env_sep'):
        os.environ['path'] = bin_dir + env_sep + env_paths;
    data_dir = config.get('data', 'DATA_DIR', '.')
    cache_dir = config.get('data', 'CACHE_DIR', 'cache')

    def __init__(self, name, dtype = 'svm', train_file = '', test_file = ''):
        self.name = name
        self.dtype = dtype

        if train_file == '':
            train_file = '{0}{1}{0}_train'.format(name, os.sep)
        self.train_file = self.get_data_path(train_file)
        if os.path.exists(self.train_file) == False:
            raise Exception('file %s not found, DATA_ROOT not set?' %(self.train_file))

        if test_file == '':
            test_file = '{0}{1}{0}_test'.format(name, os.sep)
        self.test_file = self.get_data_path(test_file)
        if os.path.exists(self.test_file) == False:
            raise Exception('file %s not found, DATA_ROOT not set?' %(self.test_file))

        self.work_dir = os.path.join(self.cache_dir, self.name)
        if os.path.exists(self.work_dir) == False:
            os.makedirs(self.work_dir)

        #prepare the dataset
        self.__analyze_dataset()

    def get_data_path(self, path):
        if os.path.isabs(path):
            return path
        else:
            return os.path.join(self.data_dir, path)

    def __get_cmd_path(self, cmd_name):
        if sys.platform == 'win32':
            cmd_name += ".exe"
        return cmd_name

    def __analyze_dataset(self):
        """analyze the dataset to obtain dim and class number
        """
        info_file = os.path.join(self.work_dir, self.name + '_info.txt')
    
        #if not analyzed before, analyze
        if os.path.exists(info_file) == False :
            exe_path = self.__get_cmd_path('analyze')
            print 'calculate dimension of %s' %self.train_file
            cmd = '{0} -i \"{1}\" -s {2} -o {3} '.format(exe_path, self.train_file, self.dtype, info_file)
            print cmd
            if os.system(cmd) != 0:
                raise Exception('analysis of %s failed' %(self.train_file))

        #parse data num
        pattern = re.compile(r'data number\s*:\s*(\d+)')
        result_list = pattern.findall(open(info_file,'r').read())
        if len(result_list) != 1:
            print result_list
            print 'parse data number failed'
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
            print result_list
            print 'parse class num failed'
            sys.exit()
    
        self.class_num = (int)(result_list[0])

    def train_path(self):
        return self.train_file

    def train_cache_path(self):
        return self.cache_file(self.train_file, self.dtype)

    def train_slice_path(self, slice_id):
        slice_path =  os.path.join(self.work_dir, self.name + '.cv.slice.%d.bin' %(slice_id))
        if os.path.exists(slice_path) == False:
            raise Exception("slice path %s not found, called split_file already?" %(slice_path))

    def train_rand_path(self):
        return self.shuffle_file(self.train_file, self.dtype)

    def test_path(self):
        return self.test_file

    def test_cache_path(self):
        return self.cache_file(self.test_file, self.dtype)

    def cache_file(self, path, dtype):
        if dtype == 'bin':
            return path
        else:
            cache_path = os.path.join(self.work_dir, os.path.splitext(os.path.basename(path))[0] + '.bin')
            if os.path.exists(cache_path):
                return cache_path
            exe_path = self.__get_cmd_path('converter')
            print 'cache file  %s to %s' %(path, cache_path)
            cmd = '{0} -i \"{1}\" -s {2} -o \"{3}\" -d bin'.format(exe_path, path, dtype, cache_path)
            print cmd
            if os.system(cmd) != 0:
                raise Exception('convert data %s to %s format failed' %(path, dtype))
            return cache_path

    def shuffle_file(self, path, dtype):
        output_path = os.path.join(self.work_dir, os.path.splitext(os.path.basename(path))[0] + '.shuffle.' + dtype)
        if os.path.exists(output_path):
            return output_path
        exe_path = self.__get_cmd_path('shuffle')
        print 'shuffle file  %s to %s' %(path, output_path)
        cmd = '{0} -i \"{1}\" -s {2} -o \"{3}\"'.format(exe_path, path, dtype, output_path)
        print cmd
        if os.system(cmd) != 0:
            raise Exception('shuffle data %s failed' %(path))
        return output_path

    def split_file(self, path, stype, dtype, split_num):
        """Split file into slices
        Parameters:
        path: string
            source file path
        stype: string
            source file type
        dtype: string
            splited file type
        split_num: int
            number of splits 
        """
        output_prefix = os.path.join(self.work_dir, os.path.splitext(os.path.basename(path))[0] + ".split.")
        if os.path.exists(output_prefix +  '0.' + dtype):
            return None
        exe_path = self.__get_cmd_path('split')
        print 'split file  %s to %d slices' %(path, split_num)
        cmd = '{0} -i \"{1}\" -s {2} -n {3} -o \"{4}\" -d {5} -r'.format(exe_path, path, stype, split_num, output_prefix, dtype)
        if os.system(cmd) != 0:
            raise Exception('split data %s failed' %(path))

if __name__ == '__main__':
    a1a = DataSet('a1a', train_file = 'a1a', test_file = 'a1a.t')
    print a1a.train_path()
    print a1a.train_cache_path()
    print a1a.train_rand_path()
    a1a.split_file(a1a.train_path(), a1a.dtype, "bin", 4)
    print a1a.train_slice_path(4)

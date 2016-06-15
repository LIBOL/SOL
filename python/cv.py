#!/usr/bin/env python
"""Cross validation"""

import sys
import os
import logging
import re
import numpy as np

from dataset import DataSet
import search_space
from lsol_core import Model

class CV(object):
    """cross validation class
    """
    def __init__(self,  dataset, fold_num, cv_params, extra_param = []):
        """Create a new cross validation instance
        Parameters:
        dataset: dataset
            data to be used for cross validation
        fold_num: int
            number of folds to conduct cross validation
        cv_params: list[(name, range)]
            parameter string, with format like '[('a', '1:2:8'),(b, '2:2:16')]'
        extra_param: list, [(string, string)]
            extra string for training, with format [('a','1')]
        """
        self.dataset = dataset
        self.fold_num = int(fold_num)
        if self.fold_num < 2 :
            raise ValueError('fold number must bigger than 2!')

        self.cv_params = cv_params
        self.search_space = search_space.SearchSpace(cv_params)
        #the last column is for average
        self.train_scores = np.zeros((self.search_space.size, self.fold_num + 1))
        self.val_scores = np.zeros((self.search_space.size, self.fold_num + 1))
        self.extra_param =  extra_param

#    #cross validation
    def train_val(self, model_name):
        """train and validate on the dataset
        """
        #split the dataset
        self.dataset.split_file(self.fold_num, "bin")

        #cross validation
        for val_fold_id in range(0,self.fold_num):
            print '---------------------------'
            print 'Cross Validation on Model %s with Data %s: Fold %d/%d' %(model_name, self.dataset.name, val_fold_id, self.fold_num)
            print '---------------------------'
            train_accu_list, val_accu_list = self.__train_val_one_fold(model_name, val_fold_id)
            self.train_scores[:, val_fold_id] = train_accu_list
            self.val_scores[:, val_fold_id] = val_accu_list

        self.train_scores[:, self.fold_num] = np.average(self.train_scores, axis = 1)
        self.val_scores[:, self.fold_num] = np.average(self.val_scores, axis = 1)

    def get_best_param(self):
        """Get the best parameters
        """
        max_param_id = np.argmax(self.val_scores[:, self.fold_num])
        return self.search_space.get_param(max_param_id), max_param_id

    def save_results(self, output_path):
        max_param, max_parma_id = self.get_best_param()
        with open(output_path,'w') as wfh:
            wfh.write('Cross Validation Parameters: %s\n' %(str(self.cv_params)))
            wfh.write('Extra Model Parameters: %s\n' %(str(self.extra_param)))
            wfh.write('Best Result: %s:\t%.4f\t%.4f\n' %(str(max_param),
                self.train_scores[max_parma_id, self.fold_num],
                self.val_scores[max_parma_id, self.fold_num]))

            wfh.write("Validation Accuracies\n")
            for y in xrange(self.val_scores.shape[0]):
                wfh.write('%s:' %(str(self.search_space.get_param(y))))
                for x in xrange(self.val_scores.shape[1]):
                    wfh.write('\t%.4f' % (self.val_scores[y,x]))
                wfh.write('\n')

            wfh.write("Train Accuracies\n")
            for y in xrange(self.train_scores.shape[0]):
                wfh.write('%s:' %(str(self.search_space.get_param(y))))
                for x in xrange(self.train_scores.shape[1]):
                    wfh.write('\t%.4f' % (self.train_scores[y,x]))
                wfh.write('\n')

        logging.info('cross validation result written to %s' %output_path)

    @staticmethod
    def load_results(result_path):
        """load the result file of cross validation to get the best parameters"""
        logging.info('loading cross validation  results from %s' %result_path)
        with open(result_path, 'r') as fh:
            pattern = re.compile(r'Best Result: \[.*\]:.*')
            result_list = pattern.findall(fh.read())
            if len(result_list) != 1:
                raise Exception('parsing cross validation file %s failed' %(result_path))
            parma_str = result_list[0]
            pattern = re.compile(r'\(\'\s*(.*)\s*\'\s*,\s+(.*)\)')
            result_list = pattern.findall(parma_str)
            return [result_list[i] for i in xrange(len(result_list))]

    def __train_val_one_fold(self, model_name, val_fold_id):
        """ cross validation on one fold of data
        Parameters:
        model_name: string
            name of the model to be tuned
        val_fold_id: int
            fold id that is used as val data
        Return:
            list of (train accuracy, validation accuracy)
        """
        train_accu_list = []
        val_accu_list = []
        #parameters
        params = []
        for k in range(0, self.search_space.size):
            params = self.search_space.get_param(k)
            for param in self.extra_param:
                params.append(param)

            with Model(model_name, self.dataset.class_num, params = params) as model:
            	train_paths = [self.dataset.split_path(i) for i in xrange(self.fold_num) if i != val_fold_id]
            	train_accu = model.train(train_paths, self.dataset.slice_type, self.dataset.pass_num)
            	val_accu = model.test(self.dataset.split_path(val_fold_id), self.dataset.slice_type)

            	print 'Results of Cross Validation on Model %s with Data %s: Fold %d/%d' %(model_name, self.dataset.name, val_fold_id, self.fold_num)
            	print '\tParameter Setting: %s' %(str(params))
            	print '\tTraining Accuracy: %f' %(train_accu)
            	print '\tValidation Accuracy: %f' %(val_accu)
            	train_accu_list.append(train_accu)
            	val_accu_list.append(val_accu)
        return train_accu_list, val_accu_list

if __name__ == '__main__':
    a1a = DataSet('a1a', data_path = 'a1a')
    cv = CV(a1a,5,[('eta','0.5:10:12')])
    cv.train_val('sgd');
    best_param = cv.get_best_param()[0]
    print best_param
    cv_output_path = os.path.join(cv.dataset.work_dir, 'cv-sgd.txt')
    cv.save_results(cv_output_path)
    best_param = cv.load_results(cv_output_path)
    print best_param

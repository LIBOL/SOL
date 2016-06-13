#!/usr/bin/env python
"""Cross validation"""

import sys
import os
import numpy as np

import dataset
import search_space 
from lsol_core import Model

class CV(object):
    """cross validation class
    """
    __slots__ = ('dataset','fold_num', 'search_space','train_scores', 'val_scores', 'extra_param')

    def __init__(self,  dataset, fold_num, param_str, extra_param = None):
        """Create a new cross validation instance
        Parameters:
        dataset: dataset
            data to be used for cross validation
        fold_num: int
            number of folds to conduct cross validation
        param_str: string
            parameter string, with format like 'a[1:2:8];b[2:2:16]'
        extra_param: list, [(string, string)]
            extra string for training, with format [(a,1)]
        """
        self.dataset = dataset
        self.fold_num = int(fold_num)
        if self.fold_num < 2 :
            raise ValueError('fold number must bigger than 2!')

        self.search_space = search_space.SearchSpace(param_str)
        #the last column is for average
        self.train_scores = np.zeros((self.search_space.size, self.fold_num + 1))
        self.val_scores = np.zeros((self.search_space.size, self.fold_num + 1))
        self.extra_param =  extra_param 

#    #cross validation
    def train_val(self, model_name):
        """train and validate on the dataset
        """
        #split the dataset
        self.dataset.split(self.fold_num)

        #cross validation
        for val_fold_id in range(0,self.fold_num):
            print '---------------------------'
            print 'Cross Validation on Model %s with Data %s: Fold %d/%d' %(model_name, self.dataset.name, val_fold_id, self.fold_num)
            print '---------------------------'
            train_accu_list, val_accu_list = self.__train_val_one_fold(test_fold_id)
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
            wfh.write('Best Result: {0}:\t{1}\t{2}\n'.format(str(min_param),
                self.train_scores[max_parma_id, self.fold_num],
                self.val_scores[max_parma_id, self.fold_num]))

            wfh.write("Validation Accuracies\n")
            for y in xrange(self.val_scores.shape[0]):
                wfh.write('%s:' %(str(self.search_space.get_param(y))))
                for x in xrange(self.val_scores.shape[1]):
                    wfh.write('\t%.2f' % (self.val_scores[y,x]))
                wfh.write('\n')

            wfh.write("Train Accuracies\n")
            for y in xrange(self.train_scores.shape[0]):
                wfh.write('%s:' %(str(self.search_space.get_param(y))))
                for x in xrange(self.train_scores.shape[1]):
                    wfh.write('\t%.2f' % (self.train_scores[y,x]))
                wfh.write('\n')

        print '\ncross validation result written to %s' %output_path

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

            #create model
            model = Model(model_name, self.dataset.class_num, params)
            #train
            train_accu = model.train(self.dataset.cv_train(val_fold_id), self.dataset.type, self.dataset.pass_num)
            val_accu = model.test(self.dataset.cv_val(val_fold_id), self.dataset.type)

            print 'Fold: %d' %(val_fold_id)
            print '\tParameter Setting: %s' %(str(params))
            print '\tTraining Accuracy: %f' %(train_accu)
            print '\tValidation Accuracy: %f' %(val_accu)
            train_accu_list.append(train_accu)
            val_accu_list.append(val_accu)
        return train_accu_list, val_accu_list

if __name__ == '__main__':
    handler = CV('aut','Ada_RDA','3','-eta 0.5:10:12 -delta 0.5:2:1')
    handler.run()

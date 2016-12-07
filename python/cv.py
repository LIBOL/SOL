#!/usr/bin/env python
"""Cross validation"""

import sys
import os
import logging
import re
import numpy as np
import ast
from multiprocessing import Process, Queue

from dataset import DataSet
from pysol import SOL

class SearchSpace(object):
    """Search space of all parameters
    """

    def __init__(self, params):
        """Create a search space with the given parameter ranges

        Parameters
        ----------
        params: dict{param, range}
            parameter string, with format like {'a': [1,2,8],'b': [2, 2,16]}
        """

        params = params.items()
        self.dim = len(params)
        param_nums = [len(v[1]) for v in params]
        self.size = np.prod(param_nums)

        self.search_space = []
        for param_id in xrange(self.size):
            param = {}
            for d in xrange(self.dim):
                idx = param_id % param_nums[d]
                param_id /= param_nums[d]
                param[params[d][0]] = params[d][1][idx]
            self.search_space.append(param)

    def get_param(self, idx):
        return self.search_space[idx]

def train_val_executor(task_queue,
                      result_queue):
    while True:
        task = task_queue.get()
        if task == None:
            break
        param_idx = task[0]
        model_name = task[1]
        dt = task[2]
        params = task[3]
        fold_num = task[4]
        val_fold_id = task[5]
        m = SOL(algo=model_name, class_num=dt.class_num, **params)

        for p in xrange(dt.pass_num):
            for i in xrange(fold_num):
                if i == val_fold_id:
                    continue
                train_accu = m.fit(dt.split_path(i), dt.slice_type)
        val_accu = m.score(dt.split_path(val_fold_id), dt.slice_type)

        logging.info('Cross validation of %s on %s, Fold %d/%d: \n\t\
                     params: %s\n\t\
                     Training Accuracy: %f, Validation Accuracy: %f',
                     model_name, dt.name, val_fold_id, fold_num,
                     str(params), train_accu, val_accu)

        result_queue.put((param_idx, train_accu, val_accu))
    task_queue.put(None)

class CV(object):
    """cross validation class
    """

    def __init__(self, dataset, fold_num, cv_params, extra_params={},
                 process_num=1):
        """Create a new cross validation instance

        Parameters
        ----------
        dataset: DataSet
            dataset to be used for cross validation
        fold_num: int
            number of folds to conduct cross validation
        cv_params: dict{param, range}
            parameter string, with format like "'{'a': [1,2,4],'b':[2,2,8]}'"
        extra_params: dict{param, val}
            extra param for training, with format {'a':'1'}
        process_num: int
            number of processes to do cross validation
        """

        self.dataset = dataset
        self.fold_num = int(fold_num)
        if self.fold_num < 2:
            raise ValueError('fold number must bigger than 2!')

        self.cv_params = cv_params
        self.search_space = SearchSpace(cv_params)
        #the last column is for average
        self.train_scores = np.zeros((self.search_space.size, self.fold_num + 1))
        self.val_scores = np.zeros((self.search_space.size, self.fold_num + 1))
        self.extra_params = extra_params
        self.process_num = process_num

    def train_val(self, model_name):
        """train and validate on the dataset

        Parameters
        ----------
        model_name: str
            name of the algorithm to use

        Return
            None
        ------

        """

        #split the dataset
        self.dataset.split_file(self.fold_num, "bin")

        #cross validation
        for val_fold_id in range(0, self.fold_num):

            logging.info('Cross Validation on Model %s with Data %s: Fold %d/%d'
                         % (model_name, self.dataset.name, val_fold_id, self.fold_num))

            train_accu_list, val_accu_list = self.__train_val_one_fold( model_name, val_fold_id)
            self.train_scores[:, val_fold_id] = train_accu_list
            self.val_scores[:, val_fold_id] = val_accu_list

        #calculate the average
        self.train_scores[:, self.fold_num] = np.sum(self.train_scores, axis=1) / self.fold_num
        self.val_scores[:, self.fold_num] = np.sum(self.val_scores, axis=1) / self.fold_num

    def __train_val_one_fold(self, model_name, val_fold_id):
        """ cross validation on one fold of data

        Parameters
        ----------

        model_name: str
            name of the algorithm to use
        val_fold_id: int
            fold id that is used as val data

        Return
        ------
            list of (train accuracy, validation accuracy)
        """

        task_queue = Queue()
        result_queue =Queue()


        for k in xrange(self.search_space.size):
            params = self.search_space.get_param(k).copy()
            params.update(self.extra_params)
            task_queue.put((k, model_name, self.dataset, params, self.fold_num, val_fold_id))

        task_queue.put(None)

        train_accu_list = [0 for k in xrange(self.search_space.size)]
        val_accu_list = [0 for k in xrange(self.search_space.size)]
        processes = []
        for k in xrange(self.process_num):
            p = Process(target=train_val_executor, args=(task_queue, result_queue,))
            p.start()
            processes.append(p)

        for k in xrange(self.search_space.size):
            param_idx, train_accu, val_accu = result_queue.get()
            train_accu_list[param_idx] = train_accu
            val_accu_list[param_idx] = val_accu

        for p in processes:
            p.join()

        return train_accu_list, val_accu_list

    def get_best_param(self):
        """Get the best parameters

        Return
        ------
            dict: best parameters and the parameter id
        """

        best_param_id = np.argmax(self.val_scores[:, self.fold_num])
        return self.search_space.get_param(best_param_id), best_param_id

    def save_results(self, output_path):
        """Save the cross validation results to a file

        Parameter
        ---------
        output_path: str
            path to the file to save the results

        """

        best_param, best_parma_id = self.get_best_param()

        with open(output_path, 'w') as wfh:
            wfh.write('Cross Validation Parameters: %s\n' %
                      (str(self.cv_params)))
            wfh.write('Extra Model Parameters: %s\n' % (str(self.extra_params)))
            wfh.write('Best Result: %s:\t%.4f\t%.4f\n' % (
                str(best_param),
                self.train_scores[best_parma_id, self.fold_num],
                self.val_scores[best_parma_id, self.fold_num]))

            wfh.write("Validation Accuracies\n")
            for y in xrange(self.val_scores.shape[0]):
                wfh.write('%s:' % (str(self.search_space.get_param(y))))
                for x in xrange(self.val_scores.shape[1]):
                    wfh.write('\t%.4f' % (self.val_scores[y, x]))
                wfh.write('\n')

            wfh.write("Train Accuracies\n")
            for y in xrange(self.train_scores.shape[0]):
                wfh.write('%s:' % (str(self.search_space.get_param(y))))
                for x in xrange(self.train_scores.shape[1]):
                    wfh.write('\t%.4f' % (self.train_scores[y, x]))
                wfh.write('\n')

        logging.info('cross validation result written to %s' % output_path)

    @staticmethod
    def load_results(result_path):
        """load the result file of cross validation to get the best parameters

        Return
        ------
        dict: cross validation parameters
        """

        logging.info('loading cross validation  results from %s' % result_path)
        with open(result_path, 'r') as fh:
            pattern = re.compile(r'Best Result: (\{.*\}):.*')
            result_list = pattern.findall(fh.read())
            assert len(result_list) == 1
            return ast.literal_eval(result_list[0])


if __name__ == '__main__':
    logger = logging.getLogger('')

    numeric_level = getattr(logging, "INFO", None)
    logger.setLevel(numeric_level)
    formatter = logging.Formatter(
        "%(threadName)s: %(asctime)s- %(levelname)s: %(message)s")

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    a1a = DataSet('a1a', data_path='../data/a1a')
    cv = CV(a1a, 5,
            { 'eta': np.logspace(-2,2,5, base=2),
             'delta': np.logspace(-2,2,5, base=2)
            },
            process_num=4)
    cv.train_val('ada-fobos')
    best_param = cv.get_best_param()[0]
    print best_param
    cv_output_path = os.path.join(cv.dataset.work_dir, 'cv.txt')
    cv.save_results(cv_output_path)
    best_param = cv.load_results(cv_output_path)
    print best_param

#This script is to run experiment automatically to test the performance of the algorithm

import os
import os.path as osp
import sys
import logging
import time
from operator import itemgetter

import numpy as np
from  sklearn import datasets
from sklearn.model_selection import GridSearchCV
from vowpalwabbit.sklearn_vw import VWClassifier

def vw_exe():
    """path to vw executable"""

    if sys.platform == 'win32':
        return 'vw.exe'
    else:
        return'vw'

def convert_to_vw(input_path, output_path, class_num=2):
    """convert data into vw format """

    logging.info('convert %s to %s', input_path, output_path)

    with open(output_path, 'w') as wfh:
        with open(input_path, 'r') as rfh:
            while True:
                line = rfh.readline().strip()
                if len(line) == 0:
                    break
                pos = 0
                while line[pos] != ' ' and line[pos] != '\t':
                    pos = pos + 1

                label = int(line[0:pos])
                if class_num != 2:
                    label += 1
                wfh.write('%s |%s\n' %(label, line[pos:]))

def calc_accuracy(test_path, predict_path, class_num):
    """calculate prediction accuracy """

    labels = []
    with open(test_path, 'r') as rfh:
        while True:
            line = rfh.readline().strip()

            if len(line) == 0:
                break
            val = int(line.split('|')[0])
            if val == 0:
                val = -1
            labels.append(val)

    with open(predict_path, 'r') as rfh:
        lines = rfh.readlines()

    if class_num == 2:
        predicts = [1 if float(v) > 0 else -1 for v in filter(None, [l.strip() for l in lines])]
    else:
        predicts = [int(v)  for v in filter(None, [l.strip() for l in lines])]

    assert len(labels) == len(predicts)
    return float(np.sum(np.array(labels) == np.array(predicts))) / len(labels)

def calc_featnum(model_path):
    """Calcualte the non zero feature number of vw model"""

    with open(model_path, 'r') as rfh:
        lines = rfh.readlines()

    feat_num = float(len(lines) - 11)
    if feat_num < 0:
        feat_num = 0

    return feat_num

def test(dtest):
    """test vw model

    Parameters
    ----------
    dteset: DataSet
        the dataset used to test the model

    Return
    ------
    tuple(test_accu, test_time)
    """

    cmd = vw_exe()

    model_path = osp.join(dtest.work_dir, 'vw.model')
    cmd += ' -t -i \"%s\"' %(model_path)

    predict_path = osp.join(dtest.work_dir, 'vw.predict')
    cmd += ' -p \"%s\"' %(predict_path)

    vw_data_path = osp.join(dtest.work_dir, osp.basename(dtest.data_path) + '.vw')
    if osp.exists(vw_data_path) is False:
        convert_to_vw(dtest.convert('svm'), vw_data_path, dtest.class_num)
    cmd += ' \"%s\"' %(vw_data_path)

    logging.info(cmd)
    start_time = time.time()
    if os.system(cmd) != 0:
        logging.error('call vw failed, vw in path?')
        sys.exit()
    test_time = time.time() - start_time
    return calc_accuracy(vw_data_path, predict_path, dtest.class_num), test_time

def train(dtrain,
          model_params={},
          output_path=None,
          cv_params=None,
          fold_num=5,
          retrain=False,
          cv_process_num=1):
    """
    train a vw model

    Parameter
    ---------
    dtrain: DataSet
        the dataset used to train the model
    model_params: dict{param, val}
        model parameters
    output_path: str
        path to save the model
    cv_params: dict{param, range}
        cross validation parameters
    fold_num: int
        number of folds to do cross validation
    retrain: bool
        whether to re-do the cross validation
    cv_process_num: int
        number of processes to do cross validation

    Return
    ------
    tuple(train accuracy, train time)
    """

    best_l = 1.0
    if cv_params is not None:
        cv_output_path = osp.join(dtrain.work_dir, 'cv-vw.txt')
        if osp.exists(cv_output_path) and retrain is False:
            with open(cv_output_path, 'r') as rfh:
                line = rfh.readline()
            best_l = float(line.split('=')[1])
        else:
            #cross validation
            logging.info("cross validation")
            logging.info("loading %s", dtrain.rand_path('svm'))
            xtrain, ytrain = datasets.load_svmlight_file(dtrain.rand_path('svm'))

            clf = GridSearchCV(estimator=VWClassifier(),
                               param_grid=cv_params,
                               n_jobs=cv_process_num,
                               cv=fold_num,
                               verbose=True)
            clf.fit(xtrain, ytrain)

            best_l = clf.best_params_['learning_rate']
            with open(cv_output_path, 'w') as wfh:
                wfh.write('Best Result: l=%f' %(best_l))

        logging.info('cross validation parameters: learning_rate=%f', (best_l))

        model_params['learning_rate'] = best_l

    cmd = vw_exe()
    if dtrain.class_num != 2:
        cmd += ' --oaa %d' %(dtrain.class_num)

    model_path = osp.join(dtrain.work_dir, 'vw.model')
    cmd += ' -f \"%s\"' %(model_path)

    if output_path != None:
        cmd += ' --readable_model \"%s\"' %(output_path)

    for k,v in model_params.iteritems():
        cmd += ' --%s \"%s\"' %(k,str(v))

    vw_data_path = dtrain.rand_path('svm') + '.vw'
    if osp.exists(vw_data_path) is False:
        convert_to_vw(dtrain.rand_path('svm'), vw_data_path, dtrain.class_num)

    cmd += ' \"%s\"' %(vw_data_path)

    logging.info(cmd)
    start_time = time.time()
    if os.system(cmd) != 0:
        logging.error('call vw failed, vw in path?')
        sys.exit()
    train_time = time.time() - start_time

    train_accu, train_test_time = test(dtrain)
    train_time += train_test_time

    logging.info("training accuracy: %.4f", train_accu)
    logging.info("training time: %.4f seconds", train_time)

    return train_accu, train_time

def train_test(dtrain,
               dtest,
               model_params={},
               cv_params=None,
               retrain=False,
               fold_num=5,
               cv_process_num=1):

    output_path = None
    if 'l1' in model_params:
        output_path = osp.join(dtrain.work_dir, 'vw-%f.readable.model' %(model_params['l1']))
    train_accu, train_time = train(dtrain,
                                   model_params,
                                   output_path=output_path,
                                   cv_params=cv_params,
                                   retrain=retrain,
                                   fold_num=fold_num,
                                   cv_process_num=cv_process_num)

    test_accu, test_time = test(dtest)
    logging.info("test accuracy: %.4f", test_accu)
    logging.info("test time: %.4f seconds", test_time)

    if output_path is None:
        return test_accu, test_time, train_accu, train_time
    else:
        clf_num = 1 if dtrain.class_num == 2 else dtrain.class_num
        return calc_featnum(output_path) / clf_num, test_accu, test_time, train_accu, train_time

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'Usage dt_name train_file test_file'
        sys.exit()

    logger = logging.getLogger('')

    numeric_level = getattr(logging, "INFO", None)
    logger.setLevel(numeric_level)

    from sol.dataset import  DataSet

    dtrain = DataSet(sys.argv[1], sys.argv[2], 'svm')
    dtest = DataSet(sys.argv[1], sys.argv[3], 'svm')

    cv_params={'learning_rate': np.logspace(-2,2,5,base=2)}
    print train_test(dtrain, dtest, cv_params=cv_params) 
    res = []
    for l1 in [0.0001, 0.001, 0.01]:
        model_params = {'l1':l1}
        res.append(train_test(dtrain, dtest, model_params))
    print res

#! /usr/bin/env python
#################################################################################
#     File Name           :     lsol.py
#     Created By          :     yuewu
#     Description         :      
#################################################################################

from lsol_core import Model

with  Model("sgd", 2) as model:
    print 'training accuracy: %f' %(model.train("../data/a1a", "svm", model_path = "model.json"))
    print 'test accuracy: %f' %(model.test("../data/a1a", "svm", output_path = "predict.txt"))

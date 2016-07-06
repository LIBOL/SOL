#! /usr/bin/env python

import ctypes
from ctypes import c_char_p, c_int, c_float, c_void_p, byref
import os
import sys

def find_lib_path():
    """Find MXNet dynamic library files.

    Returns
    -------
    lib_path : string
                found path to the library
    """
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    api_path = os.path.join(curr_path, '../dist/bin/')
    dll_path = [curr_path, api_path]
    dll_path.append(os.path.join(curr_path, '../build/bin', 'Release'))
    dll_path.append(os.path.join(curr_path, '../build/bin'))
    dll_path = [p.replace('/', os.sep) for p in dll_path]

    if os.name == "posix" and os.environ.get('LD_LIBRARY_PATH', None):
        dll_path.extend([p.strip() for p in os.environ['LD_LIBRARY_PATH'].split(":")])

    if sys.platform == 'win32':
        dll_path = [os.path.join(p, 'lsol.dll') for p in dll_path]
    elif sys.platform == 'cygwin':
        dll_path = [os.path.join(p, 'lsol.dll') for p in dll_path]
    else:
        dll_path = [os.path.join(p, 'liblsol.so') for p in dll_path]

    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
    if len(lib_path) == 0:
        raise RuntimeError('Cannot find the lsol library.\n' +
                           'List of candidates:\n' + str('\n'.join(dll_path)))

    return lib_path[0]

def load_lib():
    _LIB = ctypes.cdll.LoadLibrary(find_lib_path())

    _LIB.lsol_CreateDataIter.argtypes = [c_int, c_int]
    _LIB.lsol_CreateDataIter.restype = c_void_p

    _LIB.lsol_ReleaseDataIter.argtypes = [c_void_p]
    _LIB.lsol_ReleaseDataIter.restype = None

    _LIB.lsol_LoadData.argtypes = [c_void_p, c_char_p, c_char_p, c_int]
    _LIB.lsol_LoadData.restype = c_int

    _LIB.lsol_CreateModel.argtypes = [c_char_p, c_int]
    _LIB.lsol_CreateModel.restype = c_void_p

    _LIB.lsol_RestoreModel.argtypes = [c_char_p]
    _LIB.lsol_RestoreModel.restype = c_void_p

    _LIB.lsol_SaveModel.argtypes = [c_void_p, c_char_p]
    _LIB.lsol_SaveModel.restype = c_int

    _LIB.lsol_ReleaseModel.argtypes = [c_void_p]
    _LIB.lsol_ReleaseModel.restype = None

    _LIB.lsol_SetModelParameter.argtypes = [c_void_p, c_char_p, c_char_p]
    _LIB.lsol_SetModelParameter.restype = c_int

    _LIB.lsol_Train.argtypes = [c_void_p, c_void_p]
    _LIB.lsol_Train.restype = c_float

    _LIB.lsol_Test.argtypes = [c_void_p, c_void_p, c_char_p]
    _LIB.lsol_Test.restype = c_float

    return _LIB

class Model(object):
    # current version
    __version__ = "0.1.0"

    #Load libary by searching possible path.
    _LIB = load_lib()

    def __init__(self, model_path = None, model_name = None, class_num = None, batch_size = 256, buf_size = 2, params = []):
        """Create a new model
        Parameters:
            model_path: string
                path to model file
            model_name: string
                name of the model
            class_num: int
                number of classes
            batch_size: int
                size of minibatch
            buf_size: int
                number of minibatches for bufferring
            params: list
                model parameters
        """
        if model_path != None:
            self.model = c_void_p(Model._LIB.lsol_RestoreModel(model_path))
        elif model_name != None and class_num != None:
            self.model = c_void_p(Model._LIB.lsol_CreateModel(model_name, class_num))
            if self.model.value == None:
                raise RuntimeError("invalid parameters for Model constructor")
            for k,v in params:
                if Model._LIB.lsol_SetModelParameter(self.model, k, str(v)) != 0:
                    raise RuntimeError("set parameter %s=%s failed" %(k,v))
        else:
            raise RuntimeError("invalid parameters for Model constructor")

        if self.model.value == None:
            raise RuntimeError("invalid parameters for Model constructor")

        self.data_iter = c_void_p(Model._LIB.lsol_CreateDataIter(batch_size, buf_size))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        Model._LIB.lsol_ReleaseDataIter(byref(self.data_iter))
        Model._LIB.lsol_ReleaseModel(byref(self.model))

    def train(self, data_path, data_type, pass_num = 1, model_path = None):
        """train on the given data
        Parameters:
            data_path: string
                path to the data
            data_type: string
                type of the data, svm, bin, csv, etc.
            pass_num: int
                number of passes to iterate through the data
            output_path: string
                path to save the model
        Return:
            training error rate
        """
        if type(data_path) == str:
            data_path = [data_path]
        for dp in data_path:
            ret = Model._LIB.lsol_LoadData(self.data_iter, dp, data_type, pass_num)
            if ret != 0:
                print 'load data %s failed' %(data_path)
                return 0
        err_rate = Model._LIB.lsol_Train(self.model, self.data_iter)
        if model_path != None:
            Model._LIB.lsol_SaveModel(self.model, model_path)
        return err_rate

    def test(self, data_path, data_type, output_path = None):
        """test on the given data
        Parameters:
            data_path: string
                path to the data
            data_type: string
                type of the data, svm, bin, csv, etc.
            output_path: string
                path to save the predicted results
        Return:
            test error rate
        """
        ret = Model._LIB.lsol_LoadData(self.data_iter, data_path, data_type, 1)
        if ret != 0:
            print 'load data %s failed' %(data_path)
            return 0
        return Model._LIB.lsol_Test(self.model, self.data_iter,
                None if output_path == None else output_path)

import numpy as np
import os.path as osp
cimport numpy as np
cimport lsol

np.import_array()

cdef class Model:
    cdef void* _c_model
    cdef void* _c_data_iter

    def  __cinit__(self, const char* path_or_name, int class_num = -1, int batch_size=256, int buf_size = 2, params=[]):
        if osp.exists(path_or_name):
            self._c_model = lsol_RestoreModel(path_or_name)
        elif class_num > 0:
            self._c_model = lsol_CreateModel(path_or_name, class_num)
        else:
            raise ValueError("class_num (%d) is incorrect" %(class_num))

        if self._c_model is NULL:
            raise MemoryError()

        self._c_data_iter = lsol_CreateDataIter(batch_size, buf_size)

        if self._c_data_iter is NULL:
            raise MemoryError()
        else:
            for k,v in params:
                if lsol_SetModelParameter(self._c_model, k, str(v)) != 0:
                    raise RuntimeError("set parameter %s=%s failed" %(k,str(v)))

    def __dealloc__(self):
        if self._c_data_iter is not NULL:
            lsol_ReleaseDataIter(&self._c_data_iter)

        if self._c_model is not NULL:
            lsol_ReleaseModel(&self._c_model)

    def train(self, const char* data_path, const char* data_type, int pass_num = 1):
        """train on the given data
        Parameters:
            data_path: string
                path(paths) to the data
            data_type: string
                type of the data, svm, bin, csv, etc.
            pass_num: int
                number of passes to iterate through the data
        Return:
            training error rate
        """
        cdef int ret = lsol_LoadData(self._c_data_iter, data_path, data_type, pass_num)
        if ret != 0:
            raise RuntimeError('load data %s failed' %(data_path))
        cdef float err_rate = lsol_Train(self._c_model, self._c_data_iter)
        return err_rate

    def save(self, const char* model_path):
        lsol_SaveModel(self._c_model, model_path)

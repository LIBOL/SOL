cimport numpy as np
cimport lsol
import numpy as np

np.import_array()

cdef class CLSOL(BaseEstimator):
    cdef void* _c_model
    cdef void* _c_data_iter

    def  __cinit__(self, const char* algo = NULL, int class_num = -1, int batch_size=256, int buf_size = 2):
        """Create a new Handle for LSOL C Library

        Parameters
        ----------
        algo: str
            name of algorithm to be used for learning
        class_num: int
            number of classes for learning
        batch_size: int
            size of mini-batches in processing
        buf_size: int
            number of mini-batches for bufferring

        Returns
        -------
        (CLSOL): Returns self
        """
        if algo is not NULL
            self._c_model = lsol_CreateModel(algo, class_num)
        else:
            self._c_model = NULL

        self._c_data_iter = lsol_CreateDataIter(batch_size, buf_size)

        if self._c_data_iter is NULL:
            raise MemoryError()

    def __dealloc__(self):
        """Release Memory"""
        if self._c_data_iter is not NULL:
            lsol_ReleaseDataIter(&self._c_data_iter)

        if self._c_model is not NULL:
            lsol_ReleaseModel(&self._c_model)

    def set_params(self, params):
        """Set Model Parameters

        Parameters
        ----------
        params: dict, key is of type string, value is any type

        Returns
        ------
        (CLSOL): Returns self
        """
        assert self._c_model is not NULL, "model is not initialized"

        for k,v in params.iteritems():
            if lsol_SetModelParameter(self._c_model, k, str(v)) != 0:
                raise RuntimeError("set parameter %s=%s failed" %(k,str(v)))

    def begin_train(self):
        """Get the model ready for learning after parameter initialization"""
        assert self._c_model is not NULL, "model is not initialized"

        lsol_BeginTrain(self._c_model)

    def learn_from_file(self, const char* data_path, const char* data_type, int pass_num = 1):
        """learn model from a disk file

        Parameters
        ---------
        data_path: string
            path(paths) to the data
        data_type: string
            type of the data, svm, bin, csv, etc.
        pass_num: int
            number of passes to iterate through the data

        Returns
        -------
        float: training error rate
        """
        assert self._c_model is not NULL, "model is not initialized"

        cdef int ret = lsol_LoadData(self._c_data_iter, data_path, data_type, pass_num)
        if ret != 0:
            raise RuntimeError('load data %s failed' %(data_path))
        cdef float err_rate = lsol_Train(self._c_model, self._c_data_iter)
        return err_rate

    #def learn_from_data(self, X, np.ndarray[np.float64_t, ndim=1, mode='c'] Y,
    #        bint is_sparse, int pass_num = 1):
    #    """learn data from memory data

    #    Parameters
    #    ----------
    #    X: {array like or sparse matrix}, shape = [n_samples, n_features]
    #        Training vector, where n_samples is the number of samples and n_features is the number of features
    #    Y: array-like, shape=[n_samples]
    #        Target label vector relative to X
    #    is_sparse: bool
    #        whether X is dense or sparse
    #    pass_num: int
    #        number of passes to iterate through the data

    #    Returns
    #    -------
    #    float: training error rate
    #    """
    #    assert self._c_model is not NULL, "model is not initialized"

    #    cdef int ret = 0
    #    if is_sparse:
    #        ret = lsol_LoadCSRMatrix(self._c_data_iter,
    #                (<np.ndarray[np.float64_t, ndim=1, mode='c']>X.data).data,
    #                (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X.indices).shape,
    #                (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X.indices).data,
    #                (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X.indptr).shape,
    #                (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X.indptr).data,
    #                Y.data, (<np.int32_t>)X.shape[1])
    #    else:
    #        ret = lsol_loadArray(self._c_data_iter,
    #                (<np.ndarray[np.float64_t, ndim=2, mode='c']>X).data,
    #                Y.data,
    #                (<np.ndarray[np.float64_t, ndim=2, mode='c']>X).shape)
    #        if ret != 0:
    #            raise RuntimeError('load data failed')
    #    cdef float err_rate = lsol_Train(self._c_model, self._c_data_iter)
    #    return err_rate

    def end_train(self):
        """Finalize the model after training"""

        assert self._c_model is not NULL, "model is not initialized"

        lsol_EndTrain(self._c_model)

    def save(self, const char* model_path):
        """Save the model to a file

        Parameters
        ----------
        model_path: str
            path to save the model
        """
        assert self._c_model is not NULL, "model is not initialized"
        lsol_SaveModel(self._c_model, model_path)

    def load(self, const char* model_path):
        """Load a model from file

        Parameters
        ----------
        model_path: str
            path to load the model

        Returns
        -------
        (CLSOL): Returns self
        """
       self._c_model = lsol_RestoreModel(model_path)
       return self

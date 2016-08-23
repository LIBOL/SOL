cimport numpy as np
cimport lsol
from libc.stdlib cimport malloc,free

import numpy as np
from scipy.sparse import csr_matrix

np.import_array()

cdef void get_parameter(void* user_context,
        const char* param_name,
        const char* param_value):
    params = <object>user_context
    if params is not None:
        params[param_name] = param_value

cdef void desicion_function_callback(void* user_context,
        double label,
        double predict,
        int cls_num,
        float* scores):
    handler = <object>user_context
    handler[0].append(label)
    handler[1].append([scores[i] for i in xrange(cls_num)])

cdef void predict_callback(void* user_context,
        double label,
        double predict,
        int cls_num,
        float* scores):
    handler = <object>user_context
    handler[0].append(label)
    handler[1].append(predict)

cdef void inspect_iteration(void* user_context,
        long long data_num,
        long long iter_num,
        long long update_num,
        double err_rate ):
    handler = <object>user_context
    if handler is not None:
        handler(data_num, iter_num, update_num, err_rate)

cdef class LSOL:
    cdef void* _c_model
    cdef void* _c_data_iter
    cdef const char* algo
    cdef int class_num
    cdef bint verbose

    def  __cinit__(self, const char* algo = NULL, int class_num = -1, int
            batch_size=256, int buf_size = 2, verbose=False, **params):
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
        if algo is not NULL:
            self._c_model = lsol_CreateModel(algo, class_num)
            self.algo = algo
            self.class_num = class_num
        else:
            self._c_model = NULL
            self.algo = 'none'
            self.class_num = -1

        self._c_data_iter = lsol_CreateDataIter(batch_size, buf_size)

        if self._c_data_iter is NULL:
            raise MemoryError()

        if verbose == False:
            self.inspect_learning(None)
        self.verbose = verbose
        self.set_params(**params)

    def __dealloc__(self):
        """Release Memory"""
        if self._c_data_iter is not NULL:
            lsol_ReleaseDataIter(&self._c_data_iter)

        if self._c_model is not NULL:
            lsol_ReleaseModel(&self._c_model)

    @property
    def name(self):
        return <bytes>self.algo

    @property
    def sparsity(self):
        if self._c_model is NULL:
            return 0
        return <double>(lsol_model_sparsity(self._c_model))

    def set_params(self, **params):
        """Set Model Parameters

        Parameters
        ----------
        params: dict, key is of type string, value is any type

        Returns
        ------
        (CLSOL): Returns self
        """
        if params != None and len(params) > 0:
            assert self._c_model is not NULL, "model is not initialized"

            for k,v in params.iteritems():
                if lsol_SetModelParameter(self._c_model, k, str(v)) != 0:
                    raise RuntimeError("set parameter %s=%s failed" %(k,str(v)))

    def get_params(self):
        """Get Model Parameters

        Returns
        -------
        dict: mapping of string to string
        """
        params = dict()
        lsol_GetModelParameters(self._c_model, get_parameter, <void*>params)
        return params

    def inspect_learning(self, iterate_handler):
        if self._c_model is not NULL:
            lsol_InspectOnlineIteration(self._c_model, inspect_iteration, <void*>iterate_handler)

    def __load_data(self, param1, param2, int pass_num):
        """load data to data_iter

        Parameters
        ----------
        param1: string, data path or {array-like or sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and n_features is the number of features
        param2: string, data type or array-like, shape=[n_samples]
            Target label vector relative to X
        pass_num: int
            number of passes to iterate through the data
        """
        cdef int ret = 0

        if isinstance(param1, str):
            ret = lsol_LoadData(self._c_data_iter, <const char*>param1, <const char*>param2, pass_num)
        else:
            if param2 is None:
                param2 = np.zeros(param1.shape[0], dtype=np.float64)
            assert param1.dtype == np.float64, "only float64 data are allowed"
            assert param2.dtype == np.float64, "only float64 labels are allowed"
            X = param1
            y = param2

            if isinstance(param1, np.ndarray):
                ret = lsol_loadArray(self._c_data_iter,
                        (<np.ndarray[np.float64_t, ndim=2, mode='c']>X).data,
                        (<np.ndarray[np.float64_t, ndim=1, mode='c']>y).data,
                        (<np.ndarray[np.float64_t, ndim=2, mode='c']>X).shape,
                        (<np.ndarray[np.float64_t, ndim=2, mode='c']>X).strides,
                        pass_num)
            elif isinstance(param1, csr_matrix):
                ret= lsol_loadCsrMatrix(self._c_data_iter,
                        (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X.indices).data,
                        (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X.indptr).data,
                        (<np.ndarray[np.float64_t, ndim=1, mode='c']>X.data).data,
                        (<np.ndarray[np.float64_t, ndim=1, mode='c']>y).data,
                        (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X.indptr).shape[0] - 1,
                        pass_num)
            else:
                raise TypeError("only data path or numpy.ndarray or csr_matrix are allowed")

        if ret != 0:
            raise RuntimeError('load data failed')

    def fit(self, param1, param2, int pass_num = 1):
        """learn data from numpy array

        Parameters
        ----------
        param1: string, data path or {array-like or sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and n_features is the number of features
        param2: string, data type or array-like, shape=[n_samples]
            Target label vector relative to X
        pass_num: int
            number of passes to iterate through the data

        Returns
        -------
        CLSOL: training accuracy
        """
        assert self._c_model is not NULL, "model is not initialized"

        self.__load_data(param1, param2, pass_num)

        return 1 - lsol_Train(self._c_model, self._c_data_iter)

    def score(self, param1, param2):
        """Returns the mean accuracy on the given test data and labels

        Parameters
        ----------
        param1: str, data path or {array-like or sparse matrix}, shape = [n_samples, n_features]
            Test vector, where n_samples is the number of samples and n_features is the number of features
        param2: str, data type or array-like, shape=[n_samples]
            Target label vector relative to X

        Returns
        -------
        float: test accuracy
        """
        assert self._c_model is not NULL, "model is not initialized"

        self.__load_data(param1, param2, 1)

        return 1 - lsol_Test(self._c_model, self._c_data_iter, NULL)

    def decision_function(self, param1, param2 = None, get_labels = False):
        """Predict confidence scores for samples in X

        Parameters
        ----------
        Parameters
        ----------
        param1: str, data path or {array-like or sparse matrix}, shape = [n_samples, n_features]
            Test vector, where n_samples is the number of samples and n_features is the number of features
        param2: str, data type or None
        get_labels: bool
            Whether return labels

        Returns
        -------
        C: array, shape = [n_samples, n_classifiers]
        labels: array, shape = [n_samples] (dependes on get_labels)
        """
        assert self._c_model is not NULL, "model is not initialized"

        self.__load_data(param1, param2, 1)
        result = [[],[]]

        lsol_Predict(self._c_model, self._c_data_iter,
                desicion_function_callback, <void*>result)

        scores = np.array(result[1])
        if scores.shape[1] == 1:
            scores = scores.reshape(scores.shape[0])

        if get_labels:
            return scores, np.array(result[0])
        else:
            return scores

    def predict(self, param1, param2 = None, get_labels = False):
        """Predict class labels for samples in X

        Parameters
        ----------
        Parameters
        ----------
        param1: str, data path or {array-like or sparse matrix}, shape = [n_samples, n_features]
            Test vector, where n_samples is the number of samples and n_features is the number of features
        param2: str, data type or None
        get_labels: bool
            Whether return labels

        Returns
        -------
        C: array, shape = [n_samples]
        """
        assert self._c_model is not NULL, "model is not initialized"

        self.__load_data(param1, param2, 1)
        result = [[],[]]

        lsol_Predict(self._c_model, self._c_data_iter, predict_callback, <void*>result)

        if get_labels:
            return np.array(result[1]), np.array(result[0])
        else:
            return np.array(result[1])

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
        if self._c_model is not NULL:
            params = self.get_params()
            self.algo = params['model']
            self.class_num = int(params['cls_num'])

            if self.verbose == False:
                self.inspect_learning(None)

            return self
        else:
            return None

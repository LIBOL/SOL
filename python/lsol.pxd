cimport numpy as np

cdef extern from "lsol/c_api.h":
    void* lsol_CreateDataIter(int batch_size, int buf_size)
    void lsol_ReleaseDataIter(void** data_iter)
    int lsol_LoadData(void* data_iter, const char* path, const char* format, int pass_num)
    void* lsol_CreateModel(const char* name, int class_num)
    void* lsol_RestoreModel(const char* model_path)
    int lsol_SaveModel(void* model, const char* model_path)
    void lsol_ReleaseModel(void** model)
    int lsol_SetModelParameter(void* model, const char* param_name, const char* param_val)
    ctypedef void (*get_parameter_callback)(void* user_context, const char* param_name, const char* param_val)
    int lsol_GetModelParameters(void* model, get_parameter_callback callback, void* user_context)
    float lsol_Train(void* model, void* data_iter)
    float lsol_Test(void* model, void* data_iter, const char* output_path)
    ctypedef void (*lsol_predict_callback)(void* user_context, double label, double predict, int cls_num, float* scores)
    int lsol_Predict(void* model, void* data_iter, lsol_predict_callback callback, void* user_context)
    float lsol_model_sparsity(void* model)
    ctypedef void (*inspect_iterate_callback)(void* user_context, long long data_num, long long iter_num,
                                         long long update_num, double err_rate)
    void lsol_InspectOnlineIteration(void* model, inspect_iterate_callback callback, void* user_context)
    int lsol_loadArray(void* data_iter, char* X, char* Y, np.npy_intp* dims, np.npy_intp* strides, int pass_num)
    int lsol_loadCsrMatrix(void* data_iter, char* indices, char* indptr, char* features, char* Y, int n_samples, int pass_num)

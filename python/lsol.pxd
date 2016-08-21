cimport numpy as np

cdef extern from "lsol/c_api.h":
    cdef void* lsol_CreateDataIter(int batch_size, int buf_size)
    void lsol_ReleaseDataIter(void** data_iter)
    int lsol_LoadData(void* data_iter, const char* path, const char* format, int pass_num)
    void* lsol_CreateModel(const char* name, int class_num)
    void* lsol_RestoreModel(const char* model_path)
    int lsol_SaveModel(void* model, const char* model_path)
    void lsol_ReleaseModel(void** model)
    int lsol_SetModelParameter(void* model, const char* param_name, const char* param_val)
    float lsol_Train(void* model, void* data_iter)
    float lsol_Test(void* model, void* data_iter, const char* output_path)
    float lsol_model_sparsity(void* model)

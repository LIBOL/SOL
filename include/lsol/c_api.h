/*********************************************************************************
*     File Name           :     c_api.h
*     Created By          :     yuewu
*     Description         :
**********************************************************************************/
#ifndef LSOL_C_API_H__
#define LSOL_C_API_H__

#ifndef LSOL_EMBED_PACKAGE
#if (defined WIN32 || defined _WIN32 || defined WINCE)
#ifdef LSOL_EXPORTS
#undef LSOL_EXPORTS
#define LSOL_EXPORTS __declspec(dllexport)
#else
#define LSOL_EXPORTS __declspec(dllimport)
#endif
#else
#undef LSOL_EXPORTS
#define LSOL_EXPORTS
#endif
#else
#undef LSOL_EXPORTS
#define LSOL_EXPORTS
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// \brief  create a data iterator
///
/// \param batch_size batch size for data iteration
/// \param buf_size number of buffers to keep during iteration
///
/// \return pointer to the created instance
LSOL_EXPORTS void* lsol_CreateDataIter(int batch_size, int buf_size);

/// \brief  release the data iterator
///
/// \param data_iter pointer to data iterator pointer
LSOL_EXPORTS void lsol_ReleaseDataIter(void** data_iter);

/// \brief  load a data
///
/// \param data_iter data iteration instance
/// \param path path to the data to be loaded
/// \param format format of the data, like 'svm', 'bin', 'csv', etc.
/// \param pass_num number of passes to iterate the data
///
/// \return status code, 0 if succeed
LSOL_EXPORTS int lsol_LoadData(void* data_iter, const char* path,
                               const char* format, int pass_num);

/// \brief  create a new model for learning or prediction
///
/// \param name name of the model (algorithm)
/// \param class_num number of classes for the model
///
/// \return pointer to the created model
LSOL_EXPORTS void* lsol_CreateModel(const char* name, int class_num);

/// \brief  restore a model from a saved file
///
/// \param model_path path to the saved file
///
/// \return pointer to the created model
LSOL_EXPORTS void* lsol_RestoreModel(const char* model_path);

/// \brief  save a model to a file
///
/// \param model model to be saved
/// \param model_path path to save the model
///
/// \return status code, 0 if succeed
LSOL_EXPORTS int lsol_SaveModel(void* model, const char* model_path);

/// \brief  release model instance
///
/// \param model pointer to model pointer
LSOL_EXPORTS void lsol_ReleaseModel(void** model);

/// \brief  set model parameters
///
/// \param model pointer to the model
/// \param param_name name of the parameter
/// \param param_val value string of the parameter
///
/// \return status code, 0 if succeed
LSOL_EXPORTS int lsol_SetModelParameter(void* model, const char* param_name,
                                        const char* param_val);

/// \brief  C type of a get parameter function callback
///
/// \param user_context flexible place to handle the parameter name and value
/// \param param_name name of parameter
/// \param param_val value string of the parameter
typedef void (*lsol_get_parameter_callback)(void* user_context,
                                            const char* param_name,
                                            const char* param_val);

/// \brief  Get Model Parameters
///
/// \param model model
/// \param lsol_get_parameter_callback callback function to handle the
/// parameters
/// \param user_context flexible place to handle the parameter name and value
LSOL_EXPORTS void lsol_GetModelParameters(void* model,
                                          lsol_get_parameter_callback callback,
                                          void* user_context);
/// \brief  train a model
///
/// \param model model to be trained
/// \param data_iter data iterator
///
/// \return training accuracy
LSOL_EXPORTS float lsol_Train(void* model, void* data_iter);

/// \brief  test a model
///
/// \param model model to be tested
/// \param data_iter data iterator
/// \param output_path path to save the predicted results, if no need, leave it
/// empty
///
/// \return test accuracy
LSOL_EXPORTS float lsol_Test(void* model, void* data_iter,
                             const char* output_path);

/// \brief  C type to predict detailed scores
///
/// \param user_context flexible place to handle predicted results
/// \param label groundtruth label
/// \param predict predicted label
/// \param cls_num number of classses
/// \param scores predicted scores
typedef void (*lsol_predict_callback)(void* user_context, double label,
                                      double predict, int cls_num,
                                      float* scores);

/// \brief  predict the scores on the given data
///
/// \param model model to be tested
/// \param data_iter data iterator
/// \param callback callback to handle the predicted results
/// \param user_context flexible place to handle predicted results
///
/// \return number of samples processed
LSOL_EXPORTS int lsol_Predict(void* model, void* data_iter,
                              lsol_predict_callback callback,
                              void* user_context);

/// \brief  get the model sparsity
///
/// \param model pretrained model
///
/// \return model sparsity
LSOL_EXPORTS float lsol_model_sparsity(void* model);

/// \brief  C type to inspect iteration callback
///
/// \param user_context flexible place to handle iteration status
/// \param data_num number of data processed currently
/// \param iter_num number of iterations currently
/// \param update_num number of updates currently
/// \param err_rate training error rate currently
typedef void (*lsol_inspect_iterate_callback)(void* user_context,
                                              long long data_num,
                                              long long iter_num,
                                              long long update_num,
                                              double err_rate);

/// \brief  Get Model Parameters
///
/// \param model model
/// \param lsol_inspect_iterate_callback callback to handle the iteration status
/// \param user_context flexible place to handle the parameter name and value
LSOL_EXPORTS void lsol_InspectOnlineIteration(
    void* model, lsol_inspect_iterate_callback callback, void* user_context);

#ifdef HAS_NUMPY_DEV
#include <numpy/arrayobject.h>
LSOL_EXPORTS int lsol_loadArray(void* data_iter, char* X, char* Y,
                                npy_intp* dims, npy_intp* strides,
                                int pass_num);

LSOL_EXPORTS int lsol_loadCsrMatrix(void* data_iter, char* indices,
                                    char* indptr, char* features, char* y,
                                    int n_samples, int pass_num);
#endif

/// \brief  analyze the information of data
///
/// \param data_path path to the data
/// \param data_type type of the data ('svm', 'csv', etc.)
/// \param output_path output path to save the information
///
/// \return status code, 0 if succeed
LSOL_EXPORTS int lsol_analyze_data(const char* data_path, const char* data_type,
                                   const char* output_path);

/// \brief  convert data format
///
/// \param src_path path to the source data
/// \param src_type type of the source data
/// \param dst_path path to the destination data
/// \param dst_type type of the destination data
///
/// \return status code, 0 if succeed
LSOL_EXPORTS int lsol_convert_data(const char* src_path, const char* src_type,
                                   const char* dst_path, const char* dst_type);
/// \brief  shuffle data
///
/// \param src_path path to the source data
/// \param src_type type of the source data
/// \param dst_path path to the destination data
/// \param dst_type type of the destination data, if NULL, use src_type
///
/// \return status code, 0 if succeed
LSOL_EXPORTS int lsol_shuffle_data(const char* src_path, const char* src_type,
                                   const char* dst_path, const char* dst_type);

/// \brief  split data into parts
///
/// \param src_path path to the source data
/// \param src_type type of the source data
/// \param fold number of splits
/// \param output_prefix prefix of the output file
/// \param dst_type type of the destination data, if NULL, use src_type
/// \param shuffle whether shuffling th data before splitting, default false
///
/// \return status code, 0 if succeed
LSOL_EXPORTS int lsol_split_data(const char* src_path, const char* src_type,
                                 int fold, const char* output_prefix,
                                 const char* dst_type, bool shuffle);

#ifdef __cplusplus
}
#endif

#endif

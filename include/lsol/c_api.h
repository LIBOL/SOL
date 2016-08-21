/*********************************************************************************
*     File Name           :     c_api.h
*     Created By          :     yuewu
*     Description         :
**********************************************************************************/
#ifndef LSOL_C_API_H__
#define LSOL_C_API_H__

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

/// \brief  get the model sparsity
///
/// \param model pretrained model
///
/// \return model sparsity
LSOL_EXPORTS float lsol_model_sparsity(void* model);

/// \brief  get the training log of model
///
/// \param model pretrained model
///
/// \return string of log
LSOL_EXPORTS const char* lsol_model_train_log(void* model);
#ifdef __cplusplus
}
#endif

#endif

/*********************************************************************************
*     File Name           :     c_api.cc
*     Created By          :     yuewu
*     Description         :
**********************************************************************************/

#include "sol/c_api.h"

#include <stdexcept>
#include <fstream>
#include <cstring>

#ifdef HAS_NUMPY_DEV
#include <numpy/arrayobject.h>
#include "sol/pario/numpy_reader.h"
#include "sol/pario/csr_matrix_reader.h"
#endif

#include <json/json.h>

#include "sol/sol.h"
#include "sol/tools.h"
#include "sol/model/online_model.h"

using namespace std;
using namespace sol;
using namespace sol::pario;
using namespace sol::model;

void* sol_CreateDataIter(int batch_size, int buf_size) {
  return (void*)(new DataIter(batch_size, buf_size));
}

void sol_ReleaseDataIter(void** data_iter) {
  DataIter** iter = (DataIter**)(data_iter);
  DeletePointer(*iter);
}

void* sol_CreateDataWriter(const char* path, const char* format, int feat_dim) {
  DataWriter* writer = DataWriter::Create(format);
  if (writer == nullptr) {
    return nullptr;
  }
  if (writer->Open(path) != Status_OK) {
    delete writer;
    return nullptr;
  }
  if (strcmp(format, "csv") == 0) {
    // for csv, get extra info
    if (feat_dim == 0) {
      fprintf(stderr, "figuring out feature dimension failed\n");
      delete writer;
      return nullptr;
    }
    index_t feat_dim2 = (index_t)feat_dim;
    writer->SetExtraInfo((char*)(&feat_dim2));
  }

  return (void*)writer;
}

void sol_ReleaseDataWriter(void** data_writer) {
  DataWriter** writer = (DataWriter**)(data_writer);
  (*writer)->Close();
  DeletePointer(*writer);
}

int sol_LoadData(void* data_iter, const char* path, const char* format,
                 int pass_num) {
  DataIter* iter = (DataIter*)(data_iter);
  return iter->AddReader(path, format, pass_num);
}

int sol_WriteData(void* data_writer, void* data_iter) {
  DataWriter* writer = (DataWriter*)(data_writer);
  DataIter* iter = (DataIter*)(data_iter);
  MiniBatch* mb = nullptr;
  while (true) {
    mb = iter->Next(mb);
    if (mb == nullptr) break;

    for (int i = 0; i < mb->size(); ++i) {
      writer->Write((*mb)[i]);
    }
  }
  return Status_OK;
}

void* sol_CreateModel(const char* name, int class_num) {
  return (void*)(Model::Create(name, class_num));
}

void* sol_RestoreModel(const char* model_path) {
  return (void*)(Model::Load(model_path));
}

int sol_SaveModel(void* model, const char* model_path) {
  Model* m = (Model*)(model);
  return m->Save(model_path);
}

void sol_ReleaseModel(void** model) {
  Model** m = (Model**)(model);
  DeletePointer(*m);
}

int sol_SetModelParameter(void* model, const char* param_name,
                          const char* param_val) {
  Model* m = (Model*)(model);
  try {
    m->SetParameter(param_name, param_val);
  }
  catch (invalid_argument& err) {
    fprintf(stderr, "%s\n", err.what());
    return Status_Invalid_Argument;
  }
  return Status_OK;
}

void GetModelParameters(const Json::Value& model_info,
                        sol_get_parameter_callback callback,
                        void* user_context) {
  for (Json::Value::const_iterator iter = model_info.begin();
       iter != model_info.end(); ++iter) {
    if (iter->type() == Json::objectValue) {
      GetModelParameters(*iter, callback, user_context);
    } else {
      callback(user_context, iter.name().c_str(), iter->asString().c_str());
    }
  }
}
void sol_GetModelParameters(void* model, sol_get_parameter_callback callback,
                            void* user_context) {
  if (callback == nullptr) return;

  Model* m = (Model*)(model);
  Json::Value model_info;
  m->model_info(model_info);
  GetModelParameters(model_info, callback, user_context);
}

float sol_Train(void* model, void* data_iter) {
  Model* m = (Model*)(model);
  DataIter* iter = (DataIter*)(data_iter);
  return m->Train(*iter);
}

float sol_Test(void* model, void* data_iter, const char* output_path) {
  Model* m = (Model*)(model);
  DataIter* iter = (DataIter*)(data_iter);
  if (output_path != nullptr) {
    ofstream out_file(output_path, ios::out);
    return m->Test(*iter, &out_file);
  } else {
    return m->Test(*iter, nullptr);
  }
}

int sol_Predict(void* model, void* data_iter, sol_predict_callback callback,
                void* user_context) {
  Model* m = (Model*)(model);
  DataIter* iter = (DataIter*)(data_iter);

  if (m->model_updated()) m->EndTrain();

  float* score_buf = new float[m->clf_num()];
  MiniBatch* mb = nullptr;
  int data_num = 0;
  while (1) {
    mb = iter->Next(mb);
    if (mb == nullptr) break;
    for (int i = 0; i < mb->size(); ++i) {
      DataPoint& x = (*mb)[i];
      m->PreProcess(x);
      // predict
      label_t label = m->Predict(x, score_buf);
      callback(user_context, x.label(), label, m->clf_num(), score_buf);
    }
    data_num += mb->size();
  }
  delete[] score_buf;
  return data_num;
}

float sol_model_sparsity(void* model) {
  Model* m = (Model*)(model);
  return m->model_sparsity();
}

SOL_EXPORTS void sol_InspectOnlineIteration(
    void* model, sol_inspect_iterate_callback callback, void* user_context) {
  if (callback == nullptr) return;

  OnlineModel* m = (OnlineModel*)(model);
  m->set_iterate_callback(callback, user_context);
}

#ifdef HAS_NUMPY_DEV
int sol_loadArray(void* data_iter, char* X, char* Y, npy_intp* dims,
                  npy_intp* strides, int pass_num) {
  string path = NumpyReader::GeneratePath((double*)X, (double*)Y, int(dims[0]),
                                          int(dims[1]), int(strides[0]));
  DataIter* iter = (DataIter*)(data_iter);
  return iter->AddReader(path, "numpy", pass_num);
}

int sol_loadCsrMatrix(void* data_iter, char* indices, char* indptr,
                      char* features, char* y, int n_samples, int pass_num) {
  string path = CsrMatrixReader::GeneratePath(
      (int*)indices, (int*)indptr, (double*)features, (double*)y, n_samples);
  DataIter* iter = (DataIter*)(data_iter);
  return iter->AddReader(path, "csr_matrix", pass_num);
}
#endif

int sol_analyze_data(const char* data_path, const char* data_type,
                     const char* output_path) {
  return analyze(data_path, data_type, output_path);
}

int sol_convert_data(const char* src_path, const char* src_type,
                     const char* dst_path, const char* dst_type,
                     bool binarize, float binarize_thresh) {
  return convert(src_path, src_type, dst_path, dst_type,
      binarize, binarize_thresh);
}

int sol_shuffle_data(const char* src_path, const char* src_type,
                     const char* dst_path, const char* dst_type) {
  return shuffle(src_path, src_type, dst_path,
                 dst_type == NULL ? "" : dst_type);
}

int sol_split_data(const char* src_path, const char* src_type, int fold,
                   const char* output_prefix, const char* dst_type,
                   bool shuffle) {
  return split(src_path, src_type, fold, output_prefix, dst_type, shuffle);
}

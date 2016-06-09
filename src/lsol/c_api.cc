/*********************************************************************************
*     File Name           :     c_api.cc
*     Created By          :     yuewu
*     Description         :
**********************************************************************************/

#include "lsol/c_api.h"

#include <stdexcept>
#include <fstream>

#include "lsol/lsol.h"

using namespace std;
using namespace lsol;
using namespace lsol::pario;
using namespace lsol::model;

void* lsol_CreateDataIter(int batch_size, int buf_size) {
  return (void*)(new DataIter(batch_size, buf_size));
}

void lsol_ReleaseDataIter(void** data_iter) {
  DataIter** iter = (DataIter**)(data_iter);
  DeletePointer(*iter);
}

int lsol_LoadData(void* data_iter, const char* path, const char* format,
                  int pass_num) {
  DataIter* iter = (DataIter*)(data_iter);
  return iter->AddReader(path, format, pass_num);
}

void* lsol_CreateModel(const char* name, int class_num) {
  return (void*)(Model::Create(name, class_num));
}

void* lsol_RestoreModel(const char* model_path) {
  return (void*)(Model::Load(model_path));
}

int lsol_SaveModel(void* model, const char* model_path) {
  Model* m = (Model*)(model);
  return m->Save(model_path);
}

void lsol_ReleaseModel(void** model) {
  Model** m = (Model**)(model);
  DeletePointer(*m);
}

int lsol_SetModelParameter(void* model, const char* param_name,
                           const char* param_val) {
  Model* m = (Model*)(model);
  try {
    m->SetParameter(param_name, param_val);
  } catch (invalid_argument& err) {
    fprintf(stderr, "%s\n", err.what());
    return Status_Invalid_Argument;
  }
  return Status_OK;
}

float lsol_Train(void* model, void* data_iter) {
  Model* m = (Model*)(model);
  DataIter* iter = (DataIter*)(data_iter);
  return m->Train(*iter);
}

float lsol_Test(void* model, void* data_iter, const char* output_path) {
  Model* m = (Model*)(model);
  DataIter* iter = (DataIter*)(data_iter);
  if (output_path != nullptr) {
    ofstream out_file(output_path, ios::out);
    return m->Test(*iter, &out_file);
  } else {
    return m->Test(*iter, nullptr);
  }
}

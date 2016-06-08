/*********************************************************************************
*     File Name           :     model.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-16 22:54]
*     Last Modified       :     [2016-03-09 19:22]
*     Description         :     base class for model
**********************************************************************************/

#include "lsol/model/model.h"

#include "lsol/util/util.h"
#include "lsol/util/error_code.h"

using namespace std;

namespace lsol {
namespace model {

Model* Model::Create(const std::string& name, int class_num) {
  auto create_func = CreateObject<Model>(std::string(name) + "_model");
  return create_func == nullptr ? nullptr : create_func(class_num);
}

Model::Model(int class_num, const std::string& type)
    : class_num_(class_num),
      clf_num_(class_num == 2 ? 1 : class_num),
      loss_(nullptr),
      type_(type) {
  Check(class_num > 1);
}

Model::~Model() {
  if (this->loss_ != nullptr) {
    delete this->loss_;
    this->loss_ = nullptr;
  }
}

void Model::SetParameter(const std::string& name, const std::string& value) {
  if (name == "loss") {
    DeletePointer(this->loss_);
    this->loss_ = loss::Loss::Create(value);
    Check(this->loss_ != nullptr);
  } else {
    ostringstream oss;
    oss << "unknown parameter " << name;
    throw runtime_error(oss.str());
  }
}

int Model::Save(const string& path) const {
  ofstream out_file(path.c_str(), ios::out);
  if (!out_file) {
    fprintf(stderr, "open file %s failed\n", path.c_str());
    return Status_IO_Error;
  }
  Json::Value root;
  this->GetModelInfo(root);
  this->GetModelParam(root);
  Json::StyledStreamWriter writer;
  writer.write(out_file, root);
  out_file.close();
  return Status_OK;
}

Model* Model::Load(const string& path) {
  int ret = Status_OK;
  Model* model = nullptr;
  ifstream in_file(path.c_str(), ios::in);
  if (!in_file) {
    fprintf(stderr, "open file %s failed\n", path.c_str());
    return nullptr;
  }
  Json::Value root;
  Json::Reader reader;
  if (reader.parse(in_file, root) == false) {
    fprintf(stderr, "parse model file %s failed\n", path.c_str());
    ret = Status_Invalid_Format;
  }
  in_file.close();
  if (ret != Status_OK) return model;
  string cls_name = root.get("model", "").asString();
  int cls_num = root.get("cls_num", "0").asInt();
  model = Model::Create(cls_name, cls_num);
  if (model == nullptr) {
    fprintf(stderr, "create model failed: no model named %s\n",
            cls_name.c_str());
    return model;
  }

  ret = model->SetModelInfo(root);
  if (ret != Status_OK) {
    DeletePointer(model);
  } else {
    ret = model->SetModelParam(root);
  }
  if (ret != Status_OK) {
    DeletePointer(model);
  }
  return model;
}

void Model::GetModelInfo(Json::Value& root) const {
  root["model"] = this->name();
  root["cls_num"] = this->class_num();
  root["clf_num"] = this->clf_num();
}

int Model::SetModelInfo(const Json::Value& root) {
  try {
    Check(root.get("model", "").asString() == this->name());
    Check(root.get("cls_num", "").asInt() == this->class_num());
    Check(root.get("clf_num", "").asInt() == this->clf_num());
  } catch (invalid_argument& err) {
    fprintf(stderr, "set model info failed: %s\n", err.what());
    return Status_Invalid_Argument;
  }
  return Status_OK;
}

inline string Model::model_info() const {
  Json::Value root;
  this->GetModelInfo(root);

  Json::StyledWriter writer;
  return writer.write(root);
}

}  // namespace model
}  // namespace lsol

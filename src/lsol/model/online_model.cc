/*********************************************************************************
*     File Name           :     online_model.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-18 15:37]
*     Last Modified       :     [2016-02-18 23:26]
*     Description         :     online model
**********************************************************************************/

#include "lsol/model/online_model.h"
#include "lsol/util/str_util.h"

using namespace std;
using namespace lsol::pario;

namespace lsol {
namespace model {
OnlineModel::OnlineModel(int class_num, const std::string& type)
    : Model(class_num, type), bias_eta0_(0), dim_(1), eta_(1.f) {
  this->set_initial_t(0);
  this->lazy_update_ = false;
}

void OnlineModel::SetParameter(const std::string& name,
                               const std::string& value) {
  if (name == "bias_eta") {
    this->bias_eta0_ = stof(value);
    Check(bias_eta0_ >= 0);
  } else if (name == "t") {
    this->set_initial_t(stoi(value));
  } else if (name == "dim") {
    this->update_dim(stoi(value));
  } else if (name == "lazy_update") {
    this->lazy_update_ = value == "true" ? true : false;
  } else {
    Model::SetParameter(name, value);
  }
}

float OnlineModel::Train(DataIter& data_iter) {
  fprintf(stdout, "Model Information: \n%s\n", this->model_info().c_str());
  this->BeginTrain();
  float err_num(0);
  size_t data_num = 0;
  size_t show_step = 1;  // show information every show_step
  size_t show_count = 2;

  fprintf(stdout,
          "Training Process....\nIterate No.\t\tError Rate\t\tUpdate No.\n");

  float* predicts = new float[this->clf_num()];
  MiniBatch* mb = nullptr;
  while (1) {
    mb = data_iter.Next(mb);
    if (mb == nullptr) break;
    // data_num += mb->size();
    for (int i = 0; i < mb->size(); ++i) {
      DataPoint& x = (*mb)[i];
      this->PreProcess(x);
      // predict
      label_t label = this->Iterate(x, predicts);
      if (label != x.label()) {
        // printf("%d\n", data_num + 1);
        err_num++;
      }
      ++data_num;

      if (data_num >= show_count) {
        fprintf(stdout, "%llu\t\t\t%.6f\t\t%llu\n", data_num,
                float(double(err_num) / data_num), this->update_num());
        show_count = (size_t(1) << ++show_step);
      }
    }
  }

  fprintf(stdout, "%llu\t\t\t%.6f\t\t%llu\n", data_num,
          float(double(err_num) / data_num), this->update_num());
  this->EndTrain();
  delete[] predicts;

  return float(double(err_num) / data_num);
}

label_t OnlineModel::Iterate(const pario::DataPoint& x, float* predict) {
  this->update_dim(x.dim());
  ++this->cur_iter_num_;
  return 0;
}

void OnlineModel::GetModelInfo(Json::Value& root) const {
  Model::GetModelInfo(root);
  root["online"]["bias_eta"] = this->bias_eta0_;
  root["online"]["t"] = this->cur_iter_num_;
  root["online"]["dim"] = this->dim_;
  root["online"]["lazy_update"] = this->lazy_update_ ? "true" : "false";
}

int OnlineModel::SetModelInfo(const Json::Value& root) {
  Model::SetModelInfo(root);
  const Json::Value& online_settings = root["online"];
  if (online_settings.isNull()) {
    fprintf(stderr, "no online info found for online model\n");
    return Status_Invalid_Format;
  }
  try {
    for (Json::Value::const_iterator iter = online_settings.begin();
         iter != online_settings.end(); ++iter) {
      this->SetParameter(iter.name(), iter->asString());
    }
  } catch (std::invalid_argument& err) {
    fprintf(stderr, "set model info failed: %s\n", err.what());
    return Status_Invalid_Argument;
  }
  return Status_OK;
}

void OnlineModel::set_initial_t(int initial_t) {
  Check(initial_t >= 0);
  this->initial_t_ = initial_t;
  this->cur_iter_num_ = initial_t;
}
}  // namespace model
}  // namespace lsol

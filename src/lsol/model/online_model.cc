/*********************************************************************************
*     File Name           :     online_model.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-18 15:37]
*     Last Modified       :     [2016-02-18 23:26]
*     Description         :     online model
**********************************************************************************/

#include "lsol/model/online_model.h"

#include <cmath>

#include "lsol/util/str_util.h"

using namespace std;
using namespace lsol::pario;

namespace lsol {
namespace model {
OnlineModel::OnlineModel(int class_num, const std::string& type)
    : Model(class_num, type), eta0_(10), dim_(1), aggressive_(true) {
  this->set_power_t(0.5);
  this->set_initial_t(0);
}

void OnlineModel::SetParameter(const std::string& name,
                               const std::string& value) {
  if (name == "power_t") {
    this->set_power_t(stof(value));
  } else if (name == "eta") {
    this->eta0_ = stof(value);
    Check(eta0_ >= 0);
  } else if (name == "t0") {
    this->set_initial_t(stoi(value));
  } else if (name == "aggressive") {
    this->aggressive_ = value == "true" ? true : false;
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

  fprintf(stdout, "Training Process....\nIterate No.\t\t\tError Rate\t\t\n");

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
        fprintf(stdout, "%llu\t\t\t\t%.6f\n", data_num,
               float(double(err_num) / data_num));
        show_count = (size_t(1) << ++show_step);
      }
    }
  }
  fprintf(stdout, "%llu\t\t\t\t%.6f\n", data_num, float(double(err_num) / data_num));
  this->EndTrain();

  delete[] predicts;
  return float(double(err_num) / data_num);
}

label_t OnlineModel::Iterate(const pario::DataPoint& x, float* predict) {
  this->update_dim(x.dim());
  ++this->cur_iter_num_;
  this->eta_ = this->eta0_ / this->pow_(this->cur_iter_num_, this->power_t_);
  return 0;
}

void OnlineModel::GetModelInfo(Json::Value& root) const {
  Model::GetModelInfo(root);
  root["online"]["power_t"] = this->power_t_;
  root["online"]["eta"] = this->eta0_;
  root["online"]["t0"] = this->initial_t_;
  root["online"]["t"] = this->cur_iter_num_;
  root["online"]["dim"] = this->dim_;
}

int OnlineModel::SetModelInfo(const Json::Value& root) {
  Model::SetModelInfo(root);
  const Json::Value& online_settings = root["online"];
  if (online_settings.isNull()) {
    fprintf(stderr, "no online info found for online model\n");
    return Status_Invalid_Format;
  }
  this->set_power_t(online_settings["power_t"].asFloat());
  this->eta0_ = online_settings["eta"].asFloat();
  this->set_initial_t(online_settings["t0"].asInt());
  this->cur_iter_num_ = online_settings["t"].asInt();
  this->update_dim(online_settings["dim"].asInt() - 1);
  return Status_OK;
}

// calculate power t
float pow_const(int iter, float power_t) { return 1; }
float pow_sqrt(int iter, float power_t) { return sqrtf(float(iter)); }
float pow_linear(int iter, float power_t) { return float(iter); }
float pow_general(int iter, float power_t) {
  return powf((float)iter, power_t);
}

void OnlineModel::set_power_t(float power_t) {
  Check(power_t >= 0);
  this->power_t_ = power_t;
  if (power_t == 0)
    this->pow_ = pow_const;
  else if (power_t == 0.5)
    this->pow_ = pow_sqrt;
  else if (power_t == 1)
    this->pow_ = pow_linear;
  else
    this->pow_ = pow_general;
}

void OnlineModel::set_initial_t(int initial_t) {
  Check(initial_t >= 0);
  this->initial_t_ = initial_t;
  this->cur_iter_num_ = initial_t;
}
}  // namespace model
}  // namespace lsol

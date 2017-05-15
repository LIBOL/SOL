/*********************************************************************************
*     File Name           :     online_model.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-18 15:37]
*     Last Modified       :     [2017-05-09 16:29]
*     Description         :     online model
**********************************************************************************/

#include "sol/model/online_model.h"
#include "sol/util/str_util.h"
#include "sol/util/util.h"

#include <cmath>
#include <iostream>
#include <sstream>

using namespace std;
using namespace sol::pario;

namespace sol {
namespace model {

OnlineModel::OnlineModel(int class_num, const std::string& type)
    : Model(class_num, type), bias_eta0_(0), dim_(1), eta_(1.f) {
  this->cur_data_num_ = 0;
  this->cur_err_num_ = 0;
  this->set_initial_t(0);
  this->lazy_update_ = false;
  // active learning
  this->active_smoothness_ = 0;
  // cost sensitive learning
  this->cost_sensitive_learning_ = false;
}

void OnlineModel::SetParameter(const std::string& name,
                               const std::string& value) {
  if (name == "bias_eta") {
    this->bias_eta0_ = stof(value);
    Check(bias_eta0_ >= 0);
  } else if (name == "t") {
    this->set_initial_t(stoi(value));
    if (this->regularizer_ != nullptr) {
      this->regularizer_->SetParameter("t0", value);
    }
  } else if (name == "dim") {
    this->update_dim(stoi(value));
  } else if (name == "lazy_update") {
    this->lazy_update_ = value == "true" ? true : false;
  } else if (name == "active_smoothness") {
    this->active_smoothness_ = stof(value);
    Check(active_smoothness_ > 0);
  } else if (name == "cost_margin") {
    if (this->clf_num_ != 1) {
      throw invalid_argument(
          "cost sensitive learning is only allowed in binary classification "
          "yet");
    }
    this->cost_margin_ = stof(value);
    Check(cost_margin_ > 0);
    this->cost_sensitive_learning_ = true;
    this->require_reinit_ = true;
  } else {
    Model::SetParameter(name, value);
  }
}

void OnlineModel::BeginTrain() {
  if (this->cost_sensitive_learning_) {
    if ((this->loss_->type() & loss::Loss::Type::HINGE) == 0 ||
        (this->loss_->type() & loss::Loss::Type::BC) == 0) {
      throw invalid_argument(
          "only hinge-based loss functions of binary classification are "
          "allowed in cost sensitive learning");
    }
  }

  Model::BeginTrain();
}

float OnlineModel::Train(DataIter& data_iter) {
  if (this->require_reinit_) {
    // re-init the model
    try {
      this->BeginTrain();
    } catch (invalid_argument& err) {
      fprintf(stderr, "%s\n", err.what());
      return 0;
    }
  }

  size_t next_show_time = size_t(-1);

  if (this->iter_displayer_ != nullptr) {
    if (this->iter_callback_ == DefaultIterateFunction &&
        this->cur_data_num_ == 0) {
      cout << "Training Process....\nData No.\tIterate No.\tError Rate\tUpdate "
              "No.\n";
    }
    next_show_time = this->iter_displayer_->next();
  }

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
      if (this->Iterate(x, predicts) != x.label()) ++this->cur_err_num_;

      if (this->cur_data_num_ >= next_show_time &&
          this->iter_callback_ != nullptr) {
        float err_rate = float(this->cur_err_num_) / this->cur_data_num_;
        this->iter_callback_(this->iter_callback_user_context_,
                             this->cur_data_num_, this->cur_iter_num(),
                             this->update_num(), err_rate);
        next_show_time = this->iter_displayer_->next();
      }
    }
  }

  float err_rate = float(this->cur_err_num_) / this->cur_data_num_;
  if (this->iter_displayer_ != nullptr && this->iter_callback_ != nullptr) {
    this->iter_callback_(this->iter_callback_user_context_, this->cur_data_num_,
                         this->cur_iter_num(), this->update_num(), err_rate);
  }
  delete[] predicts;
  this->model_updated_ = true;

  return err_rate;
}

label_t OnlineModel::Iterate(const pario::DataPoint& x, float* predict) {
  this->update_dim(x.dim());
  ++this->cur_iter_num_;
  ++this->cur_data_num_;
  return 0;
}

void OnlineModel::GetModelInfo(Json::Value& root) const {
  Model::GetModelInfo(root);
  root["online"]["bias_eta"] = this->bias_eta0_;
  root["online"]["t"] = this->cur_iter_num_;
  root["online"]["dim"] = this->dim_;
  root["online"]["lazy_update"] = this->lazy_update_ ? "true" : "false";
  if (this->active_smoothness_ >= 1) {
    root["online"]["active_smoothness"] = this->active_smoothness_;
  }
  if (this->cost_sensitive_learning_) {
    root["online"]["cost_margin"] = this->cost_margin_;
  }
}

int OnlineModel::SetModelInfo(const Json::Value& root) {
  Model::SetModelInfo(root);
  const Json::Value& online_settings = root["online"];
  if (online_settings.isNull()) {
    cerr << "no online info found for online model\n";
    return Status_Invalid_Format;
  }
  try {
    for (Json::Value::const_iterator iter = online_settings.begin();
         iter != online_settings.end(); ++iter) {
      this->SetParameter(iter.name(), iter->asString());
    }
  } catch (std::invalid_argument& err) {
    cerr << "set model info failed: " << err.what() << "\n";
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
}  // namespace sol

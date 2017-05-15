/*********************************************************************************
*     File Name           :     stochastic_model.cc
*     Created By          :     yuewu
*     Creation Date       :     [2017-05-09 15:10]
*     Last Modified       :     [2017-05-09 17:39]
*     Description         :
**********************************************************************************/

#include "sol/model/stochastic_model.h"
#include "sol/util/str_util.h"
#include "sol/util/util.h"

#include <cmath>
#include <iostream>
#include <sstream>

using namespace std;
using namespace sol::pario;

namespace sol {
namespace model {

StochasticModel::StochasticModel(int class_num, const std::string& type)
    : Model(class_num, type),
      bias_eta0_(0),
      cur_batch_num_(0),
      cur_err_num_(0),
      dim_(1),
      eta_(1.f) {
  this->set_initial_t(0);
}

void StochasticModel::SetParameter(const std::string& name,
                                   const std::string& value) {
  switch (str2int(name)) {
    case "bias_eta"_I:
      this->bias_eta0_ = stof(value);
      Check(bias_eta0_ >= 0);
      break;
    case "t"_I:
      this->set_initial_t(stoi(value));
      if (this->regularizer_ != nullptr) {
        this->regularizer_->SetParameter("t0", value);
      }
      break;
    case "dim"_I:
      this->update_dim(stoi(value));
      break;
    default:
      Model::SetParameter(name, value);
      break;
  }
}

float StochasticModel::Train(DataIter& data_iter) {
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
        this->cur_batch_num_ == 0) {
      cout << "Training Process....\nData No.\tIterate No.\tError/Loss\tUpdate "
              "No.\n";
    }
    next_show_time = this->iter_displayer_->next();
  }

  MiniBatch* mb = nullptr;
  int max_batch_size = 0;
  label_t* predicts = nullptr;
  float* scores = nullptr;
  float err = 0;

  while (true) {
    mb = data_iter.Next(mb);
    if (mb == nullptr) break;
    // preprocess
    for (int i = 0; i < mb->size(); ++i) {
      this->PreProcess((*mb)[i]);
    }

    if (mb->size() > max_batch_size) {
      if (predicts != nullptr) delete[] predicts;
      if (scores != nullptr) delete[] scores;
      max_batch_size = mb->size();
      predicts = new label_t[max_batch_size];
      scores = new float[this->clf_num() * max_batch_size];
    }

    err = this->Iterate(*mb, predicts, scores);
    if (this->cur_batch_num_ >= next_show_time &&
        this->iter_callback_ != nullptr) {
      this->iter_callback_(this->iter_callback_user_context_,
                           this->cur_batch_num_, this->cur_iter_num(),
                           this->update_num(), err);
      next_show_time = this->iter_displayer_->next();
    }
  }

  if (this->iter_displayer_ != nullptr && this->iter_callback_ != nullptr) {
    this->iter_callback_(this->iter_callback_user_context_,
                         this->cur_batch_num_, this->cur_iter_num(),
                         this->update_num(), err);
  }
  delete[] predicts;
  delete[] scores;
  this->model_updated_ = true;

  return err;
}

float StochasticModel::Iterate(const pario::MiniBatch& mb, label_t* predicts,
                               float* scores) {
  index_t new_dim = 0;
  for (int i = 0; i < mb.size(); ++i) {
    if (mb[i].dim() > new_dim) new_dim = mb[i].dim();
  }
  this->update_dim(new_dim);
  ++this->cur_iter_num_;
  ++this->cur_batch_num_;
  return 0;
}

void StochasticModel::GetModelInfo(Json::Value& root) const {
  Model::GetModelInfo(root);
  root["stochastic"]["bias_eta"] = this->bias_eta0_;
  root["stochastic"]["t"] = this->cur_iter_num_;
  root["stochastic"]["dim"] = this->dim_;
}

int StochasticModel::SetModelInfo(const Json::Value& root) {
  Model::SetModelInfo(root);
  const Json::Value& settings = root["stochastic"];
  if (settings.isNull()) {
    cerr << "no settings found for stochatic model\n";
    return Status_Invalid_Format;
  }
  try {
    for (Json::Value::const_iterator iter = settings.begin();
         iter != settings.end(); ++iter) {
      this->SetParameter(iter.name(), iter->asString());
    }
  } catch (std::invalid_argument& err) {
    cerr << "set model info failed: " << err.what() << "\n";
    return Status_Invalid_Argument;
  }
  return Status_OK;
}

void StochasticModel::set_initial_t(int initial_t) {
  Check(initial_t >= 0);
  this->initial_t_ = initial_t;
  this->cur_iter_num_ = initial_t;
}
}  // namespace model
}  // namespace sol

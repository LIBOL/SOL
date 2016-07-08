/*********************************************************************************
*     File Name           :     online_model.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-18 15:37]
*     Last Modified       :     [2016-02-18 23:26]
*     Description         :     online model
**********************************************************************************/

#include "lsol/model/online_model.h"
#include "lsol/util/str_util.h"
#include "lsol/util/util.h"

#include <sstream>

using namespace std;
using namespace lsol::pario;

namespace lsol {
namespace model {

class ExpIterDisplayer : public OnlineModel::IterDisplayer {
 public:
  ExpIterDisplayer(size_t base = 2)
      : next_show_time_(base), base_(base), show_step_(1) {}

  virtual inline size_t next_show_time() { return next_show_time_; }
  virtual inline void next() {
    ++show_step_;
    next_show_time_ = size_t(pow(double(this->base_), this->show_step_));
  }

 protected:
  size_t next_show_time_;
  size_t base_;
  size_t show_step_;
};

class StepIterDisplayer : public OnlineModel::IterDisplayer {
 public:
  StepIterDisplayer(size_t step = 2) : next_show_time_(step), step_(step) {}

  virtual size_t next_show_time() { return next_show_time_; }
  virtual void next() { this->next_show_time_ += this->step_; }

 protected:
  size_t next_show_time_;
  size_t step_;
};

OnlineModel::OnlineModel(int class_num, const std::string& type)
    : Model(class_num, type),
      bias_eta0_(0),
      dim_(1),
      eta_(1.f),
      iter_displayer_(nullptr) {
  this->set_initial_t(0);
  this->lazy_update_ = false;
  this->iter_displayer_ = new ExpIterDisplayer(2);
}

OnlineModel::~OnlineModel() { DeletePointer(this->iter_displayer_); }

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
  } else if (name == "exp_show") {
    DeletePointer(this->iter_displayer_);
    this->iter_displayer_ = new ExpIterDisplayer(stoi(value));
  } else if (name == "step_show") {
    DeletePointer(this->iter_displayer_);
    this->iter_displayer_ = new StepIterDisplayer(stoi(value));
  } else {
    Model::SetParameter(name, value);
  }
}

float OnlineModel::Train(DataIter& data_iter) {
  fprintf(stdout, "Model Information: \n%s\n", this->model_info().c_str());
  this->BeginTrain();
  ostringstream log_oss;
  size_t err_num(0);
  size_t data_num = 0;
  size_t next_show_time = size_t(-1);

  if (this->iter_displayer_ != nullptr) {
    next_show_time = this->iter_displayer_->next_show_time();
    fprintf(stdout,
            "Training Process....\nIterate No.\t\tError Rate\t\tUpdate No.\n");
    log_oss << "Iterate No.\tError No.\tUpdate No.\n";
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
      if (this->Iterate(x, predicts) != x.label()) err_num++;
      ++data_num;

      if (data_num >= next_show_time) {
        float err_rate = float(err_num) / data_num;
        fprintf(stdout, "%llu\t\t\t%.6f\t\t%llu\n", data_num, err_rate,
                this->update_num());
        log_oss << data_num << "\t" << err_rate << "\t" << this->update_num()
                << "\n";
        this->iter_displayer_->next();
        next_show_time = this->iter_displayer_->next_show_time();
      }
    }
  }

  if (this->iter_displayer_ != nullptr) {
    float err_rate = float(err_num) / data_num;
    fprintf(stdout, "%llu\t\t\t%.6f\t\t%llu\n", data_num, err_rate,
            this->update_num());
    log_oss << data_num << "\t" << err_rate << "\t" << this->update_num()
            << "\n";
  }
  this->EndTrain();
  delete[] predicts;
  this->train_log_ = log_oss.str();

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

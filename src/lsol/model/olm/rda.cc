/*********************************************************************************
*     File Name           :     rda.cc
*     Created By          :     yuewu
*     Description         :     Regularized Dual Averaging
**********************************************************************************/
#include "lsol/model/olm/rda.h"

#include <cmath>
#include <iostream>

using namespace std;
using namespace lsol;
using namespace lsol::math;
using namespace lsol::math::expr;

namespace lsol {

namespace model {

RDA::RDA(int class_num)
    : OnlineLinearModel(class_num), sigma_(1.f), ut_(nullptr) {
  this->ut_ = new math::Vector<real_t>[this->clf_num_];
  for (int i = 0; i < this->clf_num_; ++i) {
    this->ut_[i].resize(this->dim_);
    this->ut_[i] = 0;
  }
}
RDA::~RDA() { DeleteArray(this->ut_); }

void RDA::SetParameter(const std::string& name, const std::string& value) {
  if (name == "sigma") {
    this->sigma_ = stof(value);
    Check(sigma_ > 0);
  } else {
    OnlineLinearModel::SetParameter(name, value);
  }
}

label_t RDA::TrainPredict(const pario::DataPoint& dp, float* predicts) {
  const auto& x = dp.data();
  real_t t = real_t(cur_iter_num_ - 1 + 1e-20);
  eta_ = 1.f / (t * sigma_);
  for (int c = 0; c < this->clf_num_; ++c) {
    w(c) = -eta_ * ut_[c].slice(x);
    w(c)[0] = -bias_eta() * ut_[c][0];
  }

  return OnlineLinearModel::TrainPredict(dp, predicts);
}

void RDA::Update(const pario::DataPoint& dp, const float*, float loss) {
  const auto& x = dp.data();
  for (int c = 0; c < this->clf_num_; ++c) {
    if (g(c) == 0) continue;

    ut_[c] += g(c) * x;
    ut_[c][0] += g(c);
  }
}

void RDA::EndTrain() {
  real_t t = real_t(cur_iter_num_ + 1e-20);
  eta_ = 1.f / (t * sigma_);
  for (int c = 0; c < this->clf_num_; ++c) {
    w(c) = -eta_ * ut_[c];
    w(c)[0] = -bias_eta() * ut_[c][0];
  }
  OnlineLinearModel::EndTrain();
}

void RDA::update_dim(index_t dim) {
  if (dim > this->dim_) {
    for (int c = 0; c < this->clf_num_; ++c) {
      this->ut_[c].resize(dim);
      this->ut_[c].slice_op([](real_t& val) { val = 0; }, this->dim_);
    }

    OnlineLinearModel::update_dim(dim);
  }
}

void RDA::GetModelInfo(Json::Value& root) const {
  OnlineLinearModel::GetModelInfo(root);
  root["online"]["sigma"] = this->sigma_;
}

void RDA::GetModelParam(Json::Value& root) const {
  OnlineLinearModel::GetModelParam(root);

  for (int c = 0; c < this->clf_num_; ++c) {
    ostringstream oss_name;
    oss_name << "ut[" << c << "]";
    ostringstream oss_value;
    oss_value << this->ut_[c] << "\n";
    root[oss_name.str()] = oss_value.str();
  }
}

int RDA::SetModelParam(const Json::Value& root) {
  OnlineLinearModel::SetModelParam(root);

  for (int c = 0; c < this->clf_num_; ++c) {
    ostringstream oss_name;
    oss_name << "ut[" << c << "]";
    istringstream iss_value(root[oss_name.str()].asString());
    iss_value >> this->ut_[c];
  }

  return Status_OK;
}

RegisterModel(RDA, "rda", "l2^2 regularized dual averaging");

RDA_L1::RDA_L1(int class_num) : RDA(class_num) { this->regularizer_ = &l1_; }

label_t RDA_L1::TrainPredict(const pario::DataPoint& dp, float* predicts) {
  const auto& x = dp.data();
  real_t t = real_t(cur_iter_num_ - 1.f + 1e-20);
  real_t trunc_thresh = l1_.lambda() * t;
  eta_ = 1.f / (t * sigma_);
  for (int c = 0; c < this->clf_num_; ++c) {
    // trucate weights
    w(c) = -eta_ * expr::truncate(ut_[c].slice(x), trunc_thresh);
    // truncate bias
    w(c)[0] = -bias_eta() * expr::truncate(ut_[c][0], trunc_thresh);
  }

  return OnlineLinearModel::TrainPredict(dp, predicts);
}

void RDA_L1::EndTrain() {
  real_t t = real_t(cur_iter_num_ + 1e-20);
  real_t trunc_thresh = l1_.lambda() * t;
  eta_ = 1.f / (t * sigma_);
  for (int c = 0; c < this->clf_num_; ++c) {
    w(c) = -eta_ * expr::truncate(ut_[c], trunc_thresh);
    w(c)[0] = -bias_eta() * expr::truncate(ut_[c][0], trunc_thresh);
  }
  OnlineLinearModel::EndTrain();
}

RegisterModel(RDA_L1, "rda-l1", "mixed l1-l2^2 regularized dual averaging");

ERDA_L1::ERDA_L1(int class_num) : RDA(class_num), rou_(0.f) {
  this->regularizer_ = &l1_;
}

void ERDA_L1::SetParameter(const std::string& name, const std::string& value) {
  if (name == "rou") {
    this->rou_ = stof(value);
    Check(rou_ >= 0);
  } else {
    RDA::SetParameter(name, value);
  }
}

label_t ERDA_L1::TrainPredict(const pario::DataPoint& dp, float* predicts) {
  const auto& x = dp.data();
  // gamma is represented by sigma
  real_t t = real_t(cur_iter_num_ - 1 + 1e-20);
  eta_ = 1.f / (sqrtf(t) * sigma_);
  real_t trunc_thresh = l1_.lambda() * t + sigma_ * rou_ * sqrtf(t);
  for (int c = 0; c < this->clf_num_; ++c) {
    // trucate weights
    w(c) = -eta_ * expr::truncate(ut_[c].slice(x), trunc_thresh);
    // truncate bias
    w(c)[0] = -bias_eta() * expr::truncate(ut_[c][0], trunc_thresh);
  }

  return OnlineLinearModel::TrainPredict(dp, predicts);
}

void ERDA_L1::EndTrain() {
  real_t t = real_t(cur_iter_num_ + 1e-20);
  eta_ = 1.f / (sqrtf(t) * sigma_);
  real_t trunc_thresh = l1_.lambda() * t + sigma_ * rou_ * sqrtf(t);
  for (int c = 0; c < this->clf_num_; ++c) {
    w(c) = -eta_ * expr::truncate(ut_[c], trunc_thresh);
    w(c)[0] = -bias_eta() * expr::truncate(ut_[c][0], trunc_thresh);
  }
  OnlineLinearModel::EndTrain();
}

void ERDA_L1::GetModelInfo(Json::Value& root) const {
  RDA::GetModelInfo(root);
  root["online"]["rou"] = this->rou_;
}

RegisterModel(ERDA_L1, "erda-l1",
              "mixed l1-l2^2 enhanced regularized dual averaging");

}  // namespace model
}  // namespace lsol

/*********************************************************************************
*     File Name           :     sop.cc
*     Created By          :     yuewu
*     Description         :     second order perceptron
**********************************************************************************/
#include "lsol/model/olm/sop.h"
#include "lsol/loss/bool_loss.h"

using namespace std;
using namespace lsol;

namespace lsol {

namespace model {
SOP::SOP(int class_num) : OnlineLinearModel(class_num), a_(1.f) {
  this->S_.resize(this->dim_);
  this->X_.resize(this->dim_);
  this->X_ = this->a_;

  this->v_.resize(this->dim_);
  this->v_ = 0;
  // loss
  if (class_num == 2) {
    this->SetParameter("loss", "bool");
  } else {
    throw invalid_argument("SOP does not support multiclass classification!");
  }
}

void SOP::SetParameter(const std::string& name, const std::string& value) {
  if (name == "a") {
    this->a_ = stof(value);
    this->X_ = this->a_;
  } else if (name == "loss") {
    OnlineLinearModel::SetParameter(name, value);
    if ((this->loss_->type() & loss::Loss::Type::BOOL) == 0) {
      throw invalid_argument("only bool-based loss functions are allowed");
    }
  } else {
    OnlineLinearModel::SetParameter(name, value);
  }
}

label_t SOP::Predict(const pario::DataPoint& dp, float* predicts) {
  const auto& x = dp.data();
  S_ = X_;
  S_ += L2(x);
  w(0) = v_ / S_;
  return OnlineLinearModel::Predict(dp, predicts);
}

void SOP::Update(const pario::DataPoint& dp, const float*, float) {
  const auto& x = dp.data();
  this->eta_ = 1.f;

  v_ -= g(0) * x;
  v_[0] -= bias_eta() * g(0);
  X_ = S_;
}

void SOP::update_dim(index_t dim) {
  if (dim >= this->dim_) {
    this->S_.resize(dim);

    this->X_.resize(dim);
    float a = this->a_;
    this->X_.slice_op([a](real_t& val) { val = a; }, this->dim_);

    this->v_.resize(dim);
    this->v_.slice_op([](real_t& val) { val = 0.f; }, this->dim_);
    OnlineLinearModel::update_dim(dim);
  }
}

void SOP::GetModelInfo(Json::Value& root) const {
  OnlineLinearModel::GetModelInfo(root);
  root["online"]["a"] = this->a_;
}

void SOP::GetModelParam(Json::Value& root) const {
  OnlineLinearModel::GetModelParam(root);

  ostringstream oss_X;
  oss_X << this->X_ << "\n";
  root["covariance"] = oss_X.str();

  ostringstream oss_v;
  oss_v << this->v_ << "\n";
  root["v"] = oss_v.str();
}

int SOP::SetModelParam(const Json::Value& root) {
  OnlineLinearModel::SetModelParam(root);

  istringstream iss_X(root["covariance"].asString());
  iss_X >> this->X_;

  istringstream iss_v(root["v"].asString());
  iss_v >> this->v_;
  return Status_OK;
}

RegisterModel(SOP, "sop", "second order perceptron");

}  // namespace model
}  // namespace lsol

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
SOP::SOP(int class_num) : OnlineLinearModel(class_num), a_(1.f), v_(nullptr) {
  this->S_.resize(this->dim_);
  this->X_.resize(this->dim_);

  this->v_ = new math::Vector<real_t>[this->clf_num_];
  for (int i = 0; i < this->clf_num_; ++i) {
    this->v_[i].resize(this->dim_);
    this->v_[i] = 0;
  }
  // loss
  if (class_num == 2) {
    this->SetParameter("loss", "bool");
  } else {
    this->SetParameter("loss", "maxscore-bool");
  }
}

SOP::~SOP() { DeleteArray(this->v_); }

void SOP::SetParameter(const std::string& name, const std::string& value) {
  if (name == "a") {
    this->a_ = stof(value);
  } else if (name == "loss") {
    OnlineLinearModel::SetParameter(name, value);
    if ((this->loss_->type() & loss::Loss::Type::BOOL) == 0) {
      throw invalid_argument("only bool-based loss functions are allowed");
    }
  } else {
    OnlineLinearModel::SetParameter(name, value);
  }
}

void SOP::BeginTrain() {
  OnlineLinearModel::BeginTrain();
  this->X_ = this->a_;
}

label_t SOP::Predict(const pario::DataPoint& dp, float* predicts) {
  const auto& x = dp.data();
  S_ = X_;
  S_ += L2(x);
  for (int c = 0; c < this->clf_num_; ++c) {
    w(c) = v_[c] / S_;
  }
  return OnlineLinearModel::Predict(dp, predicts);
}

void SOP::Update(const pario::DataPoint& dp, const float*, float) {
  const auto& x = dp.data();
  this->eta_ = 1.f;

  for (int c = 0; c < this->clf_num_; ++c) {
    if (g(c) == 0) continue;
    v_[c] -= g(c) * x;
    // update bias
    v_[c][0] -= bias_eta() * g(c);
  }
  X_ = S_;
}

void SOP::update_dim(index_t dim) {
  if (dim >= this->dim_) {
    this->S_.resize(dim);
    for (real_t* iter = this->S_.begin() + this->dim_; iter != this->S_.end();
         ++iter)
      *iter = this->a_;

    this->X_.resize(dim);
    for (real_t* iter = this->X_.begin() + this->dim_; iter != this->X_.end();
         ++iter)
      *iter = this->a_;

    for (int i = 0; i < this->clf_num_; ++i) {
      this->v_[i].resize(dim);
      // set the new value to zero
      for (real_t* iter = this->v_[i].begin() + this->dim_;
           iter != this->v_[i].end(); ++iter)
        *iter = 0;
    }
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
  root["diag_covariance"] = oss_X.str();

  ostringstream oss_v;
  for (int c = 0; c < this->clf_num_; ++c) {
    oss_v << this->v_[c] << "\n";
  }
  root["v"] = oss_v.str();
}

int SOP::SetModelParam(const Json::Value& root) {
  OnlineLinearModel::SetModelParam(root);

  istringstream iss_X(root["diag_covariance"].asString());
  iss_X >> this->X_;

  istringstream iss_v(root["v"].asString());
  for (int c = 0; c < this->clf_num_; ++c) iss_v >> this->v_[c];
  return Status_OK;
}

RegisterModel(SOP, "sop", "second order perceptron");

}  // namespace model
}  // namespace lsol

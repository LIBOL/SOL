/*********************************************************************************
*     File Name           :     eccw.cc
*     Created By          :     yuewu
*     Description         :     Exact Convex Confidence Weighted Online Learning
**********************************************************************************/

#include "lsol/model/olm/eccw.h"
#include <cmath>

using namespace std;
using namespace lsol;
using namespace lsol::math;

namespace lsol {

namespace model {
ECCW::ECCW(int class_num)
    : OnlineLinearModel(class_num), hinge_base_(nullptr), a_(1.f) {
  this->Sigma_.resize(this->dim_);

  this->Sigma_ = this->a_;
  this->set_phi(0.5244f);

  // loss
  if (class_num == 2) {
    this->SetParameter("loss", "hinge");
  } else {
    this->SetParameter("loss", "maxscore-hinge");
  }
}

void ECCW::SetParameter(const std::string& name, const std::string& value) {
  if (name == "a") {
    this->a_ = stof(value);
    this->Sigma_ = this->a_;
  } else if (name == "phi") {
    this->set_phi(stof(value));
  } else if (name == "loss") {
    OnlineLinearModel::SetParameter(name, value);
    if ((this->loss_->type() & loss::Loss::Type::HINGE) == 0) {
      throw invalid_argument("only hinge-based loss functions are allowed");
    }
    this->hinge_base_ = static_cast<loss::HingeBase*>(this->loss_);
  } else {
    OnlineLinearModel::SetParameter(name, value);
  }
}

label_t ECCW::Predict(const pario::DataPoint& dp, float* predicts) {
  const auto& x = dp.data();
  label_t predict_label = OnlineLinearModel::Predict(dp, predicts);
  vi_ = expr::dotmul(this->Sigma_, L2(x));
  if (bias_eta0_ != 0) vi_ += Sigma_[0];

  this->hinge_base_->set_margin(phi_ * vi_);
  return predict_label;
}

void ECCW::Update(const pario::DataPoint& dp, const float*, float loss) {
  const auto& x = dp.data();
  float mi = phi_ * vi_ - loss;
  float tmp = mi * phi_ * phi_;
  float alpha_i =
      (-mi * psi_ + sqrtf(tmp * tmp * 0.25f + vi_ * phi_ * phi_ * xi_)) /
      (vi_ * xi_);
  float ui =
      0.5f * (-alpha_i * vi_ * phi_ +
              sqrtf(alpha_i * alpha_i * vi_ * vi_ * phi_ * phi_ + 4.f * vi_));
  ui *= ui;
  float beta_i = alpha_i * phi_ / (sqrtf(ui) + vi_ * alpha_i * phi_);

  this->eta_ = alpha_i;
  for (int c = 0; c < this->clf_num_; ++c) {
    if (g(c) == 0) continue;
    w(c) -= eta_ * g(c) * this->Sigma_ * x;
    // update bias
    w(c)[0] -= bias_eta() * g(c) * this->Sigma_[0];
  }
  tmp = alpha_i * phi_ * sqrtf(ui);
  this->Sigma_ /= (1.f + tmp * this->Sigma_ * L2(x));
  this->Sigma_[0] /= (1.f + tmp * this->Sigma_[0]);
}

void ECCW::update_dim(index_t dim) {
  if (dim >= this->dim_) {
    this->Sigma_.resize(dim);
    for (real_t* iter = this->Sigma_.begin() + this->dim_;
         iter != this->Sigma_.end(); ++iter)
      *iter = this->a_;

    OnlineLinearModel::update_dim(dim);
  }
}
void ECCW::set_phi(float phi) {
  this->phi_ = phi;
  this->psi_ = 1 + phi * phi / 2.f;
  this->xi_ = 1 + phi * phi;
}

void ECCW::GetModelInfo(Json::Value& root) const {
  OnlineLinearModel::GetModelInfo(root);
  root["online"]["a"] = this->a_;
  root["online"]["phi"] = this->phi_;
}

void ECCW::GetModelParam(Json::Value& root) const {
  OnlineLinearModel::GetModelParam(root);

  ostringstream oss_Sigma;
  oss_Sigma << this->Sigma_ << "\n";
  root["Sigma"] = oss_Sigma.str();
}

int ECCW::SetModelParam(const Json::Value& root) {
  OnlineLinearModel::SetModelParam(root);

  istringstream iss_Sigma(root["Sigma"].asString());
  iss_Sigma >> this->Sigma_;
  return Status_OK;
}

RegisterModel(ECCW, "eccw", "exact convex confidence weighted online learning");

}  // namespace model
}  // namespace lsol

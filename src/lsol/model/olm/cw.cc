/*********************************************************************************
*     File Name           :     cw.cc
*     Created By          :     yuewu
*     Description         :     Confidence Weighted Online Learning
**********************************************************************************/

#include "lsol/model/olm/cw.h"

using namespace std;
using namespace lsol;
using namespace lsol::math;

namespace lsol {

namespace model {
CW::CW(int class_num)
    : OnlineLinearModel(class_num),
      a_(1.f),
      phi_(0.5244f),
      hinge_base_(nullptr) {
  this->Sigma_.resize(this->dim_);
  this->Sigma_ = this->a_;
  // loss
  if (class_num == 2) {
    this->SetParameter("loss", "hinge");
  } else {
    this->SetParameter("loss", "maxscore-hinge");
  }
}

void CW::SetParameter(const std::string& name, const std::string& value) {
  if (name == "a") {
    this->a_ = stof(value);
    this->Sigma_ = this->a_;
  } else if (name == "phi") {
    this->phi_ = stof(value);
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

label_t CW::Predict(const pario::DataPoint& dp, float* predicts) {
  const auto& x = dp.data();
  label_t predict_label = OnlineLinearModel::Predict(dp, predicts);
  this->Vi_ = expr::dotmul(this->Sigma_, L2(x));
  if (this->bias_eta0_ != 0) this->Vi_ += this->Sigma_[0];

  this->hinge_base_->set_margin(this->phi_ * this->Vi_);
  return predict_label;
}

void CW::Update(const pario::DataPoint& dp, const float*, float loss) {
  const auto& x = dp.data();
  float Mi = phi_ * Vi_ - loss;
  float tmp = (1 + 2 * phi_ * Mi);
  float alpha_i =
      (-(1 + 2 * phi_ * Mi) + sqrtf(tmp * tmp - 8 * phi_ * (Mi - phi_ * Vi_))) /
      (4 * phi_ * Vi_);

  this->eta_ = alpha_i;
  for (int c = 0; c < this->clf_num_; ++c) {
    if (g(c) == 0) continue;
    w(c) -= this->eta_ * g(c) * this->Sigma_ * x;
    // update bias
    w(c)[0] -= bias_eta() * g(c) * this->Sigma_[0];
  }
  tmp = 2 * alpha_i * phi_;
  this->Sigma_ /= (1.f + tmp * this->Sigma_ * L2(x));
  this->Sigma_[0] /= (1.f + tmp * this->Sigma_[0]);
}

void CW::update_dim(index_t dim) {
  if (dim >= this->dim_) {
    this->Sigma_.resize(dim);
    for (real_t* iter = this->Sigma_.begin() + this->dim_;
         iter != this->Sigma_.end(); ++iter)
      *iter = this->a_;

    OnlineLinearModel::update_dim(dim);
  }
}

void CW::GetModelInfo(Json::Value& root) const {
  OnlineLinearModel::GetModelInfo(root);
  root["online"]["a"] = this->a_;
  root["online"]["phi"] = this->phi_;
}

void CW::GetModelParam(Json::Value& root) const {
  OnlineLinearModel::GetModelParam(root);

  ostringstream oss_Sigma;
  oss_Sigma << this->Sigma_ << "\n";
  root["Sigma"] = oss_Sigma.str();
}

int CW::SetModelParam(const Json::Value& root) {
  OnlineLinearModel::SetModelParam(root);

  istringstream iss_Sigma(root["Sigma"].asString());
  iss_Sigma >> this->Sigma_;
  return Status_OK;
}

RegisterModel(CW, "cw", "confidence weighted online learning");

}  // namespace model
}  // namespace lsol

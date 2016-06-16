/*********************************************************************************
*     File Name           :     alma2.cc
*     Created By          :     yuewu
*     Description         :
*     Description         :     Approximate Large Margin Algorithm with norm 2
**********************************************************************************/

#include "lsol/model/olm/alma2.h"

using namespace std;
using namespace lsol::math::expr;

namespace lsol {

namespace model {

ALMA2::ALMA2(int class_num)
    : OnlineLinearModel(class_num), hinge_base_(nullptr) {
  this->C_ = sqrtf(2.f);
  this->SetParameter("p", "2");
  this->SetParameter("alpha", "0.9");
  // aggressive
  OnlineLinearModel::SetParameter("aggressive", "true");
  // loss
  if (class_num == 2) {
    this->SetParameter("loss", "hinge");
  } else {
    this->SetParameter("loss", "maxscore-hinge");
  }
  // norm
  OnlineLinearModel::SetParameter("norm", "L2");
  // k
  this->k_ = 1;
}

void ALMA2::SetParameter(const std::string& name, const std::string& value) {
  if (name == "p") {
    this->p_ = stoi(value);
    this->square_p1_ = std::sqrtf(float(this->p_ - 1));
    this->eta0_ = this->C_ / this->square_p1_;
  } else if (name == "alpha") {
    this->alpha_ = stof(value);
    this->B_ = 1 / this->alpha_;
    Check(alpha_ > 0);
    Check(alpha_ <= 1);
  } else if (name == "C") {
    this->C_ = stof(value);
    this->eta0_ = this->C_ / this->square_p1_;
  } else if (name == "k") {
    this->k_ = stoi(value);
    Check(this->k_ > 0);
  } else if (name == "power_t") {
    OnlineLinearModel::SetParameter(name, value);
    Check(this->power_t_ == 0.5);
  } else if (name == "aggressive") {
    OnlineLinearModel::SetParameter(name, value);
    Check(this->aggressive_ == true);
  } else if (name == "loss") {
    OnlineLinearModel::SetParameter(name, value);
    if ((this->loss_->type() & loss::Loss::Type::HINGE) == 0) {
      throw invalid_argument("only hinge-based loss functions are allowed");
    }
    this->hinge_base_ = static_cast<loss::HingeBase*>(this->loss_);
  } else if (name == "norm") {
    OnlineLinearModel::SetParameter(name, value);
    Check(this->norm_type_ == op::OpType::kL2);
  } else {
    OnlineLinearModel::SetParameter(name, value);
  }
}

void ALMA2::CalculateLearningRate() {
  this->eta_ = this->eta0_ / sqrtf(float(this->k_));
  this->bias_eta_ = this->bias_eta0_ * this->eta_;
}

void ALMA2::BeginTrain() {
  OnlineLinearModel::BeginTrain();
  float margin = (1.f - this->alpha_) * this->B_ * this->square_p1_ /
                 sqrtf(float(this->k_));
  this->hinge_base_->set_margin(margin);
}

void ALMA2::Update(const pario::DataPoint& x) {
  for (int c = 0; c < this->clf_num_; ++c) {
    if (this->gradients_[c] == 0) continue;
    math::Vector<real_t>& w = this->weights(c);
    w -= this->eta_ * this->gradients_[c] * x.data();
    // update bias
    w[0] -= this->bias_eta_ * this->gradients_[c];

    real_t w_norm = sqrt(reduce<op::plus>(L2(w)));
    if (w_norm > 1) w /= w_norm;
  }
  ++this->k_;
  float margin = (1.f - this->alpha_) * this->B_ * this->square_p1_ /
                 sqrtf(float(this->k_));
  this->hinge_base_->set_margin(margin);
}

void ALMA2::GetModelInfo(Json::Value& root) const {
  OnlineLinearModel::GetModelInfo(root);
  root["online"]["p"] = this->p_;
  root["online"]["alpha"] = this->alpha_;
  root["online"]["C"] = this->C_;
  root["online"]["k"] = this->k_;
}

RegisterModel(ALMA2, "alma2", "Approximate Large Margin Algorithm with norm 2");

}  // namespace model
}  // namespace lsol

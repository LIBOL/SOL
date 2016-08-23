/*********************************************************************************
*     File Name           :     alma2.cc
*     Created By          :     yuewu
*     Description         :
*     Description         :     Approximate Large Margin Algorithm with norm 2
**********************************************************************************/

#include "lsol/model/olm/alma2.h"
#include <cmath>

using namespace std;
using namespace lsol::math::expr;

namespace lsol {

namespace model {

ALMA2::ALMA2(int class_num)
    : OnlineLinearModel(class_num), hinge_base_(nullptr) {
  this->C_ = sqrtf(2.f);
  this->SetParameter("p", "2");
  this->SetParameter("alpha", "0.9");
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
    this->square_p1_ = sqrtf(float(this->p_ - 1));
    this->require_reinit_ = true;
  } else if (name == "alpha") {
    this->alpha_ = stof(value);
    this->B_ = 1 / this->alpha_;
    Check(alpha_ > 0);
    Check(alpha_ <= 1);
    this->require_reinit_ = true;
  } else if (name == "C") {
    this->C_ = stof(value);
  } else if (name == "k") {
    this->k_ = stoi(value);
    Check(this->k_ > 0);
    this->require_reinit_ = true;
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

void ALMA2::BeginTrain() {
  OnlineLinearModel::BeginTrain();
  float margin = (1.f - alpha_) * B_ * square_p1_ / sqrtf(float(k_));
  this->hinge_base_->set_margin(margin);
}

void ALMA2::Update(const pario::DataPoint& dp, const float*, float) {
  const auto& x = dp.data();
  this->eta_ = C_ / (square_p1_ * sqrtf(float(k_)));

  for (int c = 0; c < this->clf_num_; ++c) {
    if (g(c) == 0) continue;
    w(c) -= eta_ * g(c) * x;
    // update bias
    w(c)[0] -= bias_eta() * g(c);

    real_t w_norm = sqrt(Norm2(w(c)));
    if (w_norm > 1) w(c) /= w_norm;
  }
  ++k_;
  float margin = (1.f - alpha_) * B_ * square_p1_ / sqrtf(float(k_));
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

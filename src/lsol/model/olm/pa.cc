/*********************************************************************************
*     File Name           :     pa.cc
*     Created By          :     yuewu
*     Description         :
**********************************************************************************/

#include "lsol/model/olm/pa.h"
#include <algorithm>

using namespace std;
using namespace lsol::math::expr;

namespace lsol {

namespace model {

PA::PA(int class_num) : OnlineLinearModel(class_num) {
  this->eta_coeff_ = class_num == 2 ? 1.f : 2.f;
}
void PA::Update(const pario::DataPoint& x, const float*, float loss) {
  this->eta_ = loss / (this->eta_coeff_ * reduce<op::plus>(L2(x.data())));

  for (int c = 0; c < this->clf_num_; ++c) {
    if (this->gradients_[c] == 0) continue;
    math::Vector<real_t>& w = this->weights(c);
    w -= this->eta_ * this->gradients_[c] * x.data();
    // update bias
    w[0] -= this->bias_eta() * this->gradients_[c];
  }
}

RegisterModel(PA, "pa", "Online Passive Aggressive");

void PAI::SetParameter(const std::string& name, const std::string& value) {
  if (name == "C") {
    this->C_ = stof(value);
  } else {
    PA::SetParameter(name, value);
  }
}

void PAI::Update(const pario::DataPoint& x, const float*, float loss) {
  this->eta_ = (std::min)(
      this->C_, loss / (this->eta_coeff_ * reduce<op::plus>(L2(x.data()))));

  for (int c = 0; c < this->clf_num_; ++c) {
    if (this->gradients_[c] == 0) continue;
    math::Vector<real_t>& w = this->weights(c);
    w -= this->eta_ * this->gradients_[c] * x.data();
    // update bias
    w[0] -= this->bias_eta() * this->gradients_[c];
  }
}
void PAI::GetModelInfo(Json::Value& root) const {
  PA::GetModelInfo(root);
  root["online"]["C"] = this->C_;
}

RegisterModel(PAI, "pa1", "Online Passive Aggressive-1");

void PAII::SetParameter(const std::string& name, const std::string& value) {
  if (name == "C") {
    this->C_ = stof(value);
  } else {
    PA::SetParameter(name, value);
  }
}

void PAII::Update(const pario::DataPoint& x, const float*, float loss) {
  this->eta_ = loss / (this->eta_coeff_ * reduce<op::plus>(L2(x.data())) +
                       0.5f / this->C_);
  for (int c = 0; c < this->clf_num_; ++c) {
    if (this->gradients_[c] == 0) continue;
    math::Vector<real_t>& w = this->weights(c);
    w -= this->eta_ * this->gradients_[c] * x.data();
    // update bias
    w[0] -= this->bias_eta() * this->gradients_[c];
  }
}

void PAII::GetModelInfo(Json::Value& root) const {
  PA::GetModelInfo(root);
  root["online"]["C"] = this->C_;
}
RegisterModel(PAII, "pa2", "Online Passive Aggressive-2");

}  // namespace model
}  // namespace lsol

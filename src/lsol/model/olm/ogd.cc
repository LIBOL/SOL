/*********************************************************************************
*     File Name           :     ogd.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-18 21:37]
*     Last Modified       :     [2016-03-09 19:22]
*     Description         :     Stochastic Gradient Descent
**********************************************************************************/

#include "lsol/model/olm/ogd.h"
#include <cmath>

using namespace std;

namespace lsol {

namespace model {
OGD::OGD(int class_num) : OnlineLinearModel(class_num), eta0_(1) {
  this->set_power_t(0.5);
}
void OGD::SetParameter(const std::string& name, const std::string& value) {
  if (name == "power_t") {
    this->set_power_t(stof(value));
  } else if (name == "eta") {
    this->eta0_ = stof(value);
    Check(eta0_ >= 0);
  } else {
    OnlineLinearModel::SetParameter(name, value);
  }
}
void OGD::Update(const pario::DataPoint& dp, const float*, float) {
  const auto& x = dp.data();
  eta_ = eta0_ / this->pow_(this->cur_iter_num_, this->power_t_);

  for (int c = 0; c < this->clf_num_; ++c) {
    if (g(c) == 0) continue;
    w(c) -= this->eta_ * g(c) * x;
    // update bias
    w(c)[0] -= bias_eta() * g(c);
  }
}

void OGD::GetModelInfo(Json::Value& root) const {
  OnlineLinearModel::GetModelInfo(root);
  root["online"]["power_t"] = this->power_t_;
  root["online"]["eta"] = this->eta0_;
}

// calculate power t
float pow_const(int iter, float power_t) { return 1; }
float pow_sqrt(int iter, float power_t) { return sqrtf(float(iter)); }
float pow_linear(int iter, float power_t) { return float(iter); }
float pow_general(int iter, float power_t) {
  return powf((float)iter, power_t);
}

void OGD::set_power_t(float power_t) {
  Check(power_t >= 0);
  this->power_t_ = power_t;
  if (power_t == 0)
    this->pow_ = pow_const;
  else if (power_t == 0.5)
    this->pow_ = pow_sqrt;
  else if (power_t == 1)
    this->pow_ = pow_linear;
  else
    this->pow_ = pow_general;
}

RegisterModel(OGD, "ogd", "Online Gradient Descent");

}  // namespace model
}  // namespace lsol

/*********************************************************************************
*     File Name           :     ada_fobos.cc
*     Created By          :     yuewu
*     Description         :     Adaptive Subgradient FOBOS
**********************************************************************************/
#include "lsol/model/olm/ada_fobos.h"

#include <cmath>
#include <iostream>

#include "lsol/loss/hinge_loss.h"

using namespace std;
using namespace lsol;
using namespace lsol::math;

namespace lsol {

namespace model {

AdaFOBOS::AdaFOBOS(int class_num) : OnlineLinearModel(class_num), delta_(10.f) {
  this->H_ = new math::Vector<real_t>[this->clf_num_];
  for (int i = 0; i < this->clf_num_; ++i) {
    this->H_[i].resize(this->dim_);
    this->H_[i] = 0;
  }
}

AdaFOBOS::~AdaFOBOS() { DeleteArray(this->H_); }

void AdaFOBOS::SetParameter(const std::string& name, const std::string& value) {
  if (name == "delta") {
    this->delta_ = stof(value);
  } else if (name == "eta") {
    this->eta_ = stof(value);
    Check(eta_ >= 0);
  } else {
    OnlineLinearModel::SetParameter(name, value);
  }
}

void AdaFOBOS::Update(const pario::DataPoint& dp, const float*, float loss) {
  const auto& x = dp.data();
  for (int c = 0; c < this->clf_num_; ++c) {
    if (g(c) == 0) continue;

    H_[c] = Sqrt(L2(H_[c]) + L2(g(c) * L1(x)));
    H_[c][0] = sqrtf(H_[c][0] * H_[c][0] + g(c) * g(c));

    w(c) -= eta_ * g(c) * x / (this->delta_ + H_[c]);
    // update bias
    w(c)[0] -= bias_eta() * g(c) / (this->delta_ + H_[c][0]);
  }
}

void AdaFOBOS::update_dim(index_t dim) {
  if (dim >= this->dim_) {
    for (int c = 0; c < this->clf_num_; ++c) {
      this->H_[c].resize(dim);
      this->H_[c].slice_op([](real_t& val) { val = 0; }, this->dim_);
    }

    OnlineLinearModel::update_dim(dim);
  }
}

void AdaFOBOS::GetModelInfo(Json::Value& root) const {
  OnlineLinearModel::GetModelInfo(root);
  root["online"]["delta"] = this->delta_;
}

void AdaFOBOS::GetModelParam(Json::Value& root) const {
  OnlineLinearModel::GetModelParam(root);

  for (int c = 0; c < this->clf_num_; ++c) {
    ostringstream oss_name;
    oss_name << "H[" << c << "]";
    ostringstream oss_value;
    oss_value << this->H_[c] << "\n";
    root[oss_name.str()] = oss_value.str();
  }
}

int AdaFOBOS::SetModelParam(const Json::Value& root) {
  OnlineLinearModel::SetModelParam(root);

  for (int c = 0; c < this->clf_num_; ++c) {
    ostringstream oss_name;
    oss_name << "H[" << c << "]";
    istringstream iss_value(root[oss_name.str()].asString());
    iss_value >> this->H_[c];
  }
  return Status_OK;
}

RegisterModel(AdaFOBOS, "ada-fobos", "Adaptive Subgradient FOBOS");

}  // namespace model
}  // namespace lsol

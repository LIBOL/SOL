/*********************************************************************************
*     File Name           :     ada_rda.cc
*     Created By          :     yuewu
*     Description         :     Adaptive Subgradient RDA
**********************************************************************************/
#include "lsol/model/olm/ada_rda.h"

#include <cmath>
#include <iostream>

#include "lsol/loss/hinge_loss.h"

using namespace std;
using namespace lsol;
using namespace lsol::math;

namespace lsol {

namespace model {

AdaRDA::AdaRDA(int class_num) : OnlineLinearModel(class_num), delta_(10.f) {
  this->H_ = new math::Vector<real_t>[this->clf_num_];
  this->ut_ = new math::Vector<real_t>[this->clf_num_];
  for (int i = 0; i < this->clf_num_; ++i) {
    this->H_[i].resize(this->dim_);
    this->H_[i] = 0;

    this->ut_[i].resize(this->dim_);
    this->ut_[i] = 0;
  }
}

AdaRDA::~AdaRDA() {
  DeleteArray(this->H_);
  DeleteArray(this->ut_);
}

void AdaRDA::SetParameter(const std::string& name, const std::string& value) {
  if (name == "delta") {
    this->delta_ = stof(value);
  } else if (name == "eta") {
    this->eta_ = stof(value);
    Check(eta_ >= 0);
  } else {
    OnlineLinearModel::SetParameter(name, value);
  }
}

void AdaRDA::Update(const pario::DataPoint& dp, const float*, float loss) {
  const auto& x = dp.data();
  for (int c = 0; c < this->clf_num_; ++c) {
    if (g(c) == 0) continue;

    H_[c] = Sqrt(L2(H_[c]) + L2(g(c) * L1(x)));
    H_[c][0] = sqrtf(H_[c][0] * H_[c][0] + g(c) * g(c));

    ut_[c] += g(c) * x;
    ut_[c][0] += g(c);

    w(c) = -eta_ * ut_[c] / (this->delta_ + H_[c]);
    // update bias
    w(c)[0] = -bias_eta() * ut_[c][0] / (this->delta_ + H_[c][0]);
  }
}

void AdaRDA::update_dim(index_t dim) {
  if (dim >= this->dim_) {
    for (int c = 0; c < this->clf_num_; ++c) {
      this->H_[c].resize(dim);
      this->H_[c].slice_op([](real_t& val) { val = 0; }, this->dim_);

      this->ut_[c].resize(dim);
      this->ut_[c].slice_op([](real_t& val) { val = 0; }, this->dim_);
    }

    OnlineLinearModel::update_dim(dim);
  }
}

void AdaRDA::GetModelInfo(Json::Value& root) const {
  OnlineLinearModel::GetModelInfo(root);
  root["online"]["delta"] = this->delta_;
}

void AdaRDA::GetModelParam(Json::Value& root) const {
  OnlineLinearModel::GetModelParam(root);

  for (int c = 0; c < this->clf_num_; ++c) {
    ostringstream oss_name;
    oss_name << "H[" << c << "]";
    ostringstream oss_value;
    oss_value << this->H_[c] << "\n";
    root[oss_name.str()] = oss_value.str();
  }

  for (int c = 0; c < this->clf_num_; ++c) {
    ostringstream oss_name;
    oss_name << "ut[" << c << "]";
    ostringstream oss_value;
    oss_value << this->ut_[c] << "\n";
    root[oss_name.str()] = oss_value.str();
  }
}

int AdaRDA::SetModelParam(const Json::Value& root) {
  OnlineLinearModel::SetModelParam(root);

  for (int c = 0; c < this->clf_num_; ++c) {
    ostringstream oss_name;
    oss_name << "H[" << c << "]";
    istringstream iss_value(root[oss_name.str()].asString());
    iss_value >> this->H_[c];
  }
  for (int c = 0; c < this->clf_num_; ++c) {
    ostringstream oss_name;
    oss_name << "ut[" << c << "]";
    istringstream iss_value(root[oss_name.str()].asString());
    iss_value >> this->ut_[c];
  }

  return Status_OK;
}

RegisterModel(AdaRDA, "ada-rda", "Adaptive Subgradient RDA");

}  // namespace model
}  // namespace lsol

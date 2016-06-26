/*********************************************************************************
*     File Name           :     ada_rda.cc
*     Created By          :     yuewu
*     Description         :     Adaptive Subgradient RDA
**********************************************************************************/
#include "lsol/model/olm/ada_rda.h"

#include <cmath>
#include <iostream>

using namespace std;
using namespace lsol;
using namespace lsol::math;
using namespace lsol::math::expr;

namespace lsol {

namespace model {

AdaRDA::AdaRDA(int class_num) : OnlineLinearModel(class_num), delta_(10.f) {
  this->H_ = new math::Vector<real_t>[this->clf_num_];
  this->ut_ = new math::Vector<real_t>[this->clf_num_];
  for (int i = 0; i < this->clf_num_; ++i) {
    this->H_[i].resize(this->dim_);
    this->H_[i] = this->delta_;

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
    Check(delta_ > 0);
    for (int c = 0; c < this->clf_num_; ++c) this->H_[c] = this->delta_;
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

    H_[c] = Sqrt(L2(H_[c] - delta_) + L2(g(c) * x)) + delta_;
    H_[c][0] =
        sqrtf((H_[c][0] - delta_) * (H_[c][0] - delta_) + g(c) * g(c)) + delta_;

    ut_[c] += g(c) * x;
    ut_[c][0] += g(c);

    w(c) = -eta_ * ut_[c].slice(x) / H_[c];
    // update bias
    w(c)[0] *= bias_eta0_;
  }
}

void AdaRDA::EndTrain() {
  for (int c = 0; c < this->clf_num_; ++c) {
    w(c) = -eta_ * ut_[c] / H_[c];
    // update bias
    w(c)[0] *= bias_eta0_;
  }
  OnlineLinearModel::EndTrain();
}

void AdaRDA::update_dim(index_t dim) {
  if (dim >= this->dim_) {
    float delta = this->delta_;
    for (int c = 0; c < this->clf_num_; ++c) {
      this->H_[c].resize(dim);
      this->H_[c].slice_op([delta](real_t& val) { val = delta; }, this->dim_);

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

AdaRDA_L1::AdaRDA_L1(int class_num) : AdaRDA(class_num) {
  this->regularizer_ = &l1_;
}

label_t AdaRDA_L1::TrainPredict(const pario::DataPoint& dp, float* predicts) {
  const auto& x = dp.data();
  float trunc_thresh = l1_.lambda() * cur_iter_num_;
  for (int c = 0; c < this->clf_num_; ++c) {
    // trucate weights
    w(c) = -eta_ * expr::truncate(ut_[c].slice(x), trunc_thresh) / H_[c];
    // truncate bias
    w(c)[0] = -bias_eta() * expr::truncate(ut_[c][0], trunc_thresh) / H_[c][0];
  }

  return OnlineLinearModel::TrainPredict(dp, predicts);
}

void AdaRDA_L1::EndTrain() {
  float trunc_thresh = l1_.lambda() * cur_iter_num_;
  for (int c = 0; c < this->clf_num_; ++c) {
    w(c) = -eta_ * expr::truncate(ut_[c], trunc_thresh) / H_[c];
    w(c)[0] *= bias_eta0_;
  }
  OnlineLinearModel::EndTrain();
}

RegisterModel(AdaRDA_L1, "ada-rda-l1",
              "Adaptive Subgradient RDA with l1 regularization");

}  // namespace model
}  // namespace lsol

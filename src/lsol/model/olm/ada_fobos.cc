/*********************************************************************************
*     File Name           :     ada_fobos.cc
*     Created By          :     yuewu
*     Description         :     Adaptive Subgradient FOBOS
**********************************************************************************/
#include "lsol/model/olm/ada_fobos.h"

#include <cmath>

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
    this->H_[i] = this->delta_;
  }
}

AdaFOBOS::~AdaFOBOS() { DeleteArray(this->H_); }

void AdaFOBOS::SetParameter(const std::string& name, const std::string& value) {
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

void AdaFOBOS::Update(const pario::DataPoint& dp, const float*, float loss) {
  const auto& x = dp.data();
  for (int c = 0; c < this->clf_num_; ++c) {
    if (g(c) == 0) continue;

    H_[c] = Sqrt(L2(H_[c] - delta_) + L2(g(c) * x)) + delta_;
    H_[c][0] =
        sqrtf((H_[c][0] - delta_) * (H_[c][0] - delta_) + g(c) * g(c)) + delta_;

    w(c) -= eta_ * g(c) * x / H_[c];
    // update bias
    w(c)[0] -= bias_eta() * g(c) / H_[c][0];
  }
}

void AdaFOBOS::update_dim(index_t dim) {
  if (dim > this->dim_) {
    real_t delta = real_t(this->delta_);
    for (int c = 0; c < this->clf_num_; ++c) {
      this->H_[c].resize(dim);
      this->H_[c].slice_op([delta](real_t& val) { val = delta; }, this->dim_);
    }

    OnlineLinearModel::update_dim(dim);
  }
}

void AdaFOBOS::GetModelInfo(Json::Value& root) const {
  OnlineLinearModel::GetModelInfo(root);
  root["online"]["delta"] = this->delta_;
  root["online"]["eta"] = this->eta_;
}

void AdaFOBOS::GetModelParam(std::ostream& os) const {
  OnlineLinearModel::GetModelParam(os);

  for (int c = 0; c < this->clf_num_; ++c) {
    os << "H[" << c << "]: " << this->H_[c] << "\n";
  }
}

int AdaFOBOS::SetModelParam(std::istream& is) {
  OnlineLinearModel::SetModelParam(is);

  string line;
  for (int c = 0; c < this->clf_num_; ++c) {
    is >> line >> this->H_[c];
  }
  return Status_OK;
}

RegisterModel(AdaFOBOS, "ada-fobos", "Adaptive Subgradient FOBOS");

AdaFOBOS_L1::AdaFOBOS_L1(int class_num) : AdaFOBOS(class_num) {
  this->regularizer_ = &l1_;
}

label_t AdaFOBOS_L1::TrainPredict(const pario::DataPoint& dp, float* predicts) {
  const auto& x = dp.data();
  real_t t = real_t(cur_iter_num_ - 1);
  for (int c = 0; c < this->clf_num_; ++c) {
    // trucate weights
    w(c) =
        expr::truncate(w(c).slice(x), (eta_ * l1_.lambda()) *
                                          (t - l1_.last_update_time()) / H_[c]);
    // truncate bias
    w(c)[0] = expr::truncate(w(c)[0], bias_eta() * l1_.lambda() / H_[c][0]);
  }

  return OnlineLinearModel::TrainPredict(dp, predicts);
}

void AdaFOBOS_L1::EndTrain() {
  real_t t = real_t(cur_iter_num_);
  for (int c = 0; c < this->clf_num_; ++c) {
    // trucate weights
    w(c) = expr::truncate(
        w(c), (eta_ * l1_.lambda()) * (t - l1_.last_update_time()) / H_[c]);
    // truncate bias
    w(c)[0] = expr::truncate(w(c)[0], bias_eta() * l1_.lambda() / H_[c][0]);
  }
  AdaFOBOS::EndTrain();
}

RegisterModel(AdaFOBOS_L1, "ada-fobos-l1",
              "Adaptive Subgradient FOBOS with l1 regularization");

}  // namespace model
}  // namespace lsol

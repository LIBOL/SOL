/*********************************************************************************
*     File Name           :     ada_fobos.cc
*     Created By          :     yuewu
*     Description         :     Adaptive Subgradient FOBOS
**********************************************************************************/
#include "sol/model/olm/ada_fobos.h"

#include <cmath>

#include "sol/loss/hinge_loss.h"

using namespace std;
using namespace sol;
using namespace sol::math;

namespace sol {

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

AdaFOBOS_OFS::AdaFOBOS_OFS(int class_num) : AdaFOBOS(class_num) {
  this->regularizer_ = &(this->l0_);

  if (this->clf_num_ > 1) {
    this->H_sum_ = new Vector<real_t>;
  } else {
    this->H_sum_ = this->H_;
  }
}

AdaFOBOS_OFS::~AdaFOBOS_OFS() {
  if (this->clf_num_ > 1) DeletePointer(this->H_sum_);
}

void AdaFOBOS_OFS::SetParameter(const std::string& name, const std::string& value) {
  if (name == "B") {
    AdaFOBOS::SetParameter("lambda", value);
  } else {
    AdaFOBOS::SetParameter(name, value);
  }
}

void AdaFOBOS_OFS::BeginTrain() {
  AdaFOBOS::BeginTrain();
  index_t B = static_cast<index_t>(this->l0_.lambda());
  if (B > 0) {
    if (this->clf_num_ > 1) {
      this->H_sum_->resize(this->dim_);
    }

    if (this->dim_ < B + 1) this->update_dim(B + 1);

    if (this->clf_num_ > 1) {
      (*this->H_sum_) = 0;
      for (int i = 0; i < this->clf_num_; ++i) {
        (*this->H_sum_) += H_[i];
      }
    }
    this->min_heap_.Init(this->dim_ - 1, B, this->H_sum_->data() + 1);
  }
}

void AdaFOBOS_OFS::Update(const pario::DataPoint& dp, const float* predict,
                  float loss) {
  // number of features to select
  index_t B = static_cast<index_t>(this->l0_.lambda());
  if (B == 0) return AdaFOBOS::Update(dp, predict, loss);

  const auto& x = dp.data();
  // update H and heap
  size_t feat_num = x.indexes().size();
  if (this->clf_num_ == 1) {
    if (g(0) == 0) return;
    math::Vector<real_t>& H = this->H_[0];
    H[0] =
      sqrtf((H[0] - delta_) * (H[0] - delta_) + g(0) * g(0)) + delta_;

    for (size_t i = 0; i < feat_num; ++i) {
      index_t idx = x.index(i);
      // update H
      H[idx] -= delta_;
      H[idx] = sqrt(H[idx] * H[idx] + g(0) * g(0) * x.value(i) * x.value(i));
      H[idx] += delta_;
      // update heap
      index_t pos = this->min_heap_.get_pos(idx - 1);
      if (pos < B) {
        this->min_heap_.AdjustHeap(pos, B - 1);
      }
    }
  } else {
    // update sigma
    for (int c = 0; c < this->clf_num_; ++c) {
      if (g(c) == 0) continue;
      H_[c] = Sqrt(L2(H_[c] - delta_) + L2(g(c) * x)) + delta_;
      H_[c][0] =
        sqrtf((H_[c][0] - delta_) * (H_[c][0] - delta_) + g(c) * g(c)) + delta_;
    }

    auto& H_sum = (*this->H_sum_);
    for (size_t i = 0; i < feat_num; ++i) {
      index_t idx = x.index(i);
      H_sum[idx] = 0;
      for (int c = 0; c < this->clf_num_; ++c) {
        H_sum[idx] += H_[c][idx];
      }

      index_t pos = this->min_heap_.get_pos(idx - 1);
      if (pos < B) {
        this->min_heap_.AdjustHeap(pos, B - 1);
      }
    }
  }
  //update weights
  for (int c = 0; c < this->clf_num_; ++c) {
    if (g(c) == 0) continue;

    w(c) -= eta_ * g(c) * x / H_[c];
    // update bias
    w(c)[0] -= bias_eta() * g(c) / H_[c][0];
  }

  //truncate weights
  for (size_t i = 0; i < feat_num; ++i) {
    index_t idx = x.index(i);
    index_t ret_idx = this->min_heap_.UpdateHeap(idx - 1);
    if (ret_idx != invalid_index) {
      ++ret_idx;
      for (int c = 0; c < this->clf_num_; ++c) {
        w(c)[ret_idx] = 0;
      }
    }
  }
}

void AdaFOBOS_OFS::update_dim(index_t dim) {
  if (dim > this->dim_) {
    math::Vector<real_t>& H_sum = (*this->H_sum_);
    if (this->clf_num_ > 1) {
      H_sum.resize(dim);
      float init_h = float(this->clf_num_) * delta_;
      H_sum.slice_op([init_h](real_t& val) { val = init_h; },
                         this->dim_);
    }

    AdaFOBOS::update_dim(dim);
    this->min_heap_.set_N(dim - 1, H_sum.data() + 1);
  }
}

RegisterModel(AdaFOBOS_OFS, "adafobos-ofs", "Second Order Adaptive Online Feature Selection");

}  // namespace model
}  // namespace sol

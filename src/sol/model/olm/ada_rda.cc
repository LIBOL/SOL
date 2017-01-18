/*********************************************************************************
*     File Name           :     ada_rda.cc
*     Created By          :     yuewu
*     Description         :     Adaptive Subgradient RDA
**********************************************************************************/
#include "sol/model/olm/ada_rda.h"

#include <cmath>
#include <iostream>

using namespace std;
using namespace sol;
using namespace sol::math;
using namespace sol::math::expr;

namespace sol {

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
  if (dim > this->dim_) {
    real_t delta = real_t(this->delta_);
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
  root["online"]["eta"] = this->eta_;
}

void AdaRDA::GetModelParam(std::ostream& os) const {
  OnlineLinearModel::GetModelParam(os);

  for (int c = 0; c < this->clf_num_; ++c) {
    os << "H[" << c << "]: " << this->H_[c] << "\n";
  }

  for (int c = 0; c < this->clf_num_; ++c) {
    os << "ut[" << c << "]: " << this->ut_[c] << "\n";
  }
}

int AdaRDA::SetModelParam(std::istream& is) {
  OnlineLinearModel::SetModelParam(is);

  string line;
  for (int c = 0; c < this->clf_num_; ++c) {
    is >> line >> this->H_[c];
  }
  for (int c = 0; c < this->clf_num_; ++c) {
    is >> line >> this->ut_[c];
  }

  return Status_OK;
}

RegisterModel(AdaRDA, "ada-rda", "Adaptive Subgradient RDA");

AdaRDA_L1::AdaRDA_L1(int class_num) : AdaRDA(class_num) {
  this->regularizer_ = &l1_;
}

label_t AdaRDA_L1::TrainPredict(const pario::DataPoint& dp, float* predicts) {
  const auto& x = dp.data();
  real_t trunc_thresh = real_t(l1_.lambda() * cur_iter_num_);
  for (int c = 0; c < this->clf_num_; ++c) {
    // trucate weights
    w(c) = -eta_ * expr::truncate(ut_[c].slice(x), trunc_thresh) / H_[c];
    // truncate bias
    w(c)[0] = -bias_eta() * expr::truncate(ut_[c][0], trunc_thresh) / H_[c][0];
  }

  return OnlineLinearModel::TrainPredict(dp, predicts);
}

void AdaRDA_L1::EndTrain() {
  real_t trunc_thresh = real_t(l1_.lambda() * cur_iter_num_);
  for (int c = 0; c < this->clf_num_; ++c) {
    w(c) = -eta_ * expr::truncate(ut_[c], trunc_thresh) / H_[c];
    w(c)[0] *= bias_eta0_;
  }
  OnlineLinearModel::EndTrain();
}

RegisterModel(AdaRDA_L1, "ada-rda-l1",
              "Adaptive Subgradient RDA with l1 regularization");


AdaRDA_OFS::AdaRDA_OFS(int class_num) : AdaRDA(class_num) {
  this->regularizer_ = &(this->l0_);

  if (this->clf_num_ > 1) {
    this->H_sum_ = new Vector<real_t>;
  } else {
    this->H_sum_ = this->H_;
  }
}

AdaRDA_OFS::~AdaRDA_OFS() {
  if (this->clf_num_ > 1) DeletePointer(this->H_sum_);
}

void AdaRDA_OFS::SetParameter(const std::string& name, const std::string& value) {
  if (name == "B") {
    AdaRDA::SetParameter("lambda", value);
  } else {
    AdaRDA::SetParameter(name, value);
  }
}

void AdaRDA_OFS::BeginTrain() {
  AdaRDA::BeginTrain();
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

void AdaRDA_OFS::Update(const pario::DataPoint& dp, const float* predict,
                  float loss) {
  // number of features to select
  index_t B = static_cast<index_t>(this->l0_.lambda());
  if (B == 0) return AdaRDA::Update(dp, predict, loss);

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

    ut_[c] += g(c) * x;
    ut_[c][0] += g(c);

    w(c) = -eta_ * ut_[c].slice(x) / H_[c];
    // update bias
    w(c)[0] *= bias_eta0_;
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

void AdaRDA_OFS::EndTrain() {
  for (int c = 0; c < this->clf_num_; ++c) {
    w(c) = -eta_ * ut_[c] / H_[c];
    // update bias
    w(c)[0] *= bias_eta0_;
  }

  //truncate weights
  //for (size_t i = 2; i < this->dim_; ++i) {
  for (size_t i = 2; i < 3; ++i) {
    printf("%d\n", i);
    index_t ret_idx = this->min_heap_.UpdateHeap(i);
    //if (ret_idx != invalid_index) {
    //  ++ret_idx;
    //  for (int c = 0; c < this->clf_num_; ++c) {
    //    w(c)[ret_idx] = 0;
    //  }
    //}
  }

  OnlineLinearModel::EndTrain();
}

void AdaRDA_OFS::update_dim(index_t dim) {
  if (dim > this->dim_) {
    math::Vector<real_t>& H_sum = (*this->H_sum_);
    if (this->clf_num_ > 1) {
      H_sum.resize(dim);
      float init_h = float(this->clf_num_) * delta_;
      H_sum.slice_op([init_h](real_t& val) { val = init_h; },
                         this->dim_);
    }

    AdaRDA::update_dim(dim);
    this->min_heap_.set_N(dim - 1, H_sum.data() + 1);
  }
}

RegisterModel(AdaRDA_OFS, "adarda-ofs", "Second Order Adaptive Online Feature Selection");

}  // namespace model
}  // namespace sol

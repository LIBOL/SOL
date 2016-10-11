/*********************************************************************************
*     File Name           :     fofs.cc
*     Created By          :     yuewu
*     Description         :     First Order Online Feature Selection
**********************************************************************************/

#include "sol/model/olm/fofs.h"

using namespace std;
using namespace sol::math;

namespace sol {
namespace model {

FOFS::FOFS(int class_num) : OnlineLinearModel(class_num), lambda_(0.f), B_(0) {}

FOFS::~FOFS() {}

void FOFS::SetParameter(const std::string& name, const std::string& value) {
  if (name == "eta") {
    this->eta_ = stof(value);
    Check(eta_ >= 0);
  } else if (name == "lambda") {
    this->lambda_ = stof(value);
    Check(this->lambda_ >= 0);
  } else if (name == "B") {
    this->B_ = static_cast<index_t>(stoi(value));
    Check(this->B_ > 0);
  } else {
    OnlineLinearModel::SetParameter(name, value);
  }
}

void FOFS::BeginTrain() {
  OnlineLinearModel::BeginTrain();
  this->norm_coeff_ = 1.f / sqrtf(this->lambda_);
  this->momentum_ = 1 - this->lambda_ * this->eta_;

  if (this->B_ > 0) {
    math::Vector<real_t>& abs_w = this->abs_weights_;
    // make sure abs_w is of the same dimension of w
    abs_w.resize(this->dim_);

    if (this->dim_ < this->B_ + 1) this->update_dim(this->B_ + 1);

    abs_w = 0;

    for (int i = 0; i < this->clf_num_; ++i) {
      abs_w += L1(w(i));
    }
    this->min_heap_.Init(this->dim_ - 1, this->B_, abs_w.data() + 1);
  }
}

void FOFS::Update(const pario::DataPoint& dp, const float* predict,
                  float loss) {
  // update with sgd
  const auto& x = dp.data();

  for (int c = 0; c < this->clf_num_; ++c) {
    w(c) *= this->momentum_;
    if (g(c) == 0) continue;
    w(c) -= this->eta_ * g(c) * x;
    // update bias
    w(c)[0] -= bias_eta() * g(c);

    real_t coeff = this->norm_coeff_ / sqrt(Norm2(w(c)));
    if (coeff < 1) {
      w(c) *= coeff;
    }
  }

  if (this->B_ > 0) {
    math::Vector<real_t>& abs_w = this->abs_weights_;
    // update abosulte weights
    abs_w = L1(w(0).slice(dp.data()));
    for (int c = 1; c < this->clf_num_; ++c) {
      abs_w += L1(w(c).slice(dp.data()));
    }

    // update heap
    this->min_heap_.BuildHeap();
    index_t valid_dim = this->dim_ - 1;  // ignore bias
    for (index_t i = 0; i < valid_dim; ++i) {
      index_t ret_idx = this->min_heap_.UpdateHeap(i);
      if (ret_idx != invalid_index) {
        ++ret_idx;
        for (int c = 0; c < this->clf_num_; ++c) {
          w(c)[ret_idx] = 0;
        }
        abs_w[ret_idx] = 0;
      }
    }
  }
}
void FOFS::update_dim(index_t dim) {
  if (dim > this->dim_) {
    math::Vector<real_t>& abs_w = this->abs_weights_;
    abs_w.resize(dim);
    abs_w.slice_op([](real_t& val) { val = 0.f; }, this->dim_);
    this->min_heap_.set_N(dim - 1, abs_w.data() + 1);

    OnlineLinearModel::update_dim(dim);
  }
}

void FOFS::GetModelInfo(Json::Value& root) const {
  OnlineLinearModel::GetModelInfo(root);
  root["online"]["eta"] = this->eta_;
  root["online"]["lambda"] = this->lambda_;
  root["online"]["B"] = this->B_;
}

RegisterModel(FOFS, "fofs", "First Order Online Feature Selection");
}  // namespace mdoel
}  // namespace sol

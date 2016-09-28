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

FOFS::FOFS(int class_num)
    : OnlineLinearModel(class_num),
      lambda_(0.f),
      B_(0),
      abs_weights_(nullptr),
      min_heap_(nullptr) {
  this->abs_weights_ = new Vector<real_t>[this->clf_num_];
  this->min_heap_ = new MinHeap[this->clf_num_];

  for (int i = 0; i < this->clf_num_; ++i) {
    this->abs_weights_[i].resize(this->dim_);
    this->abs_weights_[i] = 0;
  }
}

FOFS::~FOFS() {
  DeleteArray(this->abs_weights_);
  DeleteArray(this->min_heap_);
}

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
    if (this->dim_ < this->B_ + 1) this->update_dim(this->B_ + 1);

    for (int i = 0; i < this->clf_num_; ++i) {
      this->abs_weights_[i] = L1(w(i));
      this->min_heap_[i].Init(this->dim_ - 1, this->B_,
                              this->abs_weights_[i].data() + 1);
    }
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
    for (int c = 0; c < this->clf_num_; ++c) {
      // update abosulte weights
      this->abs_weights_[c] = L1(w(c));

      // update heap
      this->min_heap_[c].BuildHeap();
      index_t valid_dim = this->dim_ - 1;  // ignore bias
      for (index_t i = 0; i < valid_dim; ++i) {
        index_t ret_idx = this->min_heap_[c].UpdateHeap(i);
        if (ret_idx != invalid_index) {
          ++ret_idx;
          w(c)[ret_idx] = 0;
          this->abs_weights_[c][ret_idx] = 0;
        }
      }
    }
  }
}
void FOFS::update_dim(index_t dim) {
  if (dim > this->dim_) {
    for (int c = 0; c < this->clf_num_; ++c) {
      math::Vector<real_t>& abs_w = this->abs_weights_[c];
      abs_w.resize(dim);
      abs_w.slice_op([](real_t& val) { val = 0.f; }, this->dim_);
      this->min_heap_[c].set_N(dim - 1, abs_w.data() + 1);
    }
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

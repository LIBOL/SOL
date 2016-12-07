/*********************************************************************************
*     File Name           :     arow.cc
*     Created By          :     yuewu
*     Description         :     Adaptive Regularization of Weight Vectors
**********************************************************************************/

#include "sol/model/olm/arow.h"
#include "sol/loss/hinge_loss.h"

using namespace std;
using namespace sol;
using namespace sol::math;

namespace sol {

namespace model {
AROW::AROW(int class_num)
    : OnlineLinearModel(class_num), r_(1.f), Sigmas_(nullptr) {
  this->Sigmas_ = new math::Vector<real_t>[this->clf_num_];

  for (int i = 0; i < this->clf_num_; ++i) {
    this->Sigmas_[i].resize(this->dim_);
    this->Sigmas_[i] = 1.f;
  }

  // loss
  if (class_num == 2) {
    this->SetParameter("loss", "hinge");
  } else {
    this->SetParameter("loss", "maxscore-hinge");
  }
}

AROW::~AROW() { DeleteArray(this->Sigmas_); }

void AROW::SetParameter(const std::string& name, const std::string& value) {
  if (name == "r") {
    this->r_ = stof(value);
  } else if (name == "loss") {
    OnlineLinearModel::SetParameter(name, value);
    if ((this->loss_->type() & loss::Loss::Type::HINGE) == 0) {
      throw invalid_argument("only hinge-based loss functions are allowed");
    }
  } else {
    OnlineLinearModel::SetParameter(name, value);
  }
}

void AROW::Update(const pario::DataPoint& dp, const float*, float loss) {
  const auto& x = dp.data();
  float beta_t = 0.f;
  //(\delta \psi)(x,i) = -g(i) * x
  for (int c = 0; c < this->clf_num_; ++c) {
    if (g(c) == 0) continue;
    float gc2 = g(c) * g(c);
    beta_t += expr::dotmul(this->Sigmas_[c], L2(x)) * gc2;
    if (this->bias_eta0_ != 0) beta_t += this->Sigmas_[c][0] * gc2;
  }
  beta_t = 1.f / (beta_t + r_);
  float alpha_t = loss * beta_t;
  this->eta_ = alpha_t;

  for (int c = 0; c < this->clf_num_; ++c) {
    if (g(c) == 0) continue;
    math::Vector<real_t>& Sigma = this->Sigmas_[c];
    w(c) -= this->eta_ * g(c) * Sigma * x;
    // update bias
    w(c)[0] -= this->bias_eta() * g(c) * Sigma[0];

    // update sigma
    float r1 = g(c) * g(c) / r_;
    Sigma /= (1.f + Sigma * L2(x) * r1);
    Sigma[0] /= (1.f + Sigma[0] * r1);
  }
}

void AROW::update_dim(index_t dim) {
  if (dim > this->dim_) {
    for (int c = 0; c < this->clf_num_; ++c) {
      math::Vector<real_t>& Sigma = this->Sigmas_[c];
      Sigma.resize(dim);
      Sigma.slice_op([](real_t& val) { val = 1.f; }, this->dim_);
    }

    OnlineLinearModel::update_dim(dim);
  }
}

void AROW::GetModelInfo(Json::Value& root) const {
  OnlineLinearModel::GetModelInfo(root);
  root["online"]["r"] = this->r_;
}

void AROW::GetModelParam(std::ostream& os) const {
  OnlineLinearModel::GetModelParam(os);

  for (int c = 0; c < this->clf_num_; ++c) {
    os << "Sigma[" << c << "]: " << this->Sigmas_[c] << "\n";
  }
}

int AROW::SetModelParam(std::istream& is) {
  OnlineLinearModel::SetModelParam(is);

  string line;
  for (int c = 0; c < this->clf_num_; ++c) {
    is >> line >> this->Sigmas_[c];
  }

  return Status_OK;
}

RegisterModel(AROW, "arow", "Adaptive Regularization of Weight Vectors");

SOFS::SOFS(int class_num) : AROW(class_num) {
  this->regularizer_ = &(this->l0_);

  if (this->clf_num_ > 1) {
    this->Sigma_sum_ = new Vector<real_t>;
  } else {
    this->Sigma_sum_ = this->Sigmas_;
  }
}

SOFS::~SOFS() {
  if (this->clf_num_ > 1) DeletePointer(this->Sigma_sum_);
}

void SOFS::SetParameter(const std::string& name, const std::string& value) {
  if (name == "B") {
    AROW::SetParameter("lambda", value);
  } else {
    AROW::SetParameter(name, value);
  }
}

void SOFS::BeginTrain() {
  AROW::BeginTrain();
  index_t B = static_cast<index_t>(this->l0_.lambda());
  if (B > 0) {
    if (this->clf_num_ > 1) {
      this->Sigma_sum_->resize(this->dim_);
    }

    if (this->dim_ < B + 1) this->update_dim(B + 1);

    if (this->clf_num_ > 1) {
      (*this->Sigma_sum_) = 0;
      for (int i = 0; i < this->clf_num_; ++i) {
        (*this->Sigma_sum_) += Sigma(i);
      }
    }
    this->max_heap_.Init(this->dim_ - 1, B, this->Sigma_sum_->data() + 1);
  }
}

void SOFS::Update(const pario::DataPoint& dp, const float* predict,
                  float loss) {
  // number of features to select
  index_t B = static_cast<index_t>(this->l0_.lambda());
  if (B == 0) return AROW::Update(dp, predict, loss);

  const auto& x = dp.data();
  float beta_t = 0.f;
  //(\delta \psi)(x,i) = -g(i) * x
  for (int c = 0; c < this->clf_num_; ++c) {
    if (g(c) == 0) continue;
    float gc2 = g(c) * g(c);
    beta_t += expr::dotmul(this->Sigmas_[c], L2(x)) * gc2;
    if (this->bias_eta0_ != 0) beta_t += this->Sigmas_[c][0] * gc2;
  }
  beta_t = 1.f / (beta_t + r_);
  float alpha_t = loss * beta_t;
  this->eta_ = alpha_t;

  // update weights
  for (int c = 0; c < this->clf_num_; ++c) {
    if (g(c) == 0) continue;
    math::Vector<real_t>& Sigma = this->Sigmas_[c];
    w(c) -= this->eta_ * g(c) * Sigma * x;
    // update bias
    w(c)[0] -= this->bias_eta() * g(c) * Sigma[0];
  }
  // update sigma and heap
  size_t feat_num = x.indexes().size();
  if (this->clf_num_ == 1) {
    if (g(0) == 0) return;
    math::Vector<real_t>& Sigma = this->Sigmas_[0];
    float r1 = g(0) * g(0) / r_;
    Sigma[0] /= (1.f + Sigma[0] * r1);
    for (size_t i = 0; i < feat_num; ++i) {
      index_t idx = x.index(i);
      // update sigma
      Sigma[idx] /= (1.f + Sigma[idx] * x.value(i) * x.value(i) * r1);
      // update heap
      index_t pos = this->max_heap_.get_pos(idx - 1);
      if (pos < B) {
        this->max_heap_.AdjustHeap(pos, B - 1);
      }
    }
  } else {
    // update sigma
    for (int c = 0; c < this->clf_num_; ++c) {
      if (g(c) == 0) continue;
      math::Vector<real_t>& Sigma = this->Sigmas_[c];
      float r1 = g(c) * g(c) / r_;
      Sigma /= (1.f + Sigma * L2(x) * r1);
      Sigma[0] /= (1.f + Sigma[0] * r1);
    }

    auto& sigma_sum = (*this->Sigma_sum_);
    for (size_t i = 0; i < feat_num; ++i) {
      index_t idx = x.index(i);
      sigma_sum[idx] = 0;
      for (int c = 0; c < this->clf_num_; ++c) {
        sigma_sum[idx] += this->Sigmas_[c][idx];
      }

      index_t pos = this->max_heap_.get_pos(idx - 1);
      if (pos < B) {
        this->max_heap_.AdjustHeap(pos, B - 1);
      }
    }
  }

  for (size_t i = 0; i < feat_num; ++i) {
    index_t idx = x.index(i);
    index_t ret_idx = this->max_heap_.UpdateHeap(idx - 1);
    if (ret_idx != invalid_index) {
      ++ret_idx;
      for (int c = 0; c < this->clf_num_; ++c) {
        w(c)[ret_idx] = 0;
      }
    }
  }
}

void SOFS::update_dim(index_t dim) {
  if (dim > this->dim_) {
    math::Vector<real_t>& sigma_sum = (*this->Sigma_sum_);
    if (this->clf_num_ > 1) {
      sigma_sum.resize(dim);
      float class_num = float(this->clf_num_);
      sigma_sum.slice_op([class_num](real_t& val) { val = class_num; },
                         this->dim_);
    }

    AROW::update_dim(dim);
    this->max_heap_.set_N(dim - 1, sigma_sum.data() + 1);
  }
}

RegisterModel(SOFS, "sofs", "Second Order Online Feature Selection");

}  // namespace model

}  // namespace sol

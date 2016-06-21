/*********************************************************************************
*     File Name           :     arow.cc
*     Created By          :     yuewu
*     Description         :     Adaptive Regularization of Weight Vectors
**********************************************************************************/

#include "lsol/model/olm/arow.h"
#include "lsol/loss/hinge_loss.h"

using namespace std;
using namespace lsol;
using namespace lsol::math;

namespace lsol {

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
  if (dim >= this->dim_) {
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

void AROW::GetModelParam(Json::Value& root) const {
  OnlineLinearModel::GetModelParam(root);

  for (int c = 0; c < this->clf_num_; ++c) {
    ostringstream oss_name;
    oss_name << "Sigma[" << c << "]";
    ostringstream oss_value;
    oss_value << this->Sigmas_[c] << "\n";
    root[oss_name.str()] = oss_value.str();
  }
}

int AROW::SetModelParam(const Json::Value& root) {
  OnlineLinearModel::SetModelParam(root);

  for (int c = 0; c < this->clf_num_; ++c) {
    ostringstream oss_name;
    oss_name << "Sigma[" << c << "]";
    istringstream iss_value(root[oss_name.str()].asString());
    iss_value >> this->Sigmas_[c];
  }

  return Status_OK;
}

RegisterModel(AROW, "arow", "Adaptive Regularization of Weight Vectors");

}  // namespace model
}  // namespace lsol

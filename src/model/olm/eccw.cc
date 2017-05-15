/*********************************************************************************
*     File Name           :     eccw.cc
*     Created By          :     yuewu
*     Description         :     Exact Convex Confidence Weighted Online Learning
**********************************************************************************/

#include "sol/model/olm/eccw.h"
#include <cmath>

using namespace std;
using namespace sol;
using namespace sol::math;

namespace sol {

namespace model {
ECCW::ECCW(int class_num)
    : OnlineLinearModel(class_num),
      hinge_base_(nullptr),
      a_(1.f),
      Sigmas_(nullptr) {
  this->Sigmas_ = new math::Vector<real_t>[this->clf_num_];
  for (int i = 0; i < this->clf_num_; ++i) {
    this->Sigmas_[i].resize(this->dim_);
    this->Sigmas_[i] = this->a_;
  }
  this->set_phi(0.5244f);

  // loss
  if (class_num == 2) {
    this->SetParameter("loss", "hinge");
  } else {
    this->SetParameter("loss", "maxscore-hinge");
  }
}

ECCW::~ECCW() { DeleteArray(this->Sigmas_); }

void ECCW::SetParameter(const std::string& name, const std::string& value) {
  if (name == "a") {
    this->a_ = stof(value);
    for (int i = 0; i < this->clf_num_; ++i) {
      this->Sigmas_[i] = this->a_;
    }
  } else if (name == "phi") {
    this->set_phi(stof(value));
    this->require_reinit_ = true;
  } else if (name == "loss") {
    OnlineLinearModel::SetParameter(name, value);
    if ((this->loss_->type() & loss::Loss::Type::HINGE) == 0) {
      throw invalid_argument("only hinge-based loss functions are allowed");
    }
    this->hinge_base_ = static_cast<loss::HingeBase*>(this->loss_);
  } else if (name == "bias_eta") {
    OnlineLinearModel::SetParameter(name, value);
    this->require_reinit_ = true;
  } else {
    OnlineLinearModel::SetParameter(name, value);
  }
}

void ECCW::BeginTrain() {
  OnlineLinearModel::BeginTrain();
  math::Vector<real_t>* sigmas = this->Sigmas_;
  float* vt = &this->vi_;
  float bias_eta = this->bias_eta();
  float phi = this->phi_;

  this->hinge_base_->set_margin([sigmas, vt, bias_eta, phi](
      const pario::DataPoint& dp, float* predict, label_t predict_label,
      float* gradient, int cls_num) {
    const auto& x = dp.data();
    *vt = 0.f;
    //(\delta \psi)(x,i) = -g(i) * x
    for (int c = 0; c < cls_num; ++c) {
      if (gradient[c] == 0) continue;
      float gc2 = gradient[c] * gradient[c];
      *vt += expr::dotmul(sigmas[c], L2(x)) * gc2;
      if (bias_eta != 0) *vt += sigmas[c][0] * gc2;
    }
    return phi * *vt;
  });
}

void ECCW::Update(const pario::DataPoint& dp, const float*, float loss) {
  const auto& x = dp.data();
  float mi = phi_ * vi_ - loss;
  float tmp = mi * phi_ * phi_;
  float alpha_i =
      (-mi * psi_ + sqrtf(tmp * tmp * 0.25f + vi_ * phi_ * phi_ * xi_)) /
      (vi_ * xi_);
  float ui =
      0.5f * (-alpha_i * vi_ * phi_ +
              sqrtf(alpha_i * alpha_i * vi_ * vi_ * phi_ * phi_ + 4.f * vi_));
  ui *= ui;
  // float beta_i = alpha_i * phi_ / (sqrtf(ui) + vi_ * alpha_i * phi_);

  this->eta_ = alpha_i;
  tmp = alpha_i * phi_ * sqrtf(ui);
  for (int c = 0; c < this->clf_num_; ++c) {
    if (g(c) == 0) continue;
    w(c) -= eta_ * g(c) * Sigmas_[c] * x;
    // update bias
    w(c)[0] -= bias_eta() * g(c) * Sigmas_[c][0];
    // update sigma
    Sigmas_[c] /= (1.f + tmp * Sigmas_[c] * L2(x));
    Sigmas_[c][0] /= (1.f + tmp * Sigmas_[c][0]);
  }
}

void ECCW::update_dim(index_t dim) {
  if (dim > this->dim_) {
    float a = this->a_;
    for (int c = 0; c < this->clf_num_; ++c) {
      math::Vector<real_t>& Sigma = this->Sigmas_[c];
      Sigma.resize(dim);
      Sigma.slice_op([a](real_t& val) { val = a; }, this->dim_);
    }

    OnlineLinearModel::update_dim(dim);
  }
}
void ECCW::set_phi(float phi) {
  this->phi_ = phi;
  this->psi_ = 1 + phi * phi / 2.f;
  this->xi_ = 1 + phi * phi;
}

void ECCW::GetModelInfo(Json::Value& root) const {
  OnlineLinearModel::GetModelInfo(root);
  root["online"]["a"] = this->a_;
  root["online"]["phi"] = this->phi_;
}

void ECCW::GetModelParam(std::ostream& os) const {
  OnlineLinearModel::GetModelParam(os);

  for (int c = 0; c < this->clf_num_; ++c) {
    os << "Sigma[" << c << "]: " << this->Sigmas_[c] << "\n";
  }
}

int ECCW::SetModelParam(std::istream& is) {
  OnlineLinearModel::SetModelParam(is);

  string line;
  for (int c = 0; c < this->clf_num_; ++c) {
    is >> line >> this->Sigmas_[c];
  }

  return Status_OK;
}

RegisterModel(ECCW, "eccw", "exact convex confidence weighted online learning");

}  // namespace model
}  // namespace sol

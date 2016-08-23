/*********************************************************************************
*     File Name           :     cw.cc
*     Created By          :     yuewu
*     Description         :     Confidence Weighted Online Learning
**********************************************************************************/

#include "lsol/model/olm/cw.h"
#include <cmath>

using namespace std;
using namespace lsol;
using namespace lsol::math;

namespace lsol {

namespace model {
CW::CW(int class_num)
    : OnlineLinearModel(class_num),
      hinge_base_(nullptr),
      a_(1.f),
      phi_(0.5244f) {
  this->Sigmas_ = new math::Vector<real_t>[this->clf_num_];
  for (int i = 0; i < this->clf_num_; ++i) {
    this->Sigmas_[i].resize(this->dim_);
    this->Sigmas_[i] = this->a_;
  }

  // loss
  if (class_num == 2) {
    this->SetParameter("loss", "hinge");
  } else {
    this->SetParameter("loss", "maxscore-hinge");
  }
}

CW::~CW() { DeleteArray(this->Sigmas_); }

void CW::SetParameter(const std::string& name, const std::string& value) {
  if (name == "a") {
    this->a_ = stof(value);
    for (int i = 0; i < this->clf_num_; ++i) {
      this->Sigmas_[i] = this->a_;
    }
  } else if (name == "phi") {
    this->phi_ = stof(value);
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

void CW::BeginTrain() {
  OnlineLinearModel::BeginTrain();
  math::Vector<real_t>* sigmas = this->Sigmas_;
  float* vt = &this->Vi_;
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

void CW::Update(const pario::DataPoint& dp, const float*, float loss) {
  const auto& x = dp.data();
  float Mi = phi_ * Vi_ - loss;
  float tmp = (1 + 2 * phi_ * Mi);
  float alpha_i =
      (-(1 + 2 * phi_ * Mi) + sqrtf(tmp * tmp - 8 * phi_ * (Mi - phi_ * Vi_))) /
      (4 * phi_ * Vi_);

  this->eta_ = alpha_i;
  tmp = 2 * alpha_i * phi_;
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

void CW::update_dim(index_t dim) {
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

void CW::GetModelInfo(Json::Value& root) const {
  OnlineLinearModel::GetModelInfo(root);
  root["online"]["a"] = this->a_;
  root["online"]["phi"] = this->phi_;
}

void CW::GetModelParam(std::ostream& os) const {
  OnlineLinearModel::GetModelParam(os);

  for (int c = 0; c < this->clf_num_; ++c) {
    os << "Sigma[" << c << "]: " << this->Sigmas_[c] << "\n";
  }
}

int CW::SetModelParam(std::istream& is) {
  OnlineLinearModel::SetModelParam(is);

  string line;
  for (int c = 0; c < this->clf_num_; ++c) {
    is >> line >> this->Sigmas_[c];
  }

  return Status_OK;
}

RegisterModel(CW, "cw", "confidence weighted online learning");

}  // namespace model
}  // namespace lsol

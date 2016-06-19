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
AROW::AROW(int class_num) : OnlineLinearModel(class_num), r_(1.f) {
  this->Sigma_.resize(this->dim_);
  this->Sigma_ = 1.f;

  // loss
  if (class_num == 2) {
    this->SetParameter("loss", "hinge");
  } else {
    this->SetParameter("loss", "maxscore-hinge");
  }
}

void AROW::SetParameter(const std::string& name, const std::string& value) {
  if (name == "r") {
    this->r_ = stof(value);
    this->Sigma_ = this->r_;
  } else if (name == "loss") {
    OnlineLinearModel::SetParameter(name, value);
    if ((this->loss_->type() & loss::Loss::Type::HINGE) == 0) {
      throw invalid_argument("only hinge-based loss functions are allowed");
    }
  } else {
    OnlineLinearModel::SetParameter(name, value);
  }
}

void AROW::Update(const pario::DataPoint& x, const float*, float loss) {
  float beta_t = 1.f / (expr::dotmul(this->Sigma_, L2(x.data())) + r_);
  float alpha_t = loss * beta_t;
  this->eta_ = alpha_t;

  for (int c = 0; c < this->clf_num_; ++c) {
    if (this->gradients_[c] == 0) continue;
    math::Vector<real_t>& w = this->weights(c);
    w -= this->eta_ * this->gradients_[c] * this->Sigma_ * x.data();
    // update bias
    w[0] -= this->bias_eta() * this->gradients_[c] * this->Sigma_[0];
  }
  this->Sigma_ /= (1.f + this->Sigma_ * L2(x.data()) / this->r_);
  this->Sigma_[0] /= (1.f + this->Sigma_[0] / this->r_);
}

void AROW::update_dim(index_t dim) {
  if (dim >= this->dim_) {
    this->Sigma_.resize(dim);
    for (real_t* iter = this->Sigma_.begin() + this->dim_;
         iter != this->Sigma_.end(); ++iter)
      *iter = this->r_;

    OnlineLinearModel::update_dim(dim);
  }
}

void AROW::GetModelInfo(Json::Value& root) const {
  OnlineLinearModel::GetModelInfo(root);
  root["online"]["r"] = this->r_;
}

void AROW::GetModelParam(Json::Value& root) const {
  OnlineLinearModel::GetModelParam(root);

  ostringstream oss_Sigma;
  oss_Sigma << this->Sigma_ << "\n";
  root["Sigma"] = oss_Sigma.str();
}

int AROW::SetModelParam(const Json::Value& root) {
  OnlineLinearModel::SetModelParam(root);

  istringstream iss_Sigma(root["Sigma"].asString());
  iss_Sigma >> this->Sigma_;
  return Status_OK;
}

RegisterModel(AROW, "arow", "Adaptive Regularization of Weight Vectors");

}  // namespace model
}  // namespace lsol

/*********************************************************************************
*     File Name           :     sop.cc
*     Created By          :     yuewu
*     Description         :     second order perceptron
**********************************************************************************/
#include "sol/model/olm/sop.h"
#include "sol/loss/bool_loss.h"

using namespace std;
using namespace sol;

namespace sol {

namespace model {
SOP::SOP(int class_num) : OnlineLinearModel(class_num), a_(1.f) {
  this->X_.resize(this->dim_);
  this->X_ = 0;

  this->v_ = new math::Vector<real_t>[this->clf_num_];

  for (int i = 0; i < this->clf_num_; ++i) {
    v(i).resize(this->dim_);
    v(i) = 0;
  }

  // loss
  if (class_num == 2) {
    this->SetParameter("loss", "bool");
  } else {
    this->SetParameter("loss", "maxscore-bool");
  }
}
SOP::~SOP() { DeleteArray(this->v_); }

void SOP::SetParameter(const std::string& name, const std::string& value) {
  if (name == "a") {
    this->a_ = stof(value);
  } else if (name == "loss") {
    OnlineLinearModel::SetParameter(name, value);
    if ((this->loss_->type() & loss::Loss::Type::BOOL) == 0) {
      throw invalid_argument("only bool-based loss functions are allowed");
    }
  } else {
    OnlineLinearModel::SetParameter(name, value);
  }
}

void SOP::EndTrain() {
  for (int c = 0; c < this->clf_num_; ++c) {
    w(c) = v(c) / (a_ + X_);
  }
  OnlineLinearModel::EndTrain();
}

label_t SOP::TrainPredict(const pario::DataPoint& dp, float* predicts) {
  const auto& x = dp.data();
  for (int c = 0; c < this->clf_num_; ++c) {
    w(c) = v(c) / (a_ + X_ + L2(x));
  }
  return OnlineLinearModel::TrainPredict(dp, predicts);
}

void SOP::Update(const pario::DataPoint& dp, const float*, float) {
  const auto& x = dp.data();
  this->eta_ = 1.f;

  for (int c = 0; c < this->clf_num_; ++c) {
    if (g(c) == 0) continue;
    v(c) -= g(c) * x;
    // update bias
    v(c)[0] -= bias_eta() * g(c);
  }
  X_ += L2(x);
}

void SOP::update_dim(index_t dim) {
  if (dim > this->dim_) {
    this->X_.resize(dim);
    this->X_.slice_op([](real_t& val) { val = 0.f; }, this->dim_);

    for (int c = 0; c < this->clf_num_; ++c) {
      v(c).resize(dim);
      v(c).slice_op([](real_t& val) { val = 0.f; }, this->dim_);
    }
    OnlineLinearModel::update_dim(dim);
  }
}

void SOP::GetModelInfo(Json::Value& root) const {
  OnlineLinearModel::GetModelInfo(root);
  root["online"]["a"] = this->a_;
}

void SOP::GetModelParam(std::ostream& os) const {
  OnlineLinearModel::GetModelParam(os);
  os << "X: " << this->X_ << "\n";

  for (int c = 0; c < this->clf_num_; ++c) {
    os << "v[" << c << "]: " << v(c) << "\n";
  }
}

int SOP::SetModelParam(std::istream& is) {
  OnlineLinearModel::SetModelParam(is);

  string line;
  is >> line >> this->X_;

  for (int c = 0; c < this->clf_num_; ++c) {
    is >> line >> v(c);
  }
  return Status_OK;
}

RegisterModel(SOP, "sop", "second order perceptron");

}  // namespace model
}  // namespace sol

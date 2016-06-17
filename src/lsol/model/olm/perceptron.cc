/*********************************************************************************
*     File Name           :     perceptron.cc
*     Created By          :     yuewu
*     Description         :     Perceptron Algorithm
**********************************************************************************/

#include "lsol/model/olm/perceptron.h"
#include "lsol/loss/bool_loss.h"

using namespace std;
using namespace lsol;

namespace lsol {

namespace model {
Perceptron::Perceptron(int class_num) : OnlineLinearModel(class_num) {
  // loss
  if (class_num == 2) {
    this->SetParameter("loss", "bool");
  } else {
    this->SetParameter("loss", "maxscore-bool");
  }
}

void Perceptron::SetParameter(const std::string& name,
                              const std::string& value) {
  if (name == "loss") {
    OnlineLinearModel::SetParameter(name, value);
    if ((this->loss_->type() & loss::Loss::Type::BOOL) == 0) {
      throw invalid_argument("only bool-based loss functions are allowed");
    }
  } else {
    OnlineLinearModel::SetParameter(name, value);
  }
}

void Perceptron::Update(const pario::DataPoint& x, const float*, float) {
  this->eta_ = 1.f;

  for (int c = 0; c < this->clf_num_; ++c) {
    if (this->gradients_[c] == 0) continue;
    math::Vector<real_t>& w = this->weights(c);
    w -= this->gradients_[c] * x.data();
    // update bias
    w[0] -= this->bias_eta() * this->gradients_[c];
  }
}

RegisterModel(Perceptron, "perceptron", "perceptron algorithm");

}  // namespace model
}  // namespace lsol

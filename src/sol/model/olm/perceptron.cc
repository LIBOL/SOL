/*********************************************************************************
*     File Name           :     perceptron.cc
*     Created By          :     yuewu
*     Description         :     Perceptron Algorithm
**********************************************************************************/

#include "sol/model/olm/perceptron.h"
#include "sol/loss/bool_loss.h"

using namespace std;
using namespace sol;

namespace sol {

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

void Perceptron::Update(const pario::DataPoint& dp, const float*, float) {
  const auto& x = dp.data();
  this->eta_ = 1.f;

  for (int c = 0; c < this->clf_num_; ++c) {
    if (g(c) == 0) continue;
    w(c) -= g(c) * x;
    // update bias
    w(c)[0] -= bias_eta() * g(c);
  }
}

RegisterModel(Perceptron, "perceptron", "perceptron algorithm");

}  // namespace model
}  // namespace sol

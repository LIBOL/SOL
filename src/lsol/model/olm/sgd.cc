/*********************************************************************************
*     File Name           :     sgd.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-18 21:37]
*     Last Modified       :     [2016-03-09 19:22]
*     Description         :     Stochastic Gradient Descent
**********************************************************************************/

#include "lsol/model/olm/sgd.h"

using namespace std;

namespace lsol {

namespace model {

void SGD::Update(const pario::DataPoint& x) {
  size_t feat_num = x.indexes().size();
  for (int c = 0; c < this->clf_num_; ++c) {
    if (this->gradients_[c] == 0) continue;
    math::Vector<real_t>& w = this->weights(c);
    w -= this->eta_ * this->gradients_[c] * x.data();
    // update bias
    w[0] -= this->eta_ * this->gradients_[c];
  }
}

RegisterModel(SGD, "sgd", "Stochastic Gradient Descent");

}  // namespace model
}  // namespace lsol

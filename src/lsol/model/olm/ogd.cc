/*********************************************************************************
*     File Name           :     ogd.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-18 21:37]
*     Last Modified       :     [2016-03-09 19:22]
*     Description         :     Stochastic Gradient Descent
**********************************************************************************/

#include "lsol/model/olm/ogd.h"

using namespace std;

namespace lsol {

namespace model {

void OGD::Update(const pario::DataPoint& x, const float*, float) {
  this->eta_ = this->eta0_ / this->pow_(this->cur_iter_num_, this->power_t_);
  this->bias_eta_ = this->bias_eta0_ * this->eta_;

  for (int c = 0; c < this->clf_num_; ++c) {
    if (this->gradients_[c] == 0) continue;
    math::Vector<real_t>& w = this->weights(c);
    w -= this->eta_ * this->gradients_[c] * x.data();
    // update bias
    w[0] -= this->bias_eta_ * this->gradients_[c];
  }
}

RegisterModel(OGD, "ogd", "Online Gradient Descent");

}  // namespace model
}  // namespace lsol

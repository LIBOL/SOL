/*********************************************************************************
*     File Name           :     hinge_loss.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-14 23:44]
*     Last Modified       :     [2016-02-15 00:01]
*     Description         :     hinge loss
**********************************************************************************/

#include "lsol/loss/hinge_loss.h"

#include <algorithm>

namespace lsol {
namespace loss {

float HingeLoss::loss(label_t label, float* predict, int cls_num) {
  return (std::max)(0.0f, 1.f - *predict * label);
}

float HingeLoss::gradient(label_t label, float* predict, float* gradient,
                          int cls_num) {
  float loss = this->loss(label, predict, cls_num);
  if (loss > 0) {
    *gradient = (float)(-label);
  } else {
    *gradient = 0;
  }
  return loss;
}

RegisterLoss(HingeLoss, "hinge", "Hinge Loss");

}  // namespace loss
}  // namespace lsol

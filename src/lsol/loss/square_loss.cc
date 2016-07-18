/*********************************************************************************
*     File Name           :     square_loss.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-14 23:44]
*     Last Modified       :     [2016-02-15 00:01]
*     Description         :     square loss
**********************************************************************************/

#include "lsol/loss/square_loss.h"

#include <algorithm>
#include <limits>

namespace lsol {
namespace loss {

float SquareLoss::loss(const pario::DataPoint& dp, float* predict,
                      label_t predict_label, int cls_num) {
  return (*predict - dp.label()) * (*predict - dp.label()) * 0.5;
}

float SquareLoss::gradient(const pario::DataPoint& dp, float* predict,
                          label_t predict_label, float* gradient, int cls_num) {
  *gradient = *predict - float(dp.label());
  return *gradient * *gradient * 0.5;
}

RegisterLoss(SquareLoss, "square", "Square Loss");

}  // namespace loss
}  // namespace lsol

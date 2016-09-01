/*********************************************************************************
*     File Name           :     hinge.h
*     Created By          :     yuewu
*     Description         :     hinge loss
**********************************************************************************/
#ifndef SOL_LOSS_SQUARE_LOSS_H__
#define SOL_LOSS_SQUARE_LOSS_H__

#include <sol/loss/loss.h>
#include <functional>

namespace sol {
namespace loss {

class SOL_EXPORTS SquareLoss : public Loss {
 public:
  SquareLoss() : Loss(Type::RG) {}

 public:
  virtual float loss(const pario::DataPoint& dp, float* predict,
                     label_t predict_label, int cls_num);

  virtual float gradient(const pario::DataPoint& dp, float* predict,
                         label_t predict_label, float* gradient, int cls_num);

};  // class SquareLoss
}  // namespace loss
}  // namespace sol
#endif

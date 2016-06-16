/*********************************************************************************
*     File Name           :     hinge.h
*     Created By          :     yuewu
*     Description         :     hinge loss
**********************************************************************************/
#ifndef LSOL_LOSS_HINGE_H__
#define LSOL_LOSS_HINGE_H__

#include <lsol/loss/loss.h>

namespace lsol {
namespace loss {

class LSOL_EXPORTS HingeLoss : public Loss {
 public:
  HingeLoss() : Loss(Type::BC) {}

 public:
  virtual float loss(label_t label, float* predict, int cls_num);

  virtual float gradient(label_t label, float* predict, float* gradient,
                         int cls_num);

};  // class HingeLoss

class LSOL_EXPORTS MaxScoreHingeLoss : public Loss {
 public:
  MaxScoreHingeLoss() : Loss(Type::MC) {}

 public:
  virtual float loss(label_t label, float* predict, int cls_num);

  virtual float gradient(label_t label, float* predict, float* gradient,
                         int cls_num);
};

class LSOL_EXPORTS UniformHingeLoss : public Loss {
 public:
  UniformHingeLoss() : Loss(Type::MC) {}

 public:
  virtual float loss(label_t label, float* predict, int cls_num);

  virtual float gradient(label_t label, float* predict, float* gradient,
                         int cls_num);
};

}  // namespace loss
}  // namespace lsol
#endif

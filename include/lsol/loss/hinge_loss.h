/*********************************************************************************
*     File Name           :     hinge.h
*     Created By          :     yuewu
*     Description         :     hinge loss
**********************************************************************************/
#ifndef LSOL_LOSS_HINGE_LOSS_H__
#define LSOL_LOSS_HINGE_LOSS_H__

#include <lsol/loss/loss.h>

namespace lsol {
namespace loss {

class LSOL_EXPORTS HingeBase : public Loss {
 public:
  HingeBase(int type) : Loss(type | Type::HINGE), margin_(1.f) {}

 public:
  float margin() const { return this->margin_; }
  void set_margin(float val) { this->margin_ = val; }

 protected:
  float margin_;
};

class LSOL_EXPORTS HingeLoss : public HingeBase {
 public:
  HingeLoss() : HingeBase(Type::BC) {}

 public:
  virtual float loss(label_t label, float* predict, label_t predict_label,
                     int cls_num);

  virtual float gradient(label_t label, float* predict, label_t predict_label,
                         float* gradient, int cls_num);

};  // class HingeLoss

class LSOL_EXPORTS MaxScoreHingeLoss : public HingeBase {
 public:
  MaxScoreHingeLoss() : HingeBase(Type::MC) {}

 public:
  virtual float loss(label_t label, float* predict, label_t predict_label,
                     int cls_num);

  virtual float gradient(label_t label, float* predict, label_t predict_label,
                         float* gradient, int cls_num);
};

class LSOL_EXPORTS UniformHingeLoss : public HingeBase {
 public:
  UniformHingeLoss() : HingeBase(Type::MC) {}

 public:
  virtual float loss(label_t label, float* predict, label_t predict_label,
                     int cls_num);

  virtual float gradient(label_t label, float* predict, label_t predict_label,
                         float* gradient, int cls_num);
};

}  // namespace loss
}  // namespace lsol
#endif

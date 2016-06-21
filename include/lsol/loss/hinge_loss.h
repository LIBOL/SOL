/*********************************************************************************
*     File Name           :     hinge.h
*     Created By          :     yuewu
*     Description         :     hinge loss
**********************************************************************************/
#ifndef LSOL_LOSS_HINGE_LOSS_H__
#define LSOL_LOSS_HINGE_LOSS_H__

#include <lsol/loss/loss.h>
#include <functional>

namespace lsol {
namespace loss {

class LSOL_EXPORTS HingeBase : public Loss {
 public:
  HingeBase(int type);

 public:
  float margin() { return margin_; }
  void set_margin(float val) { this->margin_ = val; }
  void set_margin(
      const std::function<float(const pario::DataPoint&, float*, label_t,
                                float*, int)>& margin_handler) {
    this->margin_handler_ = margin_handler;
  }

 protected:
  float margin_;
  std::function<float(const pario::DataPoint&, float*, label_t, float*, int)>
      margin_handler_;
};

class LSOL_EXPORTS HingeLoss : public HingeBase {
 public:
  HingeLoss() : HingeBase(Type::BC) {}

 public:
  virtual float loss(const pario::DataPoint& dp, float* predict,
                     label_t predict_label, int cls_num);

  virtual float gradient(const pario::DataPoint& dp, float* predict,
                         label_t predict_label, float* gradient, int cls_num);

};  // class HingeLoss

class LSOL_EXPORTS MaxScoreHingeLoss : public HingeBase {
 public:
  MaxScoreHingeLoss() : HingeBase(Type::MC) {}

 public:
  virtual float loss(const pario::DataPoint& dp, float* predict,
                     label_t predict_label, int cls_num);

  virtual float gradient(const pario::DataPoint& dp, float* predict,
                         label_t predict_label, float* gradient, int cls_num);
};

class LSOL_EXPORTS UniformHingeLoss : public HingeBase {
 public:
  UniformHingeLoss() : HingeBase(Type::MC) {}

 public:
  virtual float loss(const pario::DataPoint& dp, float* predict,
                     label_t predict_label, int cls_num);

  virtual float gradient(const pario::DataPoint& dp, float* predict,
                         label_t predict_label, float* gradient, int cls_num);
};

}  // namespace loss
}  // namespace lsol
#endif

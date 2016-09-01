/*********************************************************************************
*     File Name           :     hinge.h
*     Created By          :     yuewu
*     Description         :     hinge loss
**********************************************************************************/
#ifndef SOL_LOSS_HINGE_LOSS_H__
#define SOL_LOSS_HINGE_LOSS_H__

#include <sol/loss/loss.h>
#include <functional>

namespace sol {
namespace loss {

class SOL_EXPORTS HingeBase : public Loss {
 public:
  HingeBase(int type);

 public:
  float margin() { return margin_; }
  void set_margin(float val) { this->margin_ = val; }
  void set_margin(const std::function<float(
      const pario::DataPoint&, float*, label_t, float*, int)>& margin_handler) {
    this->margin_handler_ = margin_handler;
  }

 protected:
  float margin_;
  std::function<float(const pario::DataPoint&, float*, label_t, float*, int)>
      margin_handler_;
};

class SOL_EXPORTS HingeLoss : public HingeBase {
 public:
  HingeLoss() : HingeBase(Type::BC) {}

 public:
  virtual float loss(const pario::DataPoint& dp, float* predict,
                     label_t predict_label, int cls_num);

  virtual float gradient(const pario::DataPoint& dp, float* predict,
                         label_t predict_label, float* gradient, int cls_num);

};  // class HingeLoss

class SOL_EXPORTS MaxScoreHingeLoss : public HingeBase {
 public:
  MaxScoreHingeLoss() : HingeBase(Type::MC) {}

 public:
  virtual float loss(const pario::DataPoint& dp, float* predict,
                     label_t predict_label, int cls_num);

  virtual float gradient(const pario::DataPoint& dp, float* predict,
                         label_t predict_label, float* gradient, int cls_num);
};

class SOL_EXPORTS UniformHingeLoss : public HingeBase {
 public:
  UniformHingeLoss() : HingeBase(Type::MC) {}

 public:
  virtual float loss(const pario::DataPoint& dp, float* predict,
                     label_t predict_label, int cls_num);

  virtual float gradient(const pario::DataPoint& dp, float* predict,
                         label_t predict_label, float* gradient, int cls_num);
};

}  // namespace loss
}  // namespace sol
#endif

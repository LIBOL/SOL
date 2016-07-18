/*********************************************************************************
*     File Name           :     hinge.h
*     Created By          :     yuewu
*     Description         :     hinge loss
**********************************************************************************/
#ifndef LSOL_LOSS_LOGISTIC_LOSS_H__
#define LSOL_LOSS_LOGISTIC_LOSS_H__

#include <lsol/loss/loss.h>
#include <functional>

namespace lsol {
namespace loss {

class LSOL_EXPORTS LogisticLoss : public Loss {
 public:
  LogisticLoss() : Loss(Type::BC) {}

 public:
  virtual float loss(const pario::DataPoint& dp, float* predict,
                     label_t predict_label, int cls_num);

  virtual float gradient(const pario::DataPoint& dp, float* predict,
                         label_t predict_label, float* gradient, int cls_num);

};  // class LogisticLoss

class LSOL_EXPORTS MaxScoreLogisticLoss : public Loss {
 public:
  MaxScoreLogisticLoss() : Loss(Type::MC) {}

 public:
  virtual float loss(const pario::DataPoint& dp, float* predict,
                     label_t predict_label, int cls_num);

  virtual float gradient(const pario::DataPoint& dp, float* predict,
                         label_t predict_label, float* gradient, int cls_num);
};

class LSOL_EXPORTS UniformLogisticLoss : public Loss {
 public:
  UniformLogisticLoss() : Loss(Type::MC) {}

 public:
  virtual float loss(const pario::DataPoint& dp, float* predict,
                     label_t predict_label, int cls_num);

  virtual float gradient(const pario::DataPoint& dp, float* predict,
                         label_t predict_label, float* gradient, int cls_num);
};

}  // namespace loss
}  // namespace lsol
#endif

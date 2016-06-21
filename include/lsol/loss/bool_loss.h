/*********************************************************************************
*     File Name           :     bool_loss.h
*     Created By          :     yuewu
*     Description         :     loss with yes or no
**********************************************************************************/
#ifndef LSOL_LOSS_BOOL_LOSS_H__
#define LSOL_LOSS_BOOL_LOSS_H__

#include <lsol/loss/loss.h>

namespace lsol {
namespace loss {

class LSOL_EXPORTS BoolLoss : public Loss {
 public:
  BoolLoss() : Loss(Type::BC | Type::BOOL) {}

 public:
  virtual float loss(const pario::DataPoint& dp, float* predict,
                     label_t predict_label, int cls_num);

  virtual float gradient(const pario::DataPoint& dp, float* predict,
                         label_t predict_label, float* gradient, int cls_num);

};  // class BoolLoss

class LSOL_EXPORTS MaxScoreBoolLoss : public Loss {
 public:
  MaxScoreBoolLoss() : Loss(Type::MC | Type::BOOL) {}

 public:
  virtual float loss(const pario::DataPoint& dp, float* predict,
                     label_t predict_label, int cls_num);

  virtual float gradient(const pario::DataPoint& dp, float* predict,
                         label_t predict_label, float* gradient, int cls_num);
};

class LSOL_EXPORTS UniformBoolLoss : public Loss {
 public:
  UniformBoolLoss() : Loss(Type::MC | Type::BOOL) {}

 public:
  virtual float loss(const pario::DataPoint& dp, float* predict,
                     label_t predict_label, int cls_num);

  virtual float gradient(const pario::DataPoint& dp, float* predict,
                         label_t predict_label, float* gradient, int cls_num);
};

}  // namespace loss
}  // namespace lsol
#endif

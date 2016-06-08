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
  /// \brief  calculate loss according to the label and predictions
  ///
  /// \param label true label
  /// \param predict prediction on each class
  /// \param cls_num number of classes
  ///
  /// \return loss of the prediction
  virtual float loss(label_t label, float* predict, int cls_num = 2);

  /// \brief  calculate the gradients according to the label and predictions
  ///
  /// \param label true label
  /// \param predict prediction on each class
  /// \param gradient resulted gradient on each class (without x)
  /// \param cls_num number of classes
  ///
  /// \return loss of the prediction
  virtual float gradient(label_t label, float* predict, float* gradient,
                         int cls_num = 2);

};  // class HingeLoss

}  // namespace loss
}  // namespace lsol
#endif

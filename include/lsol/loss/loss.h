/*********************************************************************************
*     File Name           :     loss.h
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-14 23:10]
*     Last Modified       :     [2016-02-18 23:33]
*     Description         :     base class for loss functions
**********************************************************************************/

#ifndef LSOL_LOSS_LOSS_H__
#define LSOL_LOSS_LOSS_H__

#include <string>

#include <lsol/util/types.h>
#include <lsol/util/reflector.h>

namespace lsol {
namespace loss {

class LSOL_EXPORTS Loss {
  DeclareReflectorBase(Loss);

 public:
  enum Type {
    // loss function for binary classification
    BC = 1,
    // loss function for binary classification
    MC = 2,
    // hinge-based loss function
    HINGE = 4,
  };

  inline static char Sign(float x) { return x >= 0.f ? 1 : -1; }

 public:
  Loss(int type) : type_(type) {}
  virtual ~Loss() {}

  int type() const { return this->type_; }

 public:
  /// \brief  calculate loss according to the label and predictions
  ///
  /// \param label true label
  /// \param predict prediction on each class
  /// \param cls_num number of classes
  ///
  /// \return loss of the prediction
  virtual float loss(label_t label, float* predict, int cls_num) = 0;

  /// \brief  calculate the gradients according to the label and predictions
  ///
  /// \param label true label
  /// \param predict prediction on each class
  /// \param gradient resulted gradient on each class (without x)
  /// \param cls_num number of classes
  ///
  /// \return loss of the prediction
  virtual float gradient(label_t label, float* predict, float* gradient,
                         int cls_num) = 0;

 public:
  const std::string& name() const { return name_; }
  void set_name(const std::string& name) { this->name_ = name; }

 protected:
  /// \brief  indicating it's a binary or multi-class loss
  int type_;
  std::string name_;
};

#define RegisterLoss(type, name, descr)                                  \
  type* type##_##CreateNewInstance() {                                   \
    type* ins = new type();                                              \
    ins->set_name(name);                                                 \
    return ins;                                                          \
  }                                                                      \
  ClassInfo __kClassInfo_##type##__(std::string(name) + "_loss",         \
                                    (void*)(type##_##CreateNewInstance), \
                                    descr);

}  // namespace loss
}  // namespace lsol

#endif

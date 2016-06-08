/*********************************************************************************
*     File Name           :     sgd.h
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-18 21:33]
*     Last Modified       :     [2016-03-09 19:22]
*     Description         :     Stochastic Gradient Descent
**********************************************************************************/

#ifndef LSOL_MODEL_OLM_SGD_H__
#define LSOL_MODEL_OLM_SGD_H__

#include <lsol/model/online_linear_model.h>

namespace lsol {
namespace model {

class SGD : public OnlineLinearModel {
 public:
  SGD(int class_num) : OnlineLinearModel(class_num) {
    this->loss_ = loss::Loss::Create("hinge");
  }
  virtual ~SGD() {}

  /// \brief  update model
  ///
  /// \param x training instance
  virtual void Update(const pario::DataPoint& x);
};  // class SGD

}  // namespace model
}  // namespace lsol
#endif

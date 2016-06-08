/*********************************************************************************
*     File Name           :     online_optimizer.h
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-18 22:39]
*     Last Modified       :     [2016-03-09 19:22]
*     Description         :     Online Optimizer
**********************************************************************************/
#ifndef LSOL_OPTIMIZER_ONLINE_OPTIMIZER_H__
#define LSOL_OPTIMIZER_ONLINE_OPTIMIZER_H__

#include <lsol/optimizer/optimizer.h>
#include <lsol/model/online_model.h>

namespace lsol {
namespace optimizer {

class LSOL_EXPORTS OnlineOptimizer : public Optimizer {
 public:
  OnlineOptimizer(model::Model *model);

  /// \brief  Train from a data set
  //
  /// \param data_iter data iterator
  //
  /// \return training error rate
  virtual float Train(pario::DataIter &data_iter);

 protected:
  lsol::model::OnlineModel *online_model_;

};  // class OnlineOptimizer

}  // namespace optimizer
}  // namespace lsol
#endif

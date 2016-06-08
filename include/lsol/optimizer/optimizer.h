/*********************************************************************************
*     File Name           :     optimizer.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-18 21:49]
*     Last Modified       :     [2016-03-09 19:22]
*     Description         :     Base class for optimizer
**********************************************************************************/

#ifndef LSOL_OPTIMIZER_OPTIMIZER_H__
#define LSOL_OPTIMIZER_OPTIMIZER_H__

#include <ostream>

#include <lsol/util/reflector.h>
#include <lsol/util/util.h>
#include <lsol/math/operator.h>
#include <lsol/model/model.h>
#include <lsol/pario/data_iter.h>

namespace lsol {
namespace optimizer {

class LSOL_EXPORTS Optimizer {
  DeclareReflectorBase(Optimizer, model::Model *model);

 public:
  Optimizer(model::Model *model)
      : model_(model),
        norm_type_(lsol::math::expr::op::OpType::kNone),
        max_index_(0) {}

  virtual ~Optimizer() {}

  void SetParameter(const std::string &name, const std::string &value);

 public:
  /// \brief  Train from a data set
  //
  /// \param data_iter data iterator
  //
  /// \return training error rate
  virtual float Train(pario::DataIter &data_iter) = 0;

  /// \brief  Test a dataset
  ///
  /// \param data_iter data iterator
  /// \param os output stream to store the predicted results
  ///
  /// \return test error rate
  float Test(pario::DataIter &data_iter, std::ostream *os);

 protected:
  void PreProcess(pario::DataPoint &x);
  void FilterFeatures(pario::DataPoint &x);

 public:
  /// \brief  load pre-selected features
  ///
  /// \param path pre-select file path
  ///
  /// \return status code, Status_OK if succeed
  int LoadPreSelFeatures(const std::string &path);

 protected:
  model::Model *model_;
  lsol::math::expr::op::OpType norm_type_;
  // max feature index
  index_t max_index_;
  // pre-selected features
  math::Vector<char> sel_feat_flags_;
};

#define RegisterOptimizer(type, name, descr)                                \
  type *type##_##CreateNewInstance(model::Model *m) { return new type(m); } \
  ClassInfo __kClassInfo_##type##__(std::string(name) + "_optimizer",       \
                                    (void *)(type##_##CreateNewInstance),   \
                                    descr);

}  // namespace optimizer
}  // namespace lsol
#endif

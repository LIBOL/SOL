/*********************************************************************************
*     File Name           :     sparse_model.h
*     Created By          :     yuewu
*     Description         :     base class for sparse models
**********************************************************************************/

#ifndef LSOL_MODEL_SPARSE_MODEL_H__
#define LSOL_MODEL_SPARSE_MODEL_H__

#include <lsol/pario/data_point.h>
#include <json/json.h>

namespace lsol {
namespace model {

class LSOL_EXPORTS Regularizer {
 public:
  Regularizer() : lambda_(0){};
  /// \brief  set model parameters
  ///
  /// \param name name of the parameter
  /// \param value value of the parameter in string
  ///
  /// \return status code, 0 if successfully
  virtual int SetParameter(const std::string &name, const std::string &value);

  /// \brief  finalize the regularization on weights
  ///
  /// \param w weight vector

  virtual void FinalizeRegularization(math::Vector<real_t> &w) {}

  /// \brief  Get Regularizer Information
  ///
  /// \param root root node of saver
  /// info
  virtual void GetRegularizerInfo(Json::Value &root) const;

 public:
  inline real_t lambda() const { return this->lambda_; }

 protected:
  // regularization weight
  real_t lambda_;
};

class LSOL_EXPORTS OnlineRegularizer : public Regularizer {
 public:
  virtual void BeginIterate(const pario::DataPoint &dp) {}
  virtual void EndIterate(const pario::DataPoint &dp, int cur_iter_num) {}
};

class LSOL_EXPORTS OnlineL1Regularizer : public OnlineRegularizer {
 public:
  OnlineL1Regularizer();

  virtual int SetParameter(const std::string &name, const std::string &value);

  virtual void FinalizeRegularization(math::Vector<real_t> &w);

 protected:
  real_t sparse_thresh_;
};

class LSOL_EXPORTS LazyOnlineL1Regularizer : public OnlineL1Regularizer {
 public:
  LazyOnlineL1Regularizer();

  virtual int SetParameter(const std::string &name, const std::string &value);

  virtual void BeginIterate(const pario::DataPoint &dp);
  virtual void EndIterate(const pario::DataPoint &dp, int cur_iter_num);

 public:
  inline const math::Vector<real_t> &last_update_time() const {
    return this->last_update_time_;
  };
  void set_initial_t(real_t t0) { this->initial_t_ = t0; }

 protected:
  real_t initial_t_;
  // record the last update time of each dimension
  math::Vector<real_t> last_update_time_;
};

}  // namespace model
}  // namespace lsol

#endif

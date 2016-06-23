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
  Regularizer() : lambda_(0.f){};
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
  inline float lambda() const { return this->lambda_; }

 protected:
  // regularization weight
  float lambda_;
};

class LSOL_EXPORTS OnlineRegularizer : public Regularizer {
 public:
  virtual void BeginIterate(const pario::DataPoint &dp) {}
  virtual void EndIterate(const pario::DataPoint &dp) {}
};

class LSOL_EXPORTS OnlineL1Regularizer : public OnlineRegularizer {
 public:
  OnlineL1Regularizer();

  virtual int SetParameter(const std::string &name, const std::string &value);

  virtual void BeginIterate(const pario::DataPoint &dp);
  virtual void EndIterate(const pario::DataPoint &dp);

  virtual void FinalizeRegularization(math::Vector<real_t> &w);

 public:
  inline real_t cur_iter_num() const { return cur_iter_num_; }
  inline const math::Vector<real_t> &last_update_time() const {
    return this->last_update_time_;
  };

 protected:
  real_t sparse_thresh_;
  real_t cur_iter_num_;
  // record the last update time of each dimension
  math::Vector<real_t> last_update_time_;
};

}  // namespace model
}  // namespace lsol

#endif

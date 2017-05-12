/*********************************************************************************
*     File Name           :     stochastic_model.h
*     Created By          :     yuewu
*     Creation Date       :     [2017-05-08 22:39]
*     Last Modified       :     [2017-05-09 17:35]
*     Description         :     stochastic model
**********************************************************************************/

#ifndef SOL_MODEL_STOCHASTIC_MODEL_H__
#define SOL_MODEL_STOCHASTIC_MODEL_H__

#include <sol/model/model.h>

#include <vector>

#include <sol/pario/data_point.h>

namespace sol {
namespace model {

class StochasticModel : public Model {
 public:
  StochasticModel(int class_num, const std::string& type);

  virtual ~StochasticModel() {}

  /// \brief  set model parameters
  ///
  /// \param name name of the parameter
  /// \param value value of the parameter in string
  virtual void SetParameter(const std::string& name, const std::string& value);

 public:
  /// \brief  Train from a data set
  //
  /// \param data_iter data iterator
  //
  /// \return training error rate
  virtual float Train(pario::DataIter& data_iter);

  /// \brief  iterate the model with one new instance
  ///
  /// \param mb training mini-batch
  /// \param predicts predicted labels over classes
  /// \param scores predicted scores over classes
  ///
  /// \return error or loss
  virtual float Iterate(const pario::MiniBatch& mb, label_t* predicts,
                        float* scores);

 protected:
  /// \brief  Get Model Information
  ///
  /// \param root root node of saver
  /// info
  virtual void GetModelInfo(Json::Value& root) const;

  /// \brief  load model from string
  ///
  /// \param root root node of model info
  ///
  /// \return status code, Status_OK if load successfully
  virtual int SetModelInfo(const Json::Value& root);

 public:
  int cur_iter_num() const { return this->cur_iter_num_; }

 protected:
  inline float bias_eta() const { return this->bias_eta0_ * this->eta_; }
  void set_initial_t(int initial_t);
  virtual void update_dim(index_t dim) { this->dim_ = dim; }

 protected:
  // initial learning rate for bias
  float bias_eta0_;
  // initial learning step
  int initial_t_;
  // current iteration number
  int cur_iter_num_;
  // current data number
  size_t cur_batch_num_;
  // current error number
  size_t cur_err_num_;

  // dimension of input feature: can be the same to feature, or with an extra
  // bias
  index_t dim_;
  // learning rate
  float eta_;
};  // class  StochasticModel

}  // namespace model
}  // namespace sol

#endif

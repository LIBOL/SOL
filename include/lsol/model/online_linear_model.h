/*********************************************************************************
*     File Name           :     online_linear_model.h
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-18 16:38]
*     Last Modified       :     [2016-03-09 19:22]
*     Description         :     online linear model
**********************************************************************************/

#ifndef LSOL_MODEL_ONLINE_LINEAR_MODEL_H__
#define LSOL_MODEL_ONLINE_LINEAR_MODEL_H__

#include <lsol/model/online_model.h>
#include <lsol/math/vector.h>

namespace lsol {
namespace model {

class OnlineLinearModel : public OnlineModel {
 public:
  OnlineLinearModel(int class_num);
  virtual ~OnlineLinearModel();

 public:
  /// \brief  iterate the model with one new instance
  ///
  /// \param x training instance
  /// \param predict predicted scores on each class
  ///
  /// \return predicted class label
  virtual label_t Iterate(const pario::DataPoint& x, float* predict);
  /// \brief  predict the label of data
  ///
  /// \param x input data
  /// \param predicts predicted scores on the data
  ///
  /// \return predicted class label
  virtual label_t Predict(const pario::DataPoint& x, float* predicts);

 protected:
  /// \brief  update model
  ///
  /// \param x training instance
  /// \param predict predicted values
  /// \param loss prediction loss
  virtual void Update(const pario::DataPoint& x, const float* predict,
                      float loss) = 0;

 protected:
  /// \brief  serialize model parameters
  ///
  /// \param root root node to save the parameters
  virtual void GetModelParam(Json::Value& root) const;

  /// \brief  load model parameters from stream
  ///
  /// \param root root node of model info
  ///
  /// \return status code, zero if ok
  virtual int SetModelParam(const Json::Value& root);

 public:
  const math::Vector<real_t>& weights(int cls_id) const {
    return this->weights_[cls_id];
  }
  math::Vector<real_t>& weights(int cls_id) { return this->weights_[cls_id]; }
  virtual void update_dim(index_t dim);

 protected:
  // the first element is zero
  math::Vector<real_t>* weights_;
  // gradients for each class
  real_t* gradients_;
};  // class OnlineLinearModel
}  // namespace model
}  // namespace lsol

#endif

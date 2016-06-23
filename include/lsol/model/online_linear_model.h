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
  virtual void BeginTrain() { OnlineModel::BeginTrain(); }

  virtual void EndTrain() {
    if (this->regularizer_ != nullptr) {
      for (int c = 0; c < this->clf_num_; ++c) {
        this->regularizer_->FinalizeRegularization(w(c));
      }
    }
    OnlineModel::EndTrain();
  }
  virtual label_t Predict(const pario::DataPoint& dp, float* predicts);

  virtual label_t Iterate(const pario::DataPoint& dp, float* predicts);

 protected:
  /// \brief  update model
  ///
  /// \param dp training instance
  /// \param predict predicted values
  /// \param loss prediction loss
  virtual void Update(const pario::DataPoint& dp, const float* predict,
                      float loss) = 0;
  virtual void update_dim(index_t dim);

  virtual label_t TrainPredict(const pario::DataPoint& dp, float* predicts);

 public:
  virtual float model_sparsity() const;

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
  const math::Vector<real_t>& w(int cls_id) const {
    return this->weights_[cls_id];
  }
  math::Vector<real_t>& w(int cls_id) { return this->weights_[cls_id]; }

  real_t g(int cls_id) const { return this->gradients_[cls_id]; }
  real_t& g(int cls_id) { return this->gradients_[cls_id]; }

 private:
  // the first element is zero
  math::Vector<real_t>* weights_;
  // gradients for each class
  real_t* gradients_;
};  // class OnlineLinearModel
}  // namespace model
}  // namespace lsol

#endif

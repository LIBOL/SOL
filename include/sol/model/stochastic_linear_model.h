/*********************************************************************************
*     File Name           :     stochastic_linear_model.h
*     Created By          :     yuewu
*     Creation Date       :     [2017-05-09 16:32]
*     Last Modified       :     [2017-05-15 09:58]
*     Description         :     stochastic linear model
**********************************************************************************/

#ifndef SOL_MODEL_STOCHASTIC_LINEAR_MODEL_H__
#define SOL_MODEL_STOCHASTIC_LINEAR_MODEL_H__

#include <sol/math/vector.h>
#include <sol/model/stochastic_model.h>

namespace sol {
namespace model {

class StochasticLinearModel : public StochasticModel {
 public:
  StochasticLinearModel(int class_num);
  virtual ~StochasticLinearModel();

 public:
  virtual label_t Predict(const pario::DataPoint& dp, float* predicts);

  virtual float Iterate(const pario::MiniBatch& mb, label_t* predicts,
                        float* scores);

 protected:
  virtual void update_dim(index_t dim);

 public:
  virtual float model_sparsity();

 protected:
  virtual void GetModelParam(std::ostream& os) const;

  virtual int SetModelParam(std::istream& is);

 public:
  const math::Vector<real_t>& w(int cls_id) const {
    return this->weights_[cls_id];
  }
  math::Vector<real_t>& w(int cls_id) { return this->weights_[cls_id]; }

  inline real_t g(int cls_id) const { return this->gradients_[cls_id]; }
  inline real_t& g(int cls_id) { return this->gradients_[cls_id]; }

 private:
  // the first element is zero
  math::Vector<real_t>* weights_;
  // gradients for each class
  real_t* gradients_;
};  // class StochasticLinearModel

}  // namespace model
}  // namespace sol

#endif

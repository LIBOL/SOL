/*********************************************************************************
*     File Name           :     perceptron.h
*     Created By          :     yuewu
*     Description         :     Perceptron algorithm
**********************************************************************************/
#ifndef SOL_MODEL_OLM_PERCEPTRON_H__
#define SOL_MODEL_OLM_PERCEPTRON_H__

#include <sol/model/online_linear_model.h>

namespace sol {
namespace model {

class Perceptron : public OnlineLinearModel {
 public:
  Perceptron(int class_num);

  virtual void SetParameter(const std::string& name, const std::string& value);

 protected:
  virtual void Update(const pario::DataPoint& dp, const float* predict,
                      float loss);
};  // class Perceptron

}  // namespace model
}  // namespace sol
#endif

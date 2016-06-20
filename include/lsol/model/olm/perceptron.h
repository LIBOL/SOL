/*********************************************************************************
*     File Name           :     perceptron.h
*     Created By          :     yuewu
*     Description         :     Perceptron algorithm
**********************************************************************************/
#ifndef LSOL_MODEL_OLM_PERCEPTRON_H__
#define LSOL_MODEL_OLM_PERCEPTRON_H__

#include <lsol/model/online_linear_model.h>

namespace lsol {
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
}  // namespace lsol
#endif

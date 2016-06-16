/*********************************************************************************
*     File Name           :     sgd.h
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-18 21:33]
*     Last Modified       :     [2016-03-09 19:22]
*     Description         :     Online Gradient Descent
**********************************************************************************/

#ifndef LSOL_MODEL_OLM_OGD_H__
#define LSOL_MODEL_OLM_OGD_H__

#include <lsol/model/online_linear_model.h>

namespace lsol {
namespace model {

class OGD : public OnlineLinearModel {
 public:
  OGD(int class_num);

  virtual void SetParameter(const std::string& name, const std::string& value);

 protected:
  virtual void Update(const pario::DataPoint& x, const float* predict,
                      float loss);
  virtual void GetModelInfo(Json::Value& root) const;

 protected:
  void set_power_t(float power_t);

 protected:
  // power_t of the decreasing coefficient of learning rate
  float power_t_;
  // initial learning rate
  float eta0_;

  float (*pow_)(int iter, float power_t);
};  // class OGD

}  // namespace model
}  // namespace lsol
#endif

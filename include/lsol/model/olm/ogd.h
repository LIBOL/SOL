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
  virtual void Update(const pario::DataPoint& dp, const float* predict,
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

/// \brief  Sparse online learning via Truncated Gradient
class STG : public OGD {
 public:
  STG(int class_num);

  virtual void SetParameter(const std::string& name, const std::string& value);
  virtual void BeginTrain();
  virtual void EndTrain();

 protected:
  virtual label_t TrainPredict(const pario::DataPoint& dp, float* predicts);
  void update_dim(index_t dim);

  virtual void GetModelInfo(Json::Value& root) const;

 protected:
  // truncate every k steps
  int k_;
  OnlineL1Regularizer l1_;
  math::Vector<real_t> last_trunc_time_;
};  // class STG

}  // namespace model
}  // namespace lsol
#endif

/*********************************************************************************
*     File Name           :     sop.h
*     Created By          :     yuewu
*     Description         :     second order perceptron
**********************************************************************************/

#ifndef LSOL_MODEL_OLM_SOP_H__
#define LSOL_MODEL_OLM_SOP_H__

#include <lsol/model/online_linear_model.h>

namespace lsol {
namespace model {

class SOP : public OnlineLinearModel {
 public:
  SOP(int class_num);
  virtual ~SOP();

  virtual void SetParameter(const std::string& name, const std::string& value);

  virtual void BeginTrain();

  virtual label_t Predict(const pario::DataPoint& dp, float* predicts);

 protected:
  virtual void Update(const pario::DataPoint& dp, const float* predict,
                      float loss);
  virtual void update_dim(index_t dim);

  virtual void GetModelInfo(Json::Value& root) const;
  virtual void GetModelParam(Json::Value& root) const;
  virtual int SetModelParam(const Json::Value& root);

 protected:
  float a_;
  math::Vector<real_t>* v_;
  math::Vector<real_t> S_;
  math::Vector<real_t> X_;

};  // class SOP

}  // namespace model
}  // namespace lsol
#endif

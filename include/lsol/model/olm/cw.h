/*********************************************************************************
*     File Name           :     cw.h
*     Created By          :     yuewu
*     Description         :     Confidence Weighted Online Learing
**********************************************************************************/

#ifndef LSOL_MODEL_OLM_CW_H__
#define LSOL_MODEL_OLM_CW_H__

#include <lsol/model/online_linear_model.h>
#include <lsol/loss/hinge_loss.h>

namespace lsol {
namespace model {

class CW : public OnlineLinearModel {
 public:
  CW(int class_num);

  virtual void SetParameter(const std::string& name, const std::string& value);

  virtual label_t Predict(const pario::DataPoint& dp, float* predicts);

 protected:
  virtual void Update(const pario::DataPoint& dp, const float* predict,
                      float loss);
  virtual void update_dim(index_t dim);

  virtual void GetModelInfo(Json::Value& root) const;
  virtual void GetModelParam(Json::Value& root) const;
  virtual int SetModelParam(const Json::Value& root);

 protected:
  loss::HingeBase* hinge_base_;
  // initial variance
  float a_;
  // inverse normal distribution threshold
  float phi_;
  // x'Sigma x
  float Vi_;
  math::Vector<real_t> Sigma_;

};  // class CW

}  // namespace model
}  // namespace lsol
#endif

/*********************************************************************************
*     File Name           :     pa.h
*     Created By          :     yuewu
*     Description         :     Online Passive Aggressive Algorithms
**********************************************************************************/
#ifndef LSOL_MODEL_OLM_PA_H__
#define LSOL_MODEL_OLM_PA_H__

#include <lsol/model/online_linear_model.h>

namespace lsol {
namespace model {

class PA : public OnlineLinearModel {
 public:
  PA(int class_num);

 protected:
  virtual void Update(const pario::DataPoint& dp, const float* predict,
                      float loss);

 protected:
  // the coeffient difference between binary and multiclass classification
  float eta_coeff_;
};  // class PA

class PAI : public PA {
 public:
  PAI(int class_num) : PA(class_num), C_(1.f) {}

  virtual void SetParameter(const std::string& name, const std::string& value);

 protected:
  virtual void Update(const pario::DataPoint& dp, const float* predict,
                      float loss);
  virtual void GetModelInfo(Json::Value& root) const;

 protected:
  float C_;
};  // class PAI

class PAII : public PA {
 public:
  PAII(int class_num) : PA(class_num), C_(1.f) {}

  virtual void SetParameter(const std::string& name, const std::string& value);

 protected:
  virtual void Update(const pario::DataPoint& dp, const float* predict,
                      float loss);
  virtual void GetModelInfo(Json::Value& root) const;

 protected:
  float C_;
};  // class PAII

}  // namespace model
}  // namespace lsol
#endif

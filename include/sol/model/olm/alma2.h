/*********************************************************************************
*     File Name           :     alma2.h
*     Created By          :     yuewu
*     Description         :     Approximate Large Margin Algorithm with norm 2
**********************************************************************************/
#ifndef SOL_MODEL_OLM_ALMA2_H__
#define SOL_MODEL_OLM_ALMA2_H__

#include <sol/model/online_linear_model.h>
#include <sol/loss/hinge_loss.h>

namespace sol {
namespace model {

class ALMA2 : public OnlineLinearModel {
 public:
  ALMA2(int class_num);

  virtual void SetParameter(const std::string& name, const std::string& value);

 public:
  virtual void BeginTrain();

 protected:
  virtual void Update(const pario::DataPoint& dp, const float* predict,
                      float loss);

  virtual void GetModelInfo(Json::Value& root) const;

 protected:
  loss::HingeBase* hinge_base_;
  int p_;
  float alpha_;
  float C_;
  float B_;
  // sqrt(p_ - 1)
  float square_p1_;
  int k_;
};  // class ALMA2

}  // namespace model
}  // namespace sol
#endif

/*********************************************************************************
*     File Name           :     alma2.h
*     Created By          :     yuewu
*     Description         :     Approximate Large Margin Algorithm with norm 2
**********************************************************************************/
#ifndef LSOL_MODEL_OLM_ALMA2_H__
#define LSOL_MODEL_OLM_ALMA2_H__

#include <lsol/model/online_linear_model.h>
#include <lsol/loss/hinge_loss.h>

namespace lsol {
namespace model {

class ALMA2 : public OnlineLinearModel {
 public:
  ALMA2(int class_num);

  /// \brief  set model parameters, disable some parameters
  ///
  /// \param name name of the parameter
  /// \param value value of the parameter in string
  virtual void SetParameter(const std::string& name, const std::string& value);

 protected:
  /// \brief  calculate the learning rate
  virtual void CalculateLearningRate();

 public:
  /// \brief  initialize the model for training
  virtual void BeginTrain();

  /// \brief  update model
  ///
  /// \param x training instance
  virtual void Update(const pario::DataPoint& x);

 protected:
  /// \brief  Get Model Information
  ///
  /// \param root root node of saver
  /// info
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
}  // namespace lsol
#endif

/*********************************************************************************
*     File Name           :     eccw.h
*     Created By          :     yuewu
*     Description         :     Exact Convex Confidence Weighted Online Learning
**********************************************************************************/
#ifndef SOL_MODEL_OLM_ECCW_H__
#define SOL_MODEL_OLM_ECCW_H__

#include <sol/model/online_linear_model.h>
#include <sol/loss/hinge_loss.h>

namespace sol {
namespace model {

class ECCW : public OnlineLinearModel {
 public:
  ECCW(int class_num);
  virtual ~ECCW();

  virtual void SetParameter(const std::string& name, const std::string& value);

  virtual void BeginTrain();

 protected:
  virtual void Update(const pario::DataPoint& dp, const float* predict,
                      float loss);
  virtual void update_dim(index_t dim);

 protected:
  void set_phi(float phi);

 protected:
  virtual void GetModelInfo(Json::Value& root) const;
  virtual void GetModelParam(std::ostream& os) const;
  virtual int SetModelParam(std::istream& is);

 protected:
  loss::HingeBase* hinge_base_;
  // initial variance
  float a_;
  // inverse normal distribution threshold
  float phi_;
  // x'Sigma x
  float vi_;
  float psi_;
  float xi_;
  math::Vector<real_t>* Sigmas_;

};  // class ECCW
}  // namespace model
}  // namespace sol
#endif

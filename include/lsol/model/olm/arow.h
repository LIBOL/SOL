/*********************************************************************************
*     File Name           :     arow.h
*     Created By          :     yuewu
*     Description         :     Adaptive Regularization of Weight Vectors
**********************************************************************************/

#ifndef LSOL_MODEL_OLM_AROW_H__
#define LSOL_MODEL_OLM_AROW_H__

#include <lsol/model/online_linear_model.h>

namespace lsol {
namespace model {

class AROW : public OnlineLinearModel {
 public:
  AROW(int class_num);

  virtual void SetParameter(const std::string& name, const std::string& value);

 protected:
  virtual void Update(const pario::DataPoint& x, const float* predict,
                      float loss);
  virtual void update_dim(index_t dim);

  virtual void GetModelInfo(Json::Value& root) const;
  virtual void GetModelParam(Json::Value& root) const;
  virtual int SetModelParam(const Json::Value& root);

 protected:
  float r_;
  math::Vector<real_t> Sigma_;

};  // class AROW

}  // namespace model
}  // namespace lsol
#endif

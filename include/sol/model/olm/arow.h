/*********************************************************************************
*     File Name           :     arow.h
*     Created By          :     yuewu
*     Description         :     Adaptive Regularization of Weight Vectors
**********************************************************************************/

#ifndef SOL_MODEL_OLM_AROW_H__
#define SOL_MODEL_OLM_AROW_H__

#include <sol/model/online_linear_model.h>

namespace sol {
namespace model {

class AROW : public OnlineLinearModel {
 public:
  AROW(int class_num);
  virtual ~AROW();

  virtual void SetParameter(const std::string& name, const std::string& value);

 protected:
  virtual void Update(const pario::DataPoint& dp, const float* predict,
                      float loss);
  virtual void update_dim(index_t dim);

  virtual void GetModelInfo(Json::Value& root) const;
  virtual void GetModelParam(std::ostream& os) const;
  virtual int SetModelParam(std::istream& is);

 protected:
  float r_;
  math::Vector<real_t>* Sigmas_;

};  // class AROW

}  // namespace model
}  // namespace sol
#endif

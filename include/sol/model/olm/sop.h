/*********************************************************************************
*     File Name           :     sop.h
*     Created By          :     yuewu
*     Description         :     second order perceptron
**********************************************************************************/

#ifndef SOL_MODEL_OLM_SOP_H__
#define SOL_MODEL_OLM_SOP_H__

#include <sol/model/online_linear_model.h>

namespace sol {
namespace model {

class SOP : public OnlineLinearModel {
 public:
  SOP(int class_num);
  virtual ~SOP();

  virtual void SetParameter(const std::string& name, const std::string& value);
  virtual void EndTrain();

 protected:
  virtual label_t TrainPredict(const pario::DataPoint& dp, float* predicts);

 protected:
  virtual void Update(const pario::DataPoint& dp, const float* predict,
                      float loss);
  virtual void update_dim(index_t dim);

  virtual void GetModelInfo(Json::Value& root) const;
  virtual void GetModelParam(std::ostream& os) const;
  virtual int SetModelParam(std::istream& is);

 protected:
  math::Vector<real_t>& v(int cls_id) { return this->v_[cls_id]; }
  const math::Vector<real_t>& v(int cls_id) const { return this->v_[cls_id]; }

 protected:
  float a_;
  math::Vector<real_t>* v_;
  math::Vector<real_t> X_;

};  // class SOP

}  // namespace model
}  // namespace sol
#endif

/*********************************************************************************
*     File Name           :     ada_fobos.h
*     Created By          :     yuewu
*     Description         :     Adaptive Subgradient FOBOS
**********************************************************************************/

#ifndef LSOL_MODEL_OLM_ADA_FOBOS_H__
#define LSOL_MODEL_OLM_ADA_FOBOS_H__

#include <lsol/model/online_linear_model.h>

namespace lsol {
namespace model {

class AdaFOBOS : public OnlineLinearModel {
 public:
  AdaFOBOS(int class_num);
  virtual ~AdaFOBOS();

  virtual void SetParameter(const std::string& name, const std::string& value);

 protected:
  virtual void Update(const pario::DataPoint& dp, const float* predict,
                      float loss);
  virtual void update_dim(index_t dim);

  virtual void GetModelInfo(Json::Value& root) const;
  virtual void GetModelParam(Json::Value& root) const;
  virtual int SetModelParam(const Json::Value& root);

 protected:
  float delta_;
  math::Vector<real_t>* H_;

};  // class AdaFOBOS

/// \brief  AdaFOBOS with l1 regularization
class AdaFOBOS_L1 : public AdaFOBOS {
 public:
  AdaFOBOS_L1(int class_num);

  virtual void BeginTrain();
  virtual void EndTrain();

 protected:
  virtual label_t TrainPredict(const pario::DataPoint& dp, float* predicts);

 protected:
  LazyOnlineL1Regularizer l1_;
};

}  // namespace model
}  // namespace lsol
#endif

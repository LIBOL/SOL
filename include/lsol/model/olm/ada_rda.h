/*********************************************************************************
*     File Name           :     ada_rda.h
*     Created By          :     yuewu
*     Description         :     Adaptive Subgradient RDA
**********************************************************************************/

#ifndef LSOL_MODEL_OLM_ADA_RDA_H__
#define LSOL_MODEL_OLM_ADA_RDA_H__

#include <lsol/model/online_linear_model.h>

namespace lsol {
namespace model {

class AdaRDA : public OnlineLinearModel {
 public:
  AdaRDA(int class_num);
  virtual ~AdaRDA();

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
  // sum of gradients
  math::Vector<real_t>* ut_;

};  // class AdaRDA

/// \brief  AdaRDA with l1 regularization
class AdaRDA_L1 : public AdaRDA {
 public:
  AdaRDA_L1(int class_num);

  virtual void EndTrain();

 protected:
  virtual label_t TrainPredict(const pario::DataPoint& dp, float* predicts);

 protected:
  OnlineL1Regularizer l1_;
};

}  // namespace model
}  // namespace lsol
#endif

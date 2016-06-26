/*********************************************************************************
*     File Name           :     rda.h
*     Created By          :     yuewu
*     Description         :     Regularized Dual Averaging
**********************************************************************************/

#ifndef LSOL_MODEL_OLM_RDA_H__
#define LSOL_MODEL_OLM_RDA_H__

#include <lsol/model/online_linear_model.h>

namespace lsol {
namespace model {

/// \brief  RDA with `l2^2` as the proximal function
class RDA : public OnlineLinearModel {
 public:
  RDA(int class_num);
  virtual ~RDA();

  virtual void SetParameter(const std::string& name, const std::string& value);
  virtual void EndTrain();

 protected:
  virtual label_t TrainPredict(const pario::DataPoint& dp, float* predicts);
  virtual void Update(const pario::DataPoint& dp, const float* predict,
                      float loss);
  virtual void update_dim(index_t dim);

  virtual void GetModelInfo(Json::Value& root) const;
  virtual void GetModelParam(Json::Value& root) const;
  virtual int SetModelParam(const Json::Value& root);

 protected:
  float sigma_;
  // sum of gradients
  math::Vector<real_t>* ut_;
};  // class RDA

/// \brief  RDA with l1 regularization
class RDA_L1 : public RDA {
 public:
  RDA_L1(int class_num);

  virtual void EndTrain();

 protected:
  virtual label_t TrainPredict(const pario::DataPoint& dp, float* predicts);

 protected:
  OnlineL1Regularizer l1_;
};

/// \brief  Enhanced RDA with l1 regularization
class ERDA_L1 : public RDA {
 public:
  ERDA_L1(int class_num);

  virtual void SetParameter(const std::string& name, const std::string& value);
  virtual void EndTrain();

 protected:
  virtual label_t TrainPredict(const pario::DataPoint& dp, float* predicts);

  virtual void GetModelInfo(Json::Value& root) const;

 protected:
  float rou_;
  OnlineL1Regularizer l1_;
};

}  // namespace model
}  // namespace lsol
#endif

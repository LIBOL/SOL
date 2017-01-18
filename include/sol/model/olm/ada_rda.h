/*********************************************************************************
*     File Name           :     ada_rda.h
*     Created By          :     yuewu
*     Description         :     Adaptive Subgradient RDA
**********************************************************************************/

#ifndef SOL_MODEL_OLM_ADA_RDA_H__
#define SOL_MODEL_OLM_ADA_RDA_H__

#include <sol/model/online_linear_model.h>
#include <sol/util/heap.h>

namespace sol {
namespace model {

class AdaRDA : public OnlineLinearModel {
 public:
  AdaRDA(int class_num);
  virtual ~AdaRDA();

  virtual void SetParameter(const std::string& name, const std::string& value);
  virtual void EndTrain();

 protected:
  virtual void Update(const pario::DataPoint& dp, const float* predict,
                      float loss);
  virtual void update_dim(index_t dim);

  virtual void GetModelInfo(Json::Value& root) const;
  virtual void GetModelParam(std::ostream& os) const;
  virtual int SetModelParam(std::istream& is);

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

class AdaRDA_OFS: public AdaRDA {
 public:
  AdaRDA_OFS(int class_num);
  virtual ~AdaRDA_OFS();

  virtual void SetParameter(const std::string& name, const std::string& value);
  virtual void BeginTrain();
  virtual void EndTrain();

 protected:
  virtual void Update(const pario::DataPoint& dp, const float* predict,
                      float loss);
  virtual void update_dim(index_t dim);

 protected:
  math::Vector<real_t>* H_sum_;
  OnlineRegularizer l0_;
  MinHeap min_heap_;
};
}  // namespace model
}  // namespace sol
#endif

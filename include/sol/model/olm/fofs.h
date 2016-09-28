/*********************************************************************************
*     File Name           :     fofs.h
*     Created By          :     yuewu
*     Description         :     First Order Online Feature Selection
**********************************************************************************/

#ifndef SOL_MODEL_OLM_FOFS_H__
#define SOL_MODEL_OLM_FOFS_H__

#include <sol/model/online_linear_model.h>
#include <sol/util/heap.h>

namespace sol {
namespace model {

class FOFS : public OnlineLinearModel {
 public:
  FOFS(int class_num);
  virtual ~FOFS();

  virtual void SetParameter(const std::string& name, const std::string& value);
  virtual void BeginTrain();

 protected:
  virtual void Update(const pario::DataPoint& dp, const float* predict,
                      float loss);
  virtual void update_dim(index_t dim);

  virtual void GetModelInfo(Json::Value& root) const;

 protected:
  float lambda_;
  index_t B_;
  math::Vector<real_t>* abs_weights_;
  MinHeap* min_heap_;

  float norm_coeff_;
  float momentum_;
};

}  // namespace model
}  // namespace sol

#endif

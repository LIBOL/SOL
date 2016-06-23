/*********************************************************************************
*     File Name           :     regularizer.cc
*     Created By          :     yuewu
*     Description         :     model regularizers
**********************************************************************************/

#include "lsol/model/regularizer.h"
#include "lsol/util/error_code.h"

#include <string>

using namespace std;
using namespace lsol::math::expr;

namespace lsol {
namespace model {

int Regularizer::SetParameter(const std::string &name,
                              const std::string &value) {
  int ret = Status_OK;
  if (name == "lambda") {
    this->lambda_ = stof(value);
  } else {
    ret = Status_Invalid_Argument;
  }
  return ret;
}
void Regularizer::GetRegularizerInfo(Json::Value &root) const {
  root["regularizer"]["lambda"] = this->lambda_;
}

OnlineL1Regularizer::OnlineL1Regularizer() : sparse_thresh_(1e-5f) {
  this->last_update_time_.resize(1);
  this->last_update_time_ = 0;
}

int OnlineL1Regularizer::SetParameter(const std::string &name,
                                      const std::string &value) {
  if (name == "sparse_thresh") {
    this->sparse_thresh_ = stof(value);
  } else if (name == "t0") {
    this->initial_t_ = stof(value);
  } else {
    return OnlineRegularizer::SetParameter(name, value);
  }
  return Status_OK;
}

void OnlineL1Regularizer::BeginIterate(const pario::DataPoint &dp) {
  // update dim
  size_t d = this->last_update_time_.dim();
  if (dp.dim() > d) {
    this->last_update_time_.resize(dp.dim());
    real_t t0 = this->initial_t_;
    this->last_update_time_.slice_op([t0](float &val) { val = t0; }, d);
  }
}

void OnlineL1Regularizer::EndIterate(const pario::DataPoint &dp,
                                     int cur_iter_num) {
  // update last update time
  const auto &x = dp.data();
  auto &last_update_time = this->last_update_time_;
  real_t time_stamp = cur_iter_num;
  x.indexes().slice_op([&last_update_time, time_stamp](const index_t &idx) {
    last_update_time[idx] = time_stamp;
  });
  this->last_update_time_[0] = time_stamp;
}

void OnlineL1Regularizer::FinalizeRegularization(math::Vector<real_t> &w) {
  float thresh = this->sparse_thresh_;
  w.slice_op([thresh](real_t &val) { truncate(val, thresh); });
}

}  // namespace model
}  // namespace lsol

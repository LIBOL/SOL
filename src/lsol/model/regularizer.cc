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

OnlineL1Regularizer::OnlineL1Regularizer() : sparse_thresh_(1e-6f) {}

int OnlineL1Regularizer::SetParameter(const std::string &name,
                                      const std::string &value) {
  if (name == "sparse_thresh") {
    this->sparse_thresh_ = stof(value);
  } else {
    return OnlineRegularizer::SetParameter(name, value);
  }
  return Status_OK;
}

void OnlineL1Regularizer::FinalizeRegularization(math::Vector<real_t> &w) {
  float thresh = this->sparse_thresh_;
  w.slice_op([thresh](real_t &val) {
    if (val < thresh && val > -thresh) val = 0;
  });
}

LazyOnlineL1Regularizer::LazyOnlineL1Regularizer() : initial_t_(0) {
  this->last_update_time_.resize(1);
  this->last_update_time_ = 0;
}

int LazyOnlineL1Regularizer::SetParameter(const std::string &name,
                                          const std::string &value) {
  if (name == "t0") {
    this->initial_t_ = stof(value);
    this->last_update_time_ = this->initial_t_;
  } else {
    return OnlineL1Regularizer::SetParameter(name, value);
  }
  return Status_OK;
}

void LazyOnlineL1Regularizer::BeginIterate(const pario::DataPoint &dp) {
  // update dim
  size_t d = this->last_update_time_.dim();
  if (dp.dim() > d) {
    this->last_update_time_.resize(dp.dim());
    real_t t0 = this->initial_t_;
    this->last_update_time_.slice_op([t0](float &val) { val = t0; }, d);
  }
}

void LazyOnlineL1Regularizer::EndIterate(const pario::DataPoint &dp,
                                         int cur_iter_num) {
  // update last update time
  const auto &x = dp.data();
  auto &last_update_time = this->last_update_time_;
  real_t time_stamp = real_t(cur_iter_num);
  x.indexes().slice_op([&last_update_time, time_stamp](const index_t &idx) {
    last_update_time[idx] = time_stamp;
  });
  this->last_update_time_[0] = time_stamp;
}

}  // namespace model
}  // namespace lsol

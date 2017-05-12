/*********************************************************************************
*     File Name           :     stochastic_linear_model.cc
*     Created By          :     yuewu
*     Creation Date       :     [2017-05-09 16:41]
*     Last Modified       :     [2017-05-10 22:33]
*     Description         :
**********************************************************************************/

#include "sol/model/stochastic_linear_model.h"

#include <algorithm>
#include <limits>
#include <random>

#include "sol/util/util.h"

using namespace std;
using namespace sol::math;
using namespace sol::pario;

namespace sol {
namespace model {

StochasticLinearModel::StochasticLinearModel(int class_num)
    : StochasticModel(class_num, "stochastic_linear"),
      weights_(nullptr),
      gradients_(nullptr) {
  this->weights_ = new Vector<real_t>[this->clf_num_];
  this->gradients_ = new real_t[this->clf_num_];

  for (int i = 0; i < this->clf_num_; ++i) {
    w(i).resize(this->dim_);
    w(i) = 0;
    g(i) = 0;
  }

  if (class_num == 2) {
    this->loss_ = loss::Loss::Create("hinge");
  } else {
    this->loss_ = loss::Loss::Create("maxscore-hinge");
  }
}

StochasticLinearModel::~StochasticLinearModel() {
  DeleteArray(this->weights_);
  DeleteArray(this->gradients_);
}

label_t StochasticLinearModel::Predict(const pario::DataPoint& dp,
                                       float* predicts) {
  const auto& x = dp.data();
  for (int c = 0; c < this->clf_num_; ++c) {
    predicts[c] = expr::dotmul(w(c), x) + w(c)[0];
  }
  if (this->clf_num_ == 1) {
    return loss::Loss::Sign(*predicts);
  } else {
    return label_t(max_element(predicts, predicts + this->clf_num_) - predicts);
  }
}
float StochasticLinearModel::Iterate(const MiniBatch& mb, label_t* predicts,
                                     float* scores) {
  StochasticModel::Iterate(mb, predicts, scores);
  ++this->update_num_;

  return 0;
}

void StochasticLinearModel::update_dim(index_t dim) {
  if (dim > this->dim_) {
    for (int i = 0; i < this->clf_num_; ++i) {
      w(i).resize(dim);
      // set the new value to zero
      w(i).slice_op([](real_t& val) { val = 0; }, this->dim_);
    }
    StochasticModel::update_dim(dim);
  }
}

float StochasticLinearModel::model_sparsity() {
  if (this->model_updated_) this->EndTrain();
  size_t non_zero_num = 0;
  for (int c = 0; c < this->clf_num_; ++c) {
    w(c).slice_op(
        [&non_zero_num](const real_t& val) {
          if (val != 0) ++non_zero_num;
        },
        1);  // ignore bias
  }
  return 1.f - float(non_zero_num / double(this->clf_num_ * (this->dim_ - 1)));
}

void StochasticLinearModel::GetModelParam(std::ostream& os) const {
  for (int c = 0; c < this->clf_num_; ++c) {
    os << "weight[" << c << "]:" << w(c) << "\n";
  }
}

int StochasticLinearModel::SetModelParam(std::istream& is) {
  string line;
  for (int c = 0; c < this->clf_num_; ++c) {
    is >> line >> w(c);
  }
  return Status_OK;
}

}  // namespace model
}  // namespace sol

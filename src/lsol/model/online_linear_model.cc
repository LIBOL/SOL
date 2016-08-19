/*********************************************************************************
*     File Name           :     online_linear_model.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-18 16:47]
*     Last Modified       :     [2016-03-09 19:23]
*     Description         :     online linear model
**********************************************************************************/
#include "lsol/model/online_linear_model.h"

#include <algorithm>
#include <random>
#include <limits>

#include "lsol/loss/hinge_loss.h"
#include "lsol/util/util.h"

using namespace std;
using namespace lsol::math;
using namespace lsol::pario;

namespace lsol {
namespace model {

OnlineLinearModel::OnlineLinearModel(int class_num)
    : OnlineModel(class_num, "online_linear"),
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

OnlineLinearModel::~OnlineLinearModel() {
  DeleteArray(this->weights_);
  DeleteArray(this->gradients_);
}

label_t OnlineLinearModel::Iterate(const DataPoint& dp, float* predicts) {
  OnlineModel::Iterate(dp, predicts);
  if (this->regularizer_ != nullptr) {
    this->online_regularizer()->BeginIterate(dp);
  }

  label_t label = this->TrainPredict(dp, predicts);
  // active learning
  if (this->active_smoothness_ > 0) {
    static random_device rd;
    static mt19937 gen(rd());
    static uniform_real_distribution<float> dis(0, 1);
    float margin = 0;
    if (this->clf_num_ == 1) {
      margin = abs(*predicts);
    } else {
      float tmp = predicts[label];
      predicts[label] = -(std::numeric_limits<float>::max)();
      margin = tmp - *max_element(predicts, predicts + this->clf_num_);
      predicts[label] = tmp;
    }
    float prob = active_smoothness_ / (active_smoothness_ + margin);
    if (prob < dis(gen)) {
      --this->cur_iter_num_;
      return label;
    }
  }
  // cost sensitive learning, binary learning with hinge loss only yet
  if (this->cost_sensitive_learning_) {
    loss::HingeBase* hinge_loss = static_cast<loss::HingeBase*>(this->loss_);
    if (dp.label() == 1) {
      hinge_loss->set_margin(this->cost_margin_);
    } else {
      hinge_loss->set_margin(1.f);
    }
  }
  float loss = this->loss_->gradient(dp, predicts, label, this->gradients_,
                                     this->clf_num_);
  if (this->lazy_update_) {
    if (label != dp.label()) {
      ++this->update_num_;
      this->Update(dp, predicts, loss);
    }
  } else if (loss > 0) {
    ++this->update_num_;
    this->Update(dp, predicts, loss);
  }

  if (this->regularizer_ != nullptr) {
    this->online_regularizer()->EndIterate(dp, this->cur_iter_num_);
  }
  return label;
}

label_t OnlineLinearModel::TrainPredict(const pario::DataPoint& dp,
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

label_t OnlineLinearModel::Predict(const pario::DataPoint& dp,
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

void OnlineLinearModel::update_dim(index_t dim) {
  if (dim > this->dim_) {
    for (int i = 0; i < this->clf_num_; ++i) {
      w(i).resize(dim);
      // set the new value to zero
      w(i).slice_op([](real_t& val) { val = 0; }, this->dim_);
    }
    OnlineModel::update_dim(dim);
  }
}

float OnlineLinearModel::model_sparsity() const {
  size_t non_zero_num = 0;
  for (int c = 0; c < this->clf_num_; ++c) {
    w(c).slice_op([&non_zero_num](const real_t& val) {
      if (val != 0) ++non_zero_num;
    }, 1);  // ignore bias
  }
  return 1.f - float(non_zero_num / double(this->clf_num_ * (this->dim_ - 1)));
}

void OnlineLinearModel::GetModelParam(std::ostream& os) const {
  for (int c = 0; c < this->clf_num_; ++c) {
    os << "weight[" << c << "]:" << w(c) << "\n";
  }
}

int OnlineLinearModel::SetModelParam(std::istream& is) {
  string line;
  for (int c = 0; c < this->clf_num_; ++c) {
    is >> line >> w(c);
  }
  return Status_OK;
}

}  // namespace model
}  // namespace lsol

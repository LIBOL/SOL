/*********************************************************************************
*     File Name           :     stochastic_regression/ssguassian.cc
*     Created By          :     yuewu
*     Creation Date       :     [2017-05-09 17:51]
*     Last Modified       :     [2017-05-15 10:08]
*     Description         :     stochastic gaussian
**********************************************************************************/
#include "sol/model/stochastic_linear_model.h"

using namespace std;
using namespace sol::math;
using namespace sol::pario;

namespace sol {
namespace model {

/// \brief  stochastic gaussian
class SGaussian : public StochasticLinearModel {
 public:
  SGaussian(int class_num);
  virtual ~SGaussian();

 protected:
  virtual void update_dim(index_t dim);

 protected:
  virtual void GetModelParam(std::ostream& os) const;

  virtual int SetModelParam(std::istream& is);

 protected:
  math::Matrix<real_t>& Sigma(int cls_id) { return this->Sigmas_[cls_id]; }
  math::Matrix<real_t>* Sigmas_;
};  // class SGaussian

SGaussian::SGaussian(int class_num) : StochasticLinearModel(class_num) {
  if (class_num != 1) {
    throw invalid_argument(
        "only single second stochastic gaussian is supported");
  }
  this->Sigmas_ = new math::Matrix<real_t>[this->clf_num_];
  for (int i = 0; i < this->clf_num_; ++i) {
    this->Sigmas_[i].resize({this->dim_, this->dim_});
  }
}

SGaussian::~SGaussian() { DeleteArray(this->Sigmas_); }

void SGaussian::update_dim(index_t dim) {
  if (dim > this->dim_) {
    for (int c = 0; c < this->clf_num_; ++c) {
      math::Matrix<real_t>& Sigma = this->Sigmas_[c];
      Sigma.resize({dim, dim});
      for (long long y = this->dim_ - 1; y >= 0; --y) {
        real_t* src_data = Sigma.data() + y * this->dim_;
        real_t* dst_data = Sigma.data(y);
        memcpy(dst_data, src_data, this->dim_ * sizeof(real_t));
        memset(src_data, 0, this->dim_ * sizeof(real_t));
      }
    }

    StochasticLinearModel::update_dim(dim);
  }
}

void SGaussian::GetModelParam(std::ostream& os) const {
  StochasticLinearModel::GetModelParam(os);

  for (int c = 0; c < this->clf_num_; ++c) {
    os << "Sigma[" << c << "]: " << this->Sigmas_[c] << "\n";
  }
}

int SGaussian::SetModelParam(std::istream& is) {
  StochasticLinearModel::SetModelParam(is);

  string line;
  for (int c = 0; c < this->clf_num_; ++c) {
    is >> line >> this->Sigmas_[c];
  }

  return Status_OK;
}

/// \brief  passive-aggressive stochastic gaussian
class MASGaussian : public SGaussian {
 public:
  using SGaussian::SGaussian;
  virtual ~MASGaussian(){};

 public:
  virtual float Iterate(const pario::MiniBatch& mb, label_t* predicts,
                        float* scores);
};  // class MASGaussian

float MASGaussian::Iterate(const pario::MiniBatch& mb, label_t* predicts,
                           float* scores) {
  SGaussian::Iterate(mb, predicts, scores);
  ++this->update_num_;
  float decay = float(this->cur_iter_num_ - 1) / this->cur_iter_num_;
  this->eta_ = decay / (mb.size() * this->cur_iter_num_);

  // update sigma
  Sigma(0) *= decay;
  for (int i = 0; i < mb.size(); ++i) {
    const auto& x = mb[i].data();
    w(0) -= x;
    Sigma(0) += outer(w(0), w(0), this->eta_);
    w(0) += x;
  }

  // update mean
  this->eta_ = float(1.0 / (mb.size() * this->cur_iter_num_));
  w(0) *= decay;
  for (int i = 0; i < mb.size(); ++i) {
    const auto& x = mb[i].data();
    w(0) += this->eta_ * x;
  }
  return 0;
}

RegisterModel(MASGaussian, "masg", "Moving Average Stochastic Gaussian");

/// \brief  passive-aggressive stochastic gaussian
class PASGaussian : public SGaussian {
 public:
  using SGaussian::SGaussian;
  virtual ~PASGaussian(){};

 public:
  virtual float Iterate(const pario::MiniBatch& mb, label_t* predicts,
                        float* scores);
};  // class PAGaussian

float PASGaussian::Iterate(const pario::MiniBatch& mb, label_t* predicts,
                           float* scores) {
  SGaussian::Iterate(mb, predicts, scores);
  ++this->update_num_;
  float decay = (this->cur_iter_num_ - 1.f) / this->cur_iter_num_;
  this->eta_ = float(1.0 / (mb.size() * this->cur_iter_num_));

  // update sigma
  Sigma(0) *= decay;
  for (int i = 0; i < mb.size(); ++i) {
    const auto& x = mb[i].data();
    w(0) -= x;
    Sigma(0) += outer(w(0), w(0), this->eta_);
    w(0) += x;
  }

  // update mean
  w(0) *= decay;
  for (int i = 0; i < mb.size(); ++i) {
    const auto& x = mb[i].data();
    w(0) += this->eta_ * x;
  }

  return 0;
}

RegisterModel(PASGaussian, "pasg", "Passive Aggressive Stochastic Gaussian");

}  // namespace model
}  // namespace sol

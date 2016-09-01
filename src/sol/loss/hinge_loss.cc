/*********************************************************************************
*     File Name           :     hinge_loss.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-14 23:44]
*     Last Modified       :     [2016-02-15 00:01]
*     Description         :     hinge loss
**********************************************************************************/

#include "sol/loss/hinge_loss.h"

#include <algorithm>
#include <limits>

namespace sol {
namespace loss {

HingeBase::HingeBase(int type) : Loss(type | Type::HINGE), margin_(1.f) {}

float HingeLoss::loss(const pario::DataPoint& dp, float* predict,
                      label_t predict_label, int cls_num) {
  return (std::max)(0.0f, this->margin_ - *predict * dp.label());
}

float HingeLoss::gradient(const pario::DataPoint& dp, float* predict,
                          label_t predict_label, float* gradient, int cls_num) {
  if (this->margin_handler_) {
    *gradient = 1.f;
    this->margin_ =
        this->margin_handler_(dp, predict, predict_label, gradient, cls_num);
  }
  float loss = (std::max)(0.0f, this->margin_ - *predict * dp.label());
  if (loss > 0) {
    *gradient = (float)(-dp.label());
  } else {
    *gradient = 0;
  }
  return loss;
}

RegisterLoss(HingeLoss, "hinge", "Hinge Loss");

float MaxScoreHingeLoss::loss(const pario::DataPoint& dp, float* predict,
                              label_t predict_label, int cls_num) {
  label_t label = dp.label();
  if (predict_label == label) {
    float tmp = predict[label];
    predict[label] = -(std::numeric_limits<float>::max)();
    predict_label =
        label_t(std::max_element(predict, predict + cls_num) - predict);
    predict[label] = tmp;
  }
  return (std::max)(0.0f,
                    this->margin_ - predict[label] + predict[predict_label]);
}

float MaxScoreHingeLoss::gradient(const pario::DataPoint& dp, float* predict,
                                  label_t predict_label, float* gradient,
                                  int cls_num) {
  label_t label = dp.label();
  if (predict_label == label) {
    float tmp = predict[label];
    predict[label] = -(std::numeric_limits<float>::max)();
    predict_label =
        label_t(std::max_element(predict, predict + cls_num) - predict);
    predict[label] = tmp;
  }

  float loss = 0;
  if (this->margin_handler_) {
    for (int i = 0; i < cls_num; ++i) gradient[i] = 0;
    gradient[predict_label] = 1;
    gradient[label] = -1;
    this->margin_ =
        this->margin_handler_(dp, predict, predict_label, gradient, cls_num);
    loss = (std::max)(0.0f,
                      this->margin_ - predict[label] + predict[predict_label]);
    if (loss <= 0) {
      gradient[predict_label] = 0;
      gradient[label] = 0;
    }
  } else {
    loss = (std::max)(0.0f,
                      this->margin_ - predict[label] + predict[predict_label]);
    if (loss > 0) {
      for (int i = 0; i < cls_num; ++i) gradient[i] = 0;
      gradient[predict_label] = 1;
      gradient[label] = -1;
    }
  }
  return loss;
}

RegisterLoss(MaxScoreHingeLoss, "maxscore-hinge", "Max-Score Hinge Loss");

float UniformHingeLoss::loss(const pario::DataPoint& dp, float* predict,
                             label_t predict_label, int cls_num) {
  label_t label = dp.label();
  float false_predict = 0;
  float false_num = 1e-12f;
  for (int i = 0; i < cls_num; ++i) {
    if (predict[label] <= predict[i]) {
      false_predict += predict[i];
      false_num += 1;
    }
  }
  false_num -= 1;
  false_predict -= predict[label];
  return (std::max)(0.0f,
                    this->margin_ - predict[label] + false_predict / false_num);
}

float UniformHingeLoss::gradient(const pario::DataPoint& dp, float* predict,
                                 label_t predict_label, float* gradient,
                                 int cls_num) {
  label_t label = dp.label();
  float false_predict = 0;
  float false_num = 1e-12f;
  for (int i = 0; i < cls_num; ++i) {
    if (predict[label] <= predict[i]) {
      false_predict += predict[i];
      false_num += 1;
    }
  }
  false_num -= 1;
  false_predict -= predict[label];

  float alpha = 1.f / false_num;
  float loss = 0;
  if (this->margin_handler_) {
    for (int i = 0; i < cls_num; ++i) {
      if (predict[label] <= predict[i]) {
        gradient[i] = alpha;
      } else {
        gradient[i] = 0;
      }
    }
    gradient[label] = -1;
    this->margin_ =
        this->margin_handler_(dp, predict, predict_label, gradient, cls_num);
    float loss = (std::max)(
        0.0f, this->margin_ - predict[label] + false_predict * alpha);
    if (loss <= 0) {
      for (int i = 0; i < cls_num; ++i) {
        gradient[i] = 0;
      }
    }
  } else {
    float loss = (std::max)(
        0.0f, this->margin_ - predict[label] + false_predict * alpha);
    if (loss > 0) {
      for (int i = 0; i < cls_num; ++i) {
        if (predict[label] <= predict[i]) {
          gradient[i] = alpha;
        } else {
          gradient[i] = 0;
        }
      }
      gradient[label] = -1;
    }
  }
  return loss;
}

RegisterLoss(UniformHingeLoss, "uniform-hinge", "Uniform Hinge Loss");

}  // namespace loss
}  // namespace sol

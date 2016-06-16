/*********************************************************************************
*     File Name           :     hinge_loss.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-14 23:44]
*     Last Modified       :     [2016-02-15 00:01]
*     Description         :     hinge loss
**********************************************************************************/

#include "lsol/loss/hinge_loss.h"

#include <algorithm>
#include <limits>

namespace lsol {
namespace loss {

float HingeLoss::loss(label_t label, float* predict, int cls_num) {
  return (std::max)(0.0f, this->margin_ - *predict * label);
}

float HingeLoss::gradient(label_t label, float* predict, float* gradient,
                          int cls_num) {
  float loss = this->loss(label, predict, cls_num);
  if (loss > 0) {
    *gradient = (float)(-label);
  } else {
    *gradient = 0;
  }
  return loss;
}

RegisterLoss(HingeLoss, "hinge", "Hinge Loss");

float MaxScoreHingeLoss::loss(label_t label, float* predict, int cls_num) {
  float max_predict = -(std::numeric_limits<float>::max)();
  for (int i = 0; i < cls_num; ++i) {
    if (i == label) continue;
    if (max_predict < predict[i]) {
      max_predict = predict[i];
    }
  }
  return (std::max)(0.0f, this->margin_ - predict[label] + max_predict);
}

float MaxScoreHingeLoss::gradient(label_t label, float* predict,
                                  float* gradient, int cls_num) {
  float max_predict = -(std::numeric_limits<float>::max)();
  int max_label = -1;
  for (int i = 0; i < cls_num; ++i) {
    gradient[i] = 0;
    if (i == label) continue;
    if (max_predict < predict[i]) {
      max_predict = predict[i];
      max_label = i;
    }
  }
  float loss = (std::max)(0.0f, this->margin_ - predict[label] + max_predict);

  if (loss > 0) {
    gradient[max_label] = 1;
    gradient[label] = -1;
  }
  return loss;
}

RegisterLoss(MaxScoreHingeLoss, "maxscore-hinge", "Max-Score Hinge Loss");

float UniformHingeLoss::loss(label_t label, float* predict, int cls_num) {
  float false_predict = 0;
  float false_num = 1e-12f;
  for (int i = 0; i < cls_num; ++i) {
    if (predict[label] <= predict[i]) {
      false_predict += predict[i];
      false_num += 1;
    }
  }
  return (std::max)(0.0f,
                    this->margin_ - predict[label] + false_predict / false_num);
}

float UniformHingeLoss::gradient(label_t label, float* predict, float* gradient,
                                 int cls_num) {
  float false_predict = 0;
  float false_num = 1e-12f;
  for (int i = 0; i < cls_num; ++i) {
    gradient[i] = 0;
    if (i == label) continue;

    if (predict[label] <= predict[i]) {
      false_predict += predict[i];
      false_num += 1;
    }
  }
  float alpha = 1.f / false_num;
  float loss =
      (std::max)(0.0f, this->margin_ - predict[label] + false_predict * alpha);

  if (loss > 0) {
    for (int i = 0; i < cls_num; ++i) {
      if (predict[label] <= predict[i]) {
        gradient[i] = alpha;
      }
    }
    gradient[label] = -1;
  }
  return loss;
}

RegisterLoss(UniformHingeLoss, "uniform-hinge", "Uniform Hinge Loss");

}  // namespace loss
}  // namespace lsol

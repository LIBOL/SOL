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

float HingeLoss::loss(label_t label, float* predict, label_t predict_label,
                      int cls_num) {
  return (std::max)(0.0f, this->margin_ - *predict * label);
}

float HingeLoss::gradient(label_t label, float* predict, label_t predict_label,
                          float* gradient, int cls_num) {
  float loss = (std::max)(0.0f, this->margin_ - *predict * label);
  if (loss > 0) {
    *gradient = (float)(-label);
  } else {
    *gradient = 0;
  }
  return loss;
}

RegisterLoss(HingeLoss, "hinge", "Hinge Loss");

float MaxScoreHingeLoss::loss(label_t label, float* predict,
                              label_t predict_label, int cls_num) {
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

float MaxScoreHingeLoss::gradient(label_t label, float* predict,
                                  label_t predict_label, float* gradient,
                                  int cls_num) {
  if (predict_label == label) {
    float tmp = predict[label];
    predict[label] = -(std::numeric_limits<float>::max)();
    predict_label =
        label_t(std::max_element(predict, predict + cls_num) - predict);
    predict[label] = tmp;
  }

  float loss =
      (std::max)(0.0f, this->margin_ - predict[label] + predict[predict_label]);

  if (loss > 0) {
    for (int i = 0; i < cls_num; ++i) gradient[i] = 0;
    gradient[predict_label] = 1;
    gradient[label] = -1;
  }
  return loss;
}

RegisterLoss(MaxScoreHingeLoss, "maxscore-hinge", "Max-Score Hinge Loss");

float UniformHingeLoss::loss(label_t label, float* predict,
                             label_t predict_label, int cls_num) {
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

float UniformHingeLoss::gradient(label_t label, float* predict,
                                 label_t predict_label, float* gradient,
                                 int cls_num) {
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
  float loss =
      (std::max)(0.0f, this->margin_ - predict[label] + false_predict * alpha);

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
  return loss;
}

RegisterLoss(UniformHingeLoss, "uniform-hinge", "Uniform Hinge Loss");

}  // namespace loss
}  // namespace lsol

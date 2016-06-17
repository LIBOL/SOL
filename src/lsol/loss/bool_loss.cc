/*********************************************************************************
*     File Name           :     bool_loss.cc
*     Created By          :     yuewu
*     Description         :     bool loss with yes or no
**********************************************************************************/

#include "lsol/loss/bool_loss.h"

#include <algorithm>

namespace lsol {
namespace loss {

float BoolLoss::loss(label_t label, float* predict, label_t predict_label,
                     int cls_num) {
  return predict_label == label ? 0.f : 1.f;
}

float BoolLoss::gradient(label_t label, float* predict, label_t predict_label,
                         float* gradient, int cls_num) {
  float loss = predict_label == label ? 0.f : 1.f;
  if (loss > 0) {
    *gradient = (float)(-label);
  } else {
    *gradient = 0;
  }
  return loss;
}

RegisterLoss(BoolLoss, "bool", "Bool Loss");

float MaxScoreBoolLoss::loss(label_t label, float* predict,
                             label_t predict_label, int cls_num) {
  return predict_label == label ? 0.f : 1.f;
}

float MaxScoreBoolLoss::gradient(label_t label, float* predict,
                                 label_t predict_label, float* gradient,
                                 int cls_num) {
  float loss = predict_label == label ? 0.f : 1.f;

  if (loss > 0) {
    for (int i = 0; i < cls_num; ++i) {
      gradient[i] = 0;
    }
    gradient[predict_label] = 1;
    gradient[label] = -1;
  }
  return loss;
}

RegisterLoss(MaxScoreBoolLoss, "maxscore-bool", "Max-Score Bool Loss");

float UniformBoolLoss::loss(label_t label, float* predict,
                            label_t predict_label, int cls_num) {
  return predict_label == label ? 0.f : 1.f;
}

float UniformBoolLoss::gradient(label_t label, float* predict,
                                label_t predict_label, float* gradient,
                                int cls_num) {
  float loss = predict_label == label ? 0.f : 1.f;

  if (loss > 0) {
    size_t false_num = std::count_if(
        predict, predict + cls_num,
        [&predict, &label](float val) { return val >= predict[label]; });
    false_num -= 1;
    float alpha = 1.f / false_num;
    for (int i = 0; i < cls_num; ++i) {
      if (predict[i] >= predict[label])
        gradient[i] = alpha;
      else
        gradient[i] = 0;
    }
    gradient[label] = -1;
  }
  return loss;
}

RegisterLoss(UniformBoolLoss, "uniform-bool", "Uniform Bool Loss");

}  // namespace loss
}  // namespace lsol

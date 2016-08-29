/*********************************************************************************
*     File Name           :     logistic_loss.cc
*     Created By          :     yuewu
*     Description         :     logistic loss with yes or no
**********************************************************************************/

#include "sol/loss/logistic_loss.h"

#include <algorithm>
#include <cmath>

namespace sol {
namespace loss {

float LogisticLoss::loss(const pario::DataPoint& dp, float* predict,
                         label_t predict_label, int cls_num) {
  float z = -*predict * dp.label();
  if (z > 100.f)
    return z;
  else if (z < -100.f)
    return 0;
  else
    return log(1.f + exp(z));
}

float LogisticLoss::gradient(const pario::DataPoint& dp, float* predict,
                             label_t predict_label, float* gradient,
                             int cls_num) {
  float z = -*predict * dp.label();
  if (z > 100.f) {
    *gradient = (float)(-dp.label());
    return z;
  } else if (z < -100.f) {
    *gradient = 0.f;
    return 0;
  } else {
    *gradient = (float)(-dp.label()) / (1.f + exp(-z));
    return log(1.f + exp(z));
  }
}

RegisterLoss(LogisticLoss, "logistic", "Logistic Loss");

float MaxScoreLogisticLoss::loss(const pario::DataPoint& dp, float* predict,
                                 label_t predict_label, int cls_num) {
  label_t label = dp.label();
  if (predict_label == label) {
    float tmp = predict[label];
    predict[label] = -(std::numeric_limits<float>::max)();
    predict_label =
        label_t(std::max_element(predict, predict + cls_num) - predict);
    predict[label] = tmp;
  }

  float z = predict[predict_label] - predict[label];
  if (z > 100.f) {
    return z;
  } else if (z < -100.f) {
    return 0;
  } else {
    return log(1.f + exp(z));
  }
}

float MaxScoreLogisticLoss::gradient(const pario::DataPoint& dp, float* predict,
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

  for (int i = 0; i < cls_num; ++i) gradient[i] = 0;
  float z = predict[predict_label] - predict[label];
  if (z > 100.f) {
    gradient[predict_label] = 1.f;
    gradient[label] = -1.f;
    return z;
  } else if (z < -100.f) {
    gradient[predict_label] = 0;
    gradient[label] = 0;
    return 0;
  } else {
    gradient[predict_label] = 1 / (1.f + exp(-z));
    gradient[label] = -gradient[predict_label];
    return log(1.f + exp(z));
  }
}

RegisterLoss(MaxScoreLogisticLoss, "maxscore-logistic",
             "Max-Score Logistic Loss");

float UniformLogisticLoss::loss(const pario::DataPoint& dp, float* predict,
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

  float z = false_predict / false_num - predict[label];
  if (z > 100.f) {
    return z;
  } else if (z < -100.f) {
    return 0;
  } else {
    return log(1.f + exp(z));
  }
}

float UniformLogisticLoss::gradient(const pario::DataPoint& dp, float* predict,
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

  float z = false_predict / false_num - predict[label];
  if (z > 100.f) {
    float alpha = 1.f / false_num;

    for (int i = 0; i < cls_num; ++i) {
      if (predict[label] <= predict[i]) {
        gradient[i] = alpha;
      } else {
        gradient[i] = 0;
      }
    }
    gradient[label] = -1.f;
    return z;
  } else if (z < -100.f) {
    for (int i = 0; i < cls_num; ++i) gradient[i] = 0;
    return 0;
  } else {
    float alpha = 1.f / (1.f + exp(-z));
    float beta = alpha / false_num;

    for (int i = 0; i < cls_num; ++i) {
      if (predict[label] <= predict[i]) {
        gradient[i] = beta;
      } else {
        gradient[i] = 0;
      }
    }
    gradient[label] = -alpha;

    return log(1.f + exp(z));
  }
}

RegisterLoss(UniformLogisticLoss, "uniform-logistic", "Uniform Logistic Loss");

}  // namespace loss
}  // namespace sol

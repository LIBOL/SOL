/*********************************************************************************
*     File Name           :     online_optimizer.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-18 22:42]
*     Last Modified       :     [2016-02-18 23:57]
*     Description         :
*     Description         :     Online Optimizer
**********************************************************************************/

#include "lsol/optimizer/online_optimizer.h"

#include <sstream>

#include "lsol/util/util.h"

using namespace std;
using namespace lsol::pario;
using namespace lsol::model;

namespace lsol {
namespace optimizer {

OnlineOptimizer::OnlineOptimizer(Model *model) : Optimizer(model) {
  this->online_model_ = dynamic_cast<OnlineModel *>(model);
}

/// \brief  Train from a data set
//
/// \param data_iter data iterator
//
/// \return training error rate
float OnlineOptimizer::Train(DataIter &data_iter) {
  this->online_model_->BeginTrain();
  fprintf(stdout, "%s\n", this->online_model_->model_info().c_str());
  float err_num(0);
  size_t data_num = 0;
  size_t show_step = 1;  // show information every show_step
  size_t show_count = 2;

  printf("Iterate No.\t\tError Rate\t\t\n");

  float *predicts = new float[this->model_->clf_num()];
  MiniBatch *mb = nullptr;
  while (1) {
    mb = data_iter.Next(mb);
    if (mb == nullptr) break;
    data_num += mb->size();
    for (int i = 0; i < mb->size(); ++i) {
      DataPoint &x = (*mb)[i];
      this->PreProcess(x);
      // predict
      label_t label = this->online_model_->Iterate(x, predicts);
      if (label != x.label()) err_num++;
    }

    if (data_num >= show_count) {
      printf("%lu\t\t\t%.6f\n", data_num, float(double(err_num) / data_num));
      show_count = (size_t(1) << ++show_step);
    }
  }
  printf("%lu\t\t\t%.6f\n", data_num, float(double(err_num) / data_num));
  this->online_model_->EndTrain();

  delete[] predicts;
  return float(double(err_num) / data_num);
}

RegisterOptimizer(OnlineOptimizer, "online", "Online Optimizer");

}  // namespace optimizer
}  // namespace lsol

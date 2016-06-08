/*********************************************************************************
*     File Name           :     optimizer.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-18 21:49]
*     Last Modified       :     [2016-03-09 19:23]
*     Description         :     Base class for optimizer
**********************************************************************************/

#include "lsol/optimizer/optimizer.h"

#include <stdexcept>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "lsol/util/error_code.h"

using namespace std;
using namespace lsol::pario;
using namespace lsol::math;
using namespace lsol::math::expr;

namespace lsol {
namespace optimizer {

Optimizer *Optimizer::Create(const std::string &name, model::Model *model) {
  auto create_func = CreateObject<Optimizer>(std::string(name) + "_optimizer");
  return create_func == nullptr ? nullptr : create_func(model);
}

void Optimizer::SetParameter(const string &name, const string &value) {
  if (name == "norm") {
    if (value == "L1") {
      this->norm_type_ = op::OpType::kL1;
    } else if (value == "L2") {
      this->norm_type_ = op::OpType::kL2;
    } else {
      ostringstream oss;
      oss << "unknown norm type " << value;
      throw invalid_argument(oss.str());
    }
  } else if (name == "presel") {
    if (this->LoadPreSelFeatures(value) != Status_OK) {
      ostringstream oss;
      oss << "load pre-selected features failed!";
      throw invalid_argument(oss.str());
    }
  }
}

float Optimizer::Test(DataIter &data_iter, std::ostream *os) {
  size_t err_num = 0;
  size_t data_num = 0;

  if (os != nullptr) {
    (*os) << "predict\tlabel\n";
  }

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
      label_t label = this->model_->Predict(x, predicts);
      if (label != x.label()) err_num++;
      if (os != nullptr) {
        (*os) << label << "\t" << x.label() << "\n";
      }
    }
  }
  delete[] predicts;
  return float(double(err_num) / data_num);
}

void Optimizer::PreProcess(DataPoint &x) {
  // filter features
  this->FilterFeatures(x);
  // normalize
  if (this->norm_type_ != op::OpType::kNone) {
    real_t norm = 1;
    switch (this->norm_type_) {
      case op::OpType::kL1:
        norm = reduce<op::plus>(L1(x.data()));
        break;
      case op::OpType::kL2:
        norm = reduce<op::plus>(L2(x.data()));
      default:
        break;
    }
    x.data() /= norm;
  }
}

void Optimizer::FilterFeatures(DataPoint &x) {
  if (this->max_index_ == 0) return;
  size_t feat_num = x.size();
  for (size_t i = 0; i < feat_num; ++i) {
    if (x.index(i) > this->max_index_ ||
        this->sel_feat_flags_[x.index(i)] == 0) {
      x.feature(i) = 0;
    }
  }
}

int Optimizer::LoadPreSelFeatures(const string &path) {
  this->max_index_ = 0;
  this->sel_feat_flags_.clear();

  ifstream in_file(path.c_str(), ios::in);
  if (!in_file) {
    fprintf(stderr, "open file %s failed\n!", path.c_str());
    return Status_IO_Error;
  }

  index_t index = 0;
  string line;
  vector<index_t> indexes;
  // load feature indexes
  while (getline(in_file, line)) {
    const char *p = line.c_str();
    while (*p == ' ' || *p == '\t') ++p;
    // skip comments
    if (*p == '#') continue;

    index = (index_t)(stoi(line));
    if (index <= 0) {
      fprintf(stderr, "parse index %s failed!\n", line.c_str());
      return Status_Invalid_Format;
    }
    indexes.push_back(index);
  }

  // find the max index
  this->max_index_ = *std::max_element(indexes.begin(), indexes.end());
  this->sel_feat_flags_.reserve(this->max_index_ + 1);
  this->sel_feat_flags_.resize(this->max_index_ + 1);
  this->sel_feat_flags_ = 0;

  for (index_t i : indexes) {
    this->sel_feat_flags_[i] = 1;
  }

  return Status_OK;
}

}  // namespace optimizer
}  // namespace lsol

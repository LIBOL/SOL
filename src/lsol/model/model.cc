/*********************************************************************************
*     File Name           :     model.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-02-16 22:54]
*     Last Modified       :     [2016-03-09 19:22]
*     Description         :     base class for model
**********************************************************************************/

#include "lsol/model/model.h"

#include <algorithm>

#include "lsol/util/util.h"
#include "lsol/util/error_code.h"

using namespace std;
using namespace lsol::math::expr;
using namespace lsol::pario;

namespace lsol {
namespace model {

Model* Model::Create(const std::string& name, int class_num) {
  auto create_func = CreateObject<Model>(std::string(name) + "_model");
  Model* ins = nullptr;
  try {
    if (create_func != nullptr) ins = create_func(class_num);
  } catch (invalid_argument& err) {
    fprintf(stderr, "%s\n", err.what());
    ins = nullptr;
  }
  return ins;
}

Model::Model(int class_num, const std::string& type)
    : class_num_(class_num),
      clf_num_(class_num == 2 ? 1 : class_num),
      loss_(nullptr),
      type_(type),
      norm_type_(op::OpType::kNone),
      regularizer_(nullptr),
      max_index_(0) {
  Check(class_num > 1);
}

Model::~Model() { DeletePointer(this->loss_); }

void Model::SetParameter(const std::string& name, const std::string& value) {
  if (name == "loss") {
    DeletePointer(this->loss_);
    this->loss_ = loss::Loss::Create(value);
    Check(this->loss_ != nullptr);
    if (this->clf_num_ == 1 &&
        ((this->loss_->type() & loss::Loss::Type::BC) == 0)) {
      throw invalid_argument(
          "onlye binary loss function can be used with binary problems");
    } else if (this->clf_num_ != 1 &&
               ((this->loss_->type() & loss::Loss::Type::MC) == 0)) {
      throw invalid_argument(
          "onlye multiclass loss function can be used with multiclass "
          "problems");
    }
  } else if (name == "norm") {
    if (value == "L1") {
      this->norm_type_ = op::OpType::kL1;
    } else if (value == "L2") {
      this->norm_type_ = op::OpType::kL2;
    } else {
      ostringstream oss;
      oss << "unknown norm type " << value;
      throw invalid_argument(oss.str());
    }
  } else if (name == "filter") {
    if (this->LoadPreSelFeatures(value) != Status_OK) {
      ostringstream oss;
      oss << "load pre-selected features failed!";
      throw invalid_argument(oss.str());
    }
  } else {
    if (this->regularizer_ == nullptr ||
        this->regularizer_->SetParameter(name, value) != Status_OK) {
      ostringstream oss;
      oss << "unknown parameter " << name;
      throw invalid_argument(oss.str());
    }
  }
}

float Model::Test(DataIter& data_iter, std::ostream* os) {
  fprintf(stdout, "Model Information: \n%s\n", this->model_info().c_str());
  fprintf(stdout, "Test Process....\nIterate No.\t\t\tError Rate\t\t\n");

  size_t err_num = 0;
  size_t data_num = 0;
  size_t show_step = 1;  // show information every show_step
  size_t show_count = 2;

  if (os != nullptr) {
    (*os) << "predict\tlabel\n";
  }

  float* predicts = new float[this->clf_num()];
  MiniBatch* mb = nullptr;
  while (1) {
    mb = data_iter.Next(mb);
    if (mb == nullptr) break;
    // data_num += mb->size();
    for (int i = 0; i < mb->size(); ++i) {
      DataPoint& x = (*mb)[i];
      this->PreProcess(x);
      // predict
      label_t label = this->Predict(x, predicts);
      if (label != x.label()) err_num++;
      if (os != nullptr) {
        (*os) << label << "\t" << x.label() << "\n";
      }
      ++data_num;
      if (data_num >= show_count) {
        printf("%llu\t\t\t\t%.6f\r", data_num,
               float(double(err_num) / data_num));
        show_count = (size_t(1) << ++show_step);
      }
    }
  }
  printf("\n");
  delete[] predicts;
  return float(double(err_num) / data_num);
}

void Model::BeginTrain() {
  if (this->loss_ == nullptr)
    throw runtime_error("loss function is not set yet!");
  this->update_num_ = 0;
}

int Model::Save(const string& path) const {
  fprintf(stdout, "save model to %s\n", path.c_str());
  ofstream out_file(path.c_str(), ios::out);
  if (!out_file) {
    fprintf(stderr, "open file %s failed\n", path.c_str());
    return Status_IO_Error;
  }
  Json::Value root;
  this->GetModelInfo(root);
  Json::StyledStreamWriter writer;
  out_file << "model info:\n";
  writer.write(out_file, root);

  out_file << "model parameters:\n";
  this->GetModelParam(out_file);

  out_file.close();
  return Status_OK;
}

Model* Model::Load(const string& path) {
  int ret = Status_OK;
  Model* model = nullptr;
  ifstream in_file(path.c_str(), ios::in);
  if (!in_file) {
    fprintf(stderr, "open file %s failed\n", path.c_str());
    return nullptr;
  }
  string line;
  getline(in_file, line);
  if (line != "model info:") {
    fprintf(stderr, "invalid model file\n");
    ret = Status_Invalid_Format;
  }
  if (ret == Status_OK) {
    ostringstream model_info;
    while (line != "model parameters:") {
      getline(in_file, line);
      model_info << line;
    }
    Json::Value root;
    Json::Reader reader;
    if (reader.parse(model_info.str(), root) == false) {
      fprintf(stderr, "parse model file %s failed\n", path.c_str());
      ret = Status_Invalid_Format;
    }

    if (ret == Status_OK) {
      string cls_name = root.get("model", "").asString();
      int cls_num = root.get("cls_num", "0").asInt();
      model = Model::Create(cls_name, cls_num);
      if (model == nullptr) {
        fprintf(stderr, "create model failed: no model named %s\n",
                cls_name.c_str());
        ret = Status_Invalid_Format;
      } else {
        try {
          ret = model->SetModelInfo(root);
        } catch (invalid_argument& err) {
          fprintf(stderr, "set model parameter failed: %s\n", err.what());
          ret = Status_Invalid_Argument;
        }
      }
    }
  }

  if (ret == Status_OK) {
    ret = model->SetModelParam(in_file);
  }
  if (ret != Status_OK) {
    DeletePointer(model);
    DeletePointer(model);
  }
  in_file.close();
  return model;
}

void Model::GetModelInfo(Json::Value& root) const {
  root["model"] = this->name();
  root["cls_num"] = this->class_num();
  root["clf_num"] = this->clf_num();
  root["loss"] = this->loss_ == nullptr ? "" : this->loss_->name();
  root["norm"] = int(this->norm_type_);

  // regularizer
  if (this->regularizer_ != nullptr) {
    this->regularizer_->GetRegularizerInfo(root);
  }
}

int Model::SetModelInfo(const Json::Value& root) {
  try {
    Check(root.get("model", "").asString() == this->name());
    Check(root.get("cls_num", "").asInt() == this->class_num());
    Check(root.get("clf_num", "").asInt() == this->clf_num());
  } catch (invalid_argument& err) {
    fprintf(stderr, "set model info failed: %s\n", err.what());
    return Status_Invalid_Argument;
  }
  // loss
  if (this->loss_ == nullptr ||
      root["loss"].asString() != this->loss_->name()) {
    this->SetParameter("loss", root["loss"].asString());
  }
  // norm
  this->norm_type_ = lsol::math::expr::op::OpType(root["norm"].asInt());

  // regularizer
  const Json::Value& relu_settings = root["regularizer"];
  if (!(relu_settings.isNull()) && this->regularizer_ != nullptr) {
    for (Json::Value::const_iterator iter = relu_settings.begin();
         iter != relu_settings.end(); ++iter) {
      if (this->regularizer_->SetParameter(iter.name(), iter->asString()) !=
          Status_OK) {
        fprintf(stderr, "unrecognized regularizer option: %s\n",
                iter.name().c_str());
        return Status_Invalid_Argument;
      }
    }
  }

  return Status_OK;
}

string Model::model_info() const {
  Json::Value root;
  this->GetModelInfo(root);

  Json::StyledWriter writer;
  return writer.write(root);
}

void Model::PreProcess(DataPoint& x) {
  // calibrate label
  x.set_label(this->CalibrateLabel(x.label()));
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

void Model::FilterFeatures(DataPoint& x) {
  if (this->max_index_ == 0) return;
  size_t feat_num = x.size();
  for (size_t i = 0; i < feat_num; ++i) {
    if (x.index(i) > this->max_index_ ||
        this->sel_feat_flags_[x.index(i)] == 0) {
      x.feature(i) = 0;
    }
  }
}

int Model::LoadPreSelFeatures(const string& path) {
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
    const char* p = line.c_str();
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
  this->max_index_ = *(std::max_element)(indexes.begin(), indexes.end());
  this->sel_feat_flags_.reserve(this->max_index_ + 1);
  this->sel_feat_flags_.resize(this->max_index_ + 1);
  this->sel_feat_flags_ = 0;

  for (index_t i : indexes) {
    this->sel_feat_flags_[i] = 1;
  }

  return Status_OK;
}

}  // namespace model
}  // namespace lsol

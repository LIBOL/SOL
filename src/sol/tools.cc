/*********************************************************************************
*     File Name           :     tools.cc
*     Created By          :     yuewu
*     Description         :     tools for sol
**********************************************************************************/

#include "sol/tools.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

#include <sol/sol.h>
#include <sol/math/vector.h>

using namespace sol::pario;
using namespace std;

namespace sol {
int analyze(const string& src_path, const string& src_type,
            const string& output_path) {
  DataIter data_iter;
  int ret = data_iter.AddReader(src_path, src_type);
  if (ret != Status_OK) return ret;

  MiniBatch* mb = nullptr;

  size_t data_num = 0;
  size_t feat_num = 0;
  size_t feat_dim = 0;
  math::Vector<char> feat_flags;
  map<int, size_t> map_class_sample_num;

  size_t print_thresh = 10000;
  while (true) {
    mb = data_iter.Next(mb);
    if (mb == nullptr) break;
    data_num += mb->size();

    for (int i = 0; i < mb->size(); ++i) {
      DataPoint& dp = (*mb)[i];

      if (feat_dim < dp.dim()) {
        feat_dim = dp.dim();
      }

      size_t prev_size = feat_flags.size();
      if (feat_dim > prev_size) {
        feat_flags.resize(feat_dim);
        // set the new value to zero
        for (char* iter = feat_flags.begin() + prev_size;
             iter != feat_flags.end(); ++iter) {
          *iter = 0;
        }
      }
      for (size_t i = 0; i < dp.size(); i++) {
        feat_flags[dp.index(i)] = 1;
      }
      feat_num += dp.size();

      int label = dp.label();
      if (map_class_sample_num.find(label) != map_class_sample_num.end()) {
        map_class_sample_num[label] += 1;
      } else {
        map_class_sample_num[label] = 1;
      }
    }

    if (data_num > print_thresh) {
      cout << data_num << " examples analyzed\r";
      print_thresh += 10000;
    }
  }
  cout << data_num << " examples analyzed\n";

  size_t valid_dim = 0;
  for (size_t i = 0; i < feat_flags.size(); i++) {
    if (feat_flags[i] == 1) valid_dim++;
  }
  FileWriter fw;
  if ((ret = fw.Open(output_path.c_str(), "w")) != Status_OK) {
    cerr << "Write analysis result to " << output_path << " failed\n";
    return ret;
  }
  fw.Printf("data number  : %lu\n", data_num);
  fw.Printf("feat number  : %lu\n", feat_num);
  fw.Printf("dimension    : %lu\n", feat_dim - 1);
  fw.Printf("nonzero feat : %lu\n", valid_dim);
  fw.Printf("class num    : %lu\n", map_class_sample_num.size());
  if (feat_dim > 0) {
    fw.Printf("data sparsity: %.2lf%%\n", 100 - valid_dim * 100.0 / feat_dim);
  }
  for (auto& iter : map_class_sample_num) {
    fw.Printf("data number of class %d : %lu\n", iter.first, iter.second);
  }
  return ret;
}

int convert(const string& src_path, const string& src_type,
            const string& dst_path, const string& dst_type) {
  DataIter iter;
  int ret = iter.AddReader(src_path, src_type);
  if (ret != Status_OK) return ret;

  DataWriter* writer = DataWriter::Create(dst_type);
  if (writer == nullptr) {
    ret = Status_Invalid_Argument;
    return ret;
  }
  ret = writer->Open(dst_path);
  if (ret != Status_OK) {
    delete writer;
    return ret;
  }

  MiniBatch* mb = nullptr;

  if (dst_type == "csv") {
    // for csv, get extra info
    cout << "figuring out feature dimension\n";
    index_t feat_dim = 0;
    while (true) {
      mb = iter.Next(mb);
      if (mb == nullptr) break;

      for (int i = 0; i < mb->size(); ++i) {
        DataPoint& pt = (*mb)[i];
        if (feat_dim < pt.dim()) feat_dim = pt.dim();
      }
    }
    writer->SetExtraInfo((char*)(&feat_dim));
    if (feat_dim == 0) {
      cerr << "figuring out feature dimension failed\n";
      return Status_Invalid_Format;
    }
    ret = iter.AddReader(src_path, src_type);
    if (ret != Status_OK) return ret;
  }
  size_t data_num = 0;
  size_t print_thresh = 1000;
  while (true) {
    mb = iter.Next(mb);
    if (mb == nullptr) break;
    data_num += mb->size();
    if (data_num % 1000 > print_thresh) {
      cout << data_num << " examples converted\r";
      print_thresh += 1000;
    }

    for (int i = 0; i < mb->size(); ++i) {
      writer->Write((*mb)[i]);
    }
  }
  cout << data_num << " examples converted\n";
  writer->Close();
  delete writer;
  return ret;
}

int shuffle(const std::string& src_path, const std::string& src_type,
            const std::string& output_path, const std::string& output_type_) {
  string output_type = output_type_;
  if (output_type.length() == 0) output_type = src_type;

  DataIter iter;
  int ret = iter.AddReader(src_path, src_type);
  if (ret != Status_OK) return ret;

  DataWriter* writer = DataWriter::Create(output_type);
  if (writer == nullptr) {
    ret = Status_Invalid_Argument;
    return ret;
  }
  ret = writer->Open(output_path);
  if (ret != Status_OK) {
    delete writer;
    return ret;
  }

  MiniBatch* mb = nullptr;
  vector<DataPoint*> data_list;

  size_t data_num = 0;
  index_t feat_dim = 0;
  while (true) {
    mb = iter.Next(mb);
    if (mb == nullptr) break;
    for (int i = 0; i < mb->size(); ++i) {
      DataPoint* dp = new DataPoint();
      (*mb)[i].Clone(*dp);
      data_list.push_back(dp);

      if (feat_dim < dp->dim()) feat_dim = dp->dim();
    }
    data_num += mb->size();
  }

  cout << data_num << " examples loaded\n";
  random_device rd;
  mt19937 g(rd());
  std::shuffle(data_list.begin(), data_list.end(), g);
  writer->SetExtraInfo((char*)(&feat_dim));
  for (auto& data : data_list) {
    writer->Write(*data);
    delete data;
  }
  writer->Close();
  delete writer;
  return ret;
}

int split(const string& src_path, const string& src_type, int fold_num,
          const string& output_prefix, const string& dst_type, bool shuffle) {
  DataIter iter;
  int ret = iter.AddReader(src_path, src_type);
  if (ret != Status_OK) return ret;

  MiniBatch* mb = nullptr;
  vector<DataPoint*> data_list;

  size_t data_num = 0;
  index_t feat_dim = 0;
  while (true) {
    mb = iter.Next(mb);
    if (mb == nullptr) break;
    for (int i = 0; i < mb->size(); ++i) {
      DataPoint* dp = new DataPoint();
      (*mb)[i].Clone(*dp);
      data_list.push_back(dp);

      if (feat_dim < dp->dim()) feat_dim = dp->dim();
    }
    data_num += mb->size();
  }

  cout << data_num << " examples loaded\n";
  if (data_num == 0) return ret;

  if (shuffle) {
    random_device rd;
    mt19937 g(rd());
    std::shuffle(data_list.begin(), data_list.end(), g);
  }

  size_t data_split_num = size_t(ceil(data_list.size() / float(fold_num)));
  size_t data_idx = 0;
  size_t end_idx = data_split_num;

  for (int i = 0; i < fold_num; ++i) {
    DataWriter* writer = DataWriter::Create(dst_type);
    if (writer == nullptr) {
      ret = Status_Invalid_Argument;
      break;
    }
    ostringstream output_path;
    output_path << output_prefix << i << "." << dst_type;
    ret = writer->Open(output_path.str());
    if (ret != Status_OK) {
      delete writer;
      break;
    }
    fprintf(stderr, "write fold %d to %s\n", i, output_path.str().c_str());
    writer->SetExtraInfo((char*)(&feat_dim));

    for (; data_idx < end_idx && data_idx < data_num; ++data_idx) {
      writer->Write(*(data_list[data_idx]));
      delete data_list[data_idx];
    }
    end_idx += data_split_num;
    writer->Close();
    delete writer;
  }
  return ret;
}
}

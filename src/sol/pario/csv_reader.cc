/*********************************************************************************
*     File Name           : /home/yuewu/work/sol/src/sol/pario/csv_reader.cc
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-13 19:39]
*     Last Modified       :     [2016-02-12 21:23]
*     Description         :
**********************************************************************************/
#include "sol/pario/csv_reader.h"

#include <cstdlib>

#include "sol/pario/numeric_parser.h"

namespace sol {
namespace pario {

CSVReader::CSVReader() : DataFileReader() { this->feat_dim_ = 0; }

int CSVReader::Open(const std::string& path, const char* mode) {
  int ret = DataFileReader::Open(path);
  if (ret == Status_OK) {
    ret = this->LoadFeatDim();
  }
  this->is_good_ = ret == Status_OK ? true : false;
  return ret;
}

void CSVReader::Rewind() {
  DataFileReader::Rewind();
  // read the first line for csv
  this->file_reader_.ReadLine(this->read_buf_, this->read_buf_size_);
}

int CSVReader::Next(DataPoint& dst_data) {
  int ret = this->file_reader_.ReadLine(this->read_buf_, this->read_buf_size_);
  if (ret != Status_OK) return ret;

  char* iter = this->read_buf_, *endptr = nullptr;
  if (*iter == '\0') {
    fprintf(stderr, "incorrect line\n");
    return Status_Invalid_Format;
  }

  dst_data.Clear();
  // 1. parse label
  dst_data.set_label(label_t(NumericParser::ParseInt(iter, endptr)));
  if (endptr == iter) {
    fprintf(stderr, "parse label failed.\n");
    this->is_good_ = false;
    return Status_Invalid_Format;
  }
  iter = endptr;

  // 2. parse features
  dst_data.Reserve(this->feat_dim_);
  index_t index = 1;
  while (*iter != '\0') {
    if (*iter != ',') {
      fprintf(stderr, "incorrect input file (%s)!\n", iter);
      this->is_good_ = false;
      return Status_Invalid_Format;
    }
    ++iter;

    real_t feat = NumericParser::ParseFloat(iter, endptr);
    if (endptr == iter) {
      fprintf(stderr, "parse feature value (%s) failed!\n", iter);
      this->is_good_ = false;
      return Status_Invalid_Format;
    }
    iter = endptr;

    if (feat != 0) {
      dst_data.AddNewFeat(index, feat);
    }
    ++index;
  }

  return ret;
}

int CSVReader::LoadFeatDim() {
  int ret = this->file_reader_.ReadLine(this->read_buf_, this->read_buf_size_);
  if (ret != Status_OK) return ret;
  char* p = this->read_buf_;
  this->feat_dim_ = 0;
  while (*p != '\0') {
    if (*p++ == ',') ++this->feat_dim_;
  }
  ++this->feat_dim_;
  return ret;
}

RegisterDataReader(CSVReader, "csv", "csv format data reader");

}  // namespace pario
}  // namespace sol

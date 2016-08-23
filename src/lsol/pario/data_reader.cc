/*********************************************************************************
*     File Name           :     data_reader.cc
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-11 21:38]
*     Last Modified       :     [2015-11-13 21:32]
*     Description         :     Interface for data reader (svm,binary, etc.)
**********************************************************************************/

#include "lsol/pario/data_reader.h"

#include <cstdlib>

#include "lsol/util/error_code.h"

using namespace std;

namespace lsol {
namespace pario {

DataReader* DataReader::Create(const std::string& type) {
  auto create_func = CreateObject<DataReader>(std::string(type) + "_reader");
  return create_func == nullptr ? nullptr : create_func();
}

DataReader::DataReader() {}
DataReader::~DataReader() {}

DataFileReader::DataFileReader() {
  this->read_buf_size_ = 4096;
  this->read_buf_ = (char*)malloc(this->read_buf_size_ * sizeof(char));
  this->is_good_ = true;
}

DataFileReader::~DataFileReader() {
  this->Close();
  if (this->read_buf_ != nullptr) {
    free(this->read_buf_);
  }
}

int DataFileReader::Open(const string& path, const char* mode) {
  this->Close();
  this->file_path_ = path;
  int ret = this->file_reader_.Open(path.c_str(), mode);
  this->is_good_ = ret == Status_OK ? true : false;

  return ret;
}

}  // namespace pario
}  // namespace lsol

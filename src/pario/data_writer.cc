/*********************************************************************************
*     File Name           :     data_writer.cc
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-14 15:01]
*     Last Modified       :     [2015-11-14 15:03]
*     Description         :     Interface for data writer (svm,binary, etc.)
**********************************************************************************/

#include "sol/pario/data_writer.h"

#include <cstdlib>

#include "sol/util/error_code.h"

using namespace std;

namespace sol {
namespace pario {

DataWriter* DataWriter::Create(const std::string& type) {
  auto create_func = CreateObject<DataWriter>(std::string(type) + "_writer");
  return create_func == nullptr ? nullptr : create_func();
}

DataWriter::DataWriter() { this->is_good_ = true; }

DataWriter::~DataWriter() { this->Close(); }

int DataWriter::Open(const string& path, const char* mode) {
  this->Close();
  this->file_path_ = path;
  int ret = this->file_writer_.Open(path.c_str(), mode);
  this->is_good_ = ret == Status_OK ? true : false;

  return ret;
}

}  // namespace pario
}  // namespace sol

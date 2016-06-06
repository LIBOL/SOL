/*********************************************************************************
*     File Name           :     csv_writer.cc
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-14 15:10]
*     Last Modified       :     [2015-11-14 15:46]
*     Description         :     writer of csv format data
**********************************************************************************/

#include "lsol/pario/csv_writer.h"

#include <sstream>

namespace lsol {
namespace pario {

int CSVWriter::Write(const DataPoint& data) {
  size_t feat_num = data.indexes().size();
  this->file_writer_.Printf("%d", data.label());

  size_t i = 0;
  index_t j = 1;
  for (; i < feat_num && j <= this->feat_dim_; ++j) {
    if (data.indexes(i) == j) {
      this->file_writer_.Printf(",%g", data.features(i++));
    } else {
      this->file_writer_.Printf(",0");
    }
  }
  for (; j <= this->feat_dim_; ++j) this->file_writer_.Printf(",0");

  this->file_writer_.Printf("\n");
  return Status_OK;
}

int CSVWriter::SetExtraInfo(const char* extra_info) {
  if (this->Good() == false) {
    return Status_IO_Error;
  }
  this->feat_dim_ = *((index_t*)(extra_info));
  std::ostringstream oss;
  oss << "class";
  for (index_t i = 0; i < this->feat_dim_; ++i) {
    oss << ",v" << i;
  }
  this->file_writer_.Printf("%s\n", oss.str().c_str());
  return Status_OK;
}

RegisterDataWriter(CSVWriter, "csv", "csv format data writer");

}  // namespace pario
}  // namespace lsol

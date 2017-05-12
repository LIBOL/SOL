/*********************************************************************************
*     File Name           : /home/yuewu/work/sol/src/sol/pario/svm_reader.cc
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-11 22:26]
*     Last Modified       :     [2017-05-12 13:15]
*     Description         :     reader of lilbsvm format data
**********************************************************************************/

#include "sol/pario/svm_reader.h"

#include <cstdlib>

#include "sol/pario/numeric_parser.h"

namespace sol {
namespace pario {

int SVMReader::Next(DataPoint &dst_data) {
  int ret = Status_OK;

  while (true) {
    ret = this->file_reader_.ReadLine(this->read_buf_, this->read_buf_size_);
    if (ret != Status_OK) return ret;
    if (this->read_buf_[0] != '#') break;  // comment line
  }

  char *iter = this->read_buf_, *endptr = nullptr;
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
  while (*iter != '\0') {
    index_t index = (index_t)(NumericParser::ParseUint(iter, endptr));
    if (endptr == iter) {
      // parse index failed
      fprintf(stderr, "parse index value (%s) failed!\n", iter);
      this->is_good_ = false;
      return Status_Invalid_Format;
    }
    iter = endptr;
    if (*iter != ':') {
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

    dst_data.AddNewFeat(index, feat);
  }
  dst_data.Sort();

  return ret;
}

RegisterDataReader(SVMReader, "svm", "libsvm format data reader");

}  // namespace pario
}  // namespace sol

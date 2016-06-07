/*********************************************************************************
*     File Name           : /home/yuewu/work/lsol/src/lsol/pario/svm_reader.cc
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-11 22:26]
*     Last Modified       :     [2015-11-14 17:16]
*     Description         :     reader of lilbsvm format data
**********************************************************************************/

#include "lsol/pario/svm_reader.h"

#include <cstdlib>

#include "lsol/pario/numeric_parser.h"

namespace lsol {
namespace pario {

int SVMReader::Next(DataPoint &dst_data) {
  int ret = this->file_reader_.ReadLine(this->read_buf_, this->read_buf_size_);
  if (ret != Status_OK) return ret;

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

  return ret;
}

RegisterDataReader(SVMReader, "svm", "libsvm format data reader");

}  // namespace pario
}  // namespace lsol

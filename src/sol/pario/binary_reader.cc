/*********************************************************************************
*     File Name           :     binary_reader.cc
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-13 20:31]
*     Last Modified       :     [2015-11-14 17:16]
*     Description         :     binary format data reader
**********************************************************************************/

#include "sol/pario/binary_reader.h"

#include <cstdlib>

#include "sol/pario/compress.h"
#include "sol/util/error_code.h"

namespace sol {
namespace pario {

int BinaryReader::Open(const std::string& path, const char* mode) {
  return DataFileReader::Open(path, "rb");
}

int BinaryReader::Next(DataPoint& dst_data) {
  dst_data.Clear();
  label_t label;
  int ret = this->file_reader_.Read((char*)&label, sizeof(label_t));
  if (ret != Status_OK) return ret;
  dst_data.set_label(label);

  size_t feat_num;
  ret = this->file_reader_.Read((char*)&feat_num, sizeof(feat_num));
  if (ret != Status_OK) {
    fprintf(stderr, "load feature number failed!\n");
    this->is_good_ = false;
    return false;
  }
  if (feat_num > 0) {
    size_t code_len = 0;
    ret = this->file_reader_.Read((char*)&code_len, sizeof(size_t));
    if (ret != Status_OK) {
      fprintf(stderr, "read coded index length failed!\n");
      return Status_Invalid_Format;
    }
    this->comp_codes_.resize(code_len);
    ret = this->file_reader_.Read(this->comp_codes_.begin(), code_len);
    if (ret != Status_OK) {
      fprintf(stderr, "read coded index failed!\n");
      return Status_Invalid_Format;
    }
    dst_data.Resize(feat_num);
    decomp_index(this->comp_codes_, dst_data.indexes());
    if (dst_data.indexes().size() != feat_num) {
      fprintf(stderr, "decoded index number is not correct!\n");
      return Status_Invalid_Format;
    }

    ret = this->file_reader_.Read((char*)dst_data.features().begin(),
                                  sizeof(real_t) * feat_num);
    if (ret != Status_OK) {
      fprintf(stderr, "load features failed!\n");
      return Status_Invalid_Format;
    }
  }
  return ret;
}

RegisterDataReader(BinaryReader, "bin", "binary format data reader");

}  // namespace pario
}  // namespace sol

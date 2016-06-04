/*********************************************************************************
*     File Name           :     binary_writer.cc
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-14 15:34]
*     Last Modified       :     [2015-11-14 15:49]
*     Description         :     binary format data writer
**********************************************************************************/

#include "lsol/pario/binary_writer.h"

#include <cstdlib>

#include "lsol/pario/compress.h"
#include "lsol/util/error_code.h"

namespace lsol {
namespace pario {

int BinaryWriter::Open(const std::string& path, const char* mode) {
    return DataWriter::Open(path, "wb");
}

int BinaryWriter::Write(const DataPoint& data) {
    label_t label = data.label();
    this->file_writer_.Write((char*)&label, sizeof(label));
    size_t feat_num = data.indexes().size();

    this->file_writer_.Write((char*)&feat_num, sizeof(feat_num));
    if (feat_num > 0) {
        this->comp_codes_.Clear();
        comp_index(data.indexes(), this->comp_codes_);
        size_t code_len = this->comp_codes_.size();
        this->file_writer_.Write((char*)&(code_len), sizeof(code_len));
        this->file_writer_.Write(this->comp_codes_.begin(), code_len);
        this->file_writer_.Write((char*)(data.features().begin()),
                                 sizeof(real_t) * feat_num);
    }
    return Status_OK;
}

RegisterDataWriter(BinaryWriter, "bin", "binary format data writer");

}  // namespace pario
}  // namespace lsol

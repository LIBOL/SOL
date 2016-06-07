/*********************************************************************************
*     File Name           :     svm_writer.cc
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-14 15:04]
*     Last Modified       :     [2015-11-14 16:17]
*     Description         :     writer of lilbsvm format data
**********************************************************************************/

#include "lsol/pario/svm_writer.h"

#include <cstdlib>

namespace lsol {
namespace pario {

int SVMWriter::Write(const DataPoint &data) {
    size_t feat_num = data.indexes().size();
    this->file_writer_.Printf("%d", data.label());
    for (size_t i = 0; i < feat_num; ++i) {
        this->file_writer_.Printf(" %d:%g", data.index(i), data.feature(i));
    }
    this->file_writer_.Printf("\n");
    return Status_OK;
}

RegisterDataWriter(SVMWriter, "svm", "libsvm format data writer");

}  // namespace pario
}  // namespace lsol

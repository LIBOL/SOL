/*********************************************************************************
*     File Name           :     svm_writer.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-14 15:03]
*     Last Modified       :     [2015-11-14 15:04]
*     Description         :     writer of lilbsvm format data
**********************************************************************************/

#ifndef LSOL_PARIO_SVM_WRITER_H__
#define LSOL_PARIO_SVM_WRITER_H__

#include <lsol/pario/data_writer.h>

namespace lsol {
namespace pario {

class LSOL_EXPORTS SVMWriter : public DataWriter {
 public:
  /// \brief  Write a new data into the file
  ///
  /// \param data Data to be saved
  ///
  /// \return Status code,  Status_OK if succeed
  virtual int Write(const DataPoint& data);

};  // class SVMWriter

}  // namespace pario
}  // namespace lsol

#endif

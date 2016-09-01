/*********************************************************************************
*     File Name           :     svm_writer.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-14 15:03]
*     Last Modified       :     [2015-11-14 15:04]
*     Description         :     writer of lilbsvm format data
**********************************************************************************/

#ifndef SOL_PARIO_SVM_WRITER_H__
#define SOL_PARIO_SVM_WRITER_H__

#include <sol/pario/data_writer.h>

namespace sol {
namespace pario {

class SOL_EXPORTS SVMWriter : public DataWriter {
 public:
  /// \brief  Write a new data into the file
  ///
  /// \param data Data to be saved
  ///
  /// \return Status code,  Status_OK if succeed
  virtual int Write(const DataPoint& data);

};  // class SVMWriter

}  // namespace pario
}  // namespace sol

#endif

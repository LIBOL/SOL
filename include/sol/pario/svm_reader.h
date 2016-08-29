/*********************************************************************************
*     File Name           :     svm_reader.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-11 22:15]
*     Last Modified       :     [2015-11-13 20:43]
*     Description         :     reader of lilbsvm format data
**********************************************************************************/

#ifndef SOL_PARIO_SVM_READER_H__
#define SOL_PARIO_SVM_READER_H__

#include <sol/pario/data_reader.h>

namespace sol {
namespace pario {

class SOL_EXPORTS SVMReader : public DataFileReader {
 public:
  /// \brief  Read next data point
  ///
  /// \param dst_data Destination data point
  ///
  /// \return  Status code, Status_OK if everything ok, Status_EndOfFile if
  /// read to file end
  virtual int Next(DataPoint& dst_data);
};  // class SVMReader

}  // namespace pario
}  // namespace sol

#endif

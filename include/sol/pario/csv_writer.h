/*********************************************************************************
*     File Name           :     csv_writer.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-14 15:09]
*     Last Modified       :     [2015-11-14 15:46]
*     Description         :     writer of csv format data
**********************************************************************************/

#ifndef SOL_PARIO_CSV_WRITER_H__
#define SOL_PARIO_CSV_WRITER_H__

#include <sol/pario/data_writer.h>

namespace sol {
namespace pario {

class SOL_EXPORTS CSVWriter : public DataWriter {
 public:
  /// \brief  Write a new data into the file
  ///
  /// \param data Data to be saved
  ///
  /// \return Status code,  Status_OK if succeed
  virtual int Write(const DataPoint& data);

  /// \brief  Set extra information for the output format, for example header
  /// of csv
  ///
  /// \param extra_info extra info
  ///
  /// \return Status code,  Status_OK if succeed
  virtual int SetExtraInfo(const char* extra_info);

 protected:
  // dimension of data
  index_t feat_dim_;

};  // class CSVWriter

}  // namespace pario
}  // namespace sol

#endif

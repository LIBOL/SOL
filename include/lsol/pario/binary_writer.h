/*********************************************************************************
*     File Name           :     binary_writer.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-14 15:32]
*     Last Modified       :     [2015-11-14 15:33]
*     Description         :     binary format data writer
**********************************************************************************/

#ifndef LSOL_PARIO_BINARY_READER_H__
#define LSOL_PARIO_BINARY_READER_H__

#include <lsol/pario/data_writer.h>
#include <lsol/pario/array1d.h>

namespace lsol {
namespace pario {

class LSOL_EXPORTS BinaryWriter : public DataWriter {
 public:
  /// \brief  Open a new file
  ///
  /// \param path Path to the file, '-' when if use stdin
  /// \param mode open mode, "wb"
  ///
  /// \return Status code,  Status_OK if succeed
  virtual int Open(const std::string& path, const char* mode = "wb");

 public:
  /// \brief  Write a new data into the file
  ///
  /// \param data Data to be saved
  ///
  /// \return Status code,  Status_OK if succeed
  virtual int Write(const DataPoint& data);

 private:
  // compressed codes of indexes
  Array1d<char> comp_codes_;
};  // class BinaryWriter

}  // namespace pario
}  // namespace lsol

#endif

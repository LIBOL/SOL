/*********************************************************************************
*     File Name           :     binary_writer.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-14 15:32]
*     Last Modified       :     [2015-11-14 15:33]
*     Description         :     binary format data writer
**********************************************************************************/

#ifndef SOL_PARIO_BINARY_READER_H__
#define SOL_PARIO_BINARY_READER_H__

#include <sol/pario/data_writer.h>
#include <sol/math/vector.h>

namespace sol {
namespace pario {

class SOL_EXPORTS BinaryWriter : public DataWriter {
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
  math::Vector<char> comp_codes_;
};  // class BinaryWriter

}  // namespace pario
}  // namespace sol

#endif

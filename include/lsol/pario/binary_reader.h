/*********************************************************************************
*     File Name           :     binary_reader.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-13 20:28]
*     Last Modified       :     [2015-11-13 20:52]
*     Description         :     binary format data reader
**********************************************************************************/

#ifndef LSOL_PARIO_BINARY_READER_H__
#define LSOL_PARIO_BINARY_READER_H__

#include <lsol/pario/data_reader.h>
#include <lsol/math/vector.h>

namespace lsol {
namespace pario {

class LSOL_EXPORTS BinaryReader : public DataFileReader {
 public:
  /// \brief  Open a new file
  ///
  /// \param path Path to the file, '-' when if use stdin
  /// \param mode open mode, "r" or "rb"
  ///
  /// \return Status code,  Status_OK if succeed
  virtual int Open(const std::string& path, const char* mode = "rb");

 public:
  /// \brief  Read next data point
  ///
  /// \param dst_data Destination data point
  ///
  /// \return  Status code, Status_OK if everything ok, Status_EndOfFile if
  /// read to file end
  virtual int Next(DataPoint& dst_data);

 private:
  // compressed codes of indexes
  math::Vector<char> comp_codes_;
};  // class BinaryReader

}  // namespace pario
}  // namespace lsol

#endif

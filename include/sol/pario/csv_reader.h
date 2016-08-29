/*********************************************************************************
*     File Name           :     csv_reader.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-13 19:35]
*     Last Modified       :     [2015-11-13 20:49]
*     Description         :     reader for csv data format
**********************************************************************************/
#ifndef SOL_PARIO_CSV_READER_H__
#define SOL_PARIO_CSV_READER_H__

#include <sol/pario/data_reader.h>

namespace sol {
namespace pario {

class SOL_EXPORTS CSVReader : public DataFileReader {
 public:
  CSVReader();

 public:
  /// \brief  Open a new file
  ///
  /// \param path Path to the file, '-' when if use stdin
  /// \param mode open mode, "r" or "rb"
  ///
  /// \return Status code,  Status_OK if succeed
  virtual int Open(const std::string& path, const char* mode = "r");

  /// \brief  Rewind the dataset to the beginning of the file
  virtual void Rewind();

 public:
  /// \brief  Read next data point
  ///
  /// \param dst_data Destination data point
  ///
  /// \return  Status code, Status_OK if everything ok, Status_EndOfFile if
  /// read to file end
  virtual int Next(DataPoint& dst_data);

 private:
  /// \brief  load the head info of csv data
  ///
  /// \return  Status code, Status_OK if everything ok
  int LoadFeatDim();

 protected:
  // dimension of data
  index_t feat_dim_;
};  // class CSVReader

}  // namespace pario
}  // namespace sol

#endif

/*********************************************************************************
*     File Name           :     data_reader.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-11 21:31]
*     Last Modified       :     [2015-11-13 21:25]
*     Description         :     Interface for data reader (svm,binary, etc.)
**********************************************************************************/

#ifndef SOL_PARIO_DATA_READER_H__
#define SOL_PARIO_DATA_READER_H__

#include <string>

#include <sol/util/reflector.h>
#include <sol/util/error_code.h>
#include <sol/pario/file_reader.h>
#include <sol/pario/data_point.h>

namespace sol {
namespace pario {

class SOL_EXPORTS DataReader {
  DeclareReflectorBase(DataReader);

 public:
  DataReader();
  virtual ~DataReader();

 public:
  /// \brief  Open a new file
  ///
  /// \param path Path to the file, '-' when if use stdin
  /// \param mode open mode, "r" or "rb"
  ///
  /// \return Status code,  Status_OK if succeed
  virtual int Open(const std::string& path, const char* mode = "r") = 0;

  /// \brief Close the reader
  virtual void Close() = 0;

  /// \brief  Check the status of the data handler
  ///
  /// \return True if everything is ok
  virtual bool Good() = 0;

  /// \brief  Rewind the dataset to the beginning of the file
  virtual void Rewind() = 0;

 public:
  /// \brief  Read next data point
  ///
  /// \param dst_data Destination data point
  ///
  /// \return  Status code, Status_OK if everything ok, Status_EndOfFile if
  /// read to file end
  virtual int Next(DataPoint& dst_data) = 0;
};

class SOL_EXPORTS DataFileReader : public DataReader {
 public:
  DataFileReader();
  virtual ~DataFileReader();

 public:
  virtual int Open(const std::string& path, const char* mode = "r");

  /// \brief Close the reader
  virtual void Close() { this->file_reader_.Close(); }

  /// \brief  Check the status of the data handler
  ///
  /// \return True if everything is ok
  virtual bool Good() { return this->is_good_ && this->file_reader_.Good(); }

  /// \brief  Rewind the dataset to the beginning of the file
  virtual void Rewind() { this->file_reader_.Rewind(); }

 public:
  /// \brief  Read next data point
  ///
  /// \param dst_data Destination data point
  ///
  /// \return  Status code, Status_OK if everything ok, Status_EndOfFile if
  /// read to file end
  virtual int Next(DataPoint& dst_data) = 0;

 protected:
  FileReader file_reader_;
  /// \brief  flag to denote whether any parse error occurs
  bool is_good_;
  /// \brief  read buffer
  char* read_buf_;
  int read_buf_size_;
  /// \brief  path to the opened file
  std::string file_path_;

 public:
  const std::string& file_path() const { return file_path_; }
};

#define RegisterDataReader(type, name, descr)                            \
  type* type##_##CreateNewInstance() { return new type(); }              \
  ClassInfo __kClassInfo_##type##__(std::string(name) + "_reader",       \
                                    (void*)(type##_##CreateNewInstance), \
                                    descr);
}  // namespace pario
}  // namespace sol

#endif

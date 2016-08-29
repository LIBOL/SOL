/*********************************************************************************
*     File Name           :     data_writer.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-14 14:46]
*     Last Modified       :     [2015-11-14 15:14]
*     Description         :     Interface for data writer (svm,binary, etc.)
**********************************************************************************/

#ifndef SOL_PARIO_DATA_WRITER_H__
#define SOL_PARIO_DATA_WRITER_H__

#include <string>

#include <sol/util/reflector.h>
#include <sol/util/error_code.h>
#include <sol/pario/file_writer.h>
#include <sol/pario/data_point.h>

namespace sol {
namespace pario {

class SOL_EXPORTS DataWriter {
  DeclareReflectorBase(DataWriter);

 public:
  DataWriter();
  virtual ~DataWriter();

 public:
  /// \brief  Open a new file
  ///
  /// \param path Path to the file, '-' when if use stdout
  /// \param mode open mode, "w" or "wb"
  ///
  /// \return Status code,  Status_OK if succeed
  virtual int Open(const std::string& path, const char* mode = "w");

  /// \brief Close the reader
  virtual void Close() { this->file_writer_.Close(); }

  /// \brief  Check the status of the data handler
  ///
  /// \return True if everything is ok
  virtual bool Good() { return this->is_good_ && this->file_writer_.Good(); }

 public:
  /// \brief  Write a new data into the file
  ///
  /// \param data Data to be saved
  ///
  /// \return Status code,  Status_OK if succeed
  virtual int Write(const DataPoint& data) = 0;

  /// \brief  Set extra information for the output format, for example header
  /// of csv
  ///
  /// \param extra_info extra info
  ///
  /// \return Status code,  Status_OK if succeed
  virtual int SetExtraInfo(const char* extra_info) {
    return Status_OK;
  };

 protected:
  FileWriter file_writer_;
  /// \brief  flag to denote whether any parse error occurs
  bool is_good_;
  /// \brief  path to the opened file
  std::string file_path_;

 public:
  const std::string& file_path() const { return file_path_; }
};

#define RegisterDataWriter(type, name, descr)                            \
  type* type##_##CreateNewInstance() { return new type(); }              \
  ClassInfo __kClassInfo_##type##__(std::string(name) + "_writer",       \
                                    (void*)(type##_##CreateNewInstance), \
                                    descr);
}  // namespace pario
}  // namespace sol

#endif

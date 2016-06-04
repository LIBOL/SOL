/*********************************************************************************
*     File Name           :     data_writer.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-14 14:46]
*     Last Modified       :     [2015-11-14 15:14]
*     Description         :     Interface for data writer (svm,binary, etc.)
**********************************************************************************/

#ifndef LSOL_PARIO_DATA_WRITER_H__
#define LSOL_PARIO_DATA_WRITER_H__

#include <string>

#include "lsol/util/reflector.h"
#include "lsol/util/error_code.h"
#include "lsol/pario/file_writer.h"
#include "lsol/pario/data_point.h"

namespace lsol {
namespace pario {

class LSOL_EXPORTS DataWriter {
public:
    /// \brief  Create a new data writer according to the name of the reader
    ///
    /// \param type Type of data format (svm, csv, etc.)
    ///
    /// \return Pointer to the created data writer instance
    static DataWriter* Create(const std::string& type);

public:
    /// \brief  Create New instance of data reader protocol
    /// \return pointer to the base type of new instance
    typedef DataWriter* (*CreateFunction)();

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
    inline virtual void Close() { this->file_writer_.Close(); }

    /// \brief  Check the status of the data handler
    ///
    /// \return True if everything is ok
    inline virtual bool Good() {
        return this->is_good_ && this->file_writer_.Good();
    }

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
    virtual int SetExtraInfo(const char* extra_info) { return Status_OK; };

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
  type* type##_##CreateNewInstance() { return new type(); }                  \
  ClassInfo __kClassInfo_##type##__(std::string(name) + "_writer", (void*)(type##_##CreateNewInstance), \
                                  descr);
}  // namespace pario
}  // namespace lsol

#endif

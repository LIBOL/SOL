/*********************************************************************************
*     File Name           :     file_writer.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-10-17 10:51]
*     Last Modified       :     [2015-11-14 15:25]
*     Description         :     basic file writer
**********************************************************************************/

#ifndef LSOL_PARIO_FILE_WRITER_H__
#define LSOL_PARIO_FILE_WRITER_H__

#include <cstdio>

#include "lsol/util/types.h"

namespace lsol {
namespace pario {

class LSOL_EXPORTS FileWriter {
public:
    FileWriter();
    FileWriter(const char* path, const char* mode);
    ~FileWriter();

public:
    /**
     * \brief  open a file to write
     *
     * \param path Path to the file, set to '-' if write to stdout
     * \param mode 'w', 'wb', 'w+', 'w+b', 'a', 'ab,', 'a+', or 'a+b'
     *
     * \return Status code, Status_OK if succeed
     */
    int Open(const char* path, const char* mode);

    /**
     * \brief  Close the file
     */
    void Close();

    /**
     * Good : Test if the file writer is good
     *
     * \return: true of good
     */
    bool Good();

public:
    /**
     * \brief  Write the data with specified length to file
     *
     * \param src_buf source buffer to store the data
     * \param length Length of data in size of char to be written
     *
     * \return Status code, Status_OK if succeed
     */
    int Write(char* src_buf, size_t length);

    /// \brief  Wrapper for fprintf
    ///
    /// \param format format string
    /// \param ... Formated data
    ///
    /// \return Status code, Status_OK if succeed
    int Printf(const char* format, ...);

private:
    FILE* file_;
};  // class FileWriter

}  // namespace pario
}  // namespace lsol

#endif

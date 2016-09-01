/*********************************************************************************
*     File Name           :     file_reader.h
*     Created By          :     yuewu
*     Creation Date       :     [2015-10-16 23:31]
*     Last Modified       :     [2015-11-12 18:01]
*     Description         :     basic file reader
**********************************************************************************/

#ifndef SOL_PARIO_FILE_READER_H__
#define SOL_PARIO_FILE_READER_H__

#include <cstdio>
#include <sol/util/types.h>

namespace sol {
namespace pario {

class SOL_EXPORTS FileReader {
  enum ReadMode {
    kUnknown = 0,
    kText = 1,
    kBinary = 2
  };

 public:
  FileReader();
  FileReader(const char* path, const char* mode);
  ~FileReader();

 public:
  /**
   * \brief  open a file to read
   *
   * \param path Path to the file, set to '-' if read from stdin
   * \param mode 'r' or 'rb'
   *
   * \return Status code, Status_OK if succeed
   */
  int Open(const char* path, const char* mode);

  /**
   * \brief  Close the file
   */
  void Close();

  /**
   * \brief  Rewind the file reader to the beginning of the file
   */
  void Rewind();

  /**
   * Good : Test if the file reader is good
   *
   * \return: true of good
   */
  bool Good();

 public:
  /**
   * \brief  Read the data from file with specified length
   *
   * \param length Length of data in size of char to be read
   * \param dst Destination buffer to store the data
   *
   * \return Status code, Status_OK if succeed
   */
  int Read(char* dst, size_t length);

  /**
   * \brief  Read a line from
   *
   * \param dst Destination buffer to store the data
   * \param dst_len length of the destination buffer, note `dst` may be
   * reallocated to store one line
   *
   * \return   Status code, Status_OK if succeed
   */
  int ReadLine(char*& dst, int& dst_len);

 private:
  FILE* file_;
  ReadMode mode_;
};  // class FileReader

}  // namespace pario
}  // namespace sol
#endif

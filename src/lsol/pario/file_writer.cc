/*********************************************************************************
*     File Name           :     file_writer.cc
*     Created By          :     yuewu
*     Creation Date       :     [2015-10-17 11:02]
*     Last Modified       :     [2015-11-14 15:49]
*     Description         :     basic file writer
**********************************************************************************/

#include "lsol/pario/file_writer.h"

#include <cstring>
#include <cstdarg>
#include <iostream>

#include "lsol/util/util.h"
#include "lsol/util/error_code.h"

using namespace std;

namespace lsol {
namespace pario {

FileWriter::FileWriter() : file_(nullptr) {}
FileWriter::FileWriter(const char* path, const char* mode) : file_(nullptr) {
  this->Open(path, mode);
}

FileWriter::~FileWriter() { this->Close(); }

int FileWriter::Open(const char* path, const char* mode) {
  this->Close();
  // open file
  if (strcmp(path, "-") == 0) {
    this->file_ = stdout;
    path = "stdout";
  } else {
    this->file_ = open_file(path, mode);
  }

  if (this->file_ == nullptr || this->Good() == false) {
    this->Close();
    fprintf(stderr, "Error: open file (%s) failed.\n", path);
    return Status_IO_Error;
  }

  return Status_OK;
}

void FileWriter::Close() {
  if (this->file_ != nullptr && this->file_ != stdout) {
    fclose(this->file_);
  }
  this->file_ = nullptr;
}

bool FileWriter::Good() {
  // we do not need to handle eof here, when eof is set, ferror still returns
  // 0
  return this->file_ != nullptr && ferror(this->file_) == 0;
}

int FileWriter::Write(char* src_buf, size_t length) {
  size_t write_len = fwrite(src_buf, 1, length, this->file_);
  if (write_len == length) {
    return Status_OK;
  } else {
	  cerr << "Error " << Status_IO_Error << ": only " << write_len << " bytes are written while " << length << " bytes are specified.\n";
    return Status_IO_Error;
  }
}

int FileWriter::Printf(const char* format, ...) {
  va_list argptr;
  va_start(argptr, format);

  int ret = Status_OK;
  if ((ret = vfprintf(this->file_, format, argptr)) < 0) {
    fprintf(stderr, "vfprintf failed in %s:%d\n", __FILE__, __LINE__);
    ret = Status_IO_Error;
  }
  va_end(argptr);
  return ret;
}

}  // namespace pario
}  // namespace lsol

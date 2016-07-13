/*********************************************************************************
*     File Name           :     file_reader.cc
*     Created By          :     yuewu
*     Creation Date       :     [2015-10-16 23:51]
*     Last Modified       :     [2015-11-13 20:26]
*     Description         :
**********************************************************************************/

#include "lsol/pario/file_reader.h"

#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <memory>
#include <iostream>

#include "lsol/util/util.h"
#include "lsol/util/error_code.h"

using namespace std;

namespace lsol {
namespace pario {

FileReader::FileReader() : file_(nullptr), mode_(kUnknown) {}
FileReader::FileReader(const char* path, const char* mode)
    : file_(nullptr), mode_(kUnknown) {
  this->Open(path, mode);
}

FileReader::~FileReader() { this->Close(); }

int FileReader::Open(const char* path, const char* mode) {
  this->Close();
  // set mode
  if (strcmp(mode, "r") == 0) {
    this->mode_ = kText;
  } else if (strcmp(mode, "rb") == 0) {
    this->mode_ = kBinary;
  } else {
    fprintf(stderr, "unrecognized mode (%s).\n", mode);
    return Status_Invalid_Argument;
  }

  // open file
  if (strcmp(path, "-") == 0) {
    this->file_ = stdin;
    path = "stdin";
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

void FileReader::Close() {
  if (this->file_ != nullptr && this->file_ != stdin) {
    fclose(this->file_);
  }
  this->file_ = nullptr;
  this->mode_ = kUnknown;
}

void FileReader::Rewind() {
  if (this->file_ != nullptr) rewind(this->file_);
}

bool FileReader::Good() {
  // we do not need to handle eof here, when eof is set, ferror still returns
  // 0
  return this->file_ != nullptr && ferror(this->file_) == 0;
}

int FileReader::Read(char* dst, size_t length) {
  size_t read_len = fread(dst, 1, length, this->file_);
  if (read_len == length) {
    return Status_OK;
  } else if (feof(this->file_)) {
    return Status_EndOfFile;
  } else {
    cerr << "Error " << Status_IO_Error << ": only " << read_len
         << " bytes are read while " << length << " bytes are specified.\n";
    return Status_IO_Error;
  }
}

int FileReader::ReadLine(char*& dst, int& dst_len) {
  if (this->mode_ != kText) {
    throw logic_error(
        "ReadLine can only be called when only file is opened with `text` "
        "mode.\n");
  }
  int len(0);
  if (fgets(dst, dst_len, this->file_) == nullptr) {
    if (feof(this->file_)) {
      return Status_EndOfFile;
    } else {
      fprintf(stderr, "Error %d: read line failed\n", Status_IO_Error);
      return Status_IO_Error;
    }
  }
  while (strrchr(dst, '\n') == nullptr) {
    dst_len *= 2;
    dst = (char*)realloc(dst, dst_len);
    len = int(strlen(dst));
    if (fgets(dst + len, dst_len - len, this->file_) == nullptr) break;
  }
  return Status_OK;
}

}  // namespace pario
}  // namespace lsol

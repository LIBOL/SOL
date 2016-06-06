/*********************************************************************************
*     File Name           :     test_file_reader.cc
*     Created By          :     yuewu
*     Creation Date       :     [2015-10-17 00:26]
*     Last Modified       :     [2015-10-21 11:43]
*     Description         :     test file reader
**********************************************************************************/
#include <cstring>
#include <string>
#include <cstdlib>

#include "lsol/pario/file_reader.h"
#include "lsol/util/error_code.h"

using namespace lsol;
using namespace lsol::pario;
using namespace std;

int main(int argc, char** args) {
// check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
  int tmpFlag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
  tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
  _CrtSetDbgFlag(tmpFlag);
//_CrtSetBreakAlloc(368);
#endif

  FileReader reader;
  string path = "data/a1a";
  if (argc > 1) {
    path = args[1];
  }

  reader.Open(path.c_str(), "r");
  if (reader.Good() == false) {
    fprintf(stderr, "open file (%s) failed\n", path.c_str());
    return -1;
  }
  printf("test readline\n");
  int buf_len = 1024;
  char* buf = new char[buf_len];
  size_t file_len = 0;
  for (int i = 0; i < 10; ++i) {
    fprintf(stderr, "\tread round %d\t", i);
    file_len = 0;
    while (reader.ReadLine(buf, buf_len) == Status_OK) {
      file_len += strlen(buf);
    }
    if (reader.Good()) {
      printf("%lu bytes read\n", file_len);
      reader.Rewind();
    }
  }
  int status = 0;
  if (reader.Good() == false) {
    status = -1;
  } else {
    reader.Rewind();
  }
  if (status == 0) {
    fprintf(stderr, "test read, file length: %lu\n", file_len);
    buf = (char*)realloc(buf, file_len);
    for (int i = 0; i < 10; ++i) {
      fprintf(stderr, "\tread round %d\t", i);
      while (reader.Read(buf, file_len / 2) == Status_OK) {
      }
      if (reader.Good()) {
        fprintf(stderr, "%lu bytes read\n", file_len);
        reader.Rewind();
      }
    }
    if (reader.Good() == false) {
      status = -1;
    } else {
      reader.Rewind();
    }
  }

  delete[] buf;
  fprintf(stderr, "program exited with code %d\n", status);
  return status;
}

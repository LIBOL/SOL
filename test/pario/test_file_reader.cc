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
#include <iostream>

#include "sol/pario/file_reader.h"
#include "sol/util/error_code.h"

using namespace sol;
using namespace sol::pario;
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
    cerr << "open file (" << path << ") failed\n";
    return -1;
  }
  cout << "test readline\n";
  int buf_len = 1024;
  char* buf = new char[buf_len];
  size_t file_len = 0;
  for (int i = 0; i < 10; ++i) {
    cerr << "\tread round " << i << "\t";
    file_len = 0;
    while (reader.ReadLine(buf, buf_len) == Status_OK) {
      file_len += strlen(buf);
    }
    if (reader.Good()) {
      cout << file_len << " bytes read\n";
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
    cerr << "test read, file length: " << file_len << "\n";
    buf = (char*)realloc(buf, file_len);
    for (int i = 0; i < 10; ++i) {
      cerr << "\tread round " << i << "\t";
      while (reader.Read(buf, file_len / 2) == Status_OK) {
      }
      if (reader.Good()) {
        cerr << file_len << " bytes read\n";
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
  cerr << "program exited with code " << status << "\n";
  return status;
}

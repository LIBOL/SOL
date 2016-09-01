/*********************************************************************************
*     File Name           :     ../../../test/test_file_writer.cc
*     Created By          :     yuewu
*     Creation Date       :     [2015-10-17 11:13]
*     Last Modified       :     [2015-10-17 11:25]
*     Description         :     test file writer
**********************************************************************************/

#include <cstring>
#include <string>
#include <cstdlib>
#include <iostream>

#include "sol/pario/file_writer.h"
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

  FileWriter writer;
  string out_path = "data/a1a.out";
  if (argc > 1) {
    out_path = args[1];
  }

  string in_path = "data/a1a";
  if (argc > 2) {
    in_path = args[2];
  }

  FileReader reader(in_path.c_str(), "r");
  if (reader.Good() == false) {
    fprintf(stderr, "open file (%s) failed\n", in_path.c_str());
    return -1;
  }

  writer.Open(out_path.c_str(), "w");
  if (writer.Good() == false) {
    fprintf(stderr, "open file (%s) failed\n", out_path.c_str());
    return -1;
  }
  int buf_len = 1024;
  char* buf = new char[buf_len];
  size_t file_len = 0;
  while (reader.ReadLine(buf, buf_len) == Status_OK) {
    file_len += strlen(buf);
    writer.Write(buf, strlen(buf));
  }
  int status = 0;
  if (writer.Good() == false || reader.Good() == false) {
    status = -1;
  } else {
    cerr << file_len << "bytes read and write\n";
  }

  delete[] buf;
  fprintf(stderr, "program exited with code %d\n", status);
  return status;
}

/*********************************************************************************
*     File Name           :     test_binary.cc
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-14 16:36]
*     Last Modified       :     [2015-11-14 17:50]
*     Description         :     test binary reader and writer
**********************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <vector>

#include "lsol/pario/data_reader.h"
#include "lsol/pario/data_writer.h"
#include "lsol/util/util.h"

using namespace lsol;
using namespace lsol::pario;
using namespace std;

#define CHECK_EQ(x, y) assert(std::abs((x) - (y)) < 1e-6)

int test_binary(vector<DataPoint>& dps) {
  const char* out_path = "tmp_test_binary_writer.bin";
  DataWriter* writer = DataWriter::Create("bin");
  if (writer == nullptr) {
    cerr << "create binary writer failed!\n";
    return -1;
  }
  if (writer->Open(out_path) != Status_OK) {
    return -1;
  }
  for (const DataPoint& dp : dps) {
    writer->Write(dp);
  }
  delete writer;

  DataReader* reader = DataReader::Create("bin");
  if (reader == nullptr) {
    cerr << "create binary reader failed!\n";
    return -1;
  }
  if (reader->Open(out_path) != Status_OK) {
    return -1;
  }
  vector<DataPoint> dps2;
  DataPoint dp2;
  while (reader->Next(dp2) == Status_OK) {
    dps2.push_back(dp2);
  }

  delete reader;

  // check if dps and dps2 are the same
  if (dps.size() != dps2.size()) {
    cerr << "check svm writer faileds: data point size not the same ("
         << dps.size() << " vs " << dps2.size() << ")!\n";
    return Status_Error;
  }
  for (size_t i = 0; i < dps.size(); ++i) {
    if (dps[i].label() != dps2[i].label()) {
      cerr << "check svm writer failed: label of instance " << i
           << " not the same (" << dps[i].label() << " vs " << dps2[i].label()
           << ")\n";
      return Status_Error;
    }
    for (size_t j = 0; j < dps[i].indexes().size(); ++j) {
      if (dps[i].index(j) != dps2[i].index(j)) {
        cerr << "check svm writer failed: index " << j << " of instance " << i
             << " not the same\n";
        return Status_Error;
      }
      if (dps[i].feature(j) != dps2[i].feature(j)) {
        cerr << "check svm writer failed: feature " << j << " of instance " << i
             << " not the same\n";
        return Status_Error;
      }
    }
  }
  delete_file(out_path);
  return Status_OK;
}

int main() {
// check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
  int tmpFlag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
  tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
  _CrtSetDbgFlag(tmpFlag);
//_CrtSetBreakAlloc(368);
#endif

  const char* path = "data/a1a";
  vector<DataPoint> dps;
  DataReader* reader = DataReader::Create("svm");
  if (reader == nullptr) {
    cerr << "create svm reader failed!\n";
    return -1;
  }
  if (reader->Open(path) != Status_OK) {
    return -1;
  }
  DataPoint dp;
  while (reader->Next(dp) == Status_OK) {
    dps.push_back(dp);
  }

  delete reader;

  int ret = 0;
  if ((ret = test_binary(dps)) == 0) {
    cout << "check binary reader succeed!\n";
  }

  return ret;
}

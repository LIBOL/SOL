/*********************************************************************************
*     File Name           :     test_csv_reader.cc
*     Created By          :     yuewu
*     Creation Date       :     [2015-11-13 19:52]
*     Last Modified       :     [2015-11-14 16:43]
*     Description         :     test csv reader
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

int test_csv_reader(const char* path, vector<DataPoint>& dps) {
  printf("load and parse data\n");
  DataReader* reader = DataReader::Create("csv");
  if (reader == nullptr) {
    fprintf(stderr, "create csv reader failed!\n");
    return -1;
  }
  if (reader->Open(path) != Status_OK) {
    return -1;
  }
  int ret = Status_OK;
  for (int i = 0; i < 5; ++i) {
    DataPoint dp;
    printf("parse line %d\n", i);
    ret = reader->Next(dp);
    switch (i) {
      case 0:
        assert(ret == Status_OK);
        //"-1,0.1,-0.1	,1.5e-1	,-1.5e1 ,-1.5e-1 \n"
        CHECK_EQ(dp.label(), -1);
        CHECK_EQ(dp.feature(0), 0.1);
        CHECK_EQ(dp.feature(1), -0.1);
        CHECK_EQ(dp.feature(2), 1.5e-1);
        CHECK_EQ(dp.feature(3), -1.5e1);
        CHECK_EQ(dp.feature(4), -1.5e-1);
        break;
      case 1:
        assert(ret == Status_OK);
        //"1  ,0.1  ,-0.1	,1.5e-1	,-1.5e1 ,-1.5e-1 \n"
        CHECK_EQ(dp.label(), 1);
        CHECK_EQ(dp.feature(0), 0.1);
        CHECK_EQ(dp.feature(1), -0.1);
        CHECK_EQ(dp.feature(2), 1.5e-1);
        CHECK_EQ(dp.feature(3), -1.5e1);
        CHECK_EQ(dp.feature(4), -1.5e-1);
        break;
      case 2:
        //"1\t,0.1  :-0.1	,1.5e-1	4:-1.5e1 5:-1.5e-1\n"
        assert(ret == Status_Invalid_Format);
        break;
      case 3:
        assert(ret == Status_OK);
        //"1  ,0.1  ,-0.1	, 1.5e-1	,\t-1.5e1 ,-1.5e-1";
        CHECK_EQ(dp.label(), 1);
        CHECK_EQ(dp.feature(0), 0.1);
        CHECK_EQ(dp.feature(1), -0.1);
        CHECK_EQ(dp.feature(2), 1.5e-1);
        CHECK_EQ(int(dp.index(3)), 4);
        CHECK_EQ(dp.feature(3), -1.5e1);
        CHECK_EQ(dp.feature(4), -1.5e-1);
        break;
      case 4:
        assert(ret == Status_EndOfFile);
        break;
      default:
        break;
    }
    if (ret == Status_OK) dps.push_back(dp);
  }

  delete reader;
  return 0;
}

int test_csv_writer(vector<DataPoint>& dps) {
  const char* out_path = "tmp_test_csv_writer.csv";
  DataWriter* writer = DataWriter::Create("csv");
  if (writer == nullptr) {
    fprintf(stderr, "create csv writer failed!\n");
    return -1;
  }
  index_t max_dim = 0;
  for (const DataPoint& dp : dps) {
    if (dp.dim() > max_dim) {
      max_dim = dp.dim();
    }
  }
  if (writer->Open(out_path) != Status_OK) {
    return -1;
  }
  writer->SetExtraInfo((const char*)(&max_dim));
  for (const DataPoint& dp : dps) {
    writer->Write(dp);
  }
  delete writer;

  DataReader* reader = DataReader::Create("csv");
  if (reader == nullptr) {
    fprintf(stderr, "create csv reader failed!\n");
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
    fprintf(stderr, "check csv writer failed!\n");
    return Status_Error;
  }
  for (size_t i = 0; i < dps.size(); ++i) {
    if (dps[i].label() != dps2[i].label()) {
      fprintf(stderr,
              "check csv writer failed: label of instance %lu not the same\n",
              i);
      return Status_Error;
    }
    for (size_t j = 0; j < dps[i].indexes().size(); ++j) {
      if (dps[i].index(j) != dps2[i].index(j)) {
        fprintf(stderr,
                "check csv writer failed: index %lu of instance %lu "
                "not the "
                "same\n",
                j, i);
        return Status_Error;
      }
      if (dps[i].feature(j) != dps2[i].feature(j)) {
        fprintf(stderr,
                "check csv writer failed: feature %lu of instance %lu "
                "not the "
                "same\n",
                j, i);
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

  const char* test_data =
      "class,v1, v2, v3, \t v4, v5\t\n"
      "-1,0.1,-0.1	,1.5e-1	,-1.5e1 ,-1.5e-1 \n"
      "1  ,0.1  ,-0.1	,1.5e-1	,-1.5e1 ,-1.5e-1 \n"
      "1\t,0.1  :-0.1	,1.5e-1	4:-1.5e1 5:-1.5e-1\n"
      "1  ,0.1  ,-0.1	, 1.5e-1	,\t-1.5e1 ,-1.5e-1";

  printf("write test data to disk\n");

  const char* out_path = "tmp_test_csv_reader.csv";
  ofstream out_file(out_path, ios::out);
  if (!out_file) {
    fprintf(stderr, "open %s failed!\n", out_path);
    return -1;
  }
  out_file << test_data;
  out_file.close();

  vector<DataPoint> dps;
  int ret = 0;
  if ((ret = test_csv_reader(out_path, dps)) == 0) {
    printf("%lu features loaded\n", dps.size());
    for (const DataPoint& dp : dps) {
      printf("%d", dp.label());
      for (size_t i = 0; i < dp.indexes().size(); ++i) {
        printf(" %d:%g", dp.index(i), dp.feature(i));
      }
      printf("\n");
    }
    printf("check csv reader succeed!\n");
  }
  delete_file(out_path, true);

  if ((ret = test_csv_writer(dps)) == Status_OK) {
    printf("check csv writer succeed!\n");
  }
  return ret;
}

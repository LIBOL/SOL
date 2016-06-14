/*********************************************************************************
*     File Name           :     concat.cc
*     Created By          :     yuewu
*     Description         :     concatenate datasets
**********************************************************************************/
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

#include <lsol/lsol.h>
#include <lsol/util/str_util.h>
#include <cmdline/cmdline.h>

using namespace lsol;
using namespace lsol::pario;
using namespace std;

int main(int argc, char** argv) {
// check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
  int tmpFlag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
  tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
  _CrtSetDbgFlag(tmpFlag);
//_CrtSetBreakAlloc(231);
#endif

  cmdline::parser parser;
  parser.add<string>("input", 'i', "input data paths, separated by ';'", true);
  parser.add<string>("input_type", 's', "input data type", true);
  parser.add<string>("output", 'o', "output data path", true);
  parser.add<string>("output_type", 'd', "output data type");

  parser.parse_check(argc, argv);

  string src_path = parser.get<string>("input");
  string src_type = parser.get<string>("input_type");
  string dst_path = parser.get<string>("output");
  string dst_type = parser.get<string>("output_type");

  DataWriter* writer = DataWriter::Create(dst_type);
  if (writer == nullptr) {
    return Status_Invalid_Argument;
  }
  int ret = writer->Open(dst_path);
  if (ret != Status_OK) {
    delete writer;
    return ret;
  }

  DataIter iter;
  MiniBatch* mb = nullptr;
  const vector<string>& input_list = split(src_path, ';');

  if (dst_type == "csv") {
    fprintf(stdout, "analyzing feature dimension\n");
    index_t feat_dim = 0;
    for (const string& input_path : input_list) {
      ret = iter.AddReader(input_path, src_type);
      if (ret != Status_OK) return ret;
    }
    while (true) {
      mb = iter.Next(mb);
      if (mb == nullptr) break;
      for (size_t i = 0; i < mb->size(); ++i) {
        DataPoint& dp = (*mb)[i];
        if (feat_dim < dp.dim()) feat_dim = dp.dim();
      }
    }
    fprintf(stdout, "total dimension: %llu\n", size_t(feat_dim));
    writer->SetExtraInfo((char*)(&feat_dim));
  }

  size_t data_num = 0;
  int print_thresh = 10000;
  for (const string& input_path : input_list) {
    ret = iter.AddReader(input_path, src_type);
    if (ret != Status_OK) return ret;
  }
  while (true) {
    mb = iter.Next(mb);
    if (mb == nullptr) break;
    data_num += mb->size();
    for (size_t i = 0; i < mb->size(); ++i) {
      writer->Write((*mb)[i]);
    }

    if (data_num > print_thresh) {
      fprintf(stdout, "%llu examples concatenated\r", data_num);
      print_thresh += 10000;
    }
  }

  writer->Close();
  delete writer;
  fprintf(stdout, "%llu examples concatenated to %s\n", data_num,
          dst_path.c_str());
  return ret;
}

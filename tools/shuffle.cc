/*********************************************************************************
*     File Name           :     shuffle.cc
*     Created By          :     yuewu
*     Description         :     shuffle file
**********************************************************************************/

#include <string>
#include <vector>
#include <algorithm>
#include <random>

#include <lsol/lsol.h>
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
  parser.add<string>("input", 'i', "input data path", true);
  parser.add<string>("input_type", 's', "input data type", true);
  parser.add<string>("output", 'o', "output data path", false, "", "-");

  parser.parse_check(argc, argv);

  string src_path = parser.get<string>("input");
  string src_type = parser.get<string>("input_type");
  string output_path = parser.get<string>("output");

  DataIter iter;
  int ret = iter.AddReader(src_path, src_type);
  if (ret != Status_OK) return ret;

  DataWriter* writer = DataWriter::Create(src_type);
  if (writer == nullptr) {
    ret = Status_Invalid_Argument;
    return ret;
  }
  ret = writer->Open(output_path);
  if (ret != Status_OK) {
    delete writer;
    return ret;
  }

  MiniBatch* mb = nullptr;
  vector<DataPoint*> data_list;

  size_t data_num = 0;
  index_t feat_dim = 0;
  while (true) {
    mb = iter.Next(mb);
    if (mb == nullptr) break;
    for (size_t i = 0; i < mb->size(); ++i) {
      DataPoint* dp = new DataPoint();
      (*mb)[i].Clone(*dp);
      data_list.push_back(dp);

      if (feat_dim < dp->dim()) feat_dim = dp->dim();
    }
    data_num += mb->size();
  }

  fprintf(stdout, "%llu examples loaded\n", data_num);
  random_device rd;
  mt19937 g(rd());
  std::shuffle(data_list.begin(), data_list.end(), g);
  writer->SetExtraInfo((char*)(&feat_dim));
  for (auto& data : data_list) {
    writer->Write(*data);
    delete data;
  }
  writer->Close();
  delete writer;
  return ret;
}

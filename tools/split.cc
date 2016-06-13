/*********************************************************************************
*     File Name           :     shuffle.cc
*     Created By          :     yuewu
*     Description         :     split file into folds
**********************************************************************************/

#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>

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
  parser.add<int>("fold", 'n', "split number", true);
  parser.add<string>("output_prefix", 'o', "output prefix", true);
  parser.add<string>("output_type", 'd', "output data type");
  parser.add("shuffle", 'r', "shuffle the input file");

  parser.parse_check(argc, argv);

  string src_path = parser.get<string>("input");
  string src_type = parser.get<string>("input_type");
  int fold_num = parser.get<int>("fold");
  string output_prefix = parser.get<string>("output_prefix");
  string dst_type = parser.get<string>("output_type");
  bool shuffle = parser.exist("shuffle") ? true : false;

  DataIter iter;
  int ret = iter.AddReader(src_path, src_type);
  if (ret != Status_OK) return ret;

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
  if (data_num == 0) return ret;

  if (shuffle) {
    random_device rd;
    mt19937 g(rd());
    std::shuffle(data_list.begin(), data_list.end(), g);
  }

  size_t data_split_num = size_t(ceil(data_list.size() / float(fold_num)));
  size_t data_idx = 0;
  size_t end_idx = data_split_num;

  for (int i = 0; i < fold_num; ++i) {
    DataWriter* writer = DataWriter::Create(dst_type);
    if (writer == nullptr) {
      ret = Status_Invalid_Argument;
      break;
    }
    ostringstream output_path;
	output_path << output_prefix << i << "." << dst_type;
    ret = writer->Open(output_path.str());
    if (ret != Status_OK) {
      delete writer;
      break;
    }
	fprintf(stderr, "write fold %d to %s\n", i, output_path.str().c_str());
    writer->SetExtraInfo((char*)(&feat_dim));

    for (; data_idx < end_idx && data_idx < data_num; ++data_idx) {
      writer->Write(*(data_list[data_idx]));
	  delete data_list[data_idx];
    }
    end_idx += data_split_num;
    writer->Close();
    delete writer;
  }
  return ret;
}

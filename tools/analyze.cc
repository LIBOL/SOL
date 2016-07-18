/*********************************************************************************
*     File Name           :     analyze.cc
*     Created By          :     yuewu
*     Description         :     analyze data
**********************************************************************************/
#include <string>
#include <map>

#include <lsol/lsol.h>
#include <lsol/math/vector.h>
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

  DataIter data_iter;
  int ret = data_iter.AddReader(src_path, src_type);
  if (ret != Status_OK) return ret;

  MiniBatch* mb = nullptr;

  size_t data_num = 0;
  size_t feat_num = 0;
  size_t feat_dim = 0;
  math::Vector<char> feat_flags;
  map<int, size_t> map_class_sample_num;

  size_t print_thresh = 10000;
  while (true) {
    mb = data_iter.Next(mb);
    if (mb == nullptr) break;
    data_num += mb->size();

    for (int i = 0; i < mb->size(); ++i) {
      DataPoint& dp = (*mb)[i];

      if (feat_dim < dp.dim()) {
        feat_dim = dp.dim();
      }

      size_t prev_size = feat_flags.size();
      if (feat_dim > prev_size) {
        feat_flags.resize(feat_dim);
        // set the new value to zero
        for (char* iter = feat_flags.begin() + prev_size;
             iter != feat_flags.end(); ++iter) {
          *iter = 0;
        }
      }
      for (size_t i = 0; i < dp.size(); i++) {
        feat_flags[dp.index(i)] = 1;
      }
      feat_num += dp.size();

      int label = dp.label();
      if (map_class_sample_num.find(label) != map_class_sample_num.end()) {
        map_class_sample_num[label] += 1;
      } else {
        map_class_sample_num[label] = 1;
      }
    }

    if (data_num > print_thresh) {
      cout << data_num << " examples analyzed\r";
      print_thresh += 10000;
    }
  }
  cout << data_num << " examples analyzed\n";

  size_t valid_dim = 0;
  for (size_t i = 0; i < feat_flags.size(); i++) {
    if (feat_flags[i] == 1) valid_dim++;
  }
  FileWriter fw;
  if ((ret = fw.Open(output_path.c_str(), "w")) != Status_OK) {
    cerr << "Write analysis result to " << output_path << " failed\n";
    return ret;
  }
  fw.Printf("data number  : %lu\n", data_num);
  fw.Printf("feat number  : %lu\n", feat_num);
  fw.Printf("dimension    : %lu\n", feat_dim - 1);
  fw.Printf("nonzero feat : %lu\n", valid_dim);
  fw.Printf("class num    : %lu\n", map_class_sample_num.size());
  if (feat_dim > 0) {
    fw.Printf("data sparsity: %.2lf%%\n", 100 - valid_dim * 100.0 / feat_dim);
  }
  for (auto& iter : map_class_sample_num) {
    fw.Printf("data number of class %d : %lu\n", iter.first, iter.second);
  }
  return ret;
}

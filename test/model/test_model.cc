/*********************************************************************************
*     File Name           :     test_model.cc
*     Created By          :     yuewu
*     Description         :     test the accuracy of models
**********************************************************************************/
#include <iostream>
#include <map>
#include <fstream>
#include <cmath>
#include <lsol/lsol.h>
#include <lsol/util/str_util.h>

using namespace std;
using namespace lsol;
using namespace lsol::pario;
using namespace lsol::model;

struct Result {
  float accu;
  size_t update_num;
  Result() : accu(0), update_num(0) {}
  Result(float accu, size_t update_num) : accu(accu), update_num(update_num) {}
};

int test_algo(const std::string& algo, const std::string& train_path,
              const std::string& dtype, int cls_num,
              map<string, string>& params, float& accu, size_t& update_num);

int main(int argc, char** argv) {
// check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
  int tmpFlag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
  tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
  _CrtSetDbgFlag(tmpFlag);
//_CrtSetBreakAlloc(231);
#endif
  if (argc < 5 || strcmp(argv[1], "-h") == 0 ||
      strcmp(argv[1], "--help") == 0) {
    fprintf(stdout,
            "Usage: test_model data_path data_type class_num ground_truth_path "
            "[params=val]\n");
    return -1;
  }
  int class_num = stoi(argv[3]);
  map<string, string> params;
  for (int argi = 5; argi < argc; ++argi) {
    vector<string> parts = split(argv[argi], '=');
    if (parts.size() != 2) {
      fprintf(stderr, "unrecognized argument %s\n", argv[argi]);
      return -1;
    }
    params[strip(parts[0])] = strip(parts[1]);
  }

  // load ground truth
  map<string, Result> res_table;
  ifstream in_file(argv[4], ios::in);
  if (!in_file) {
    fprintf(stderr, "load file %s failed\n", argv[3]);
    return -1;
  }
  int ret = 0;
  string line;
  while (getline(in_file, line)) {
    if (line.length() == 0) break;
    vector<string> parts = split(line, ' ');
    if (parts.size() != 3) {
      fprintf(stderr, "invalid line %s\n", line.c_str());
      ret = 1;
      break;
    }
    res_table[strip(parts[0])] = Result(stof(parts[1]), stoi(parts[2]));
  }
  in_file.close();
  if (ret != 0) return ret;

  float accu;
  size_t update_num;
  for (const auto& pair : res_table) {
    if ((ret = test_algo(pair.first, argv[1], argv[2], class_num, params, accu,
                         update_num)) != Status_OK) {
		cerr << "test algorithm " << pair.first << " failed\n";
      break;
    }
    if (abs(accu - pair.second.accu) > 1e-6 ||
        update_num != pair.second.update_num) {
		cerr << "test algorithm " << pair.first << " failed, accu(" << accu << "), update_num(" << update_num << "), expected: accu(" << pair.second.accu << "), update_num(" << pair.second.update_num << ")\n";
      ret = Status_Error;
      break;
    }
  }
  if (ret == Status_OK) {
    cerr<< "test model succeed\n";
  }
  return ret;
}
float roundp(float f, int precision) {
  float temp = powf(10, float(precision));
  return int(roundf(f * temp)) / temp;
}
int test_algo(const std::string& algo, const std::string& train_path,
              const std::string& dtype, int cls_num,
              map<string, string>& params, float& accu, size_t& update_num) {
	cout << "Test algorithm: " << algo << "\n";
  shared_ptr<Model> model;
  model.reset(Model::Create(algo, cls_num));
  if (model == nullptr) return Status_Invalid_Argument;
  for (const auto& param : params) {
    try {
      model->SetParameter(param.first, param.second);
    } catch (invalid_argument& err) {
      fprintf(stderr, "%s\n", err.what());
      return Status_Invalid_Argument;
    }
  }

  // load data
  DataIter iter;
  int ret = iter.AddReader(train_path, dtype);
  if (ret != Status_OK) return ret;

  double start_time = lsol::get_current_time();
  accu = roundp(model->Train(iter), 4);
  update_num = model->update_num();
  double end_time = lsol::get_current_time();
  fprintf(stdout, "Test algorithm: %s\n", algo.c_str());
  fprintf(stdout, "training accuracy: %.4f\n", accu);
  fprintf(stdout, "training time: %.3f seconds\n", end_time - start_time);
  return Status_OK;
}

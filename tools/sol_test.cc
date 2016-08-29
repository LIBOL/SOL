/*********************************************************************************
*     File Name           :     sol_test.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-03-09 15:17]
*     Last Modified       :     [2016-03-09 17:36]
*     Description         :     test entry of sol
**********************************************************************************/

#include <string>
#include <cstdlib>
#include <memory>

#include <sol/sol.h>
#include <sol/util/str_util.h>
#include <cmdline/cmdline.h>

using namespace sol;
using namespace sol::pario;
using namespace sol::model;
using namespace std;

void getparser(int argc, char** argv, cmdline::parser&);
int test(cmdline::parser& parser);

int main(int argc, char** argv) {
// check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
  int tmpFlag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
  tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
  _CrtSetDbgFlag(tmpFlag);
//_CrtSetBreakAlloc(231);
#endif

  cmdline::parser parser;
  getparser(argc, argv, parser);

  // create model
  return test(parser);
}

int test(cmdline::parser& parser) {
  string model_path, input_path, output_path;
  if (parser.rest().size() == 2) {
    model_path = parser.rest()[0];
    input_path = parser.rest()[1];
  } else if (parser.rest().size() == 3) {
    model_path = parser.rest()[0];
    input_path = parser.rest()[1];
    output_path = parser.rest()[2];
  } else {
    fprintf(stderr, "%s\n", parser.usage().c_str());
    return Status_Invalid_Argument;
  }

  shared_ptr<Model> model(Model::Load(model_path));
  if (model == nullptr) return Status_Invalid_Argument;
  if (parser.exist("filter")) {
    try {
      model->SetParameter("filter", parser.get<string>("filter"));
    }
    catch (invalid_argument& err) {
      fprintf(stderr, "%s\n", err.what());
      return Status_Invalid_Argument;
    }
  }

  // load data
  DataIter iter(parser.get<int>("batchsize"), parser.get<int>("bufsize"));
  int ret = iter.AddReader(input_path, parser.get<string>("format"));
  if (ret != Status_OK) return ret;

  double start_time = sol::get_current_time();
  float err_rate;
  if (!output_path.empty()) {
    ofstream out_file(output_path.c_str(), ios::out);
    err_rate = model->Test(iter, &out_file);
    out_file.close();
  } else {
    err_rate = model->Test(iter, nullptr);
  }
  double end_time = sol::get_current_time();
  fprintf(stdout, "test accuracy: %.4f\n", 1.f - err_rate);
  fprintf(stdout, "test time: %.3f seconds\n", end_time - start_time);
  return Status_OK;
}

void getparser(int argc, char** argv, cmdline::parser& parser) {
  // pario related options
  parser.add<string>("format", 'f', "dataset format", false, "", "svm",
                     cmdline::oneof<string>("csv", "svm", "bin"));
  parser.add<int>("batchsize", 'b', "batch size", false, "", 256);
  parser.add<int>("bufsize", 0, "number of buffered minibatches", false, "", 2);

  parser.add<string>("filter", 0, "filtered features", false);
  parser.add("help", 'h', "print this message");
  parser.footer("model_file test_file [output_file]");

  bool ok = parser.parse(argc, argv);

  if ((argc == 1 && !ok) || parser.exist("help")) {
    fprintf(stderr, "%s\n", parser.usage().c_str());
    exit(0);
  }

  if (!ok) {
    fprintf(stderr, "%s\n", parser.error().c_str());
    fprintf(stderr, "%s\n", parser.usage().c_str());
    exit(1);
  }
}

/*********************************************************************************
*     File Name           :     lsol.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-03-09 15:17]
*     Last Modified       :     [2016-03-09 17:36]
*     Description         :     test entry of lsol
**********************************************************************************/

#include <string>
#include <cstdlib>
#include <memory>

#include <lsol/lsol.h>
#include <lsol/util/str_util.h>
#include <cmdline/cmdline.h>

using namespace lsol;
using namespace lsol::pario;
using namespace lsol::model;
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
  shared_ptr<Model> model;
  if (parser.exist("model")) {
    model.reset(Model::Load(parser.get<string>("model")));
  } else {
    fprintf(stderr, "--model should be specified!\n");
    return Status_Invalid_Argument;
  }
  if (model == nullptr) return Status_Invalid_Argument;
  if (parser.exist("filter")) {
    try {
      model->SetParameter("filter", parser.get<string>("filter"));
    } catch (invalid_argument& err) {
      fprintf(stderr, "%s\n", err.what());
      return Status_Invalid_Argument;
    }
  }

  // load data
  DataIter iter(parser.get<int>("batchsize"), parser.get<int>("bufsize"));
  int ret =
      iter.AddReader(parser.get<string>("input"), parser.get<string>("format"));
  if (ret != Status_OK) return ret;

  float accu;
  if (parser.exist("output")) {
    ofstream out_file(parser.get<string>("output").c_str(), ios::out);
    accu = model->Test(iter, &out_file);
  } else {
    accu = model->Test(iter, nullptr);
  }
  fprintf(stdout, "test accuracy: %.4f\n", accu);
  return Status_OK;
}

void getparser(int argc, char** argv, cmdline::parser& parser) {
  // input & output
  parser.add<string>("input", 'i', "input file", true);
  parser.add<string>("model", 'm', "model to preload, required for test", true);
  parser.add<string>("output", 'o',
                     "output model(train) or predict results(test)", false);

  // pario related options
  parser.add<string>("format", 'f', "dataset format", false, "", "svm",
                     cmdline::oneof<string>("csv", "svm", "bin"));
  parser.add<int>("batchsize", 'b', "batch size", false, "", 256);
  parser.add<int>("bufsize", 0, "number of buffered minibatches", false, "",
                  2);

  parser.add<string>("filter", 0, "filtered features", false);
  parser.add("help", 'h', "print this message");

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

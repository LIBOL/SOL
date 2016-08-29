/*********************************************************************************
*     File Name           :     sol_train.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-03-09 15:17]
*     Last Modified       :     [2016-03-09 17:36]
*     Description         :     train entry of sol
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
int train(cmdline::parser& parser);

int main(int argc, char** argv) {
// check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
  int tmpFlag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
  tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
  _CrtSetDbgFlag(tmpFlag);
//_CrtSetBreakAlloc(793);
#endif

  cmdline::parser parser;
  getparser(argc, argv, parser);
  return train(parser);
}

int train(cmdline::parser& parser) {
  string input_path, output_path;
  if (parser.rest().size() == 1) {
    input_path = parser.rest()[0];
  } else if (parser.rest().size() == 2) {
    input_path = parser.rest()[0];
    output_path = parser.rest()[1];
  } else {
    fprintf(stderr, "%s\n", parser.usage().c_str());
    return Status_Invalid_Argument;
  }

  shared_ptr<Model> model;
  if (parser.exist("model")) {
    model.reset(Model::Load(parser.get<string>("model")));
  } else if (parser.get<string>("algo").length()) {
    model.reset(
        Model::Create(parser.get<string>("algo"), parser.get<int>("classes")));
  } else {
    fprintf(stderr, "either --model or --algo should be specified!\n");
    return Status_Invalid_Argument;
  }
  if (model == nullptr) return Status_Invalid_Argument;

  const string& model_params = parser.get<string>("params");
  if (model_params.length() > 0) {
    for (const string& opt : split(model_params, ';')) {
      const vector<string>& opt_pair = split(opt, '=');
      if (opt_pair.size() != 2) {
        fprintf(stderr, "invalid params: %s\n", opt.c_str());
        return Status_Invalid_Argument;
      }
      try {
        model->SetParameter(strip(opt_pair[0]), strip(opt_pair[1]));
      }
      catch (invalid_argument& err) {
        fprintf(stderr, "%s\n", err.what());
        return Status_Invalid_Argument;
      }
    }
  }

  // load data
  DataIter iter(parser.get<int>("batchsize"), parser.get<int>("bufsize"));
  int ret = iter.AddReader(input_path, parser.get<string>("format"),
                           parser.get<int>("pass"));
  if (ret != Status_OK) return ret;

  cout << "Model Information: \n" << model->model_info() << "\n";
  double start_time = sol::get_current_time();
  float err_rate = model->Train(iter);
  double end_time = sol::get_current_time();
  fprintf(stdout, "training accuracy: %.4f\n", 1.f - err_rate);
  fprintf(stdout, "training time: %.3f seconds\n", end_time - start_time);
  fprintf(stdout, "model sparsity: %.4f%%\n", model->model_sparsity() * 100.f);

  // save model
  if (!output_path.empty()) {
    model->Save(output_path);
    fprintf(stdout, "save time: %.3f seconds\n", get_current_time() - end_time);
  }

  return Status_OK;
}

/// \brief  show classes of a group in a group
void showinfo(const std::string& group) {
  ClassFactory::ClsInfoMapType info_map = ClassFactory::ClassInfoMap();
  for (auto& pair : info_map) {
    const vector<string>& parts = split(pair.first, '_');
    if (parts.back() == group) {
      fprintf(stdout, "%s:\t%s\n", parts[0].c_str(),
              pair.second->descr().c_str());
    }
  }
}

void getparser(int argc, char** argv, cmdline::parser& parser) {
  parser.add<string>(
      "show", 's', "show related information(model, loss, reader, writer)",
      false, "", "",
      cmdline::oneof<string>("", "model", "loss", "reader", "writer"));

  // input & output
  parser.add<string>("format", 'f', "dataset format", false, "io", "svm",
                     cmdline::oneof<string>("csv", "svm", "bin"));
  parser.add<int>("classes", 'c', "class number", false, "io", 2);
  parser.add<int>("pass", 'p', "number of passes", false, "io", 1);
  parser.add<string>("dim", 'd', "dimension of features", false, "io");
  parser.add<int>("batchsize", 'b', "batch size", false, "io", 256);
  parser.add<int>("bufsize", 0, "number of buffered minibatches", false, "io",
                  2);

  // model setting
  parser.add<string>("algo", 'a', "learning algorithm", false, "model", "ogd");
  parser.add<string>("model", 'm', "path to pre-trained model", false, "model");
  parser.add<string>(
      "params", 0, "model parameters, in the format 'param=val;param=val;...'",
      false, "model");

  parser.add("help", 'h', "print this message");
  parser.footer("train_file [model_file]");

  bool ok = parser.parse(argc, argv);
  if (parser.exist("show")) {
    showinfo(parser.get<string>("show"));
    exit(0);
  }

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

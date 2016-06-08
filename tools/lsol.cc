/*********************************************************************************
*     File Name           :     lsol.cc
*     Created By          :     yuewu
*     Creation Date       :     [2016-03-09 15:17]
*     Last Modified       :     [2016-03-09 17:36]
*     Description         :     main entry of lsol
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
int train(cmdline::parser& parser);
int test(cmdline::parser& parser);

int main(int argc, char** argv) {
  cmdline::parser parser;
  getparser(argc, argv, parser);

  // create model
  if (parser.get<string>("task") == "train") {
    return train(parser);
  } else if (parser.get<string>("task") == "test") {
    return test(parser);
  } else {
    fprintf(stderr, "unrecognized task: %s\n",
            parser.get<string>("task").c_str());
    return Status_Invalid_Argument;
  }
}

int train(cmdline::parser& parser) {
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

  try {
    auto& group_options = parser.group_options("model-param");
    for (auto& opt : group_options) {
      if (opt->has_set()) {
        model->SetParameter(opt->name(), parser.get<string>(opt->name()));
      }
    }
  } catch (invalid_argument& err) {
    fprintf(stderr, "%s\n", err.what());
    return Status_Invalid_Argument;
  } catch (cmdline::cmdline_error& err) {
    fprintf(stderr, "%s\n", err.what());
    return Status_Invalid_Argument;
  }

  // load data
  DataIter iter(parser.get<int>("batchsize"), parser.get<int>("bufsize"));
  int ret =
      iter.AddReader(parser.get<string>("input"), parser.get<string>("format"),
                     parser.get<int>("pass"));
  if (ret != Status_OK) return ret;

  model->Train(iter);

  // save model
  if (parser.exist("output")) {
    model->Save(parser.get<string>("output"));
  }

  return Status_OK;
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

  // load data
  DataIter iter(parser.get<int>("batchsize"), parser.get<int>("bufsize"));
  int ret =
      iter.AddReader(parser.get<string>("input"), parser.get<string>("format"));
  if (ret != Status_OK) return ret;

  if (parser.exist("output")) {
    ofstream out_file(parser.get<string>("output").c_str(), ios::out);
    model->Test(iter, &out_file);
  } else {
    model->Test(iter, nullptr);
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
  parser.add<string>("input", 'i', "input file", true);
  parser.add<string>("task", 't', "task(train or test)", false, "", "train",
                     cmdline::oneof<string>("train", "test"));
  parser.add<string>("model", 'm', "model to preload, required for test",
                     false);
  parser.add<string>("output", 'o',
                     "output model(train) or predict results(test)", false);

  // pario related options
  parser.add<string>("format", 'f', "dataset format", false, "io", "svm",
                     cmdline::oneof<string>("csv", "svm", "bin"));
  parser.add<string>("dim", 0, "dimension of features", false, "io");
  parser.add<int>("batchsize", 'b', "batch size", false, "io", 256);
  parser.add<int>("bufsize", 0, "number of buffered minibatches", false, "io",
                  2);
  parser.add<int>("pass", 'p', "number of passes", false, "io", 1);

  // loss setting
  parser.add<string>("loss", 'l', "loss function type", false, "loss", "");

  // model setting
  parser.add<int>("classes", 'c', "class number", false, "model", 2);
  parser.add<string>("algo", 'a', "learning algorithm", false, "model");

  // model parameters
  parser.add<string>("eta", 0, "learning rate", false, "model-param");
  parser.add<string>("power_t", 0, "decaying learning rate", false,
                     "model-param");
  parser.add<string>("t0", 0, "initial iteration number", false, "model-param");
  parser.add<string>("aggressive", 0, "aggressively update", false,
                     "model-param", "true",
                     cmdline::oneof<string>("true", "false"));
  parser.add<string>("lambda", 0, "regularization parameter", false,
                     "model-param");
  parser.add<string>("gamma", 0, "gamma parameter", false, "model-param");
  parser.add<string>("rou", 0, "rou parameter", false, "model-param");
  parser.add<string>("delta", 0, "delta parameter", false, "model-param");

  parser.add<string>(
      "K", 'k',
      "number of k in truncated gradient descent or feature selection", false,
      "model-param");

  parser.add<string>("filter", 0, "filtered features", false, "model-param");
  parser.add<string>("norm", 0, "normalization type", false, "model-param",
                     "none", cmdline::oneof<string>("none", "L1", "L2"));
  parser.add("help", 'h', "print this message");

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
